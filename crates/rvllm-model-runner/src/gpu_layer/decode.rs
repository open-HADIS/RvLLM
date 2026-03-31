//! T=1 single-token decode forward paths.
//!
//! Two deterministic paths, selected by ForwardPath enum at the call site:
//! - `forward_fp8_decode`: cublasLt FP8 GEMMs (requires FP8 weights + cublaslt feature)
//! - `forward_fused_decode`: maximally fused f16 GEMV kernels

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, CudaViewMut, DevicePtr, DevicePtrMut, LaunchConfig};
use half::f16;
use tracing::info;

use rvllm_core::error::{LLMError, Result};
use rvllm_gpu::cublas::CublasHandle;

use super::{GpuLayerInput, GpuLayerWeights, GpuTransformerLayer};

impl GpuTransformerLayer {
    // ================================================================
    // PATH 1: FP8 cublasLt decode (T=1)
    // ================================================================

    #[cfg(feature = "cublaslt")]
    pub(crate) fn forward_fp8_decode(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: &rvllm_gpu::cublaslt_ops::CublasLtOps,
        prev_mlp_out: Option<&CudaSlice<f16>>,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + kv_dim + kv_dim;

        let qkv_fp8 = weights.fused_qkv_fp8
            .ok_or_else(|| LLMError::GpuError("Fp8Decode requires fused_qkv_fp8 weights".into()))?;
        let gu_fp8 = weights.fused_gate_up_fp8
            .ok_or_else(|| LLMError::GpuError("Fp8Decode requires fused_gate_up_fp8 weights".into()))?;
        let down_fp8 = weights.down_proj_fp8
            .ok_or_else(|| LLMError::GpuError("Fp8Decode requires down_proj_fp8 weights".into()))?;
        let o_fp8 = weights.o_proj_fp8
            .ok_or_else(|| LLMError::GpuError("Fp8Decode requires o_proj_fp8 weights".into()))?;
        let cast_fn = self.loader.get_func("cast_f16_to_fp8", "cast_f16_to_fp8_kernel")
            .map_err(|e| LLMError::GpuError(format!("Fp8Decode requires cast_f16_to_fp8 kernel: {e}")))?;

        let fp8_scratch_ptr = input.fp8_input_scratch_ptr;
        if fp8_scratch_ptr == 0 {
            return Err(LLMError::GpuError("Fp8Decode requires non-zero fp8_input_scratch_ptr".into()));
        }

        // Step 1: RMSNorm
        let (normed, residual_ref) = if let Some(prev_mlp) = prev_mlp_out {
            let (n, r) = Self::fused_residual_rmsnorm_f16(
                &self.stream, &self.loader, input.hidden_states, prev_mlp,
                weights.input_layernorm, cfg.rms_norm_eps, 1, hidden)?;
            (n, r)
        } else {
            let n = Self::rms_norm_f16(&self.stream, &self.loader, input.hidden_states,
                weights.input_layernorm, cfg.rms_norm_eps, 1, hidden)?;
            (n, input.hidden_states.clone())
        };

        // Step 2: Cast normed -> FP8, cublasLt FP8 GEMM for QKV
        let cast_cfg = LaunchConfig {
            grid_dim: (((hidden + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        let (normed_ptr, _ng) = DevicePtr::device_ptr(&normed, &self.stream);
        let (qkv_w_ptr, _qg) = DevicePtr::device_ptr(qkv_fp8, &self.stream);
        unsafe {
            self.stream.launch_builder(&cast_fn)
                .arg(&fp8_scratch_ptr).arg(&normed_ptr).arg(&(hidden as i32))
                .launch(cast_cfg)
                .map_err(|e| LLMError::GpuError(format!("cast f16->fp8 qkv: {e}")))?;
        }
        let mut qkv = unsafe { self.stream.alloc::<f16>(qkv_dim) }
            .map_err(|e| LLMError::GpuError(format!("fp8 qkv alloc: {e}")))?;
        let (qkv_out_ptr, _qog) = DevicePtrMut::device_ptr_mut(&mut qkv, &self.stream);
        lt.fp8_gemm_a_bt_raw(1, qkv_dim, hidden, fp8_scratch_ptr, qkv_w_ptr, qkv_out_ptr)?;
        drop(_qog);

        // Step 3: QKV bias
        if let Some(bias) = weights.qkv_bias {
            let mut qkv_view = qkv.slice_mut(..qkv_dim);
            Self::add_bias_f16_view(&self.stream, &self.loader, &mut qkv_view, bias, 1, qkv_dim)?;
        }

        // Step 4: Fused RoPE + KV cache write
        Self::fused_rope_cache_write(&self.stream, &self.loader, &mut qkv,
            input, q_dim, kv_dim, num_heads, num_kv_heads, head_dim, 1)?;

        // Step 5: Attention
        let attn_out = Self::decode_attention_f16io(
            &self.stream, &self.loader,
            &qkv.slice(..q_dim),
            input.key_cache, input.value_cache,
            &input.block_tables, &input.context_lens,
            1, input.num_seqs, num_heads, num_kv_heads, head_dim,
            input.max_context_len, input.block_size)?;

        // Step 6: O-proj via FP8
        let (attn_ptr, _ag) = DevicePtr::device_ptr(&attn_out, &self.stream);
        let (o_w_ptr, _og) = DevicePtr::device_ptr(o_fp8, &self.stream);
        let cast_q_cfg = LaunchConfig {
            grid_dim: (((q_dim + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&cast_fn)
                .arg(&fp8_scratch_ptr).arg(&attn_ptr).arg(&(q_dim as i32))
                .launch(cast_q_cfg)
                .map_err(|e| LLMError::GpuError(format!("cast f16->fp8 oproj: {e}")))?;
        }
        let mut attn_proj = unsafe { self.stream.alloc::<f16>(hidden) }
            .map_err(|e| LLMError::GpuError(format!("fp8 oproj alloc: {e}")))?;
        let (ap_ptr, _apg) = DevicePtrMut::device_ptr_mut(&mut attn_proj, &self.stream);
        lt.fp8_gemm_a_bt_raw(1, hidden, q_dim, fp8_scratch_ptr, o_w_ptr, ap_ptr)?;

        // Step 7: Post-attn fused residual + norm
        drop((_ag, _og, _apg));
        let (normed2, residual) = Self::fused_residual_rmsnorm_f16(
            &self.stream, &self.loader, &residual_ref, &attn_proj,
            weights.post_attention_layernorm, cfg.rms_norm_eps, 1, hidden)?;

        // Step 8: GateUp via FP8
        let gate_up_dim = intermediate * 2;
        let (n2_ptr, _n2g) = DevicePtr::device_ptr(&normed2, &self.stream);
        let (gu_w_ptr, _gug) = DevicePtr::device_ptr(gu_fp8, &self.stream);
        unsafe {
            self.stream.launch_builder(&cast_fn)
                .arg(&fp8_scratch_ptr).arg(&n2_ptr).arg(&(hidden as i32))
                .launch(cast_cfg)
                .map_err(|e| LLMError::GpuError(format!("cast f16->fp8 gateup: {e}")))?;
        }
        let mut gate_up = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
            .map_err(|e| LLMError::GpuError(format!("fp8 gateup alloc: {e}")))?;
        let (gu_out_ptr, _guog) = DevicePtrMut::device_ptr_mut(&mut gate_up, &self.stream);
        lt.fp8_gemm_a_bt_raw(1, gate_up_dim, hidden, fp8_scratch_ptr, gu_w_ptr, gu_out_ptr)?;

        // Step 9: SiLU * mul
        drop((_n2g, _gug, _guog));
        let fused_act = Self::fused_silu_mul_f16_split(&self.stream, &self.loader, &gate_up, intermediate)?;

        // Step 10: Down via FP8
        let (fa_ptr, _fag) = DevicePtr::device_ptr(&fused_act, &self.stream);
        let (d_w_ptr, _dg) = DevicePtr::device_ptr(down_fp8, &self.stream);
        let cast_inter_cfg = LaunchConfig {
            grid_dim: (((intermediate + 255) / 256) as u32, 1, 1),
            block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&cast_fn)
                .arg(&fp8_scratch_ptr).arg(&fa_ptr).arg(&(intermediate as i32))
                .launch(cast_inter_cfg)
                .map_err(|e| LLMError::GpuError(format!("cast f16->fp8 down: {e}")))?;
        }
        let mut mlp_out = unsafe { self.stream.alloc::<f16>(hidden) }
            .map_err(|e| LLMError::GpuError(format!("fp8 down alloc: {e}")))?;
        let (mlp_ptr, _mlp_guard) = DevicePtrMut::device_ptr_mut(&mut mlp_out, &self.stream);
        lt.fp8_gemm_a_bt_raw(1, hidden, intermediate, fp8_scratch_ptr, d_w_ptr, mlp_ptr)?;
        drop((_fag, _dg));

        Ok((residual, mlp_out))
    }

    // ================================================================
    // PATH 2: Fused f16 GEMV decode (T=1)
    // ================================================================

    pub(crate) fn forward_fused_decode(
        &self,
        input: &GpuLayerInput<'_>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        prev_mlp_out: Option<&CudaSlice<f16>>,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let num_kv_heads = cfg.num_kv_heads;
        let head_dim = cfg.head_dim;
        let intermediate = cfg.intermediate_size;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let qkv_dim = q_dim + kv_dim + kv_dim;

        let profile = cfg.layer_idx == 0 && std::env::var("RVLLM_PROFILE").is_ok();
        let mut _pt = std::time::Instant::now();
        macro_rules! prof {
            ($label:expr) => {
                if profile {
                    if let Err(e) = self.stream.synchronize() { tracing::warn!("profile sync: {e}"); }
                    let e = _pt.elapsed();
                    info!("PROFILE L0 {}: {:?}", $label, e);
                    _pt = std::time::Instant::now();
                }
            };
        }
        if profile { info!("PROFILE L0 path=f16_fused"); }
        prof!("enter");

        // --- Steps 1-2: Fused add+norm+QKV GEMV ---
        let fused_qkv_w = weights.fused_qkv.expect("FusedDecode requires fused_qkv weight");
        let (mut qkv, residual_ref, bias_fused) = if let Some(prev_mlp) = prev_mlp_out {
            self.fused_add_norm_qkv(input.hidden_states, prev_mlp, weights, fused_qkv_w,
                hidden, qkv_dim)?
        } else {
            self.fused_norm_qkv(input.hidden_states, weights, fused_qkv_w,
                hidden, qkv_dim)?
        };

        prof!("norm+qkv");

        // --- Step 3: QKV bias (if not already fused into the GEMV kernel) ---
        if !bias_fused {
            if let Some(bias) = weights.qkv_bias {
                let mut qkv_view = qkv.slice_mut(..qkv_dim);
                Self::add_bias_f16_view(&self.stream, &self.loader, &mut qkv_view, bias, 1, qkv_dim)?;
            }
        }
        prof!("qkv_bias");

        // --- Steps 4-5: Fused RoPE + KV cache write ---
        Self::fused_rope_cache_write(&self.stream, &self.loader, &mut qkv,
            input, q_dim, kv_dim, num_heads, num_kv_heads, head_dim, 1)?;
        prof!("rope+cache");

        // --- Step 6: Attention (FA3) ---
        let attn_out = Self::decode_attention_f16io(
            &self.stream, &self.loader,
            &qkv.slice(..q_dim),
            input.key_cache, input.value_cache,
            &input.block_tables, &input.context_lens,
            1, input.num_seqs, num_heads, num_kv_heads, head_dim,
            input.max_context_len, input.block_size)?;
        prof!("attention");

        // --- Steps 7-10: O-proj + post-norm + GateUp + SiLU + Down ---
        let (residual, mlp_out) = self.fused_oproj_mlp(
            &attn_out, &residual_ref, weights, blas, lt,
            q_dim, hidden, intermediate)?;
        prof!("oproj+mlp");

        Ok((residual, mlp_out))
    }

    // ----------------------------------------------------------------
    // Decode-specific sub-routines (not shared with batched path)
    // ----------------------------------------------------------------

    /// Fused add + RMSNorm + QKV GEMV for layers with prev_mlp_out.
    /// Returns (qkv, residual, bias_was_fused).
    fn fused_add_norm_qkv(
        &self,
        hidden_states: &CudaSlice<f16>,
        prev_mlp: &CudaSlice<f16>,
        weights: &GpuLayerWeights<'_>,
        fused_qkv_w: &CudaSlice<f16>,
        hidden: usize,
        qkv_dim: usize,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>, bool)> {
        let norm_w = weights.input_layernorm;
        let eps = self.config.rms_norm_eps;

        // FP8 fused variant (halves weight bandwidth)
        if let (Some(fp8_w), Some(fp8_s), Ok(ref fk)) = (
            weights.fused_qkv_fp8,
            weights.fused_qkv_fp8_scale,
            self.loader.get_func("gemv_fp8", "fused_add_norm_fp8_gemv_kernel"),
        ) {
            return self.launch_fused_gemv_2out(fk, hidden_states, prev_mlp, norm_w,
                fp8_w, Some(fp8_s), None, eps, hidden, qkv_dim, false);
        }

        // f16 bias-fused variant
        if let (Some(qkv_bias), Ok(ref fk)) = (
            weights.qkv_bias,
            self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_bias_gemv"),
        ) {
            return self.launch_fused_gemv_2out(fk, hidden_states, prev_mlp, norm_w,
                fused_qkv_w, None, Some(qkv_bias), eps, hidden, qkv_dim, true);
        }

        // f16 plain variant
        if let Ok(ref fk) = self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_add_norm_qkv_gemv") {
            return self.launch_fused_gemv_2out(fk, hidden_states, prev_mlp, norm_w,
                fused_qkv_w, None, None, eps, hidden, qkv_dim, false);
        }

        Err(LLMError::GpuError(
            "No fused add+norm+QKV GEMV kernel available. \
             Required: fused_cute_add_norm_qkv_gemv or bias/fp8 variant.".into()
        ))
    }

    /// Fused RMSNorm + QKV GEMV for first layer (no prev_mlp_out).
    /// Returns (qkv, residual, bias_was_fused).
    fn fused_norm_qkv(
        &self,
        hidden_states: &CudaSlice<f16>,
        weights: &GpuLayerWeights<'_>,
        fused_qkv_w: &CudaSlice<f16>,
        hidden: usize,
        qkv_dim: usize,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>, bool)> {
        let norm_w = weights.input_layernorm;
        let eps = self.config.rms_norm_eps;

        // Bias-fused variant
        if let (Some(qkv_bias), Ok(ref fk)) = (
            weights.qkv_bias,
            self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_bias_gemv"),
        ) {
            let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
            let smem = (hidden * 4 + 8 * 4) as u32;
            let rpb = 8u32;
            if smem > 49152 {
                fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                    .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
            }
            unsafe {
                self.stream.launch_builder(fk)
                    .arg(&mut qkv_out).arg(hidden_states).arg(norm_w).arg(fused_qkv_w)
                    .arg(qkv_bias)
                    .arg(&eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                    .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                    .map_err(|e| LLMError::GpuError(format!("fused norm_qkv_bias: {e}")))?;
            }
            return Ok((qkv_out, hidden_states.clone(), true));
        }

        // Plain variant
        if let Ok(ref fk) = self.loader.get_func("fused_add_norm_qkv_gemv", "fused_cute_norm_qkv_gemv") {
            let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
                .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
            let smem = (hidden * 4 + 8 * 4) as u32;
            let rpb = 8u32;
            if smem > 49152 {
                fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                    .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
            }
            unsafe {
                self.stream.launch_builder(fk)
                    .arg(&mut qkv_out).arg(hidden_states).arg(norm_w).arg(fused_qkv_w)
                    .arg(&eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                    .launch(LaunchConfig { grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                    .map_err(|e| LLMError::GpuError(format!("fused norm_qkv: {e}")))?;
            }
            return Ok((qkv_out, hidden_states.clone(), false));
        }

        Err(LLMError::GpuError(
            "No fused norm+QKV GEMV kernel available (first layer). \
             Required: fused_cute_norm_qkv_gemv or bias variant.".into()
        ))
    }

    /// Launch a fused GEMV kernel that produces 2 outputs (qkv + residual).
    /// Handles smem setup, launch config, and multiple kernel signatures.
    fn launch_fused_gemv_2out(
        &self,
        kernel: &cudarc::driver::CudaFunction,
        hidden_states: &CudaSlice<f16>,
        prev_mlp: &CudaSlice<f16>,
        norm_w: &CudaSlice<f16>,
        weight: &(impl DevicePtr<f16> + cudarc::driver::DeviceSlice<f16>),
        fp8_scale: Option<&CudaSlice<f16>>,
        qkv_bias: Option<&CudaSlice<f16>>,
        eps: f32,
        hidden: usize,
        qkv_dim: usize,
        bias_fused: bool,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>, bool)> {
        let mut qkv_out = unsafe { self.stream.alloc::<f16>(qkv_dim) }
            .map_err(|e| LLMError::GpuError(format!("fused qkv alloc: {e}")))?;
        let mut residual_out = unsafe { self.stream.alloc::<f16>(hidden) }
            .map_err(|e| LLMError::GpuError(format!("fused residual alloc: {e}")))?;
        let smem = (hidden * 4 + 8 * 4) as u32;
        let rpb = 8u32;
        if smem > 49152 {
            kernel.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
        }
        let launch_cfg = LaunchConfig {
            grid_dim: ((qkv_dim as u32 + rpb - 1) / rpb, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: smem,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(kernel);
            builder = builder.arg(&mut qkv_out).arg(&mut residual_out)
                .arg(hidden_states).arg(prev_mlp).arg(norm_w).arg(weight);
            if let Some(scale) = fp8_scale {
                builder = builder.arg(scale);
            }
            if let Some(bias) = qkv_bias {
                builder = builder.arg(bias);
            }
            builder.arg(&eps).arg(&(hidden as i32)).arg(&(qkv_dim as i32))
                .launch(launch_cfg)
                .map_err(|e| LLMError::GpuError(format!("fused add_norm_qkv: {e}")))?;
        }
        Ok((qkv_out, residual_out, bias_fused))
    }

    /// Steps 7-10 of fused decode: O-proj + post-norm + GateUp + SiLU + Down.
    fn fused_oproj_mlp(
        &self,
        attn_out: &CudaSlice<f16>,
        residual_ref: &CudaSlice<f16>,
        weights: &GpuLayerWeights<'_>,
        blas: &CublasHandle,
        lt: Option<&crate::CublasLtRef>,
        q_dim: usize,
        hidden: usize,
        intermediate: usize,
    ) -> Result<(CudaSlice<f16>, CudaSlice<f16>)> {
        let cfg = &self.config;
        let post_norm_w = weights.post_attention_layernorm;
        let fused_gate_up_w = weights.fused_gate_up.expect("FusedDecode requires fused_gate_up weight");
        let gate_up_dim = intermediate * 2;

        // Try FP8 mega kernel: O-proj + add + norm + gateup (all FP8 weights)
        let oproj_fp8_ok = (hidden * q_dim) < 4 * 1024 * 1024;
        if let (true, Some(o_fp8), Some(_o_sc), Some(gu_fp8), Some(_gu_sc), Ok(ref mk)) = (
            oproj_fp8_ok,
            weights.o_proj_fp8, weights.o_proj_fp8_scale,
            weights.fused_gate_up_fp8, weights.fused_gate_up_fp8_scale,
            self.loader.get_func("fused_oproj_add_norm_gateup_gemv", "fused_cute_oproj_add_norm_gateup_fp8_gemv"),
        ) {
            let mut gate_up_out = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
                .map_err(|e| LLMError::GpuError(format!("fp8 mega gateup alloc: {e}")))?;
            let mut residual_out2 = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("fp8 mega residual alloc: {e}")))?;
            let smem = (hidden * 4 + 8 * 4) as u32;
            let rpb = 8u32;
            if smem > 49152 {
                mk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                    .map_err(|e| LLMError::GpuError(format!("fp8 mega smem: {e}")))?;
            }
            unsafe {
                self.stream.launch_builder(mk)
                    .arg(&mut gate_up_out).arg(&mut residual_out2)
                    .arg(attn_out)
                    .arg(o_fp8).arg(_o_sc)
                    .arg(residual_ref).arg(post_norm_w)
                    .arg(gu_fp8).arg(_gu_sc)
                    .arg(&cfg.rms_norm_eps).arg(&(q_dim as i32)).arg(&(hidden as i32)).arg(&(gate_up_dim as i32))
                    .launch(LaunchConfig { grid_dim: ((gate_up_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                    .map_err(|e| LLMError::GpuError(format!("fp8 mega oproj_gateup: {e}")))?;
            }
            let mlp = self.fused_silu_down(weights, &gate_up_out, hidden, intermediate)?;
            return Ok((residual_out2, mlp));
        }

        // Separate O-proj + fused add+norm+gateup
        let attn_proj = Self::hgemm_dispatch(&self.stream, blas, lt, attn_out, weights.o_proj, 1, hidden, q_dim, &self.loader)?;
        let fk = self.loader.get_func("fused_add_norm_gateup_gemv", "fused_cute_add_norm_gateup_gemv")
            .map_err(|e| LLMError::GpuError(format!("Required fused_cute_add_norm_gateup_gemv kernel missing: {e}")))?;
        let mut gate_up_out = unsafe { self.stream.alloc::<f16>(gate_up_dim) }
            .map_err(|e| LLMError::GpuError(format!("fused gateup alloc: {e}")))?;
        let mut residual_out2 = unsafe { self.stream.alloc::<f16>(hidden) }
            .map_err(|e| LLMError::GpuError(format!("fused residual alloc: {e}")))?;
        let smem = (hidden * 4 + 8 * 4) as u32;
        let rpb = 8u32;
        if smem > 49152 {
            fk.set_attribute(cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem as i32)
                .map_err(|e| LLMError::GpuError(format!("smem attr: {e}")))?;
        }
        unsafe {
            self.stream.launch_builder(&fk)
                .arg(&mut gate_up_out).arg(&mut residual_out2)
                .arg(residual_ref).arg(&attn_proj).arg(post_norm_w).arg(fused_gate_up_w)
                .arg(&cfg.rms_norm_eps).arg(&(hidden as i32)).arg(&(gate_up_dim as i32))
                .launch(LaunchConfig { grid_dim: ((gate_up_dim as u32 + rpb - 1) / rpb, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: smem })
                .map_err(|e| LLMError::GpuError(format!("fused add_norm_gateup: {e}")))?;
        }
        let mlp = self.fused_silu_down(weights, &gate_up_out, hidden, intermediate)?;
        Ok((residual_out2, mlp))
    }

    /// Fused SiLU + Down projection GEMV (shared by both mega and fallback paths).
    fn fused_silu_down(
        &self,
        weights: &GpuLayerWeights<'_>,
        gate_up_out: &CudaSlice<f16>,
        hidden: usize,
        intermediate: usize,
    ) -> Result<CudaSlice<f16>> {
        let gate = gate_up_out.slice(..intermediate);
        let up = gate_up_out.slice(intermediate..intermediate * 2);

        // FP8 variant
        if let (Some(d_fp8), Some(d_sc), Ok(ref sk)) = (
            weights.down_proj_fp8, weights.down_proj_fp8_scale,
            self.loader.get_func("fused_silu_down_gemv", "fused_cute_silu_down_fp8_gemv"),
        ) {
            let mut down_out = unsafe { self.stream.alloc::<f16>(hidden) }
                .map_err(|e| LLMError::GpuError(format!("fp8 silu_down alloc: {e}")))?;
            unsafe {
                self.stream.launch_builder(sk)
                    .arg(&mut down_out).arg(&gate).arg(&up)
                    .arg(d_fp8).arg(d_sc)
                    .arg(&(hidden as i32)).arg(&(intermediate as i32))
                    .launch(LaunchConfig { grid_dim: (((hidden as u32) + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (8 * 4) as u32 })
                    .map_err(|e| LLMError::GpuError(format!("fp8 silu_down: {e}")))?;
            }
            return Ok(down_out);
        }

        // f16 variant
        let sk = self.loader.get_func("fused_silu_down_gemv", "fused_cute_silu_down_gemv")
            .map_err(|e| LLMError::GpuError(format!("Required fused_cute_silu_down_gemv kernel missing: {e}")))?;
        let mut down_out = unsafe { self.stream.alloc::<f16>(hidden) }
            .map_err(|e| LLMError::GpuError(format!("silu_down alloc: {e}")))?;
        unsafe {
            self.stream.launch_builder(&sk)
                .arg(&mut down_out).arg(&gate).arg(&up).arg(weights.down_proj)
                .arg(&(hidden as i32)).arg(&(intermediate as i32))
                .launch(LaunchConfig { grid_dim: (((hidden as u32) + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: (8 * 4) as u32 })
                .map_err(|e| LLMError::GpuError(format!("silu_down: {e}")))?;
        }
        Ok(down_out)
    }
}
