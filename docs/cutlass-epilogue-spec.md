# CUTLASS Fused Epilogue Spec

## Goal

Replace cuBLAS GEMM + separate elementwise kernels with CUTLASS GEMMs that fuse the elementwise ops as epilogues. This eliminates intermediate HBM round-trips and kernel launch overhead.

## Current state (12,611 tok/s at N=128)

The T>1 batched decode path (gpu_layer.rs) does per layer:

```
1. fused_residual_rmsnorm    -> normed [128, 3584]     (1 kernel, writes to HBM)
2. cuBLAS GEMM QKV           -> qkv [128, 4608]        (1 kernel, reads normed from HBM)
3. bias broadcast             -> qkv += bias             (1 kernel)
4. RoPE                       -> q,k in-place            (1 kernel)
5. cache write                -> KV cache                (1 kernel)
6. attention                  -> attn_out [128, 3584]    (1 kernel)
7. cuBLAS GEMM O-proj        -> o_proj [128, 3584]      (1 kernel, writes to HBM)
8. fused_residual_rmsnorm    -> normed2 [128, 3584]     (1 kernel, reads o_proj from HBM)
9. cuBLAS GEMM gate+up       -> gate_up [128, 37888]    (1 kernel, writes to HBM)
10. silu_mul_interleaved      -> activated [128, 18944]  (1 kernel, reads gate_up from HBM)
11. cuBLAS GEMM down          -> down [128, 3584]        (1 kernel, writes to HBM)
```

11 kernels/layer. The HBM round-trips between steps 1-2, 7-8, 9-10 are pure waste.

## Target: CUTLASS fused epilogues

```
1. fused_residual_rmsnorm    -> normed                   (1 kernel)
2. CUTLASS GEMM QKV + bias   -> qkv (bias as epilogue)  (1 kernel, saves step 3)
4. RoPE                       -> q,k in-place            (1 kernel)
5. cache write                -> KV cache                (1 kernel)
6. attention                  -> attn_out                 (1 kernel)
7. CUTLASS GEMM O-proj + residual_add + RMSNorm -> normed2  (1 kernel, saves steps 7+8)
9. CUTLASS GEMM gate+up + SiLU*mul -> activated          (1 kernel, saves steps 9+10)
11. CUTLASS GEMM down         -> down                     (1 kernel)
```

8 kernels/layer (down from 11). Eliminates 3 HBM round-trips per layer.

## Qwen2.5-7B dimensions

```
hidden_size = 3584
num_heads = 28, num_kv_heads = 4, head_dim = 128
q_dim = 3584, kv_dim = 512, qkv_dim = 4608
intermediate_size = 18944
gate_up_dim = 37888
num_layers = 28
```

At N=128:
- QKV GEMM: [128, 3584] x [3584, 4608]^T = [128, 4608]
- O-proj GEMM: [128, 3584] x [3584, 3584]^T = [128, 3584]
- GateUp GEMM: [128, 3584] x [3584, 37888]^T = [128, 37888]
- Down GEMM: [128, 18944] x [18944, 3584]^T = [128, 3584]

## CUTLASS 3.x API (sm_90)

Use CUTLASS 3.x with the collective builder API. H100 supports:
- `SM90_TMA_LOAD` for tensor memory access
- `SM90_16x8x16_F16F16F16F16_TN` or similar tile MMA
- Custom epilogues via `cutlass::epilogue::fusion`

### Agent 1: CUTLASS GEMM + Bias epilogue (QKV)

Replace cuBLAS QKV GEMM + separate bias kernel with one CUTLASS GEMM that adds bias in the epilogue.

**CUTLASS epilogue:** `LinearCombination` with bias vector.

```cpp
// D = alpha * A @ B + beta * C + bias
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBias<
    cutlass::half_t, 8, float, float, cutlass::half_t>;
```

The bias is broadcast across the M dimension (same bias for every token).

**File:** `kernels/cutlass_qkv_bias.cu`
**Signature:** `(output, input, weight, bias, M, N, K)`
**Grid/Block:** CUTLASS handles this internally.

### Agent 2: CUTLASS GEMM + Residual Add + RMSNorm epilogue (O-proj)

This is the hardest one. After O-proj GEMM, add the residual and apply RMSNorm. RMSNorm is a row-wise reduction (sum of squares across K=3584), so it can't be a simple per-element epilogue.

**Approach:** Two-phase CUTLASS kernel:
1. GEMM with custom epilogue that writes GEMM output + residual add to a temp buffer AND computes partial sum-of-squares per tile
2. Reduction kernel that finishes the RMSNorm

OR simpler: CUTLASS GEMM with residual-add epilogue only (no RMSNorm), then separate RMSNorm kernel. This still saves one HBM round-trip (the O-proj output write+read).

```cpp
// D[m,n] = alpha * A @ B + residual[m,n]
using EpilogueOp = cutlass::epilogue::thread::LinearCombinationResidual<
    cutlass::half_t, 8, float, float>;
```

**File:** `kernels/cutlass_oproj_residual.cu`
**Signature:** `(output, input, weight, residual, M, N, K)`

### Agent 3: CUTLASS GEMM + SiLU*Mul epilogue (GateUp)

The gate+up GEMM outputs [M, 2*intermediate]. The first half is gate, second half is up. Apply SiLU to gate, multiply by up.

**Approach:** Custom CUTLASS epilogue functor:

```cpp
struct SiLUMulEpilogue {
    CUTLASS_DEVICE
    void operator()(
        cutlass::half_t* output,       // [M, intermediate]
        const cutlass::half_t* d,      // [M, 2*intermediate] from GEMM
        int m, int n, int intermediate
    ) {
        int gate_idx = m * 2 * intermediate + n;
        int up_idx = gate_idx + intermediate;
        float g = float(d[gate_idx]);
        float u = float(d[up_idx]);
        float silu_g = g / (1.0f + expf(-g));
        output[m * intermediate + n] = cutlass::half_t(silu_g * u);
    }
};
```

This eliminates the separate silu_mul kernel and the [M, 37888] intermediate write+read.

**File:** `kernels/cutlass_gateup_silu.cu`
**Signature:** `(activated_output, input, fused_gate_up_weight, M, hidden, intermediate)`

### Agent 4: CUTLASS GEMM integration into Rust dispatch

Wire the CUTLASS kernels into gpu_layer.rs. Each CUTLASS kernel is compiled to PTX/cubin and loaded via kernel_loader.

In the T>1 path:
- Replace `hgemm_dispatch` + `add_bias_broadcast` for QKV with CUTLASS QKV+bias
- Replace `hgemm_dispatch` + `fused_residual_rmsnorm` for O-proj with CUTLASS O-proj+residual
- Replace `hgemm_dispatch` + `silu_mul_interleaved` for GateUp with CUTLASS GateUp+SiLU

**Fallback:** If CUTLASS kernels aren't loaded (PTX not compiled), fall through to existing cuBLAS + separate kernel path.

**File:** `crates/rvllm-model-runner/src/gpu_layer.rs` (T>1 section only)

### Agent 5: Build integration + benchmark

1. Add CUTLASS as a git submodule or download headers
2. Compile CUTLASS kernels with nvcc (they need CUTLASS headers):
   ```
   nvcc -ptx -arch=sm_90 -O3 --use_fast_math \
     -I/path/to/cutlass/include \
     -o sm_90/cutlass_qkv_bias.ptx kernels/cutlass_qkv_bias.cu
   ```
3. Register in kernel_loader.rs
4. Benchmark before/after on H100

## Expected gains

| Fusion | Bytes saved per layer (N=128) | Time saved |
|--------|------|------|
| QKV + bias | 128 * 4608 * 2 = 1.2 MB write+read | ~0.7us |
| O-proj + residual | 128 * 3584 * 2 = 0.9 MB write+read | ~0.5us |
| GateUp + SiLU | 128 * 37888 * 2 = 9.7 MB write+read | ~5.8us |
| **Total per layer** | **11.8 MB** | **~7us** |
| **28 layers** | **330 MB** | **~196us** |

At 10ms per step, 196us = 2% improvement. The real gain is from CUTLASS's better GEMM tiling compared to cuBLAS default heuristic -- estimated 5-10% on top of the epilogue savings.

## H100 CUTLASS availability

CUTLASS headers are at `/root/cutlass/include` or need to be cloned:
```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass /root/cutlass
```

Use CUTLASS 3.x (latest) which supports sm_90 natively with TMA and WGMMA.
