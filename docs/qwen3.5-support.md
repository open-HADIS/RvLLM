# Qwen3.5 Architecture Support

## Status: Not Supported

rvLLM's `Qwen2ForCausalLM` expects standard transformer self-attention (`self_attn.{q,k,v,o}_proj`).
Qwen3.5 is a **hybrid transformer + Gated DeltaNet** architecture. ~75% of layers use linear
attention with SSM-style recurrence -- completely different tensor set, forward pass, and state
management.

---

## Architecture Overview

Qwen3.5 uses a **3:1 hybrid pattern** controlled by `full_attention_interval: 4`:

```
Layer 0:  linear_attn  (Gated DeltaNet)
Layer 1:  linear_attn
Layer 2:  linear_attn
Layer 3:  self_attn    (standard softmax attention)
Layer 4:  linear_attn
Layer 5:  linear_attn
Layer 6:  linear_attn
Layer 7:  self_attn
...
```

Every 4th layer is full softmax attention (with RoPE, KV cache, paged attention -- same as Qwen2).
The other 3 use **Gated DeltaNet**: a linear attention variant combining Mamba-style gating with
delta-rule state updates.

Paper: "Gated Delta Networks: Improving Mamba2 with Delta Rule" (arXiv 2412.06464, ICLR 2025)

### Model Sizes

| Model | Layers | Hidden | KV Heads (full attn) | Linear V Heads | Linear K Heads |
|-------|--------|--------|---------------------|----------------|----------------|
| 0.8B  | 24     | 1024   | varies              | varies         | varies         |
| 2B    | 24     | 2048   | varies              | varies         | varies         |
| 4B    | 32     | 2560   | varies              | varies         | varies         |
| 9B    | 32     | 4096   | 4                   | 32             | 16             |
| 27B   | 40     | 4096   | varies              | varies         | varies         |

All support 262K native context, extensible to ~1M via RoPE scaling.

---

## Tensor Names (from Qwen3.5-0.8B checkpoint inspection)

### Linear Attention Layers (Gated DeltaNet)

```
model.layers.{i}.linear_attn.in_proj_qkv.weight    # fused Q/K/V projection
model.layers.{i}.linear_attn.in_proj_z.weight       # gate projection (Z)
model.layers.{i}.linear_attn.out_proj.weight         # output projection
model.layers.{i}.linear_attn.conv1d.weight           # causal depthwise conv1d
model.layers.{i}.linear_attn.A_log                   # log decay matrix (discretized)
model.layers.{i}.linear_attn.dt_bias                 # time-step bias
```

### Full Attention Layers (standard, same as Qwen2)

```
model.layers.{i}.self_attn.q_proj.weight
model.layers.{i}.self_attn.k_proj.weight
model.layers.{i}.self_attn.v_proj.weight
model.layers.{i}.self_attn.o_proj.weight
```

### Common to All Layers

```
model.layers.{i}.input_layernorm.weight
model.layers.{i}.post_attention_layernorm.weight
model.layers.{i}.mlp.gate_proj.weight
model.layers.{i}.mlp.up_proj.weight
model.layers.{i}.mlp.down_proj.weight
model.embed_tokens.weight
model.norm.weight
lm_head.weight
```

---

## Config Fields (new vs Qwen2)

```json
{
  "model_type": "qwen3_5",
  "full_attention_interval": 4,
  "linear_num_value_heads": 32,
  "linear_num_key_heads": 16,
  "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                  "full_attention", ...]
}
```

- `full_attention_interval` -- every Nth layer uses full attention (default 4)
- `linear_num_value_heads` -- number of value heads in linear attention layers
- `linear_num_key_heads` -- number of key/query heads in linear attention layers
- `layer_types` -- optional explicit per-layer type list (auto-derived from interval if absent)

---

## Gated DeltaNet Forward Pass

### Per-layer (linear attention)

```
Input: hidden [B, T, D]

1. input_layernorm(hidden)

2. Projections:
   qkv = in_proj_qkv(normed)           # [B, T, 3 * d_inner]
   q, k, v = split(qkv, 3)             # each [B, T, d_inner]
   z = in_proj_z(normed)                # [B, T, d_inner] -- gate

3. Causal Conv1d:
   q = conv1d(q, kernel_size=4)         # depthwise causal convolution
   k = conv1d(k, kernel_size=4)         # (or q/k only, depends on variant)

4. L2 normalize Q, K (instead of softmax)

5. Discretize decay:
   A = -exp(A_log)                      # [d_inner, d_state] or [num_heads]
   dt = dt_bias + ...                   # time-step per token

6. Recurrence (delta rule + gating):
   For each timestep t:
     g_t = exp(alpha_t)                 # exponential gate (from A, dt)
     delta_t = beta_t * (v_t - S_{t-1}^T @ k_t)   # delta correction
     S_t = g_t * S_{t-1} + k_t outer delta_t       # state update
     y_t = S_t^T @ q_t                              # output query

   (Parallelized via chunkwise scan for training;
    sequential recurrence for autoregressive decode)

7. Output gating:
   y = y * silu(z)                      # gated output
   output = out_proj(y)                 # [B, T, D]

8. Residual: hidden += output

9. post_attention_layernorm(hidden)
10. MLP (same as Qwen2: gate/up/down with SiLU)
11. Residual: hidden += mlp_out
```

### Per-layer (full attention) -- identical to Qwen2

Standard Q/K/V projections, RoPE, paged KV cache attention, O projection, residual, MLP.

---

## Implementation Plan

### Phase 1: New Layer Primitives

**1a. Causal Conv1d** -- `crates/rvllm-model-runner/src/layers/conv1d.rs`

```rust
pub struct CausalConv1d;

impl CausalConv1d {
    /// Depthwise causal 1D convolution.
    /// input:  [num_tokens, channels]
    /// weight: [channels, 1, kernel_size]
    /// bias:   [channels] (optional)
    /// state:  [batch, channels, kernel_size-1] (rolling state for decode)
    pub fn forward(
        input: &GpuBuffer<f16>,
        weight: &GpuBuffer<f16>,
        bias: Option<&GpuBuffer<f16>>,
        state: Option<&mut GpuBuffer<f16>>,
    ) -> Result<GpuBuffer<f16>>;
}
```

- Prefill: full causal conv over sequence (left-padded, no future tokens)
- Decode: shift-register on `kernel_size - 1` state slots, single output per step
- CUDA kernel: depthwise conv1d is straightforward, one thread per (token, channel)

**1b. DeltaNet Recurrence** -- `crates/rvllm-model-runner/src/layers/deltanet.rs`

```rust
pub struct DeltaNetScan;

impl DeltaNetScan {
    /// Gated delta-rule recurrence.
    /// q, k, v: [num_tokens, num_heads, head_dim]
    /// gate:    [num_tokens, num_heads] (exponential decay per head)
    /// beta:    [num_tokens, num_heads] (update rate)
    /// state:   [batch, num_heads, d_k, d_v] (persistent across decode steps)
    pub fn forward(
        q: &GpuBuffer<f16>,
        k: &GpuBuffer<f16>,
        v: &GpuBuffer<f16>,
        gate: &GpuBuffer<f16>,
        beta: &GpuBuffer<f16>,
        state: &mut GpuBuffer<f16>,
    ) -> Result<GpuBuffer<f16>>;
}
```

- Decode path: single-step state update (simple, fast)
- Prefill path: chunkwise parallel scan or sequential (correctness first, optimize later)
- State is fixed-size per sequence -- no growing KV cache for these layers

### Phase 2: Architecture File

**`crates/rvllm-model-runner/src/architectures/qwen3_5.rs`**

```rust
pub struct Qwen3_5ForCausalLM {
    hidden_size: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f32,
    embed_tokens: GpuBuffer<f16>,
    layers: Vec<Qwen3_5Layer>,
    norm_weight: GpuBuffer<f16>,
    lm_head_weight: GpuBuffer<f16>,
}

enum Qwen3_5Attention {
    FullAttention {
        q_proj: GpuBuffer<f16>,
        k_proj: GpuBuffer<f16>,
        v_proj: GpuBuffer<f16>,
        o_proj: GpuBuffer<f16>,
    },
    LinearAttention {
        in_proj_qkv: GpuBuffer<f16>,
        in_proj_z: GpuBuffer<f16>,
        out_proj: GpuBuffer<f16>,
        conv1d_weight: GpuBuffer<f16>,
        a_log: GpuBuffer<f16>,
        dt_bias: GpuBuffer<f16>,
    },
}

struct Qwen3_5Layer {
    input_layernorm: GpuBuffer<f16>,
    post_attention_layernorm: GpuBuffer<f16>,
    attn: Qwen3_5Attention,
    gate_proj: GpuBuffer<f16>,
    up_proj: GpuBuffer<f16>,
    down_proj: GpuBuffer<f16>,
}
```

Forward pass dispatches per layer based on enum variant. Full attention layers use the existing
`AttentionBackend` + RoPE + KV cache. Linear attention layers use conv1d + DeltaNet scan with
separate per-sequence recurrent state.

### Phase 3: Weight Mapper

**`crates/rvllm-model-loader/src/mapper.rs`** -- add `"qwen3_5"` case:

```rust
"qwen3_5" => {
    // Full attention layers (same as qwen2)
    name = name.replace("self_attn.q_proj", "attn.q");
    name = name.replace("self_attn.k_proj", "attn.k");
    name = name.replace("self_attn.v_proj", "attn.v");
    name = name.replace("self_attn.o_proj", "attn.o");
    // Linear attention layers
    name = name.replace("linear_attn.in_proj_qkv", "lattn.qkv");
    name = name.replace("linear_attn.in_proj_z", "lattn.z");
    name = name.replace("linear_attn.out_proj", "lattn.o");
    name = name.replace("linear_attn.conv1d", "lattn.conv1d");
    name = name.replace("linear_attn.A_log", "lattn.a_log");
    name = name.replace("linear_attn.dt_bias", "lattn.dt_bias");
    // MLP + norms (same as qwen2)
    name = name.replace("mlp.gate_proj", "ffn.gate");
    name = name.replace("mlp.up_proj", "ffn.up");
    name = name.replace("mlp.down_proj", "ffn.down");
    name = name.replace("input_layernorm", "attn_norm");
    name = name.replace("post_attention_layernorm", "ffn_norm");
}
```

### Phase 4: Config Parsing

Extend `ModelRunnerConfig` or parse Qwen3.5-specific fields from the HF `config.json`:

- `full_attention_interval` (u32)
- `linear_num_value_heads` (usize)
- `linear_num_key_heads` (usize)
- Derive `layer_types: Vec<LayerType>` from interval + num_layers

### Phase 5: Recurrent State Management

Unlike the KV cache (which grows with context), the DeltaNet state is **fixed-size**:

```
state_per_layer = batch_size * num_heads * d_k * d_v * sizeof(f16)
conv_state_per_layer = batch_size * d_inner * (kernel_size - 1) * sizeof(f16)
```

For Qwen3.5-0.8B (d_inner ~1024, num_heads ~8, head_dim ~128, kernel=4):
- DeltaNet state: ~128 KB/seq/layer
- Conv state: ~6 KB/seq/layer
- 18 linear layers * 134 KB = ~2.4 MB/seq total recurrent state

This is tiny compared to KV cache. Options:
- Allocate alongside KV cache in `CacheEngine`
- Separate `RecurrentStateManager` that the architecture owns

### Phase 6: Architecture Registration

`crates/rvllm-model-runner/src/architectures/mod.rs`:

```rust
pub mod qwen3_5;

// in create_model():
"Qwen3_5ForCausalLM" => Ok(Box::new(qwen3_5::Qwen3_5ForCausalLM::new(weights, config)?)),
```

---

## State Management: KV Cache vs Recurrent State

| Property | Full Attention (self_attn) | Linear Attention (linear_attn) |
|----------|---------------------------|-------------------------------|
| State type | KV cache (paged blocks) | Fixed recurrent state matrix |
| Grows with context | Yes | No |
| State per layer | O(seq_len * head_dim) | O(d_k * d_v) per head (fixed) |
| Decode cost | O(seq_len) per token | O(1) per token |
| Prefill | Flash/paged attention | Chunkwise scan or sequential |
| Cache invalidation | Evict blocks | Reset state matrix |

The hybrid model needs **both**: paged KV cache for the ~25% full-attention layers, and fixed
recurrent state buffers for the ~75% linear-attention layers. This is the main architectural
challenge for the scheduler/cache engine.

---

## CUDA Kernels Needed

1. **Causal depthwise conv1d** -- simple, one thread per (token, channel), shift-register for decode
2. **DeltaNet single-step update** -- decode path: `S = g*S + k outer delta; y = S^T @ q`
3. **DeltaNet chunkwise scan** -- prefill path: parallel prefix sum over chunks (optimize later)
4. **L2 normalize** -- per-head normalization of Q/K (may already exist in norm kernels)
5. **Fused QKV split + silu gate** -- optional, for throughput

Priority: (2) is critical for decode perf. (1) is simple. (3) can be sequential initially.

---

## Open Questions

- Exact `A_log` / `dt_bias` shapes and discretization scheme in Qwen3.5-0.8B -- need to inspect
  the checkpoint tensors directly to confirm dimensions
- Whether Qwen3.5 uses `in_proj_ba` (as in some references) or `A_log` + `dt_bias` separately
  (as observed in 0.8B checkpoint) -- may vary by model size
- Conv1d kernel size -- likely 4, confirm from config or weight shape
- Whether the gate Z uses SiLU or sigmoid -- Gated DeltaNet paper uses SiLU
- Batch handling for recurrent state during continuous batching -- sequences join/leave mid-batch,
  state must be per-sequence
- Whether `linear_attn` layers in Qwen3.5-0.8B are true Gated DeltaNet or a simpler variant
  (the presence of `A_log` and `dt_bias` suggests Mamba-style discretization rather than pure
  DeltaNet exponential gating)

---

## Reference Implementation

- HuggingFace Transformers: `src/transformers/models/qwen3/modeling_qwen3.py`
- NVlabs GatedDeltaNet: `github.com/NVlabs/GatedDeltaNet`
- Flash Linear Attention (FLA): `github.com/fla-org/flash-linear-attention`
- Qwen3 technical report: arXiv 2505.09388
- Gated DeltaNet paper: arXiv 2412.06464

---

## Estimated Effort

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Conv1d layer (CPU) | Low | Simple shift-register + dot product |
| Conv1d CUDA kernel | Low | Depthwise conv, straightforward |
| DeltaNet recurrence (CPU) | Medium | Sequential scan, outer products |
| DeltaNet CUDA kernel (decode) | Medium | Single-step matmul + state update |
| DeltaNet CUDA kernel (prefill) | High | Chunkwise parallel scan |
| Architecture file | Medium | Enum dispatch, weight loading |
| Weight mapper | Low | String replacements |
| Config parsing | Low | New fields from config.json |
| Recurrent state management | Medium | Per-sequence state alloc/free |
| Scheduler integration | Medium | Hybrid cache + state tracking |

Shortest path to working inference: CPU-only DeltaNet recurrence (sequential), CPU conv1d,
reuse existing attention backend for full-attention layers. Then CUDA kernels for performance.
