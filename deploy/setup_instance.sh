#!/bin/bash
# setup_instance.sh -- Bootstrap a fresh vast.ai (or any CUDA) instance for rvLLM.
#
# Installs: Rust, CUTLASS headers, Python deps, model weights.
# Then builds rvLLM with full fused kernel support and runs benchmark.
#
# Usage:
#   # From local machine, push repo then run:
#   rsync -az --exclude target/ --exclude .git/ -e "ssh -p PORT" . root@HOST:/root/rvllm/
#   ssh -p PORT root@HOST 'bash /root/rvllm/deploy/setup_instance.sh'
#
#   # Or with options:
#   bash setup_instance.sh --model Qwen/Qwen2.5-7B --skip-bench

set -euo pipefail

# --- Defaults ---
MODEL="Qwen/Qwen2.5-7B"
SKIP_BENCH=0
GPU_MEM_UTIL=0.9

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2"; shift 2 ;;
        --skip-bench)  SKIP_BENCH=1; shift ;;
        --gpu-mem)     GPU_MEM_UTIL="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

bold() { echo -e "\033[1m>>> $*\033[0m"; }
ok()   { echo -e "\033[32m[OK]\033[0m $*"; }
err()  { echo -e "\033[31m[ERR]\033[0m $*"; }

# ============================================================
# 1. System deps
# ============================================================
bold "1/7 System deps"

# Ensure nvcc is available
if ! command -v nvcc &>/dev/null; then
    if [[ -d /usr/local/cuda/bin ]]; then
        export PATH=/usr/local/cuda/bin:$PATH
    fi
fi
if command -v nvcc &>/dev/null; then
    ok "nvcc: $(nvcc --version 2>/dev/null | grep release)"
else
    err "nvcc not found. Need CUDA toolkit."
    exit 1
fi

# GPU arch
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: $GPU_NAME (cc $CC)"

# ============================================================
# 2. Rust toolchain
# ============================================================
bold "2/7 Rust toolchain"

if command -v cargo &>/dev/null; then
    ok "cargo already installed: $(cargo --version)"
else
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
    ok "cargo installed: $(cargo --version)"
fi
export PATH="$HOME/.cargo/bin:$PATH"

# ============================================================
# 3. CUTLASS headers (needed for fused CuTE JIT kernels)
# ============================================================
bold "3/7 CUTLASS headers"

CUTLASS_DIR="/root/cutlass"
if [[ -f "$CUTLASS_DIR/include/cute/tensor.hpp" ]]; then
    ok "CUTLASS already at $CUTLASS_DIR"
else
    # Shallow clone just the headers (no build, no examples)
    rm -rf "$CUTLASS_DIR"
    git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR" 2>/dev/null
    cd "$CUTLASS_DIR"
    git sparse-checkout set include tools/util/include 2>/dev/null
    cd "$REPO_DIR"
    if [[ -f "$CUTLASS_DIR/include/cute/tensor.hpp" ]]; then
        ok "CUTLASS headers installed"
    else
        err "CUTLASS clone failed, fused kernels will be disabled"
    fi
fi

# ============================================================
# 4. Python + deps (for vLLM comparison benchmarks)
# ============================================================
bold "4/7 Python deps"

# Use system python3, install uv for fast dep management
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null
    export PATH="$HOME/.local/bin:$PATH"
fi

# Create venv with required deps
UV="$HOME/.local/bin/uv"
if [[ ! -d /root/.venv/rvllm ]]; then
    $UV venv /root/.venv/rvllm --python 3.11 2>/dev/null || $UV venv /root/.venv/rvllm 2>/dev/null
fi
source /root/.venv/rvllm/bin/activate

# Install benchmark deps + vLLM for comparison
$UV pip install aiohttp numpy huggingface_hub 2>/dev/null
ok "Python benchmark deps installed"

# vLLM for comparison (large install, skip if already present)
if python3 -c "import vllm" 2>/dev/null; then
    ok "vLLM already installed: $(python3 -c 'import vllm; print(vllm.__version__)')"
else
    bold "    Installing vLLM (this takes a few minutes)..."
    $UV pip install vllm 2>/dev/null
    if python3 -c "import vllm" 2>/dev/null; then
        ok "vLLM installed: $(python3 -c 'import vllm; print(vllm.__version__)')"
    else
        err "vLLM install failed, comparison bench will be skipped"
    fi
fi

# ============================================================
# 5. Download model weights
# ============================================================
bold "5/7 Model weights: $MODEL"

# Derive local path from model name
MODEL_LOCAL="/root/models/$(basename "$MODEL")"
if [[ -f "$MODEL_LOCAL/config.json" ]]; then
    ok "Model already at $MODEL_LOCAL"
else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL', local_dir='$MODEL_LOCAL')
print('done')
"
    # Clean HF cache to save disk
    rm -rf ~/.cache/huggingface/hub/ 2>/dev/null || true
    ok "Model downloaded to $MODEL_LOCAL"
fi

# ============================================================
# 6. Build rvLLM
# ============================================================
bold "6/7 Build rvLLM (--features cuda,cublaslt)"

cd "$REPO_DIR"
cargo build --release --features cuda,cublaslt -p rvllm-server 2>&1 | tail -5

BINARY="$REPO_DIR/target/release/rvllm"
if [[ -x "$BINARY" ]]; then
    ok "Binary: $BINARY ($(stat -c%s "$BINARY" 2>/dev/null || stat -f%z "$BINARY") bytes)"
else
    err "Build failed"
    exit 1
fi

# ============================================================
# 7. Benchmark
# ============================================================
if [[ "$SKIP_BENCH" -eq 1 ]]; then
    bold "7/7 Benchmark SKIPPED (--skip-bench)"
    echo ""
    echo "Ready to run:"
    echo "  $BINARY benchmark --model $MODEL_LOCAL --n 16,32,64,128 --output-len 512 --json"
    exit 0
fi

bold "7/7 Benchmark"

BATCH_SIZES="16,24,32,48,64,96,128"

echo ""
echo "--- rvLLM ---"
$BINARY benchmark --model "$MODEL_LOCAL" \
    --n "$BATCH_SIZES" --output-len 512 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" --json 2>&1 | \
    grep -E "tok_per_sec|\"n\"" | paste - - | \
    sed 's/.*"n": \([0-9]*\).*"tok_per_sec": \([0-9.]*\).*/  N=\1: \2 tok\/s/'

echo ""
echo "--- vLLM $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo '?') ---"
python3 - "$MODEL_LOCAL" "$BATCH_SIZES" <<'PYEOF'
import time, json, sys
from vllm import LLM, SamplingParams

model = sys.argv[1]
batch_sizes = [int(x) for x in sys.argv[2].split(",")]
output_len = 512

llm = LLM(model=model, dtype="float16", max_model_len=2048,
          gpu_memory_utilization=0.9, enforce_eager=True)
sampling = SamplingParams(max_tokens=output_len, temperature=0.0)

for n in batch_sizes:
    prompts = ["Hello, my name is"] * n
    llm.generate(prompts, sampling)  # warmup
    t0 = time.time()
    out = llm.generate(prompts, sampling)
    elapsed = time.time() - t0
    total = sum(len(o.outputs[0].token_ids) for o in out)
    tps = total / elapsed
    print(f"  N={n}: {tps:.1f} tok/s")
PYEOF

echo ""
ok "Benchmark complete"
