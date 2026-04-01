#!/bin/bash
set -euo pipefail

# Deploy rvllm to vast.ai GPU instance: clone, compile kernels, build, download model, bench.
# One script, no caching, no bullshit.
#
# Usage:
#   ./deploy/vastai-deploy.sh                     # auto-detect instance
#   ./deploy/vastai-deploy.sh <instance_id>       # explicit instance
#   BENCH_ONLY=1 ./deploy/vastai-deploy.sh        # skip build, just bench

INSTANCE_ID=${1:-$(cat deploy/.instance_id_rvllm 2>/dev/null || echo "")}
if [[ -z "$INSTANCE_ID" ]]; then
    echo "No instance ID. Run vastai-provision.sh first."
    exit 1
fi

MODEL="Qwen/Qwen2.5-7B"
BENCH_ONLY=${BENCH_ONLY:-0}

SSH_URL=$(vastai ssh-url "$INSTANCE_ID")
SSH_HOST=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f1)
SSH_PORT=$(echo "$SSH_URL" | sed 's|ssh://||' | cut -d: -f2)
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $SSH_PORT $SSH_HOST"

echo "Instance: $INSTANCE_ID ($SSH_HOST:$SSH_PORT)"
echo "Model: $MODEL"
echo ""

$SSH 'bash -l' << REMOTE
set -euo pipefail

# Kill any running server
pkill -9 -f rvllm 2>/dev/null || true
sleep 1

# Ensure toolchain
if ! command -v cargo &>/dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi

# GPU info
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
CC=\$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
ARCH="sm_\$(echo \$CC | tr -d '.')"
echo "Arch: \$ARCH"

if [[ "$BENCH_ONLY" == "1" ]]; then
    echo "=== BENCH ONLY ==="
    cd /root/rvllm
    git pull origin main
else
    # Fresh clone
    rm -rf /root/rvllm
    git clone https://github.com/m0at/rvllm.git /root/rvllm
    cd /root/rvllm

    # Compile ALL kernels
    echo "=== Compiling kernels ==="
    mkdir -p kernels/\$ARCH
    FAIL=0
    for cu in kernels/*.cu; do
        stem=\$(basename "\$cu" .cu)
        if [[ "\$stem" == "persistent_layer_decode" ]]; then
            echo "  \$stem -> cubin"
            nvcc -cubin -arch=\$ARCH -O3 --use_fast_math -rdc=true -Xptxas -v \
                -o "kernels/\$ARCH/\${stem}.cubin" "\$cu" 2>&1 | grep -E "registers|spill" || true
        elif [[ "\$stem" == cutlass_* ]]; then
            echo "  \$stem -> skip (needs CUTLASS headers)"
        else
            echo "  \$stem -> ptx"
            if ! nvcc -ptx -arch=\$ARCH -O3 --use_fast_math -o "kernels/\$ARCH/\${stem}.ptx" "\$cu" 2>/dev/null; then
                echo "    FAILED"
                FAIL=\$((FAIL+1))
            fi
        fi
    done
    echo "Kernels: \$(ls kernels/\$ARCH/ | wc -l) compiled (\$FAIL failed)"

    # Build
    echo "=== Building ==="
    CUDA_ARCH=\$ARCH cargo build --release --features cuda 2>&1 | tail -3

    # Download model
    echo "=== Downloading model ==="
    pip3 install -q huggingface_hub 2>/dev/null || true
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" 2>&1 | tail -2
fi

ls -la target/release/rvllm
echo ""

# ================================================================
# BENCHMARK
# ================================================================
export RVLLM_PTX_DIR=/root/rvllm/kernels/\$ARCH

bench_path() {
    local label="\$1"
    shift
    echo ">>> \$label"

    # Set env vars passed as args
    for arg in "\$@"; do export \$arg; done

    pkill -9 -f "rvllm serve" 2>/dev/null || true
    sleep 2

    target/release/rvllm serve --model $MODEL --port 8000 \
        --gpu-memory-utilization 0.90 --dtype half > /tmp/rvllm_bench.log 2>&1 &
    local PID=\$!

    for i in \$(seq 1 120); do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then break; fi
        if ! kill -0 \$PID 2>/dev/null; then
            echo "  SERVER DIED"
            tail -20 /tmp/rvllm_bench.log
            return 1
        fi
        sleep 1
    done

    # Warmup (single sequential request)
    curl -s -X POST http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{"model":"$MODEL","prompt":"Hello","max_tokens":32,"temperature":0.7}' \
        --max-time 60 >/dev/null 2>&1

    # Check for crash
    if ! kill -0 \$PID 2>/dev/null; then
        echo "  SERVER CRASHED during warmup"
        tail -20 /tmp/rvllm_bench.log
        return 1
    fi
    sleep 1

    # Bench: 3 runs, 256 tokens each
    for run in 1 2 3; do
        local START=\$(date +%s%N)
        local RESP=\$(curl -s -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"$MODEL","prompt":"The theory of general relativity states that","max_tokens":256,"temperature":0.0}' \
            --max-time 120)
        local END=\$(date +%s%N)
        local TOKS=\$(echo "\$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
        local MS=\$(( (END - START) / 1000000 ))
        local TPS=\$(( TOKS * 1000 / (MS + 1) ))
        echo "  run\$run: \${TOKS} tok / \${MS}ms = \${TPS} tok/s"
    done

    pkill -9 -f "rvllm serve" 2>/dev/null || true
    sleep 2

    # Unset env vars
    for arg in "\$@"; do unset \${arg%%=*}; done
}

echo "=== A/B Benchmark ==="
echo ""
bench_path "FusedDecode (default)" || true
echo ""
bench_path "PersistentDecode (DAG)" "RVLLM_PERSISTENT=1" || true
echo ""
echo "=== Done ==="
REMOTE
