#!/bin/bash
# Compile CUTLASS kernels for rvllm
# Requires: CUTLASS headers at $CUTLASS_DIR (default /root/cutlass)
# Usage: ./kernels/build_cutlass.sh [arch] [cutlass_dir]

set -euo pipefail

ARCH=${1:-sm_90}
CUTLASS_DIR=${2:-/root/cutlass}

if [ ! -d "$CUTLASS_DIR/include/cutlass" ]; then
    echo "CUTLASS not found at $CUTLASS_DIR, cloning..."
    git clone --depth 1 https://github.com/NVIDIA/cutlass "$CUTLASS_DIR"
fi

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
mkdir -p "$ARCH"

NVCC=${NVCC:-nvcc}
OK=0
FAIL=0
TOTAL=0

for f in cutlass_*.cu; do
    [ -f "$f" ] || continue
    TOTAL=$((TOTAL + 1))
    stem=${f%.cu}
    echo -n "  $f -> $ARCH/${stem}.ptx ... "
    if $NVCC --ptx -arch=$ARCH -O3 --use_fast_math \
        -I"$CUTLASS_DIR/include" \
        -I"$CUTLASS_DIR/tools/util/include" \
        -o "$ARCH/${stem}.ptx" "$f" 2>/tmp/nvcc_cutlass_${stem}.log; then
        echo "ok"
        OK=$((OK + 1))
    else
        echo "FAILED"
        FAIL=$((FAIL + 1))
        tail -3 /tmp/nvcc_cutlass_${stem}.log 2>/dev/null
    fi
done

echo ""
echo "CUTLASS kernels: ${OK}/${TOTAL} compiled (${FAIL} failed)"
echo "PTX output: $DIR/$ARCH/cutlass_*.ptx"
