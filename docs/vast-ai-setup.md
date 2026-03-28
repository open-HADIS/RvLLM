# vast.ai Instance Setup

## The PyTorch Docker Image Problem

The `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image runs a background process on startup that grabs ~70GB of GPU memory. You MUST kill it before running rvLLM.

### After every instance start/restart:

```bash
# Kill the rogue PyTorch process eating GPU memory
ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST "
  kill -9 \$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) 2>/dev/null
  sleep 3
  nvidia-smi | grep MiB | head -1  # Should show ~0MiB used
"
```

### Alternative: use a bare CUDA image

If provisioning a new instance, prefer `nvidia/cuda:12.6.3-devel-ubuntu22.04` which has no rogue process. You'll need to install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH=/root/.cargo/bin:$PATH
apt-get install -y pkg-config libssl-dev
```

## Quick Deploy

```bash
# 1. Rsync source (exclude target and .git)
rsync -avz --exclude target --exclude .git --exclude '.claude/worktrees' \
  -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
  /Users/andy/rvllm/ root@$HOST:/root/rvllm/

# 2. Build with CUDA
ssh -p $PORT root@$HOST "
  export PATH=/root/.cargo/bin:\$PATH
  cd /root/rvllm && cargo build --release --features cuda -p rvllm-server
"

# 3. Start server
ssh -p $PORT root@$HOST "
  export PATH=/root/.cargo/bin:\$PATH
  cd /root/rvllm
  RUST_LOG=info nohup ./target/release/rvllm serve \
    --model Qwen/Qwen2.5-1.5B \
    --gpu-memory-utilization 0.85 \
    --dtype half \
    --port 8000 > /tmp/rvllm.log 2>&1 &
"

# 4. Test
ssh -p $PORT root@$HOST "
  curl -s http://localhost:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{\"model\":\"Qwen/Qwen2.5-1.5B\",\"prompt\":\"Hello\",\"max_tokens\":8,\"temperature\":0}'
"
```

## Instance Management

```bash
# List instances
vastai show instances

# Stop/start (clears GPU memory from rogue processes)
vastai stop instance $ID
vastai start instance $ID

# Check SSH details
vastai show instances  # SSH Addr + SSH Port columns
```

## Common Issues

- **OOM on model load**: The rogue PyTorch process is eating GPU memory. Kill it first.
- **Address already in use**: Previous rvllm process still running. `kill -9 $(pgrep -x rvllm)`
- **cargo: command not found**: `export PATH=/root/.cargo/bin:$PATH`
- **openssl-sys build error**: `apt-get install -y pkg-config libssl-dev`
