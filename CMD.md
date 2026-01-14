# Command Reference

Commands used during profiling setup and execution. Run these to reproduce results.

## Environment Setup

```bash
# Activate conda environment
conda activate tpu-diffusion

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Verify NCCL
python -c "import torch.distributed as dist; print(f'NCCL available: {dist.is_nccl_available()}')"

# Check GPU info and availability
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi  # Check which GPUs are in use
```

## Package Installation

```bash
# Install PyTorch with CUDA 12.8 support
pip uninstall torch torchvision -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install diffusers for PyTorch-based diffusion models
pip install diffusers accelerate

# Install tensorboard for profiler visualization
pip install tensorboard
```

## Profiling Commands

### Single GPU Profiling
```bash
# IMPORTANT: Check nvidia-smi first to find free GPUs
# Only use GPUs that are not in use by other processes

# Run on a specific free GPU (e.g., GPU 5)
conda activate tpu-diffusion
cd /home/ye/workspace/tpu/research-tpu-diffusion
CUDA_VISIBLE_DEVICES=5 python profile_single_gpu.py
```

### Multi-GPU Distributed Profiling
```bash
# IMPORTANT: Only use free GPUs (check nvidia-smi first)

# 3 GPUs (e.g., GPUs 5,6,7 if free)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 profile_distributed.py

# 2 GPUs
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 profile_distributed.py
```

## TensorBoard Visualization

```bash
# View all profiling results
tensorboard --logdir ./profiling_results

# View specific single GPU run
tensorboard --logdir ./profiling_results/single_gpu/20260113_061347

# View specific distributed run
tensorboard --logdir ./profiling_results/distributed/20260113_061620_gpus3
```

## Environment Variables (if needed)

```bash
# NCCL debugging (if issues arise)
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# For Blackwell GPUs - disable P2P if issues arise
export NCCL_P2P_DISABLE=1

# Set specific GPUs
export CUDA_VISIBLE_DEVICES=5,6,7
```

## DiT (Wan2.1) Profiling Commands

```bash
# DiT distributed profiling (simulates TP/SP patterns)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 profile_dit_distributed.py

# DiT single GPU profiling (CogVideoX - requires download)
CUDA_VISIBLE_DEVICES=5 python profile_dit_single_gpu.py

# Real Wan2.1-T2V-1.3B profiling (uses actual model)
CUDA_VISIBLE_DEVICES=5 python profile_wan_real.py
```

## Wan2.1 Model Setup

```bash
# Clone Wan2.1 repo
git clone https://github.com/Wan-Video/Wan2.1.git

# Download model (huggingface-cli)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./Wan2.1-T2V-1.3B

# Patch model to use SDPA instead of flash_attn (if flash_attn not installed)
# Edit Wan2.1/wan/modules/model.py line 10:
# Change: from .attention import flash_attention
# To:     from .attention import attention as flash_attention
```

## Key Results from This Session

### UNet (SD 1.5) - Single GPU
- Mean inference time: 442.35 ms (20 steps)
- Time per step: 22.12 ms

### UNet - Distributed (GPUs 5,6,7)
- NCCL All-Reduce bandwidth: 42-126 Gbps
- TP Simulation comm percentage: 46.9%
- DistriFusion overlap potential: 1.88x speedup

### DiT (Wan2.1) - Distributed (GPUs 5,6,7) - Simulated
- NCCL All-Reduce bandwidth: 138-198 Gbps (larger tensors)
- TP Simulation comm percentage: 41.3%
- SP Simulation comm percentage: 61.0%
- DistriFusion TP overlap potential: 1.70x speedup
- DistriFusion SP overlap potential: 2.57x speedup

### Real Wan2.1-T2V-1.3B - Single GPU (GPU 5)
- Video: 17 frames @ 480x832
- Inference steps: 10
- Mean inference time: 4,587.62 ms
- Time per step: 458.76 ms
- Attention compute: 10.0%
- Linear/FFN compute: 27.7%
- Attention+Linear: 37.7% (TP-parallelizable)

## Results Files

```bash
# View UNet single GPU results
cat ./profiling_results/single_gpu/20260113_061347/results.json

# View UNet distributed results
cat ./profiling_results/distributed/20260113_061620_gpus3/all_results.json

# View DiT distributed results (simulated)
cat ./profiling_results/dit_distributed/20260113_063738_gpus3/all_results.json

# View real Wan2.1 results
cat ./profiling_results/wan_real/20260113_071418/results.json
```
