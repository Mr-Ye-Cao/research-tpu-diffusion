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

## Key Results from This Session

### Single GPU (GPU 5)
- Mean inference time: 442.35 ms (20 steps)
- Time per step: 22.12 ms

### Distributed (GPUs 5,6,7)
- NCCL All-Reduce bandwidth: 42-126 Gbps
- TP Simulation comm percentage: 46.9%
- DistriFusion overlap potential: 1.88x speedup

## Results Files

```bash
# View single GPU results
cat ./profiling_results/single_gpu/20260113_061347/results.json

# View distributed results
cat ./profiling_results/distributed/20260113_061620_gpus3/all_results.json
```
