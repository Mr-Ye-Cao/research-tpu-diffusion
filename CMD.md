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

---

## Phase 2: Video Generation Multi-GPU Profiling (2026-01-14)

### Distrifuser Repository Setup

```bash
cd /home/ye/workspace/tpu/research-tpu-diffusion

# Clone distrifuser repo
git clone https://github.com/Mr-Ye-Cao/distrifuser.git

# Create video-gen branch
cd distrifuser
git checkout -b video-gen
```

### Single GPU Wan2.1 Video Generation Profiling

```bash
cd /home/ye/workspace/tpu/research-tpu-diffusion

# Profile on a single free GPU (e.g., GPU 5)
CUDA_VISIBLE_DEVICES=5 python distrifuser/scripts/profile_wan_video.py \
    --mode single \
    --gpu_id 0 \
    --frame_num 17 \
    --num_steps 10 \
    --warmup_runs 1 \
    --profile_runs 2

# Results saved to: profiling_results/wan_single_gpu/<timestamp>/
```

### Multi-GPU NCCL Communication Profiling

```bash
cd /home/ye/workspace/tpu/research-tpu-diffusion

# Profile NCCL communication on 3 free GPUs (e.g., GPUs 5, 6, 7)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
    distrifuser/scripts/profile_wan_nccl.py

# Results saved to: profiling_results/wan_nccl/<timestamp>_gpus3/
```

### Multi-GPU Wan2.1 with USP (Requires Flash Attention)

```bash
# First, install Flash Attention for Blackwell GPUs (takes ~15 minutes)
pip install ninja  # Required for fast compilation
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0" pip install flash-attn --no-build-isolation

# Verify flash-attn works on Blackwell
python -c "import flash_attn; print(f'Flash Attention: {flash_attn.__version__}')"

# Run multi-GPU profiling with USP
cd /home/ye/workspace/tpu/research-tpu-diffusion

CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 \
    distrifuser/scripts/profile_wan_video.py \
    --mode multi \
    --ulysses_size 3 \
    --ring_size 1 \
    --frame_num 17 \
    --num_steps 10 \
    --warmup_runs 1 \
    --profile_runs 2
```

### Run Original Wan2.1 with xDiT Context Parallelism

```bash
cd /home/ye/workspace/tpu/research-tpu-diffusion/Wan2.1

# Run with Ulysses parallelism on 3 GPUs (requires Flash Attention)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 generate.py \
    --task t2v-1.3B \
    --size 480*832 \
    --ckpt_dir ../Wan2.1-T2V-1.3B \
    --prompt "Two anthropomorphic cats in boxing gear fight on a spotlighted stage." \
    --ulysses_size 3 \
    --ring_size 1 \
    --sample_steps 50 \
    --frame_num 81 \
    --t5_cpu
```

### View New Results

```bash
# View single GPU video generation results
cat profiling_results/wan_single_gpu/*/results.json | jq .

# View NCCL communication benchmark results
cat profiling_results/wan_nccl/*/results.json | jq .
```

### Key Results (2026-01-14)

#### Single GPU (Wan2.1-T2V-1.3B, 480x832, 17 frames, 10 steps)
- Mean inference time: **7,281 ms**
- Time per step: **728 ms**

#### Multi-GPU USP (3 GPUs with xFuser)
- Mean inference time: **31,517 ms**
- Time per step: **3,152 ms**
- **Result: 4.3x SLOWER than single GPU!**

#### Root Cause: xFuserLongContextAttention Overhead
| Configuration | Time | Overhead |
|---------------|------|----------|
| Flash Attention (1706 tokens) | 0.067 ms | baseline |
| xFuserLongContextAttention (same) | 0.922 ms | **13.73x** |
| Full sequence (5120 tokens) single GPU | 0.546 ms | - |
| USP (3 GPUs, 5120 tokens total) | 0.922 ms | **1.69x slower** |

#### Crossover Point Analysis (Frames vs Performance)
| Frames | Est. Tokens | Single GPU | 3 GPU USP | Speedup |
|--------|-------------|------------|-----------|---------|
| 17 | ~5K | 7,281 ms | 31,517 ms | 0.23x (4.3x slower) |
| 81 | ~24K | 33,616 ms | 45,199 ms | 0.74x (1.34x slower) |
| 129 | ~39K | 59,519 ms | 55,872 ms | **1.07x** |
| 177 | ~53K | 93,698 ms | 69,956 ms | **1.34x** |
| **241** | **~72K** | **147,802 ms** | **92,188 ms** | **1.60x** |

#### Conclusion
- **Crossover point: ~100-129 frames (~30-40K tokens)**
- **Speedup scales with sequence length**
- At 241 frames: 1.60x speedup (53% parallel efficiency)
- For short videos (< 100 frames): use single GPU
- For long videos (> 129 frames): USP provides significant speedup

### Benchmark Commands (No Profiler Overhead)

```bash
# Single GPU benchmark
CUDA_VISIBLE_DEVICES=5 python distrifuser/scripts/benchmark_wan_single.py

# Multi-GPU benchmark (USP)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 distrifuser/scripts/benchmark_wan_multi.py

# Test xFuserLongContextAttention overhead
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 distrifuser/scripts/test_usp_minimal.py
```

#### NCCL Communication (theoretical best case)
| Operation | Size (MB) | Time (ms) | Bandwidth (Gbps) |
|-----------|-----------|-----------|------------------|
| KV All-Gather | 10.00 | 0.703 | 341.4 |
| Attn All-Gather | 5.00 | 0.373 | 321.3 |
| FFN All-Reduce | 15.00 | 0.630 | 253.8 |
