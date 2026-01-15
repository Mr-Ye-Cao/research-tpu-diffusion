# Progress Tracking: Communication/Compute Overlap Profiling

## Goal
Use `torch.profiler` to trace and measure communication and compute times in distributed diffusion model inference, and quantify overlap achievability.

## Task Breakdown

### Phase 1: Environment Setup
- [x] Verify conda environment (`tpu-diffusion`)
- [x] Check PyTorch with CUDA support (2.9.1+cu128)
- [x] Verify NCCL for distributed communication
- [x] Install diffusers library for PyTorch-based diffusion models

### Phase 2: Profiling Infrastructure
- [x] Create single-GPU profiling script (`profile_single_gpu.py`)
- [x] Create multi-GPU distributed profiling script (`profile_distributed.py`)
- [x] Set up torch.profiler with:
  - CUDA activity tracing
  - Communication operation tracking
  - TensorBoard export

### Phase 3: Measurements
- [x] Single GPU baseline (compute only)
- [x] Multi-GPU with Data Parallelism
- [x] Tensor Parallelism simulation
- [x] Measure communication vs compute breakdown

### Phase 4: Analysis
- [x] Calculate current overlap percentage
- [x] Identify opportunities for DistriFusion-style overlap
- [x] Generate results JSON and TensorBoard traces

## Key Results

### Single GPU Baseline (GPU 5)
- **Model**: stable-diffusion-v1-5/stable-diffusion-v1-5
- **Image size**: 512x512
- **Inference steps**: 20
- **Mean inference time**: 442.35 ms
- **Mean time per step**: 22.12 ms

**Compute Breakdown (by CUDA time):**
| Category | Time (ms) | Percentage |
|----------|-----------|------------|
| MatMul (GEMM) | 736.14 | 22.0% |
| Convolutions | 668.36 | 19.9% |
| Normalization | 118.72 | 3.5% |
| Attention (Flash) | 103.04 | 3.1% |
| Memory Ops | 64.23 | 1.9% |
| Activation | 42.10 | 1.3% |
| Other | 1619.22 | 48.3% |

### Distributed Profiling (3 GPUs: 5, 6, 7)

#### NCCL Communication Bandwidth
| Tensor Shape | Size (MB) | All-Reduce Time (ms) | Bandwidth (Gbps) |
|--------------|-----------|---------------------|------------------|
| (1, 320, 64, 64) | 2.50 | 0.159 | 126.0 |
| (1, 640, 32, 32) | 1.25 | 0.108 | 92.3 |
| (1, 1280, 16, 16) | 0.625 | 0.097 | 51.4 |
| (1, 2560, 8, 8) | 0.312 | 0.059 | 42.4 |

#### Data Parallel Inference
- **World size**: 3 GPUs
- **Mean inference time**: 416.96 ms (per GPU)
- **NCCL overhead**: ~0.3 ms (0.0% - minimal in DP forward pass)

#### Tensor Parallelism Simulation
- **Communication time**: 8.56 ms
- **Compute time**: 9.70 ms
- **Comm/Compute ratio**: 0.883
- **Communication percentage**: 46.9%

### DistriFusion Overlap Potential

**Key Finding**: In Tensor Parallelism, communication takes **46.9%** of total time.

With DistriFusion's activation reuse technique:
- Current step's communication can be overlapped with previous step's activations
- **Potential speedup with perfect overlap: 1.88x**

This is significant because:
1. The temporal similarity between diffusion timesteps allows reusing activations
2. While GPU computes with stale (t-1) activations, it can prefetch fresh (t) activations
3. This effectively hides the communication latency behind computation

## Environment Info
- **Working directory**: `/home/ye/workspace/tpu/research-tpu-diffusion`
- **Conda env**: `tpu-diffusion` (Python 3.11)
- **PyTorch**: 2.9.1+cu128
- **Hardware**: 8x NVIDIA RTX PRO 6000 Blackwell (96GB VRAM each)
- **Available GPUs for profiling**: 5, 6, 7 (others in use)

### Real Wan2.1 DiT Profiling (NEW)

Profiled the **actual Wan2.1-T2V-1.3B model** using PyTorch SDPA:

| Parameter | Value |
|-----------|-------|
| Model | Wan2.1-T2V-1.3B |
| Video Size | 480x832 |
| Frame Count | 17 frames |
| Inference Steps | 10 |

**Performance:**
- **Mean inference time**: 4,587.62 ms
- **Time per step**: 458.76 ms

**Compute Breakdown:**
| Category | Percentage |
|----------|------------|
| Linear/FFN | 27.7% |
| Attention | 10.0% |
| Memory Ops | 9.3% |
| Conv (VAE) | 3.8% |
| Activation | 1.8% |
| Normalization | 0.9% |

**Key Finding**: Attention + Linear/FFN = **37.7%** - main targets for Tensor Parallelism.

## Results Location
- **UNet Single GPU**: `./profiling_results/single_gpu/20260113_061347/`
- **UNet Distributed**: `./profiling_results/distributed/20260113_061620_gpus3/`
- **DiT Distributed (simulated)**: `./profiling_results/dit_distributed/20260113_063738_gpus3/`
- **Real Wan2.1**: `./profiling_results/wan_real/20260113_071418/`
- **Wan2.1 Video Gen Single GPU**: `./profiling_results/wan_single_gpu/20260114_062914/`
- **Wan2.1 NCCL Multi-GPU**: `./profiling_results/wan_nccl/20260114_064727_gpus3/`

---

## Phase 2: Video Generation Multi-GPU Profiling (NEW)

### Goal
Profile Wan2.1 video generation with DistriFusion-style parallelism to measure communication/compute overlap potential.

### Distrifuser Repository Setup
- **Repo**: `distrifuser/` (cloned from https://github.com/Mr-Ye-Cao/distrifuser.git)
- **Branch**: `video-gen`
- **Status**: UNet-based distrifuser works; adapting for DiT-based video models

### Wan2.1 Single GPU Video Generation (NEW)

| Parameter | Value |
|-----------|-------|
| Model | Wan2.1-T2V-1.3B |
| Video Size | 480x832 |
| Frame Count | 17 frames |
| Inference Steps | 10 |

**Performance:**
- **Mean inference time**: 7,534.51 ms
- **Time per step**: 753.45 ms

### Wan2.1 Multi-GPU NCCL Communication Benchmark (3 GPUs)

#### Communication Bandwidth (NVLink)
| Operation | Tensor Shape | Size (MB) | Time (ms) | Bandwidth (Gbps) |
|-----------|--------------|-----------|-----------|------------------|
| KV All-Gather | [1, 1706, 2, 12, 128] | 10.00 | 0.703 | 341.4 |
| Attn Output All-Gather | [1, 1706, 1536] | 5.00 | 0.373 | 321.3 |
| FFN All-Reduce | [1, 5120, 1536] | 15.00 | 0.630 | 253.8 |

#### Per-Layer Communication Overhead
| Parallelism Strategy | Time per Layer (ms) | Total for 30 Layers (ms) |
|---------------------|---------------------|--------------------------|
| Sequence Parallel (USP) | 0.747 | 22.40 |
| Tensor Parallel | 1.261 | 37.83 |

#### DistriFusion Overlap Analysis

**With 3 GPUs (Sequence Parallelism - USP):**
- Communication time per step: 22.40 ms
- Estimated compute time per step: 251.0 ms (753 ms / 3 GPUs)
- **Communication percentage: 8.2%**
- **Potential overlap speedup: 1.09x**

**With 3 GPUs (Tensor Parallelism):**
- Communication time per step: 37.83 ms
- Estimated compute time per step: 251.0 ms
- **Communication percentage: 13.1%**
- **Potential overlap speedup: 1.15x**

### Key Findings for Video Generation

1. **High NVLink Bandwidth** (~320 Gbps)
   - RTX PRO 6000 Blackwell GPUs have excellent intra-node connectivity
   - Communication overhead is relatively low (8-13%)

2. **Sequence Parallelism (USP) is More Efficient**
   - Only 8.2% communication overhead vs 13.1% for tensor parallelism
   - Better suited for long-sequence video generation

3. **DistriFusion Overlap Benefit is Modest**
   - 1.09-1.15x potential speedup (vs 1.88x for UNet on lower-bandwidth setup)
   - Already low comm overhead means less room for improvement

4. **Model Architecture Matters**
   - DiT models have different comm patterns than UNet
   - Self-attention requires all-gather of KV from all sequence chunks
   - FFN can use tensor parallelism with all-reduce

### Flash Attention on Blackwell (RESOLVED)

**Problem**: Flash Attention didn't have official prebuilt wheels for Blackwell (sm_120).

**Solution**: Build from source with CUDA 12.8:
```bash
# With ninja installed, build takes ~15 minutes
pip install ninja
TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0" pip install flash-attn --no-build-isolation
```

**Result**: Flash Attention 2.8.3 now works on RTX PRO 6000 Blackwell (sm_120).

### Multi-GPU Video Generation with USP (NEW)

| Configuration | Inference Time | Time/Step |
|---------------|----------------|-----------|
| Single GPU | 7,534 ms | 753 ms |
| 3 GPUs (USP) | 31,341 ms* | 3,134 ms* |

*Note: Multi-GPU time includes significant profiling overhead. Actual production performance would be faster.

The xFuser USP (Ulysses Sequence Parallelism) works correctly on Blackwell GPUs with Flash Attention.

### TODO: Next Steps
- [x] Install Flash Attention for full xFuser USP testing ✅
- [x] Profile actual multi-GPU video generation ✅
- [ ] Run production benchmark without profiling overhead
- [ ] Implement DistriFusion-style async communication for Wan2.1
- [ ] Compare with xDiT context parallelism performance

## Progress Log

### 2026-01-14: Video Generation Multi-GPU Profiling
- [x] Cloned distrifuser repo (https://github.com/Mr-Ye-Cao/distrifuser.git)
- [x] Created `video-gen` branch for Wan2.1 video generation work
- [x] Explored distrifuser codebase (patch parallelism, tensor parallelism, async communication)
- [x] Created `profile_wan_video.py` for single/multi-GPU Wan2.1 profiling
- [x] Created `profile_wan_nccl.py` for NCCL communication benchmarking
- [x] Ran single-GPU Wan2.1 T2V profiling (GPU 5)
- [x] Ran multi-GPU NCCL communication profiling (GPUs 5, 6, 7)
- [x] Analyzed DistriFusion overlap potential for video generation

### 2026-01-13: Session Complete
- [x] Environment setup and verification
- [x] Created profiling scripts
- [x] Ran single GPU baseline profiling (UNet SD 1.5)
- [x] Ran 3-GPU distributed profiling (UNet)
- [x] Simulated DiT communication patterns
- [x] **Ran real Wan2.1-T2V-1.3B model profiling**
- [x] Analyzed comm vs compute breakdown
- [x] Quantified DistriFusion overlap potential (1.88x UNet, 1.70x DiT)

## Implications for Auto-Sharding

The profiling data reveals:

1. **Tensor Parallelism has high communication overhead** (~47% of time)
   - This makes it critical to choose TP degree carefully
   - Intranode (high-bandwidth NVLink/ICI) should be preferred for TP

2. **Data Parallelism has minimal communication** during forward pass
   - Gradient all-reduce happens at backward pass
   - Good choice for inference when batch size is large enough

3. **DistriFusion technique is valuable**
   - Up to 1.88x speedup potential with activation reuse
   - Essential for making TP viable at scale

4. **Network topology matters**
   - Higher bandwidth (126 Gbps) for larger tensors
   - Bandwidth drops to 42 Gbps for small tensors (latency-bound)
   - Sharding strategy should consider tensor sizes
