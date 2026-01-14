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

## Progress Log

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
