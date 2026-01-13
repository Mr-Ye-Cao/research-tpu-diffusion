# Communication/Compute Overlap Profiling Report

**Date:** January 13, 2026
**Author:** Profiling Session
**Hardware:** 8x NVIDIA RTX PRO 6000 Blackwell (96GB VRAM each)
**Software:** PyTorch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5

---

## Executive Summary

We profiled Stable Diffusion inference to measure communication vs. compute times and quantify the overlap potential using DistriFusion's activation reuse technique. **Key finding: In Tensor Parallelism, communication consumes 46.9% of execution time, enabling a potential 1.88x speedup through DistriFusion-style overlap.**

---

## 1. Objectives

1. Measure compute operation times in diffusion model inference
2. Measure NCCL communication bandwidth and latency
3. Quantify communication overhead in different parallelism strategies
4. Estimate DistriFusion overlap potential for the auto-sharding cost model

---

## 2. Experimental Setup

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Model | stable-diffusion-v1-5 |
| Image Size | 512×512 |
| Inference Steps | 20 |
| Precision | FP16 |
| Scheduler | DPMSolver++ |

### Hardware Configuration
| Resource | Specification |
|----------|---------------|
| GPUs Used | 3x RTX PRO 6000 Blackwell (GPUs 5, 6, 7) |
| VRAM per GPU | 96 GB |
| Interconnect | PCIe (intranode) |
| Driver | 570.195.03 |

---

## 3. Results

### 3.1 Single GPU Baseline

**Performance:**
- **Mean inference time:** 442.35 ms (20 steps)
- **Time per step:** 22.12 ms

**Compute Breakdown by Operation Type:**

| Operation Category | CUDA Time (ms) | Percentage |
|--------------------|----------------|------------|
| Matrix Multiply (GEMM) | 736.14 | 22.0% |
| Convolutions (cuDNN) | 668.36 | 19.9% |
| Normalization (Group/Layer) | 118.72 | 3.5% |
| Flash Attention | 103.04 | 3.1% |
| Memory Operations | 64.23 | 1.9% |
| Activations (GELU/SiLU) | 42.10 | 1.3% |
| Other (incl. profiler) | 1619.22 | 48.3% |

**Key Observation:** MatMul and Convolutions dominate compute at ~42% combined. These are the primary targets for parallelization.

---

### 3.2 NCCL Communication Bandwidth

We measured all-reduce performance across typical UNet activation tensor sizes:

| Tensor Shape | Size (MB) | Latency (ms) | Bandwidth (Gbps) |
|--------------|-----------|--------------|------------------|
| (1, 320, 64, 64) | 2.50 | 0.159 | **126.0** |
| (1, 640, 32, 32) | 1.25 | 0.108 | 92.3 |
| (1, 1280, 16, 16) | 0.625 | 0.097 | 51.4 |
| (1, 2560, 8, 8) | 0.312 | 0.059 | **42.4** |

**Key Observations:**
1. Larger tensors achieve higher bandwidth efficiency (126 Gbps)
2. Small tensors are latency-bound (42 Gbps for 0.3MB)
3. All-gather shows similar patterns with ~1.4x overhead vs all-reduce

---

### 3.3 Data Parallel Inference

| Metric | Value |
|--------|-------|
| World Size | 3 GPUs |
| Mean Inference Time | 416.96 ms |
| NCCL Overhead | 0.34 ms (0.0%) |

**Key Observation:** Data Parallelism has negligible communication overhead during forward pass. Communication only occurs during gradient synchronization (backward pass), making DP ideal for inference with sufficient batch size.

---

### 3.4 Tensor Parallelism Simulation

We simulated TP communication patterns for the UNet architecture:

| UNet Block | Seq Length | Hidden Dim | Layers | Comm (ms) | Compute (ms) |
|------------|------------|------------|--------|-----------|--------------|
| Down Block 1 | 4096 | 320 | 2 | 4.62 | 6.92 |
| Down Block 2 | 1024 | 640 | 2 | 0.66 | 0.49 |
| Down Block 3 | 256 | 1280 | 2 | 0.57 | 0.73 |
| Mid Block | 64 | 1280 | 1 | 0.14 | 0.17 |
| Up Block 1 | 256 | 1280 | 3 | 0.63 | 0.32 |
| Up Block 2 | 1024 | 640 | 3 | 0.77 | 0.32 |
| Up Block 3 | 4096 | 320 | 3 | 1.18 | 0.75 |

**Aggregate TP Results:**

| Metric | Value |
|--------|-------|
| Total Communication Time | 8.56 ms |
| Total Compute Time | 9.70 ms |
| **Comm/Compute Ratio** | **0.883** |
| **Communication Percentage** | **46.9%** |

---

## 4. DistriFusion Overlap Analysis

### 4.1 The Problem

In Tensor Parallelism, each GPU must:
1. Compute local portion of matmul/attention
2. **Communicate** (all-reduce/all-gather) to synchronize activations
3. Proceed to next layer

This creates a sequential dependency where **46.9% of time is spent on communication**.

### 4.2 DistriFusion Solution

DistriFusion exploits **temporal similarity between diffusion timesteps**:
- Activations at timestep `t` are similar to timestep `t-1`
- Instead of waiting for fresh activations, use stale (t-1) activations for compute
- Overlap current step's communication with computation using stale data

```
Traditional:  [Compute t] -> [Communicate t] -> [Compute t+1] -> ...
DistriFusion: [Compute t (stale)] + [Communicate t-1] -> [Compute t+1 (stale)] + [Communicate t] -> ...
```

### 4.3 Potential Speedup

With perfect overlap, communication time can be completely hidden:

| Scenario | Time | Speedup |
|----------|------|---------|
| No Overlap (baseline) | Compute + Comm = 18.26 ms | 1.00x |
| Perfect Overlap | max(Compute, Comm) = 9.70 ms | **1.88x** |

**Formula:** `Speedup = 1 / (1 - comm_percentage) = 1 / (1 - 0.469) = 1.88x`

---

## 5. Implications for Auto-Sharding

### 5.1 Parallelism Strategy Selection

| Strategy | Comm Overhead | Best For |
|----------|---------------|----------|
| Data Parallel | ~0% (inference) | Large batch, inference |
| Tensor Parallel | ~47% | Small batch, needs overlap |
| Sequence Parallel | Variable | Long sequences |

### 5.2 Network Topology Considerations

| Interconnect | Bandwidth | Recommendation |
|--------------|-----------|----------------|
| NVLink/ICI (intranode) | 400-900 Gbps | Prefer for TP |
| PCIe (intranode) | ~126 Gbps | Acceptable for TP |
| DCN (internode) | 10-100 Gbps | Avoid TP, use DP |

### 5.3 Cost Model Parameters

For the auto-sharding cost model, use these measured values:

```python
# Communication costs (3 GPUs, PCIe)
ALL_REDUCE_LATENCY_MS = 0.06  # Base latency
ALL_REDUCE_BW_GBPS = 126.0    # Peak bandwidth (large tensors)
ALL_REDUCE_BW_SMALL_GBPS = 42.4  # Small tensor bandwidth

# Compute costs (per diffusion step)
COMPUTE_TIME_PER_STEP_MS = 22.12

# Overlap factor (with DistriFusion)
MAX_OVERLAP_SPEEDUP = 1.88
COMM_PERCENTAGE_TP = 0.469
```

---

## 6. Recommendations

1. **Implement DistriFusion activation reuse** - Essential for making TP viable, provides up to 1.88x speedup

2. **Prefer intranode TP** - Use high-bandwidth interconnects (NVLink/ICI) for tensor parallelism to minimize communication overhead

3. **Use DP for inference** - When batch size allows, Data Parallelism has near-zero communication overhead

4. **Consider tensor size in sharding** - Larger tensors achieve better bandwidth utilization; avoid splitting small tensors across many devices

5. **Profile target hardware** - These results are for PCIe interconnect; NVLink/ICI will show different comm/compute ratios

---

## 7. Files and Reproduction

### Result Files
```
./profiling_results/
├── single_gpu/
│   └── 20260113_061347/
│       ├── results.json          # Single GPU metrics
│       └── *.json                # TensorBoard traces
└── distributed/
    └── 20260113_061620_gpus3/
        ├── all_results.json      # All distributed metrics
        └── rank_*/               # Per-rank traces
```

### Reproduction Commands
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=5 python profile_single_gpu.py

# Distributed (3 GPUs)
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 profile_distributed.py

# View results
tensorboard --logdir ./profiling_results
```

---

## 8. Conclusion

Our profiling reveals that **Tensor Parallelism incurs ~47% communication overhead** in diffusion model inference. However, **DistriFusion's activation reuse technique can achieve up to 1.88x speedup** by overlapping communication with computation. This data provides critical inputs for the auto-sharding cost model that will drive parallelism strategy selection across heterogeneous TPU/GPU networks.

---

## Appendix: Raw Data

### A.1 Single GPU Timing (3 runs)
| Run | Total (ms) | Per Step (ms) |
|-----|------------|---------------|
| 1 | 450.41 | 22.52 |
| 2 | 436.37 | 21.82 |
| 3 | 440.26 | 22.01 |
| **Mean** | **442.35** | **22.12** |

### A.2 Distributed Inference (3 runs, 3 GPUs)
| Run | Total (ms) |
|-----|------------|
| 1 | 419.37 |
| 2 | 417.78 |
| 3 | 413.74 |
| **Mean** | **416.96** |

### A.3 All-Gather Latency
| Shape | Size (MB) | Latency (ms) |
|-------|-----------|--------------|
| (1, 320, 64, 64) | 2.50 | 0.230 |
| (1, 640, 32, 32) | 1.25 | 0.133 |
| (1, 1280, 16, 16) | 0.625 | 0.094 |
| (1, 2560, 8, 8) | 0.312 | 0.067 |
