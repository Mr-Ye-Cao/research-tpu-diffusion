# Communication/Compute Overlap Profiling Report - DiT Architecture

**Date:** January 13, 2026
**Author:** Profiling Session
**Target Model:** Wan2.1 DiT (Diffusion Transformer) 1.3B
**Hardware:** 8x NVIDIA RTX PRO 6000 Blackwell (96GB VRAM each)
**Software:** PyTorch 2.9.1+cu128, CUDA 12.8, NCCL 2.27.5

---

## Executive Summary

We profiled **Wan2.1 DiT (Diffusion Transformer)** communication patterns to measure comm/compute breakdown for the auto-sharding cost model.

### Key Findings

| Parallelism Strategy | Comm % | Potential Speedup |
|---------------------|--------|-------------------|
| **Tensor Parallelism** | 41.3% | **1.70x** |
| **Sequence Parallelism** | 61.0% | **2.57x** |

DistriFusion's activation reuse can hide communication latency by overlapping current step's communication with previous step's computation.

---

## 1. Why DiT (Not UNet)?

### Architecture Comparison

| Feature | UNet (SD 1.5) | DiT (Wan2.1) |
|---------|---------------|--------------|
| Core blocks | Conv + Attention | **Transformer (Attention + FFN)** |
| Convolutions | Heavy (~50%) | Light (VAE only) |
| Attention | Moderate (~10%) | **Heavy (dominant)** |
| Best parallelism | TP for attention blocks | **TP for all transformer layers** |

### Wan2.1-1.3B Architecture Specs

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 1536 |
| FFN hidden | 6144 (4x expansion) |
| Attention heads | 24 |
| Transformer layers | ~30 |
| Total parameters | 1.3B |

---

## 2. Experimental Setup

### Hardware
| Resource | Specification |
|----------|---------------|
| GPUs Used | 3x RTX PRO 6000 Blackwell (GPUs 5, 6, 7) |
| VRAM per GPU | 96 GB |
| Interconnect | PCIe (intranode) |

### Simulation Parameters
| Parameter | Value |
|-----------|-------|
| Sequence length | 4096 tokens |
| Hidden dimension | 1536 |
| Transformer layers | 30 |
| World size | 3 GPUs |

---

## 3. Results

### 3.1 NCCL Communication Bandwidth (DiT Tensor Sizes)

DiT uses larger, more contiguous tensors than UNet:

| Tensor Shape | Size (MB) | Latency (ms) | Bandwidth (Gbps) |
|--------------|-----------|--------------|------------------|
| (1, 1024, 1536) | 3.0 | 0.174 | 137.9 |
| (1, 4096, 1536) | 12.0 | 0.523 | **183.5** |
| (1, 8192, 1536) | 24.0 | 0.989 | **194.2** |
| (1, 16384, 1536) | 48.0 | 1.936 | **198.3** |

**Key Insight:** DiT achieves **198 Gbps** vs UNet's 126 Gbps due to larger tensors.

---

### 3.2 Tensor Parallelism Simulation

In TP, model weights are split across GPUs:
- **Column Parallel**: QKV projection → all-gather
- **Row Parallel**: Output projection → all-reduce
- **FFN**: Up projection (column) + Down projection (row) → all-reduce

| Metric | Value |
|--------|-------|
| Total Communication Time | 78.98 ms |
| Total Compute Time | 112.42 ms |
| **Comm/Compute Ratio** | **0.703** |
| **Communication Percentage** | **41.3%** |

**Per-Layer Breakdown:**
- Each transformer layer: ~2.6ms comm + ~3.7ms compute
- 30 layers total

---

### 3.3 Sequence Parallelism Simulation

In SP, the sequence dimension is split:
- Each GPU handles `seq_len / world_size` tokens
- **All-gather K, V** for full sequence attention

| Metric | Value |
|--------|-------|
| Total Communication Time | 20.41 ms |
| Total Compute Time | 13.03 ms |
| **Comm/Compute Ratio** | **1.57** |
| **Communication Percentage** | **61.0%** |

**Note:** SP has higher comm overhead because attention requires full sequence.

---

### 3.4 Comparison: DiT vs UNet

| Architecture | TP Comm % | TP Speedup | SP Comm % | SP Speedup |
|--------------|-----------|------------|-----------|------------|
| **DiT (Wan2.1)** | **41.3%** | **1.70x** | 61.0% | 2.57x |
| UNet (SD 1.5) | 46.9% | 1.88x | N/A | N/A |

DiT has **lower TP overhead** due to:
1. More compute-heavy attention
2. Larger, bandwidth-efficient tensors
3. Regular transformer structure (easier to parallelize)

---

## 4. DistriFusion Overlap Analysis

### 4.1 The Technique

DistriFusion exploits **temporal similarity between diffusion timesteps**:

```
Without Overlap:
  Step t:   [Compute] ──→ [Communicate] ──→ [Wait]
  Step t+1:                                 [Compute] ──→ [Communicate]

With DistriFusion:
  Step t:   [Compute (stale)] + [Comm (fresh)]
  Step t+1: [Compute (stale)] + [Comm (fresh)]
            ↑ Communication hidden behind compute!
```

### 4.2 Speedup Potential

| Strategy | Comm % | Without Overlap | With Overlap | **Speedup** |
|----------|--------|-----------------|--------------|-------------|
| **Tensor Parallel** | 41.3% | 191.4 ms | 112.4 ms | **1.70x** |
| **Sequence Parallel** | 61.0% | 33.4 ms | 13.0 ms | **2.57x** |

### 4.3 Implementation Reference

From [DistriFuser](https://github.com/mit-han-lab/distrifuser):
- Patches HuggingFace diffusers pipeline functions
- For DiT: patch `WanPipeline` transformer blocks
- Uses `DistriConfig` for distributed parameters

---

## 5. Cost Model Parameters

For the auto-sharding cost model:

```python
# DiT Communication Costs (3 GPUs, PCIe)
DIT_ALL_REDUCE_BW_GBPS = 198.3      # Large tensors (48MB)
DIT_ALL_REDUCE_BW_SMALL_GBPS = 137.9  # Small tensors (3MB)
DIT_ALL_REDUCE_LATENCY_MS = 0.17    # Base latency

# DiT Parallelism Overhead
DIT_TP_COMM_PERCENTAGE = 0.413      # 41.3%
DIT_SP_COMM_PERCENTAGE = 0.610      # 61.0%

# DistriFusion Overlap Factors
DIT_TP_OVERLAP_SPEEDUP = 1.70
DIT_SP_OVERLAP_SPEEDUP = 2.57

# Per-layer costs (30 layers)
DIT_TP_COMM_PER_LAYER_MS = 2.63
DIT_TP_COMPUTE_PER_LAYER_MS = 3.74
```

---

## 6. Strategy Selection Guidelines

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| Intranode (NVLink/ICI) | **Tensor Parallel** | 1.70x speedup, low latency |
| Internode (DCN) | Data Parallel | Avoid high comm overhead |
| Long video sequences | Sequence Parallel | 2.57x speedup potential |
| Memory constrained | TP + SP hybrid | Split model and sequence |

### Network Topology Requirements

| Interconnect | Bandwidth | TP Viable? | SP Viable? |
|--------------|-----------|------------|------------|
| NVLink 4.0 | 900 Gbps | **Yes** | **Yes** |
| PCIe 5.0 | 200 Gbps | **Yes** | Marginal |
| ICI (TPU) | 400 Gbps | **Yes** | **Yes** |
| DCN | 10-100 Gbps | No | No |

---

## 7. Recommendations

1. **Implement DistriFusion for Wan2.1 DiT**
   - Patch transformer blocks in `WanPipeline`
   - Expected **1.70x speedup** with TP
   - Reference: DistriFuser's patching approach

2. **Prefer TP over SP for inference**
   - TP: 41% comm → 1.70x speedup
   - SP: 61% comm → harder to overlap effectively

3. **Use high-bandwidth interconnects**
   - DiT achieves 198 Gbps on PCIe
   - NVLink/ICI will be even more effective

4. **Scale with Data Parallelism across nodes**
   - DP has near-zero comm overhead for inference
   - Use for batch-level scaling

---

## 8. Files and Reproduction

### Result Files
```
./profiling_results/dit_distributed/20260113_063738_gpus3/
└── all_results.json     # All DiT profiling results
```

### Scripts
```bash
# DiT distributed profiling
CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 profile_dit_distributed.py

# View results
cat ./profiling_results/dit_distributed/*/all_results.json | python -m json.tool
```

---

## 9. Conclusion

Profiling **Wan2.1 DiT** reveals:

| Finding | Value |
|---------|-------|
| TP Communication Overhead | **41.3%** |
| SP Communication Overhead | **61.0%** |
| DistriFusion TP Speedup | **1.70x** |
| DistriFusion SP Speedup | **2.57x** |
| NCCL Bandwidth (large tensors) | **198 Gbps** |

These measurements provide critical inputs for the auto-sharding cost model to optimize parallelism strategy selection for Wan2.1 across heterogeneous TPU/GPU networks.

---

## Appendix: Raw Data

### A.1 All-Reduce Bandwidth
| Tensor Shape | Size (MB) | Time (ms) | BW (Gbps) |
|--------------|-----------|-----------|-----------|
| (1, 1024, 1536) | 3.0 | 0.174 | 137.9 |
| (1, 4096, 1536) | 12.0 | 0.523 | 183.5 |
| (1, 8192, 1536) | 24.0 | 0.989 | 194.2 |
| (1, 16384, 1536) | 48.0 | 1.936 | 198.3 |

### A.2 All-Gather Latency
| Tensor Shape | Size (MB) | Time (ms) |
|--------------|-----------|-----------|
| (1, 1024, 1536) | 3.0 | 0.22 |
| (1, 4096, 1536) | 12.0 | 0.71 |
| (1, 8192, 1536) | 24.0 | 1.37 |
| (1, 16384, 1536) | 48.0 | 2.71 |
