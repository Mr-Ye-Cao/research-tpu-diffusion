# Distributed Optimization for Diffusion Models on TPUs

## Project Overview

This project develops a distributed optimization system for diffusion models targeting heterogeneous TPU networks. The goal is to implement and evaluate multiple parallelism strategies that account for the bandwidth differences between intranode (ICI) and internode (DCN) connections in TPU clusters.

**Target Venue:** SC 2026 (SuperComputing Conference)  
**Submission Timeline:** March–April 2026

## Problem Statement

TPU networks exhibit heterogeneous communication characteristics:
- **Intranode (ICI):** High-bandwidth interconnect similar to NVLink
- **Internode (DCN):** Lower-bandwidth connections that route through CPU

When deploying diffusion models across 16+ TPUs, naive parallelism strategies (e.g., pure pipeline parallelism) fail to account for this heterogeneity, leading to suboptimal performance. The challenge is to automatically determine the optimal combination of parallelism strategies given the network topology.

## Technical Approach

### Parallelism Strategies to Implement

1. **Tensor Parallelism (TP):** Split model weights across devices
2. **Data Parallelism (DP):** Replicate model, split batch across devices  
3. **Sequence Parallelism (SP):** Split input sequence across devices — essential for Wan 2.1 due to long input sequences

### Key Insight: Diffusion-Specific Optimization

Unlike language models, diffusion models predict noise with small norm differences between adjacent timesteps. This enables:
- **Reusing previous timestep results** to reduce computation
- **Overlapping communication with computation** by leveraging temporal similarity

Reference: [DistriFusion paper](https://arxiv.org/html/2402.19481v1)

### Auto-Sharding Approach

Inspired by Stanford's [Alpa](https://github.com/alpa-projects/alpa), but at a higher abstraction level:
1. Enumerate different sharding strategies for each layer
2. Let JAX/XLA compile to low-level MLIR
3. Extract communication graphs from MLIR
4. Use cost model to select optimal sharding configuration
5. Validate with actual performance measurements

## Codebase

**Base Library:** [MaxDiffusion](https://github.com/AI-Hypercomputer/maxdiffusion) (Google's JAX-based diffusion library)  
**Working Fork:** [Multi-Sharding-MaxDiffusion](https://github.com/AlbedoWang/Multi-Sharding-MaxDiffusion)  
**Target Model:** Wan 2.1 (1.3B and 14B variants)

## Hardware Resources

| Resource | Specs | Status |
|----------|-------|--------|
| Google TRC TPUs | 16× TPU v5e | Temporarily reclaimed |
| Ye's Workstation | 8× RTX PRO 6000 Blackwell (96GB each) | Available |
| Rice Cluster | L40S nodes | Access pending (VPN/SSH forwarding needed) |

## Immediate Tasks

### Phase 1: GPU Portability (Current)
- [ ] Get MaxDiffusion running on GPU (single-GPU verified, multi-GPU untested)
- [ ] Test multi-GPU support on Rice cluster L40S nodes
- [ ] Resolve Rice cluster access (SSH forwarding through VPN)

### Phase 2: Parallelism Implementation
- [ ] Implement Tensor Parallelism in JAX
- [ ] Implement Sequence Parallelism in JAX
- [ ] Implement DistriFusion-style timestep reuse

### Phase 3: Auto-Sharding
- [ ] Build cost model for heterogeneous TPU networks
- [ ] Extract communication patterns from XLA/MLIR
- [ ] Implement sharding strategy search

## References

- [MaxDiffusion Repository](https://github.com/AI-Hypercomputer/maxdiffusion)
- [Alpa: Automating Inter- and Intra-Operator Parallelism](https://github.com/alpa-projects/alpa)
- [DistriFusion: Distributed Inference for Diffusion Models](https://arxiv.org/html/2402.19481v1)
- [Project Notes (Google Doc)](https://docs.google.com/document/d/18AJ3XkMzGlLnufPswHHnQ6uDtxf8NarsKtaC9uzQUaM/edit?pli=1&tab=t.0)

## Success Criteria

1. Demonstrate measurable speedup over naive parallelism on 16 TPUs
2. Show that auto-selected sharding outperforms manual configuration
3. Validate approach generalizes across TPU and GPU hardware
4. Publish findings at SC 2026