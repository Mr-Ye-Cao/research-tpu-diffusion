## Task: Communication/Compute Overlap Profiling with torch.profiler

**Objective:** 
Use `torch.profiler` to trace and measure communication and compute times in the distributed diffusion model inference pipeline, and quantify how much overlap is achievable between them.

**Context:**
- The project is building an auto-sharding system for diffusion models on TPU/GPU networks
- DistriFusion's core technique involves reusing activations from previous timesteps to overlap communication with computation (exploiting temporal similarity between diffusion timesteps)
- Different parallelism strategies (Tensor Parallel, Data Parallel, Sequence Parallel) have different communication requirements
- The network topology has heterogeneous bandwidth: intranode (ICI/NVLink) is high bandwidth, internode (DCN) is lower bandwidth
- Understanding the actual communication vs compute breakdown is essential for building accurate cost models that will drive the auto-sharding decisions

**What to Measure:**
- How long communication operations take
- How long compute operations take  
- Current overlap (if any) between communication and compute
- Potential for additional overlap using DistriFusion's activation reuse technique

**Why Urgent:**
Kaijian needs this profiling data to inform the cost model design, which determines how the auto-search algorithm assigns sharding strategies across the heterogeneous TPU/GPU network.