# Claude Notes

## Environment Setup

**Conda Environment:** `tpu-diffusion`
- Python 3.11
- Location: `~/miniconda3/envs/tpu-diffusion`

**Activation:**
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tpu-diffusion
```

## Project Structure

- `Multi-Sharding-MaxDiffusion/` - Git submodule of the working fork of MaxDiffusion
  - Remote: https://github.com/AlbedoWang/Multi-Sharding-MaxDiffusion.git

## Key Dependencies

- JAX 0.8.2 with CUDA 12 (`pip install -U "jax[cuda12]"`)
- Flax 0.12.2
- PyTorch 2.9.1 (CPU version)
- Transformers 4.48.1
- MaxDiffusion installed in dev mode: `cd Multi-Sharding-MaxDiffusion && pip install -e .`

## Hardware

- Local workstation: 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM each)
- Target: TPU v5e (16 devices) via Google TRC (temporarily reclaimed)
- Alternative: Rice cluster L40S nodes (access pending)

## Known Issues

### NCCL Multi-GPU Issue on Blackwell (2026-01-05) - COMPREHENSIVE INVESTIGATION

**Problem:** JAX/XLA NCCL operations fail with `corrupted comm object` on Blackwell GPUs (compute capability 12.0)

**Impact:** Multi-GPU MaxDiffusion inference fails for ALL models (Wan, SDXL, etc.); single-GPU works fine

**Status:** CONFIRMED BUG - XLA/PJRT NCCL communicator race condition on Blackwell

**Root Cause (identified 2026-01-05):**
Race condition in XLA's PJRT client where execution threads try to use NCCL communicators BEFORE initialization threads complete setup. Analysis of NCCL DEBUG logs shows:
- Thread A (execution): Tries `ncclCommCount()` on uninitialized comm → FAILS
- Thread B (init): `ncclCommInitRankConfig` START → COMPLETE (SUCCESS)
- Thread A: Still fails because it accessed before init

This is specific to Blackwell architecture - the timing difference between comm creation and usage exposes the race condition that doesn't manifest on older GPUs.

**Error signature:**
```
external/nccl_archive/src/misc/argcheck.cc:39 NCCL WARN Error: corrupted comm object detected
NCCL operation ncclCommCount(comm_, &count) failed: invalid argument
```

**Investigation findings (2026-01-05):**

1. **What WORKS:**
   - Basic NCCL collectives (all-reduce, all-gather, psum)
   - Simple nnx.Module with multi-GPU sharding
   - Transformer-like modules (2-30 layers) with sharding constraints
   - Batched `jax.device_put()` with 500+ params (~14GB)
   - All tested with `test_nccl_minimal.py`, `test_batched_sharding.py`

2. **What FAILS:**
   - MaxDiffusion inference (ALL models: Wan 1.3B, SDXL)
   - Fails with ANY number of GPUs (2, 4, 8)
   - The NCCL communicator initializes successfully but XLA tries to use it before init completes

3. **XLA Source Analysis (xla/xla/backends/gpu/collectives/):**
   - `nccl_communicator.h:207-222`: Documents NCCL thread safety requirements
   - `nccl_communicator.cc`: `SingleThreadedExecutor` used for async execution
   - `gpu_cliques.cc:114-123`: Blackwell-specific handling (max 32 channels)
   - The race condition is in `pjrt_stream_executor_client.cc:2091`

4. **Workarounds ATTEMPTED (ALL FAILED):**
   - `NCCL_P2P_DISABLE=1` - No effect
   - NCCL downgrade to 2.27.3 - No effect
   - NCCL upgrade to 2.28.9 - No effect
   - `NCCL_SHM_DISABLE=0/1`, `NCCL_NET_GDR_LEVEL=0`, `NCCL_IB_DISABLE=1` - No effect
   - `NCCL_MAX_NCHANNELS=1`, `NCCL_MIN_NCHANNELS=1` - No effect
   - `NCCL_CUMEM_HOST_ENABLE=0` - Gemini suggestion, no effect
   - `NCCL_P2P_LEVEL=PIX` - Gemini suggestion, no effect
   - `--xla_gpu_shard_autotuning=false` - No effect
   - `--xla_gpu_nccl_async_execution=true` - Flags applied but still fails
   - `--xla_gpu_nccl_blocking_communicators=false` - No effect
   - `--xla_gpu_enable_nccl_comm_splitting=false` - No effect
   - `--xla_gpu_disable_async_collectives=ALLREDUCE,...` - No effect
   - `--xla_gpu_collectives_use_persistent_cliques=true` - No effect
   - `--xla_gpu_enable_command_buffer=` (empty) - No effect
   - `--xla_gpu_experimental_enable_nccl_symmetric_buffers=true` - Different error (ncclCommWindowRegister fails)
   - `NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1` combined - No effect
   - `NCCL_CUMEM_ENABLE=0` - No effect
   - `NCCL_ASYNC_ERROR_HANDLING=0` - No effect
   - `jax_use_shardy_partitioner=False` - No effect
   - `jax[cuda13]` - Failed (requires driver >= 580, current is 570.195.3)
   - IOMMU/ACS configuration - NOT TESTED (requires admin/reboot)

4. **Related Issues Found:**
   - [RTX 5090 P2P issues (NCCL #1637)](https://github.com/NVIDIA/nccl/issues/1637) - Fixed in NCCL 2.26.2+
   - [NCCL shared memory bug on Blackwell (nccl-tests #287)](https://github.com/NVIDIA/nccl-tests/issues/287)
   - [JAX Blackwell issues (#33910)](https://github.com/jax-ml/jax/issues/33910) - Unresolved
   - [RTX PRO 6000 P2P issues (sglang #15181)](https://github.com/sgl-project/sglang/issues/15181)
   - [Level1Techs forum - Blackwell P2P workarounds](https://forum.level1techs.com/t/dual-rtx-pro-6000-blackwell-max-q-how-to-make-p2p-nccl-work/242403)

5. **Environment:**
   - JAX 0.8.2, NCCL 2.28.9, CUDA 12.8/12.9, Driver 570.195.3
   - RTX PRO 6000 Blackwell (compute capability 12.0, sm_120)
   - Python 3.11, Flax 0.12.2

**Workaround:** Use single GPU: `CUDA_VISIBLE_DEVICES=0`

**Test files:**
- `Multi-Sharding-MaxDiffusion/test_nccl_minimal.py` - Minimal tests (ALL PASS)
- `Multi-Sharding-MaxDiffusion/run_debug.sh` - Debug script for MaxDiffusion

**Next steps:**
1. **Try IOMMU/ACS configuration** - From Level1Techs forum, disabling IOMMU or using passthrough mode (`iommu=pt`) with ACS disabled in BIOS fixed similar issues for RTX PRO 6000 Blackwell users
2. **File JAX GitHub issue** with detailed reproduction steps (link to existing #30786, #33910)
3. **Upgrade GPU driver** to >= 580 for CUDA 13 support (requires admin access)
4. **Test on Rice cluster L40S nodes** (older architecture, should work)
5. **Monitor releases:**
   - JAX updates for Blackwell fixes
   - NVIDIA JAX container (currently has best Blackwell support)
   - NCCL 2.29 (Q4 2025 roadmap mentions Blackwell optimizations)
