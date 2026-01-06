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

### NCCL Multi-GPU Issue on GPU (2026-01-05) - COMPREHENSIVE INVESTIGATION

**Problem:** JAX/XLA NCCL operations fail with `corrupted comm object` on multiple GPU architectures

**Impact:** Multi-GPU MaxDiffusion inference fails for ALL models (Wan, SDXL, etc.); single-GPU works fine

**Status:** ✅ FIXED - XLA/PJRT NCCL communicator race condition (workaround found)

**CRITICAL UPDATE (2026-01-05):** Issue also affects H100 GPUs (compute capability 9.0), not just Blackwell. Tested on UCSD cluster node sn4622120245 with 8× H100 80GB HBM3.

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
   - Single-GPU MaxDiffusion inference (verified on H100)

2. **What FAILS:**
   - MaxDiffusion inference (ALL models: Wan 1.3B, SDXL)
   - Fails with ANY number of GPUs (2, 4, 8)
   - Fails on BOTH H100 and Blackwell GPUs
   - Multiple error types observed:
     - `ncclCommCount(comm_, &count) failed: invalid argument`
     - `ncclCommSplit(...) failed: invalid argument`
     - `ncclSend/ncclRecv failed: invalid argument`
   - "Call to bind failed: Address already in use" when NCCL RAS enabled (fix: `NCCL_RAS_ENABLE=0`)

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

**Workaround (OLD):** Use single GPU: `CUDA_VISIBLE_DEVICES=0`

**FIX (VERIFIED):** Add `_ = jax.devices()` immediately after `import jax` - see "FIX FOUND" section below.

**Test files:**
- `Multi-Sharding-MaxDiffusion/test_nccl_minimal.py` - Minimal tests (ALL PASS)
- `Multi-Sharding-MaxDiffusion/run_debug.sh` - Debug script for MaxDiffusion

**H100 Testing (2026-01-05, UCSD cluster):**
- Environment: sn4622120245, 8× H100 80GB HBM3, JAX 0.8.1, NCCL 2.28.9
- Single-GPU: WORKS (compile: 41s, generation: 6.4s)
- Multi-GPU (2-4 GPUs): FAILS with same NCCL errors as Blackwell
- Basic NCCL tests (psum, all_gather, sharded matmul): ALL PASS
- Model: `/scr/dataset/kaijian/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers`

**H100 Workarounds Tested (ALL FAILED):**
- `NCCL_RAS_ENABLE=0` - Fixes "Address already in use" but still fails
- `NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1` - No effect
- `NCCL_CUMEM_ENABLE=0 + NCCL_NET_GDR_LEVEL=0` - No effect
- `NCCL_LAUNCH_MODE=PARALLEL` - No effect
- `--xla_gpu_enable_nccl_comm_splitting=false` - No effect
- `--xla_gpu_enable_command_buffer=` (empty) - No effect
- `--xla_gpu_collectives_use_persistent_cliques=true` - No effect
- `JAX_USE_SHARDY_PARTITIONER=0` - No effect
- `JAX_COMPILATION_CACHE_DIR=""` - No effect
- Older JAX (0.4.35) - Incompatible with current flax/maxdiffusion
- NCCL version downgrades (2.18.3, 2.19.3, 2.27.5) - All fail with same error

**Note:** CUDA runtime 12.9 vs Driver 12.2 mismatch was suspected but NCCL downgrades did not help - issue is in JAX/XLA PJRT layer, not NCCL compatibility.

**FIX FOUND (2026-01-06):**
Add `_ = jax.devices()` immediately after `import jax` to force early device initialization:
```python
import jax
_ = jax.devices()  # Force early device initialization to avoid NCCL race condition
```
This bypasses the race condition where execution threads try to use NCCL communicators before initialization completes.

**H100 Multi-GPU SUCCESS after fix:**
- compile_time: 53.1s
- generation_time: 10.4s (2x H100, 3 steps, 5 frames)
- Video successfully exported

**Blackwell (RTX PRO 6000) Multi-GPU SUCCESS after fix (2026-01-06):**
- compile_time: 40.0s
- generation_time: 12.3s (2x RTX PRO 6000, 3 steps, 5 frames)
- Video successfully exported: `wan_output_*.mp4`
- Fix applied to: `Multi-Sharding-MaxDiffusion/src/maxdiffusion/generate_wan.py:17`

**Next steps:**
1. ✅ Apply fix to Blackwell (RTX PRO 6000) - DONE, VERIFIED WORKING
2. Test with more GPUs (4, 8) on Blackwell
3. Consider filing JAX issue about the race condition for permanent upstream fix
4. Proceed to Phase 2: Parallelism Implementation (TP, SP, DistriFusion)
