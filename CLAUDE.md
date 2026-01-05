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

### NCCL Multi-GPU Issue on Blackwell (2026-01-05)
- **Problem:** JAX/XLA NCCL operations fail with `corrupted comm object` on Blackwell GPUs (compute capability 12.0)
- **Impact:** Multi-GPU MaxDiffusion inference fails; single-GPU works fine
- **Status:** Blocked - likely requires JAX/XLA update for Blackwell support
- **Workaround:** Use single GPU with `CUDA_VISIBLE_DEVICES=0`
