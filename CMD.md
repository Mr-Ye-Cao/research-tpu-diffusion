# Commands for Verification

## Environment Activation

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tpu-diffusion
```

## 1. Verify JAX Multi-GPU Setup

```bash
python -c "
import jax
print('JAX version:', jax.__version__)
print('Backend:', jax.default_backend())
print('Device count:', jax.device_count())
print('Devices:', jax.devices())
"
```

Expected output: 8 CudaDevices

## 2. Verify Multi-GPU Computation

```bash
python -c "
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding

# Create device mesh with all 8 GPUs
devices = jax.devices()
mesh = Mesh(devices, ('dp',))

# Create sharded array
with mesh:
    x = jnp.ones((8, 1024, 1024))
    sharding = NamedSharding(mesh, PartitionSpec('dp', None, None))
    x_sharded = jax.device_put(x, sharding)

    # Run computation across all GPUs
    result = jax.jit(lambda x: x @ x.T)(x_sharded)
    print('Sharding:', x_sharded.sharding)
    print('Result shape:', result.shape)
    print('Multi-GPU computation successful!')
"
```

## 3. Verify MaxDiffusion Import

```bash
python -c "
from maxdiffusion import max_utils
import jax
print('MaxDiffusion imported successfully')
print('Available devices:', jax.device_count(), 'GPUs')
"
```

## 4. Run Wan 2.1 1.3B Single-GPU Inference (Working)

```bash
cd /home/ye/workspace/research-tpu-diffusion/Multi-Sharding-MaxDiffusion

CUDA_VISIBLE_DEVICES=0 HF_HUB_ENABLE_HF_TRANSFER=1 python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_1b.yml \
  hardware=gpu \
  attention=dot_product \
  run_name=wan-1b-gpu-test-single \
  output_dir=/tmp/maxdiffusion_output \
  num_inference_steps=3 \
  num_frames=17 \
  width=640 \
  height=480 \
  per_device_batch_size=1 \
  ici_fsdp_parallelism=1 \
  prompt="A cat walking in the garden"
```

Note: First run will download the Wan 2.1 1.3B model (~5GB).

## 4b. Multi-GPU Inference (BLOCKED - NCCL Issue)

```bash
# WARNING: Currently fails due to NCCL comm corruption on Blackwell GPUs
cd /home/ye/workspace/research-tpu-diffusion/Multi-Sharding-MaxDiffusion

HF_HUB_ENABLE_HF_TRANSFER=1 python src/maxdiffusion/generate_wan.py \
  src/maxdiffusion/configs/base_wan_1b.yml \
  hardware=gpu \
  attention=dot_product \
  run_name=wan-1b-gpu-test-multi \
  output_dir=/tmp/maxdiffusion_output \
  num_inference_steps=3 \
  num_frames=17 \
  width=640 \
  height=480 \
  per_device_batch_size=1 \
  ici_fsdp_parallelism=8 \
  prompt="A cat walking in the garden"
```

**Known Issue:** NCCL `corrupted comm object` error on RTX PRO 6000 Blackwell (compute capability 12.0).
This appears to be a JAX/XLA NCCL compatibility issue with Blackwell architecture.

## 5. Quick Sharding Test (No Model Download)

```bash
python -c "
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import numpy as np

# Simulate MaxDiffusion mesh config: ['data', 'fsdp', 'tensor']
devices = np.array(jax.devices()).reshape(1, 8, 1)
mesh = Mesh(devices, ('data', 'fsdp', 'tensor'))

print('Mesh shape:', mesh.shape)
print('Mesh devices:', mesh.devices.shape)

# Simulate model sharding (FSDP across 8 GPUs)
with mesh:
    # Simulate transformer weights [hidden, hidden]
    weights = jnp.ones((4096, 4096))
    weight_sharding = NamedSharding(mesh, P('fsdp', None))
    weights_sharded = jax.device_put(weights, weight_sharding)

    # Simulate batch of activations [batch, seq, hidden]
    acts = jnp.ones((8, 1024, 4096))
    act_sharding = NamedSharding(mesh, P('data', 'fsdp', None))
    acts_sharded = jax.device_put(acts, act_sharding)

    @jax.jit
    def forward(x, w):
        return x @ w

    result = forward(acts_sharded, weights_sharded)
    print('Weight sharding:', weights_sharded.sharding)
    print('Activation sharding:', acts_sharded.sharding)
    print('Output shape:', result.shape)
    print('FSDP sharding across 8 GPUs verified!')
"
```
