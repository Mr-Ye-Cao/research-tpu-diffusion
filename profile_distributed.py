"""
Distributed Multi-GPU Profiling Script for Diffusion Model Inference

This script profiles diffusion model inference across multiple GPUs
to measure communication vs compute breakdown and overlap potential.

Parallelism strategies tested:
1. Data Parallelism (DP) - replicate model, split batch
2. Tensor Parallelism (TP) - split model weights, requires all-reduce

Usage:
    # 2 GPUs
    torchrun --nproc_per_node=2 profile_distributed.py

    # 4 GPUs
    torchrun --nproc_per_node=4 profile_distributed.py
"""

import torch
import torch.distributed as dist
import torch.profiler
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, UNet2DConditionModel
import os
import json
from datetime import datetime
import time

# Configuration
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT = "a photo of an astronaut riding a horse on mars"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 2
NUM_PROFILE_RUNS = 3
OUTPUT_DIR = "./profiling_results/distributed"
IMAGE_SIZE = 512
BATCH_SIZE_PER_GPU = 1

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    return local_rank, world_size

def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()

def log_rank0(msg, rank):
    """Log only from rank 0."""
    if rank == 0:
        print(msg)

def setup_model_dp(local_rank, world_size):
    """Setup model with Data Parallelism (each GPU has full model copy)."""
    device = torch.device(f"cuda:{local_rank}")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    return pipe, device

def profile_communication_primitives(local_rank, world_size, device):
    """
    Profile basic NCCL communication primitives to measure bandwidth.
    This simulates the communication patterns in tensor/sequence parallelism.
    """
    results = {
        "all_reduce": [],
        "all_gather": [],
        "reduce_scatter": [],
    }

    # Test different tensor sizes (simulating activation sizes)
    # Typical UNet hidden sizes: 320, 640, 1280, 2560
    tensor_sizes = [
        (1, 320, 64, 64),    # Early blocks
        (1, 640, 32, 32),    # Mid blocks
        (1, 1280, 16, 16),   # Later blocks
        (1, 2560, 8, 8),     # Bottleneck
    ]

    for size in tensor_sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float16)
        numel = tensor.numel()
        size_mb = numel * 2 / (1024 * 1024)  # float16 = 2 bytes

        # Warmup
        for _ in range(3):
            dist.all_reduce(tensor.clone())
            torch.cuda.synchronize()

        # Profile all_reduce
        times = []
        for _ in range(10):
            t = tensor.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(t)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

        avg_time = sum(times) / len(times)
        bandwidth = size_mb / (avg_time / 1000) if avg_time > 0 else 0

        results["all_reduce"].append({
            "shape": size,
            "size_mb": size_mb,
            "avg_time_ms": avg_time,
            "bandwidth_gbps": bandwidth * 8 / 1000,  # GB/s to Gbps
        })

        # Profile all_gather
        times = []
        output_tensors = [torch.empty_like(tensor) for _ in range(world_size)]
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_gather(output_tensors, tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        results["all_gather"].append({
            "shape": size,
            "size_mb": size_mb,
            "avg_time_ms": avg_time,
        })

    return results

def profile_dp_inference(pipe, prompt, device, local_rank, world_size, output_dir):
    """Profile Data Parallel inference."""
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "parallelism": "data_parallel",
        "model": MODEL_ID,
        "world_size": world_size,
        "local_rank": local_rank,
        "device": str(device),
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "batch_size_per_gpu": BATCH_SIZE_PER_GPU,
        "total_batch_size": BATCH_SIZE_PER_GPU * world_size,
        "runs": []
    }

    # Warmup
    log_rank0(f"Warmup with {NUM_WARMUP_RUNS} runs...", local_rank)
    for _ in range(NUM_WARMUP_RUNS):
        with torch.no_grad():
            _ = pipe(
                prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                output_type="latent",
            )
        torch.cuda.synchronize()
        dist.barrier()

    # Profile
    log_rank0(f"Profiling with {NUM_PROFILE_RUNS} runs...", local_rank)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=NUM_PROFILE_RUNS,
        repeat=1
    )

    trace_dir = os.path.join(output_dir, f"rank_{local_rank}")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:

        for run_idx in range(1 + NUM_PROFILE_RUNS):
            torch.cuda.synchronize()
            dist.barrier()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            with torch.no_grad():
                _ = pipe(
                    prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                    output_type="latent",
                )

            end_event.record()
            torch.cuda.synchronize()
            dist.barrier()

            elapsed_time = start_event.elapsed_time(end_event)

            if run_idx > 0:
                results["runs"].append({
                    "run_index": run_idx,
                    "total_time_ms": elapsed_time,
                })
                log_rank0(f"  Run {run_idx}: {elapsed_time:.2f} ms", local_rank)

            prof.step()

    return prof, results

def profile_simulated_tp(device, local_rank, world_size, output_dir):
    """
    Simulate Tensor Parallelism communication patterns.

    In true TP, matrix multiplications are split across GPUs and require
    all-reduce for gradient/activation synchronization. This simulates
    that pattern to measure communication overhead.
    """
    os.makedirs(output_dir, exist_ok=True)

    log_rank0("\nSimulating Tensor Parallelism communication patterns...", local_rank)

    # Simulate UNet forward pass communication pattern
    # Each attention layer requires: input projection -> attention -> output projection
    # With TP, we need all-reduce after each parallel matmul

    # Typical UNet-2D architecture for SD:
    # - Down blocks: 2 attention layers each at resolutions 64, 32, 16
    # - Mid block: 1 attention layer at resolution 8
    # - Up blocks: 3 attention layers each at resolutions 16, 32, 64

    attention_configs = [
        # (batch, seq_len, hidden_dim, num_layers)
        (1, 64*64, 320, 2),    # Down block 1
        (1, 32*32, 640, 2),    # Down block 2
        (1, 16*16, 1280, 2),   # Down block 3
        (1, 8*8, 1280, 1),     # Mid block
        (1, 16*16, 1280, 3),   # Up block 1
        (1, 32*32, 640, 3),    # Up block 2
        (1, 64*64, 320, 3),    # Up block 3
    ]

    total_comm_time = 0
    total_compute_time = 0
    comm_breakdown = []

    for batch, seq_len, hidden_dim, num_layers in attention_configs:
        # For each attention layer, we have:
        # 1. QKV projection (3 matmuls, can be fused)
        # 2. Attention computation
        # 3. Output projection

        # With column-parallel: Q, K, V projections need all-gather
        # With row-parallel: output projection needs all-reduce

        # Simulate activation tensor for this layer
        activation = torch.randn(batch, seq_len, hidden_dim, device=device, dtype=torch.float16)
        qkv_weight = torch.randn(hidden_dim, 3 * hidden_dim // world_size, device=device, dtype=torch.float16)
        out_weight = torch.randn(hidden_dim // world_size, hidden_dim, device=device, dtype=torch.float16)

        layer_comm_time = 0
        layer_compute_time = 0

        for _ in range(num_layers):
            # Simulate QKV projection (compute)
            torch.cuda.synchronize()
            start = time.perf_counter()
            qkv = torch.matmul(activation, qkv_weight)
            torch.cuda.synchronize()
            compute_time = (time.perf_counter() - start) * 1000
            layer_compute_time += compute_time

            # All-gather for QKV (communication)
            gathered = [torch.empty_like(qkv) for _ in range(world_size)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_gather(gathered, qkv)
            torch.cuda.synchronize()
            comm_time = (time.perf_counter() - start) * 1000
            layer_comm_time += comm_time

            # Simulate attention (compute) - simplified
            full_qkv = torch.cat(gathered, dim=-1)
            q, k, v = full_qkv.chunk(3, dim=-1)
            torch.cuda.synchronize()
            start = time.perf_counter()
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (hidden_dim ** 0.5)
            attn_output = torch.matmul(torch.softmax(attn_weights, dim=-1), v)
            torch.cuda.synchronize()
            compute_time = (time.perf_counter() - start) * 1000
            layer_compute_time += compute_time

            # Output projection (compute)
            local_out = attn_output[..., local_rank * (hidden_dim // world_size):(local_rank + 1) * (hidden_dim // world_size)]
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = torch.matmul(local_out, out_weight)
            torch.cuda.synchronize()
            compute_time = (time.perf_counter() - start) * 1000
            layer_compute_time += compute_time

            # All-reduce for output (communication)
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(out)
            torch.cuda.synchronize()
            comm_time = (time.perf_counter() - start) * 1000
            layer_comm_time += comm_time

        total_comm_time += layer_comm_time
        total_compute_time += layer_compute_time

        comm_breakdown.append({
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "comm_time_ms": layer_comm_time,
            "compute_time_ms": layer_compute_time,
        })

    results = {
        "parallelism": "tensor_parallel_simulation",
        "world_size": world_size,
        "total_comm_time_ms": total_comm_time,
        "total_compute_time_ms": total_compute_time,
        "comm_compute_ratio": total_comm_time / total_compute_time if total_compute_time > 0 else 0,
        "comm_percentage": total_comm_time / (total_comm_time + total_compute_time) * 100,
        "breakdown": comm_breakdown,
    }

    log_rank0(f"\nTP Simulation Results:", local_rank)
    log_rank0(f"  Total communication time: {total_comm_time:.2f} ms", local_rank)
    log_rank0(f"  Total compute time: {total_compute_time:.2f} ms", local_rank)
    log_rank0(f"  Comm/Compute ratio: {results['comm_compute_ratio']:.3f}", local_rank)
    log_rank0(f"  Communication percentage: {results['comm_percentage']:.1f}%", local_rank)

    return results

def analyze_nccl_ops(prof, local_rank):
    """Extract NCCL communication operations from profile."""
    if local_rank != 0:
        return None

    key_averages = prof.key_averages()

    nccl_ops = []
    compute_ops = []
    total_cuda_time = 0

    for item in key_averages:
        name = item.key.lower()
        cuda_time = item.device_time_total

        total_cuda_time += cuda_time

        if any(x in name for x in ['nccl', 'allreduce', 'allgather', 'broadcast', 'reduce_scatter', 'ncclkernel']):
            nccl_ops.append({
                "name": item.key,
                "cuda_time_us": cuda_time,
                "count": item.count,
            })
        elif any(x in name for x in ['gemm', 'conv', 'matmul', 'mm', 'addmm', 'cudnn']):
            compute_ops.append({
                "name": item.key,
                "cuda_time_us": cuda_time,
                "count": item.count,
            })

    nccl_time = sum(op["cuda_time_us"] for op in nccl_ops)
    compute_time = sum(op["cuda_time_us"] for op in compute_ops)

    print("\n" + "="*80)
    print("NCCL COMMUNICATION ANALYSIS")
    print("="*80)
    print(f"\nTotal CUDA time: {total_cuda_time/1000:.2f} ms")
    print(f"NCCL communication time: {nccl_time/1000:.2f} ms ({nccl_time/total_cuda_time*100:.1f}%)")
    print(f"Compute time (matmul/conv): {compute_time/1000:.2f} ms ({compute_time/total_cuda_time*100:.1f}%)")

    if nccl_ops:
        print("\nNCCL Operations:")
        for op in sorted(nccl_ops, key=lambda x: -x["cuda_time_us"])[:10]:
            print(f"  {op['name']}: {op['cuda_time_us']/1000:.2f} ms (count: {op['count']})")

    return {
        "total_cuda_time_us": total_cuda_time,
        "nccl_time_us": nccl_time,
        "compute_time_us": compute_time,
        "nccl_ops": nccl_ops,
    }

def main():
    local_rank, world_size = setup_distributed()

    log_rank0(f"\n{'='*80}", local_rank)
    log_rank0(f"DISTRIBUTED PROFILING - {world_size} GPUs", local_rank)
    log_rank0(f"{'='*80}", local_rank)

    device = torch.device(f"cuda:{local_rank}")
    log_rank0(f"Device: {torch.cuda.get_device_name(local_rank)}", local_rank)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{OUTPUT_DIR}/{timestamp}_gpus{world_size}"

    # 1. Profile communication primitives
    log_rank0("\n[1/3] Profiling NCCL communication primitives...", local_rank)
    comm_results = profile_communication_primitives(local_rank, world_size, device)

    if local_rank == 0:
        print("\nAll-Reduce bandwidth:")
        for r in comm_results["all_reduce"]:
            print(f"  {r['shape']}: {r['avg_time_ms']:.3f} ms, {r['bandwidth_gbps']:.1f} Gbps")

    # 2. Profile Data Parallel inference
    log_rank0("\n[2/3] Profiling Data Parallel inference...", local_rank)
    pipe, device = setup_model_dp(local_rank, world_size)
    dp_output_dir = os.path.join(base_output_dir, "data_parallel")
    prof, dp_results = profile_dp_inference(pipe, PROMPT, device, local_rank, world_size, dp_output_dir)

    # Analyze NCCL operations in DP
    nccl_analysis = analyze_nccl_ops(prof, local_rank)

    # Clean up pipeline
    del pipe
    torch.cuda.empty_cache()

    # 3. Simulate Tensor Parallelism
    log_rank0("\n[3/3] Simulating Tensor Parallelism patterns...", local_rank)
    tp_output_dir = os.path.join(base_output_dir, "tensor_parallel_sim")
    tp_results = profile_simulated_tp(device, local_rank, world_size, tp_output_dir)

    # Save all results (rank 0 only)
    if local_rank == 0:
        all_results = {
            "timestamp": timestamp,
            "world_size": world_size,
            "communication_primitives": comm_results,
            "data_parallel": dp_results,
            "tensor_parallel_simulation": tp_results,
            "nccl_analysis": nccl_analysis,
        }

        results_file = os.path.join(base_output_dir, "all_results.json")
        os.makedirs(base_output_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"Results saved to: {results_file}")
        print(f"TensorBoard traces: {base_output_dir}")
        print(f"\nTo view: tensorboard --logdir {base_output_dir}")

        # DistriFusion overlap potential analysis
        print(f"\n{'='*80}")
        print("DISTRIFUSION OVERLAP POTENTIAL")
        print(f"{'='*80}")

        if tp_results['comm_percentage'] > 0:
            print(f"\nIn Tensor Parallelism:")
            print(f"  Communication takes {tp_results['comm_percentage']:.1f}% of total time")
            print(f"  With DistriFusion's activation reuse, up to {tp_results['comm_percentage']:.1f}% can be hidden")
            print(f"  by overlapping current step's communication with previous step's activations")

            potential_speedup = 1 / (1 - tp_results['comm_percentage']/100)
            print(f"\n  Potential speedup with perfect overlap: {potential_speedup:.2f}x")

    cleanup_distributed()

if __name__ == "__main__":
    main()
