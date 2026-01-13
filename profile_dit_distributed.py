"""
Distributed Multi-GPU Profiling Script for Wan2.1 DiT Model

This script profiles DiT (Diffusion Transformer) model inference across
multiple GPUs to measure communication vs compute breakdown.

DiT architecture differences from UNet:
- Heavy attention operations (benefit from sequence parallelism)
- Linear layers in FFN (benefit from tensor parallelism)
- Less convolution (VAE only)

Usage:
    CUDA_VISIBLE_DEVICES=5,6,7 torchrun --nproc_per_node=3 profile_dit_distributed.py
"""

import torch
import torch.distributed as dist
import torch.profiler
from diffusers import WanPipeline
import os
import json
from datetime import datetime
import time

# Configuration
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
PROMPT = "A cat walking on the grass"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 1
NUM_PROFILE_RUNS = 3
OUTPUT_DIR = "./profiling_results/dit_distributed"

# Video params
NUM_FRAMES = 17
HEIGHT = 480
WIDTH = 832

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

def profile_communication_primitives(local_rank, world_size, device):
    """
    Profile NCCL communication for DiT-relevant tensor sizes.
    DiT uses different activation sizes than UNet.
    """
    results = {
        "all_reduce": [],
        "all_gather": [],
    }

    # DiT activation sizes (batch, seq_len, hidden_dim)
    # Wan2.1-1.3B: hidden_dim=1536, num_heads=24
    # Sequence length depends on video resolution and patch size
    # For 480x832, patch_size=2: seq_len = (480/2)*(832/2)*(17frames/temporal_patch) ~ 17000
    tensor_sizes = [
        (1, 1024, 1536),    # Small sequence
        (1, 4096, 1536),    # Medium sequence
        (1, 8192, 1536),    # Large sequence
        (1, 16384, 1536),   # Full video sequence
    ]

    for size in tensor_sizes:
        tensor = torch.randn(size, device=device, dtype=torch.float16)
        numel = tensor.numel()
        size_mb = numel * 2 / (1024 * 1024)

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
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)
        bandwidth = size_mb / (avg_time / 1000) if avg_time > 0 else 0

        results["all_reduce"].append({
            "shape": size,
            "size_mb": size_mb,
            "avg_time_ms": avg_time,
            "bandwidth_gbps": bandwidth * 8 / 1000,
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

def profile_dit_tp_simulation(device, local_rank, world_size, output_dir):
    """
    Simulate Tensor Parallelism communication patterns for DiT.

    In DiT with TP:
    - Attention: Q, K, V projections split across GPUs -> all-gather
    - FFN: First linear split -> all-reduce, Second linear split -> all-reduce
    - Each transformer block has: Attention + FFN

    Wan2.1-1.3B architecture:
    - hidden_dim: 1536
    - num_attention_heads: 24
    - num_layers: 30 (approximately)
    - FFN hidden: 1536 * 4 = 6144
    """
    os.makedirs(output_dir, exist_ok=True)

    log_rank0("\nSimulating DiT Tensor Parallelism patterns...", local_rank)

    # DiT configuration (Wan2.1-1.3B approximate)
    hidden_dim = 1536
    ffn_hidden = hidden_dim * 4  # 6144
    num_heads = 24
    num_layers = 30

    # Sequence lengths at different video sizes
    # seq_len = num_patches = (H/patch)*(W/patch)*(T/temporal_patch)
    # For 480x832, patch=2, temporal_patch=1: seq ~ 240*416*17 = ~1.7M (too large)
    # Let's use realistic per-step sequence
    seq_len = 4096  # After spatial/temporal compression

    total_comm_time = 0
    total_compute_time = 0
    layer_breakdown = []

    for layer_idx in range(num_layers):
        layer_comm_time = 0
        layer_compute_time = 0

        # Input tensor
        x = torch.randn(1, seq_len, hidden_dim, device=device, dtype=torch.float16)

        # === Attention Block ===
        # QKV projection (column parallel)
        qkv_weight = torch.randn(hidden_dim, 3 * hidden_dim // world_size, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        start = time.perf_counter()
        qkv = torch.matmul(x, qkv_weight)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        layer_compute_time += compute_time

        # All-gather for QKV
        gathered = [torch.empty_like(qkv) for _ in range(world_size)]
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_gather(gathered, qkv)
        torch.cuda.synchronize()
        comm_time = (time.perf_counter() - start) * 1000
        layer_comm_time += comm_time

        # Attention computation (simplified)
        full_qkv = torch.cat(gathered, dim=-1)
        q, k, v = full_qkv.chunk(3, dim=-1)

        torch.cuda.synchronize()
        start = time.perf_counter()
        # Scaled dot-product attention
        scale = (hidden_dim // num_heads) ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn_out = torch.matmul(attn, v)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        layer_compute_time += compute_time

        # Output projection (row parallel)
        out_weight = torch.randn(hidden_dim // world_size, hidden_dim, device=device, dtype=torch.float16)
        local_out = attn_out[..., local_rank * (hidden_dim // world_size):(local_rank + 1) * (hidden_dim // world_size)]

        torch.cuda.synchronize()
        start = time.perf_counter()
        out = torch.matmul(local_out, out_weight)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        layer_compute_time += compute_time

        # All-reduce for output
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(out)
        torch.cuda.synchronize()
        comm_time = (time.perf_counter() - start) * 1000
        layer_comm_time += comm_time

        # === FFN Block ===
        # FFN up projection (column parallel)
        ffn_up_weight = torch.randn(hidden_dim, ffn_hidden // world_size, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        start = time.perf_counter()
        ffn_hidden_state = torch.matmul(x, ffn_up_weight)
        ffn_hidden_state = torch.nn.functional.gelu(ffn_hidden_state)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        layer_compute_time += compute_time

        # FFN down projection (row parallel)
        ffn_down_weight = torch.randn(ffn_hidden // world_size, hidden_dim, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        start = time.perf_counter()
        ffn_out = torch.matmul(ffn_hidden_state, ffn_down_weight)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        layer_compute_time += compute_time

        # All-reduce for FFN output
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_reduce(ffn_out)
        torch.cuda.synchronize()
        comm_time = (time.perf_counter() - start) * 1000
        layer_comm_time += comm_time

        total_comm_time += layer_comm_time
        total_compute_time += layer_compute_time

        if layer_idx < 3 or layer_idx == num_layers - 1:  # Log first 3 and last
            layer_breakdown.append({
                "layer": layer_idx,
                "comm_time_ms": layer_comm_time,
                "compute_time_ms": layer_compute_time,
            })

    results = {
        "parallelism": "tensor_parallel_simulation",
        "architecture": "DiT",
        "world_size": world_size,
        "hidden_dim": hidden_dim,
        "ffn_hidden": ffn_hidden,
        "num_layers": num_layers,
        "seq_len": seq_len,
        "total_comm_time_ms": total_comm_time,
        "total_compute_time_ms": total_compute_time,
        "comm_compute_ratio": total_comm_time / total_compute_time if total_compute_time > 0 else 0,
        "comm_percentage": total_comm_time / (total_comm_time + total_compute_time) * 100,
        "layer_breakdown": layer_breakdown,
    }

    log_rank0(f"\nDiT TP Simulation Results:", local_rank)
    log_rank0(f"  Layers: {num_layers}, Seq len: {seq_len}, Hidden: {hidden_dim}", local_rank)
    log_rank0(f"  Total communication time: {total_comm_time:.2f} ms", local_rank)
    log_rank0(f"  Total compute time: {total_compute_time:.2f} ms", local_rank)
    log_rank0(f"  Comm/Compute ratio: {results['comm_compute_ratio']:.3f}", local_rank)
    log_rank0(f"  Communication percentage: {results['comm_percentage']:.1f}%", local_rank)

    return results

def profile_dit_sequence_parallel_simulation(device, local_rank, world_size, output_dir):
    """
    Simulate Sequence Parallelism for DiT.

    In SP, the sequence dimension is split across GPUs.
    Communication happens at attention (all-to-all for K, V gathering).
    """
    log_rank0("\nSimulating DiT Sequence Parallelism patterns...", local_rank)

    hidden_dim = 1536
    total_seq_len = 4096
    local_seq_len = total_seq_len // world_size
    num_layers = 30

    total_comm_time = 0
    total_compute_time = 0

    for layer_idx in range(num_layers):
        # Local sequence portion
        x = torch.randn(1, local_seq_len, hidden_dim, device=device, dtype=torch.float16)

        # Local QKV computation
        qkv_weight = torch.randn(hidden_dim, 3 * hidden_dim, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        start = time.perf_counter()
        qkv = torch.matmul(x, qkv_weight)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        total_compute_time += compute_time

        # All-gather K, V for full sequence attention
        k_local = qkv[..., hidden_dim:2*hidden_dim].contiguous()
        v_local = qkv[..., 2*hidden_dim:].contiguous()

        k_gathered = [torch.empty_like(k_local) for _ in range(world_size)]
        v_gathered = [torch.empty_like(v_local) for _ in range(world_size)]

        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_gather(k_gathered, k_local)
        dist.all_gather(v_gathered, v_local)
        torch.cuda.synchronize()
        comm_time = (time.perf_counter() - start) * 1000
        total_comm_time += comm_time

        # Local attention with full K, V
        q = qkv[..., :hidden_dim]
        k_full = torch.cat(k_gathered, dim=1)
        v_full = torch.cat(v_gathered, dim=1)

        torch.cuda.synchronize()
        start = time.perf_counter()
        attn = torch.matmul(q, k_full.transpose(-2, -1))
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_full)
        torch.cuda.synchronize()
        compute_time = (time.perf_counter() - start) * 1000
        total_compute_time += compute_time

    results = {
        "parallelism": "sequence_parallel_simulation",
        "architecture": "DiT",
        "world_size": world_size,
        "total_seq_len": total_seq_len,
        "local_seq_len": local_seq_len,
        "num_layers": num_layers,
        "total_comm_time_ms": total_comm_time,
        "total_compute_time_ms": total_compute_time,
        "comm_compute_ratio": total_comm_time / total_compute_time if total_compute_time > 0 else 0,
        "comm_percentage": total_comm_time / (total_comm_time + total_compute_time) * 100,
    }

    log_rank0(f"\nDiT SP Simulation Results:", local_rank)
    log_rank0(f"  Total communication time: {total_comm_time:.2f} ms", local_rank)
    log_rank0(f"  Total compute time: {total_compute_time:.2f} ms", local_rank)
    log_rank0(f"  Communication percentage: {results['comm_percentage']:.1f}%", local_rank)

    return results

def main():
    local_rank, world_size = setup_distributed()

    log_rank0(f"\n{'='*80}", local_rank)
    log_rank0(f"DiT DISTRIBUTED PROFILING - {world_size} GPUs", local_rank)
    log_rank0(f"{'='*80}", local_rank)

    device = torch.device(f"cuda:{local_rank}")
    log_rank0(f"Device: {torch.cuda.get_device_name(local_rank)}", local_rank)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"{OUTPUT_DIR}/{timestamp}_gpus{world_size}"

    # 1. Profile communication primitives
    log_rank0("\n[1/3] Profiling NCCL communication (DiT tensor sizes)...", local_rank)
    comm_results = profile_communication_primitives(local_rank, world_size, device)

    if local_rank == 0:
        print("\nAll-Reduce bandwidth (DiT sizes):")
        for r in comm_results["all_reduce"]:
            print(f"  {r['shape']}: {r['avg_time_ms']:.3f} ms, {r['bandwidth_gbps']:.1f} Gbps")

    # 2. Simulate Tensor Parallelism
    log_rank0("\n[2/3] Simulating Tensor Parallelism for DiT...", local_rank)
    tp_output_dir = os.path.join(base_output_dir, "tensor_parallel_sim")
    tp_results = profile_dit_tp_simulation(device, local_rank, world_size, tp_output_dir)

    # 3. Simulate Sequence Parallelism
    log_rank0("\n[3/3] Simulating Sequence Parallelism for DiT...", local_rank)
    sp_results = profile_dit_sequence_parallel_simulation(device, local_rank, world_size, base_output_dir)

    # Save all results
    if local_rank == 0:
        all_results = {
            "timestamp": timestamp,
            "model": MODEL_ID,
            "architecture": "DiT (Diffusion Transformer)",
            "world_size": world_size,
            "communication_primitives": comm_results,
            "tensor_parallel_simulation": tp_results,
            "sequence_parallel_simulation": sp_results,
        }

        os.makedirs(base_output_dir, exist_ok=True)
        results_file = os.path.join(base_output_dir, "all_results.json")
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print("SUMMARY - DiT Distributed Profiling")
        print(f"{'='*80}")
        print(f"Results saved to: {results_file}")

        # DistriFusion overlap analysis
        print(f"\n{'='*80}")
        print("DISTRIFUSION OVERLAP POTENTIAL (DiT)")
        print(f"{'='*80}")

        print(f"\nTensor Parallelism:")
        print(f"  Communication: {tp_results['comm_percentage']:.1f}%")
        tp_speedup = 1 / (1 - tp_results['comm_percentage']/100)
        print(f"  Potential speedup with perfect overlap: {tp_speedup:.2f}x")

        print(f"\nSequence Parallelism:")
        print(f"  Communication: {sp_results['comm_percentage']:.1f}%")
        sp_speedup = 1 / (1 - sp_results['comm_percentage']/100)
        print(f"  Potential speedup with perfect overlap: {sp_speedup:.2f}x")

        print(f"\nDistriFusion can hide this communication by:")
        print(f"  1. Reusing activations from previous diffusion timestep")
        print(f"  2. Overlapping communication of step t with compute of step t+1")

    cleanup_distributed()

if __name__ == "__main__":
    main()
