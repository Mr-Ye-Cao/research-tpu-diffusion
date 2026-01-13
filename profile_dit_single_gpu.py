"""
Single GPU Profiling Script for Wan2.1 DiT Model

This script profiles the Wan2.1 1.3B DiT (Diffusion Transformer) model
on a single GPU to establish baseline compute times.

Wan2.1 uses DiT architecture (transformer-based) instead of UNet.
Key differences from UNet:
- More attention operations
- Less convolution operations
- Different communication patterns for parallelism

Usage:
    CUDA_VISIBLE_DEVICES=5 python profile_dit_single_gpu.py
"""

import torch
import torch.profiler
from diffusers import WanPipeline
from diffusers.utils import export_to_video
import os
import json
from datetime import datetime

# Configuration
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
PROMPT = "A cat walking on the grass"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 1
NUM_PROFILE_RUNS = 3
OUTPUT_DIR = "./profiling_results/dit_single_gpu"

# Video generation params (smaller for faster profiling)
NUM_FRAMES = 17  # Wan2.1 default
HEIGHT = 480
WIDTH = 832

def setup_model(device):
    """Load and configure the Wan2.1 DiT model."""
    print(f"Loading model {MODEL_ID}...")
    print("This may take a few minutes for first download...")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    # Enable memory optimizations
    pipe.enable_model_cpu_offload()  # Comment out if enough VRAM

    return pipe

def setup_model_full_gpu(device):
    """Load model fully on GPU (needs ~20GB+ VRAM)."""
    print(f"Loading model {MODEL_ID} (full GPU mode)...")

    pipe = WanPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    )
    pipe = pipe.to(device)

    return pipe

def warmup(pipe, prompt, num_runs=1):
    """Warmup runs to stabilize CUDA kernels."""
    print(f"Running {num_runs} warmup iterations...")
    for i in range(num_runs):
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                num_frames=NUM_FRAMES,
                height=HEIGHT,
                width=WIDTH,
                output_type="latent",
            )
        torch.cuda.synchronize()
    print("Warmup complete.")

def profile_inference(pipe, prompt, output_dir):
    """Profile the DiT inference with torch.profiler."""
    os.makedirs(output_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=NUM_PROFILE_RUNS,
        repeat=1
    )

    results = {
        "model": MODEL_ID,
        "architecture": "DiT (Diffusion Transformer)",
        "device": str(pipe.device) if hasattr(pipe, 'device') else "cuda",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "num_frames": NUM_FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "runs": []
    }

    print(f"Starting profiling with {NUM_PROFILE_RUNS} runs...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:

        for run_idx in range(1 + NUM_PROFILE_RUNS):
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            with torch.no_grad():
                _ = pipe(
                    prompt=prompt,
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    num_frames=NUM_FRAMES,
                    height=HEIGHT,
                    width=WIDTH,
                    output_type="latent",
                )

            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)

            if run_idx > 0:
                results["runs"].append({
                    "run_index": run_idx,
                    "total_time_ms": elapsed_time,
                    "time_per_step_ms": elapsed_time / NUM_INFERENCE_STEPS,
                })
                print(f"  Run {run_idx}: {elapsed_time:.2f} ms ({elapsed_time/NUM_INFERENCE_STEPS:.2f} ms/step)")

            prof.step()

    # Calculate statistics
    times = [r["total_time_ms"] for r in results["runs"]]
    results["statistics"] = {
        "mean_time_ms": sum(times) / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "mean_time_per_step_ms": sum(times) / len(times) / NUM_INFERENCE_STEPS,
    }

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")
    print(f"TensorBoard traces saved to {output_dir}")

    return prof, results

def analyze_profile(prof):
    """Analyze and print profiler results."""
    print("\n" + "="*80)
    print("PROFILER ANALYSIS - DiT (Diffusion Transformer)")
    print("="*80)

    # Key averages by CUDA time
    print("\nTop 20 operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=20
    ))

    return prof.key_averages()

def categorize_dit_operations(key_averages):
    """Categorize DiT operations into compute vs other categories."""
    categories = {
        "attention": [],      # Self-attention, cross-attention
        "linear_ffn": [],     # Linear layers, FFN
        "normalization": [],  # LayerNorm, RMSNorm
        "activation": [],     # GELU, SiLU, etc.
        "embedding": [],      # Embeddings, positional encoding
        "memory_ops": [],     # Copy, transpose, etc.
        "conv": [],           # Any conv operations (VAE)
        "other": [],
    }

    for item in key_averages:
        name = item.key.lower()
        cuda_time = item.self_device_time_total

        if any(x in name for x in ['attention', 'softmax', 'scaled_dot', 'flash', 'sdpa']):
            categories["attention"].append((item.key, cuda_time))
        elif any(x in name for x in ['linear', 'addmm', 'mm', 'gemm', 'matmul']):
            categories["linear_ffn"].append((item.key, cuda_time))
        elif any(x in name for x in ['norm', 'layer_norm', 'rms_norm', 'group_norm']):
            categories["normalization"].append((item.key, cuda_time))
        elif any(x in name for x in ['gelu', 'silu', 'relu', 'sigmoid', 'tanh', 'swish']):
            categories["activation"].append((item.key, cuda_time))
        elif any(x in name for x in ['embedding', 'embed', 'positional']):
            categories["embedding"].append((item.key, cuda_time))
        elif any(x in name for x in ['copy', 'transpose', 'permute', 'contiguous', 'view', 'reshape']):
            categories["memory_ops"].append((item.key, cuda_time))
        elif any(x in name for x in ['conv', 'cudnn']):
            categories["conv"].append((item.key, cuda_time))
        else:
            categories["other"].append((item.key, cuda_time))

    print("\n" + "="*80)
    print("DiT OPERATION CATEGORIES (by CUDA time)")
    print("="*80)

    total_time = sum(item.self_device_time_total for item in key_averages)

    breakdown = {}
    for cat_name, ops in categories.items():
        cat_time = sum(t for _, t in ops)
        pct = (cat_time / total_time * 100) if total_time > 0 else 0
        breakdown[cat_name] = {"time_ms": cat_time/1000, "percentage": pct}
        print(f"\n{cat_name}: {cat_time/1000:.2f} ms ({pct:.1f}%)")
        # Show top 3 ops in each category
        for op, t in sorted(ops, key=lambda x: -x[1])[:3]:
            print(f"  - {op[:80]}: {t/1000:.2f} ms")

    return categories, breakdown

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Setup - try full GPU first, fall back to CPU offload
    try:
        pipe = setup_model_full_gpu(device)
        print("Model loaded fully on GPU")
    except torch.cuda.OutOfMemoryError:
        print("Not enough VRAM, using CPU offload...")
        torch.cuda.empty_cache()
        pipe = setup_model(device)

    # Warmup
    warmup(pipe, PROMPT, NUM_WARMUP_RUNS)

    # Profile
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}/{timestamp}"
    prof, results = profile_inference(pipe, PROMPT, output_dir)

    # Analyze
    key_averages = analyze_profile(prof)
    categories, breakdown = categorize_dit_operations(key_averages)

    # Save breakdown
    results["operation_breakdown"] = breakdown
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - Wan2.1 DiT Model")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Architecture: DiT (Diffusion Transformer)")
    print(f"Video: {NUM_FRAMES} frames @ {WIDTH}x{HEIGHT}")
    print(f"Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"Mean inference time: {results['statistics']['mean_time_ms']:.2f} ms")
    print(f"Mean time per step: {results['statistics']['mean_time_per_step_ms']:.2f} ms")

    # DiT-specific analysis
    attention_pct = breakdown.get("attention", {}).get("percentage", 0)
    linear_pct = breakdown.get("linear_ffn", {}).get("percentage", 0)
    print(f"\nDiT Compute Profile:")
    print(f"  Attention: {attention_pct:.1f}%")
    print(f"  Linear/FFN: {linear_pct:.1f}%")
    print(f"  (Attention+Linear = {attention_pct + linear_pct:.1f}% - these benefit from TP)")

    print(f"\nTo view traces: tensorboard --logdir {output_dir}")

if __name__ == "__main__":
    main()
