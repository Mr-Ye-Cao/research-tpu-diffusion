"""
Single GPU Profiling Script for Diffusion Model Inference

This script profiles a diffusion model on a single GPU to establish
baseline compute times without any communication overhead.

Usage:
    python profile_single_gpu.py
"""

import torch
import torch.profiler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
import json
from datetime import datetime

# Configuration
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PROMPT = "a photo of an astronaut riding a horse on mars"
NUM_INFERENCE_STEPS = 20
NUM_WARMUP_RUNS = 2
NUM_PROFILE_RUNS = 3
OUTPUT_DIR = "./profiling_results/single_gpu"
IMAGE_SIZE = 512

def setup_model(device):
    """Load and configure the diffusion model."""
    print(f"Loading model {MODEL_ID}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Enable memory efficient attention if available
    if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("Enabled xformers memory efficient attention")
        except Exception as e:
            print(f"xformers not available: {e}")

    return pipe

def warmup(pipe, prompt, num_runs=2):
    """Warmup runs to stabilize CUDA kernels."""
    print(f"Running {num_runs} warmup iterations...")
    for i in range(num_runs):
        with torch.no_grad():
            _ = pipe(
                prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                output_type="latent",
            )
        torch.cuda.synchronize()
    print("Warmup complete.")

def profile_inference(pipe, prompt, output_dir):
    """Profile the diffusion inference with torch.profiler."""
    os.makedirs(output_dir, exist_ok=True)

    # Profile schedule: skip warmup, then profile active runs
    schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=NUM_PROFILE_RUNS,
        repeat=1
    )

    results = {
        "model": MODEL_ID,
        "device": str(pipe.device),
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "image_size": IMAGE_SIZE,
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

        for run_idx in range(1 + NUM_PROFILE_RUNS):  # 1 warmup + active runs
            torch.cuda.synchronize()
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

            elapsed_time = start_event.elapsed_time(end_event)  # ms

            if run_idx > 0:  # Skip warmup run
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
    print("PROFILER ANALYSIS")
    print("="*80)

    # Key averages by CUDA time
    print("\nTop 20 operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    # Key averages by CPU time
    print("\nTop 20 operations by CPU time:")
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20
    ))

    # Memory usage
    print("\nTop 10 operations by CUDA memory:")
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=10
    ))

    return prof.key_averages()

def categorize_operations(key_averages):
    """Categorize operations into compute vs other categories."""
    categories = {
        "matmul_compute": [],  # Matrix multiplications
        "conv_compute": [],   # Convolutions
        "attention": [],      # Attention operations
        "normalization": [],  # LayerNorm, GroupNorm, etc.
        "activation": [],     # ReLU, GELU, SiLU, etc.
        "memory_ops": [],     # Copy, transpose, etc.
        "other": [],
    }

    for item in key_averages:
        name = item.key.lower()
        cuda_time = item.self_device_time_total

        if any(x in name for x in ['gemm', 'matmul', 'mm', 'linear', 'addmm']):
            categories["matmul_compute"].append((item.key, cuda_time))
        elif any(x in name for x in ['conv', 'cudnn']):
            categories["conv_compute"].append((item.key, cuda_time))
        elif any(x in name for x in ['attention', 'softmax', 'scaled_dot']):
            categories["attention"].append((item.key, cuda_time))
        elif any(x in name for x in ['norm', 'layer_norm', 'group_norm', 'batch_norm']):
            categories["normalization"].append((item.key, cuda_time))
        elif any(x in name for x in ['relu', 'gelu', 'silu', 'sigmoid', 'tanh']):
            categories["activation"].append((item.key, cuda_time))
        elif any(x in name for x in ['copy', 'transpose', 'permute', 'contiguous', 'view']):
            categories["memory_ops"].append((item.key, cuda_time))
        else:
            categories["other"].append((item.key, cuda_time))

    print("\n" + "="*80)
    print("OPERATION CATEGORIES (by CUDA time)")
    print("="*80)

    total_time = sum(item.self_device_time_total for item in key_averages)

    for cat_name, ops in categories.items():
        cat_time = sum(t for _, t in ops)
        pct = (cat_time / total_time * 100) if total_time > 0 else 0
        print(f"\n{cat_name}: {cat_time/1000:.2f} ms ({pct:.1f}%)")
        # Show top 3 ops in each category
        for op, t in sorted(ops, key=lambda x: -x[1])[:3]:
            print(f"  - {op}: {t/1000:.2f} ms")

    return categories

def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    # Setup
    pipe = setup_model(device)

    # Warmup
    warmup(pipe, PROMPT, NUM_WARMUP_RUNS)

    # Profile
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}/{timestamp}"
    prof, results = profile_inference(pipe, PROMPT, output_dir)

    # Analyze
    key_averages = analyze_profile(prof)
    categories = categorize_operations(key_averages)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model: {MODEL_ID}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Inference steps: {NUM_INFERENCE_STEPS}")
    print(f"Mean inference time: {results['statistics']['mean_time_ms']:.2f} ms")
    print(f"Mean time per step: {results['statistics']['mean_time_per_step_ms']:.2f} ms")
    print(f"\nTo view traces: tensorboard --logdir {output_dir}")

if __name__ == "__main__":
    main()
