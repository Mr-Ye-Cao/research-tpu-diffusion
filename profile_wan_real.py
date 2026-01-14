"""
Real Wan2.1 DiT Model Profiling Script

This script profiles the actual Wan2.1-T2V-1.3B model on a single GPU
to measure compute operation breakdown.

Usage:
    cd /home/ye/workspace/tpu/research-tpu-diffusion
    CUDA_VISIBLE_DEVICES=5 python profile_wan_real.py
"""

import os
import sys
import json
import torch
import torch.profiler
from datetime import datetime

# Add Wan2.1 repo to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Wan2.1'))

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

# Configuration
CKPT_DIR = "./Wan2.1-T2V-1.3B"
TASK = "t2v-1.3B"
PROMPT = "A cat walking on the grass"
SIZE = "480*832"  # Smaller for faster profiling (width*height)
FRAME_NUM = 17    # Smaller for faster profiling (4n+1)
SAMPLE_STEPS = 10  # Fewer steps for profiling
OUTPUT_DIR = "./profiling_results/wan_real"

NUM_WARMUP_RUNS = 1
NUM_PROFILE_RUNS = 2


def setup_model(device):
    """Load Wan2.1 model."""
    print(f"Loading Wan2.1 model from {CKPT_DIR}...")
    print(f"Task: {TASK}")

    cfg = WAN_CONFIGS[TASK]

    # Load the Wan pipeline
    model = wan.WanT2V(
        config=cfg,
        checkpoint_dir=CKPT_DIR,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    )

    print("Model loaded successfully!")
    return model


def warmup(model, num_runs=1):
    """Warmup runs."""
    print(f"Running {num_runs} warmup iterations...")
    size_tuple = SIZE_CONFIGS[SIZE]  # e.g., (480, 832) for "480*832"
    for i in range(num_runs):
        with torch.no_grad():
            _ = model.generate(
                input_prompt=PROMPT,
                size=size_tuple,
                frame_num=FRAME_NUM,
                sampling_steps=SAMPLE_STEPS,
                seed=42,
                offload_model=False,  # Keep on GPU for consistent profiling
            )
        torch.cuda.synchronize()
    print("Warmup complete.")


def profile_inference(model, output_dir):
    """Profile Wan2.1 inference."""
    os.makedirs(output_dir, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=0,
        warmup=1,
        active=NUM_PROFILE_RUNS,
        repeat=1
    )

    results = {
        "model": "Wan2.1-T2V-1.3B",
        "architecture": "DiT (Diffusion Transformer)",
        "task": TASK,
        "size": SIZE,
        "frame_num": FRAME_NUM,
        "sample_steps": SAMPLE_STEPS,
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
                video = model.generate(
                    input_prompt=PROMPT,
                    size=SIZE_CONFIGS[SIZE],
                    frame_num=FRAME_NUM,
                    sampling_steps=SAMPLE_STEPS,
                    seed=42 + run_idx,
                    offload_model=False,
                )

            end_event.record()
            torch.cuda.synchronize()

            elapsed_time = start_event.elapsed_time(end_event)

            if run_idx > 0:
                results["runs"].append({
                    "run_index": run_idx,
                    "total_time_ms": elapsed_time,
                    "time_per_step_ms": elapsed_time / SAMPLE_STEPS,
                })
                print(f"  Run {run_idx}: {elapsed_time:.2f} ms ({elapsed_time/SAMPLE_STEPS:.2f} ms/step)")

            prof.step()

    # Calculate statistics
    times = [r["total_time_ms"] for r in results["runs"]]
    results["statistics"] = {
        "mean_time_ms": sum(times) / len(times),
        "min_time_ms": min(times),
        "max_time_ms": max(times),
        "mean_time_per_step_ms": sum(times) / len(times) / SAMPLE_STEPS,
    }

    # Save results
    results_file = os.path.join(output_dir, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")
    return prof, results


def analyze_profile(prof):
    """Analyze profiler results."""
    print("\n" + "="*80)
    print("PROFILER ANALYSIS - Wan2.1 DiT")
    print("="*80)

    print("\nTop 20 operations by CUDA time:")
    print(prof.key_averages().table(
        sort_by="self_device_time_total",
        row_limit=20
    ))

    return prof.key_averages()


def categorize_dit_operations(key_averages):
    """Categorize DiT operations."""
    categories = {
        "attention": [],
        "linear_ffn": [],
        "normalization": [],
        "activation": [],
        "embedding": [],
        "memory_ops": [],
        "conv": [],
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
    print("Wan2.1 DiT OPERATION CATEGORIES")
    print("="*80)

    total_time = sum(item.self_device_time_total for item in key_averages)

    breakdown = {}
    for cat_name, ops in categories.items():
        cat_time = sum(t for _, t in ops)
        pct = (cat_time / total_time * 100) if total_time > 0 else 0
        breakdown[cat_name] = {"time_ms": cat_time/1000, "percentage": pct}
        print(f"\n{cat_name}: {cat_time/1000:.2f} ms ({pct:.1f}%)")
        for op, t in sorted(ops, key=lambda x: -x[1])[:3]:
            print(f"  - {op[:80]}: {t/1000:.2f} ms")

    return categories, breakdown


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return

    device = torch.device("cuda:0")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Check flash_attn availability
    try:
        import flash_attn
        print(f"Flash Attention: Available (v{flash_attn.__version__})")
    except ImportError:
        print("Flash Attention: Not available (using SDPA fallback)")

    # Setup model
    model = setup_model(device)

    # Warmup
    warmup(model, NUM_WARMUP_RUNS)

    # Profile
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{OUTPUT_DIR}/{timestamp}"
    prof, results = profile_inference(model, output_dir)

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
    print("SUMMARY - Wan2.1 DiT Model (REAL)")
    print("="*80)
    print(f"Model: Wan2.1-T2V-1.3B")
    print(f"Video: {FRAME_NUM} frames @ {SIZE}")
    print(f"Inference steps: {SAMPLE_STEPS}")
    print(f"Mean inference time: {results['statistics']['mean_time_ms']:.2f} ms")
    print(f"Mean time per step: {results['statistics']['mean_time_per_step_ms']:.2f} ms")

    attention_pct = breakdown.get("attention", {}).get("percentage", 0)
    linear_pct = breakdown.get("linear_ffn", {}).get("percentage", 0)
    print(f"\nDiT Compute Profile:")
    print(f"  Attention: {attention_pct:.1f}%")
    print(f"  Linear/FFN: {linear_pct:.1f}%")
    print(f"  (Attention+Linear = {attention_pct + linear_pct:.1f}%)")

    print(f"\nTo view traces: tensorboard --logdir {output_dir}")


if __name__ == "__main__":
    main()
