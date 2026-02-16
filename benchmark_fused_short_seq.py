#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Runner for PagedAttention Kernel Optimizations (AMD MI300X)

Three benchmark types:
  kernel     - Raw kernel latency (microseconds), no model needed
  model      - End-to-end inference latency with a real model (seconds)
  throughput - Real-dataset throughput (req/s, tok/s) with ShareGPT/sonnet

Compares 5 kernel modes:
  1. baseline       - Original 2-kernel path (QKV + reduce always)
  2. fused          - Skip reduce for short seqs (seq <= 256)
  3. mfma4_all      - Use mfma4 for ALL GQA ratios + fused (seq <= 256)
  4. mfma4_all_512  - mfma4 for all GQA + 512-token partition (seq <= 512)
  5. mfma4_all_1024 - mfma4 for all GQA + 1024-token partition (seq <= 1024)

Each mode runs in a separate subprocess with the appropriate env vars,
since the kernel caches env var values in static variables.

Usage:
    # === KERNEL BENCHMARK (default) ===
    python benchmark_fused_short_seq.py kernel

    # Run specific modes only
    python benchmark_fused_short_seq.py kernel --modes baseline fused

    # Custom sweep
    python benchmark_fused_short_seq.py kernel \
        --seq-lengths 64 128 256 512 1024 \
        --batch-sizes 16 32

    # === MODEL BENCHMARK ===
    # Test all predefined models (GQA=4 and GQA=7)
    python benchmark_fused_short_seq.py model --preset all

    # Test specific presets
    python benchmark_fused_short_seq.py model --preset llama3-8b qwen2-7b

    # Test custom model(s)
    python benchmark_fused_short_seq.py model \
        --model Qwen/Qwen2.5-7B-Instruct \
               NousResearch/Hermes-3-Llama-3.1-8B

    # Model benchmark with custom configs
    python benchmark_fused_short_seq.py model \
        --preset llama3-8b \
        --input-lens 64 128 256 512 \
        --output-len 64 --batch-size 32

    # === THROUGHPUT BENCHMARK (real conversations) ===
    # Test all models with ShareGPT dataset
    python benchmark_fused_short_seq.py throughput --preset all \
        --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json

    # Specific model with sonnet dataset
    python benchmark_fused_short_seq.py throughput \
        --preset llama3-8b \
        --dataset-name sonnet \
        --dataset-path vllm/benchmarks/sonnet.txt

    # === COMMON OPTIONS ===
    # Include markdown table
    python benchmark_fused_short_seq.py kernel --markdown

    # Load and display previous results
    python benchmark_fused_short_seq.py --load ../benchmark_results/results.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime


# ============================================================================
# Model profiles: well-known models with their GQA configurations
# ============================================================================

MODEL_PROFILES = {
    "llama3-8b": {
        "name": "NousResearch/Hermes-3-Llama-3.1-8B",
        "description": "Llama-3.1 8B (GQA=4, 32q/8kv)",
        "gqa_ratio": 4,
        "query_heads": 32,
        "kv_heads": 8,
    },
    "qwen2-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5 7B (GQA=7, 28q/4kv)",
        "gqa_ratio": 7,
        "query_heads": 28,
        "kv_heads": 4,
    },
    "mistral-7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral 7B v0.3 (GQA=4, 32q/8kv)",
        "gqa_ratio": 4,
        "query_heads": 32,
        "kv_heads": 8,
    },
}


# ============================================================================
# Mode definitions: name -> env var settings
# ============================================================================

MODES = {
    "baseline": {
        "label": "Baseline (original 2-kernel)",
        "short": "Baseline",
        "env": {
            "VLLM_ROCM_FUSED_SHORT_SEQ": "0",
            "VLLM_ROCM_MFMA4_ALL": "0",
            "VLLM_ROCM_PARTITION_512": "0",
            "VLLM_ROCM_PARTITION_1024": "0",
        },
        "partition_size": 256,
    },
    "fused": {
        "label": "Fused short-seq (skip reduce, seq<=256)",
        "short": "Fused",
        "env": {
            "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
            "VLLM_ROCM_MFMA4_ALL": "0",
            "VLLM_ROCM_PARTITION_512": "0",
            "VLLM_ROCM_PARTITION_1024": "0",
        },
        "partition_size": 256,
    },
    "mfma4_all": {
        "label": "mfma4-all GQA + fused (seq<=256)",
        "short": "mfma4-all",
        "env": {
            "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
            "VLLM_ROCM_MFMA4_ALL": "1",
            "VLLM_ROCM_PARTITION_512": "0",
            "VLLM_ROCM_PARTITION_1024": "0",
        },
        "partition_size": 256,
    },
    "mfma4_all_512": {
        "label": "mfma4-all GQA + fused 512 (seq<=512)",
        "short": "mfma4+512",
        "env": {
            "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
            "VLLM_ROCM_MFMA4_ALL": "1",
            "VLLM_ROCM_PARTITION_512": "1",
            "VLLM_ROCM_PARTITION_1024": "0",
        },
        "partition_size": 512,
    },
    "mfma4_all_1024": {
        "label": "mfma4-all GQA + fused 1024 (seq<=1024)",
        "short": "mfma4+1024",
        "env": {
            "VLLM_ROCM_FUSED_SHORT_SEQ": "1",
            "VLLM_ROCM_MFMA4_ALL": "1",
            "VLLM_ROCM_PARTITION_512": "0",
            "VLLM_ROCM_PARTITION_1024": "1",
        },
        "partition_size": 1024,
    },
}

ALL_MODE_NAMES = list(MODES.keys())

# Results are saved OUTSIDE the repo to avoid cluttering git
# Default: ../benchmark_results/ (sibling to repo root)
DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_results"
)


# ============================================================================
# Git helpers
# ============================================================================

def get_git_info(repo_dir: str = ".") -> dict:
    """Get git branch and commit hash."""
    info = {"branch": "unknown", "commit": "unknown"}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        info["branch"] = result.stdout.strip() or "unknown"
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        info["commit"] = result.stdout.strip() or "unknown"
    except Exception:
        pass
    return info


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_single_benchmark(
    mode_name: str,
    seq_len: int,
    batch_size: int,
    num_runs: int = 3,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
) -> float:
    """Run the paged attention benchmark for a given mode.

    Returns average latency in microseconds.
    """
    mode = MODES[mode_name]

    env = os.environ.copy()
    env.update(mode["env"])

    # benchmark_paged_attention.py is in benchmarks/kernels/ relative to repo root
    benchmark_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "benchmarks", "kernels", "benchmark_paged_attention.py",
    )

    cmd = [
        sys.executable,
        benchmark_script,
        "--version", "v2",
        "--custom-paged-attn",
        "--seq-len", str(seq_len),
        "--batch-size", str(batch_size),
        "--num-query-heads", str(num_query_heads),
        "--num-kv-heads", str(num_kv_heads),
        "--head-size", str(head_size),
    ]

    latencies = []
    for run in range(num_runs):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=120,
            )

            match = re.search(r"Kernel running time:\s+([\d.]+)\s*us",
                              result.stdout)
            if match:
                latencies.append(float(match.group(1)))
            else:
                print(f"    Warning: Could not parse output "
                      f"(run {run+1}/{num_runs})")
                if result.stderr:
                    lines = result.stderr.strip().split("\n")
                    print(f"    stderr: {lines[-1][:120]}")
        except subprocess.TimeoutExpired:
            print(f"    Warning: Run {run+1} timed out")
        except Exception as e:
            print(f"    Error: {e}")

    if not latencies:
        return 0.0

    return sum(latencies) / len(latencies)


def is_fused_active(mode_name: str, seq_len: int) -> bool:
    """Check if the fused (single-partition) path is active for this config."""
    mode = MODES[mode_name]
    if mode["env"]["VLLM_ROCM_FUSED_SHORT_SEQ"] == "0":
        return False
    return seq_len <= mode["partition_size"]


def run_benchmark_suite(
    mode_names: list,
    seq_lengths: list,
    batch_sizes: list,
    num_runs: int = 3,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
) -> dict:
    """Run benchmark across all modes, seq lengths, and batch sizes."""

    total = len(seq_lengths) * len(batch_sizes) * len(mode_names)
    current = 0

    vllm_dir = os.path.dirname(os.path.abspath(__file__))
    git_info = get_git_info(vllm_dir)

    print("\n" + "=" * 80)
    print("PagedAttention Kernel Optimization Benchmark (AMD MI300X)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Branch:            {git_info['branch']} ({git_info['commit']})")
    print(f"  Modes:             {mode_names}")
    print(f"  Sequence lengths:  {seq_lengths}")
    print(f"  Batch sizes:       {batch_sizes}")
    print(f"  Runs per config:   {num_runs}")
    print(f"  Query heads:       {num_query_heads}")
    print(f"  KV heads:          {num_kv_heads} "
          f"(GQA={num_query_heads // num_kv_heads})")
    print(f"  Head size:         {head_size}")
    print(f"  Total benchmarks:  {total}")
    print("-" * 80)

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            print(f"\n--- seq_len={seq_len}, batch_size={batch_size} ---")

            for mode_name in mode_names:
                current += 1
                mode = MODES[mode_name]
                fused = is_fused_active(mode_name, seq_len)
                tag = " [FUSED]" if fused else ""

                print(f"  [{current}/{total}] {mode['short']:<12}{tag}",
                      end="", flush=True)

                latency_us = run_single_benchmark(
                    mode_name, seq_len, batch_size, num_runs,
                    num_query_heads, num_kv_heads, head_size,
                )

                print(f"  -> {latency_us:.3f} us")

                results.append({
                    "mode": mode_name,
                    "mode_label": mode["short"],
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "latency_us": round(latency_us, 3),
                    "fused_active": fused,
                    "partition_size": mode["partition_size"],
                    "num_query_heads": num_query_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_size": head_size,
                })

    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "kernel",
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "modes": mode_names,
            "seq_lengths": seq_lengths,
            "batch_sizes": batch_sizes,
            "num_runs": num_runs,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "gqa_ratio": num_query_heads // num_kv_heads,
        },
        "results": results,
    }

    return data


# ============================================================================
# Model (End-to-End) Benchmark
# ============================================================================

def run_single_model_benchmark(
    mode_name: str,
    model: str,
    input_len: int,
    output_len: int,
    batch_size: int,
    num_iters: int = 10,
    dtype: str = "half",
    max_model_len: int = 4096,
    enforce_eager: bool = True,
) -> dict:
    """Run vllm bench latency for a given mode.

    Returns dict with avg_latency (seconds) and percentiles.
    """
    mode = MODES[mode_name]

    env = os.environ.copy()
    env.update(mode["env"])

    # Write results to a temp JSON file so we can parse structured output
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "latency",
        "--model", model,
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--batch-size", str(batch_size),
        "--num-iters", str(num_iters),
        "--num-iters-warmup", "3",
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--output-json", tmp.name,
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600,  # 10 min timeout for model loading + inference
        )

        if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
            with open(tmp.name) as f:
                data = json.load(f)
            return {
                "avg_latency": data.get("avg_latency", 0),
                "percentiles": data.get("percentiles", {}),
            }

        # Fallback: parse stdout
        match = re.search(r"Avg latency:\s+([\d.]+)\s*seconds", result.stdout)
        if match:
            return {"avg_latency": float(match.group(1)), "percentiles": {}}

        print(f"    Warning: Could not parse output")
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            for line in lines[-3:]:
                print(f"    stderr: {line[:150]}")
        return {"avg_latency": 0, "percentiles": {}}

    except subprocess.TimeoutExpired:
        print(f"    Warning: Timed out after 600s")
        return {"avg_latency": 0, "percentiles": {}}
    except Exception as e:
        print(f"    Error: {e}")
        return {"avg_latency": 0, "percentiles": {}}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def run_model_benchmark_suite(
    mode_names: list,
    models: list,
    input_lens: list,
    output_len: int = 64,
    batch_size: int = 32,
    num_iters: int = 10,
    dtype: str = "half",
    max_model_len: int = 4096,
    enforce_eager: bool = True,
) -> dict:
    """Run model latency benchmark across all models, modes, and input lengths."""

    total = len(models) * len(input_lens) * len(mode_names)
    current = 0

    vllm_dir = os.path.dirname(os.path.abspath(__file__))
    git_info = get_git_info(vllm_dir)

    print("\n" + "=" * 80)
    print("Model Latency Benchmark (AMD MI300X)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Branch:            {git_info['branch']} ({git_info['commit']})")
    print(f"  Models:            {models}")
    print(f"  Modes:             {mode_names}")
    print(f"  Input lengths:     {input_lens}")
    print(f"  Output length:     {output_len}")
    print(f"  Batch size:        {batch_size}")
    print(f"  Iterations:        {num_iters}")
    print(f"  Dtype:             {dtype}")
    print(f"  Max model len:     {max_model_len}")
    print(f"  Enforce eager:     {enforce_eager}")
    print(f"  Total benchmarks:  {total}")
    print("-" * 80)

    results = []

    for model in models:
        # Look up GQA info from profiles
        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = (f" (GQA={pval['gqa_ratio']}, "
                                f"{pval['query_heads']}q/{pval['kv_heads']}kv)")
                break

        print(f"\n{'=' * 80}")
        print(f"Model: {model}{profile_info}")
        print(f"{'=' * 80}")

        for input_len in input_lens:
            print(f"\n--- input_len={input_len}, output_len={output_len}, "
                  f"batch_size={batch_size} ---")

            for mode_name in mode_names:
                current += 1
                mode = MODES[mode_name]
                fused = is_fused_active(mode_name, input_len)
                tag = " [FUSED]" if fused else ""

                print(f"  [{current}/{total}] {mode['short']:<12}{tag}",
                      end="", flush=True)

                res = run_single_model_benchmark(
                    mode_name, model, input_len, output_len,
                    batch_size, num_iters, dtype, max_model_len,
                    enforce_eager,
                )

                avg = res["avg_latency"]
                if avg > 0:
                    print(f"  -> {avg:.4f}s ({avg*1000:.1f}ms)")
                else:
                    print(f"  -> FAILED")

                results.append({
                    "mode": mode_name,
                    "mode_label": mode["short"],
                    "model": model,
                    "input_len": input_len,
                    "output_len": output_len,
                    "batch_size": batch_size,
                    "avg_latency_s": round(avg, 6),
                    "avg_latency_ms": round(avg * 1000, 2),
                    "percentiles": res["percentiles"],
                    "fused_active": fused,
                    "partition_size": mode["partition_size"],
                })

    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "model",
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "models": models,
            "modes": mode_names,
            "input_lens": input_lens,
            "output_len": output_len,
            "batch_size": batch_size,
            "num_iters": num_iters,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
        },
        "results": results,
    }

    return data


# ============================================================================
# Throughput (Real Dataset) Benchmark
# ============================================================================

def run_single_throughput_benchmark(
    mode_name: str,
    model: str,
    dataset_name: str,
    dataset_path: str,
    num_prompts: int = 100,
    dtype: str = "half",
    max_model_len: int = 4096,
    enforce_eager: bool = True,
) -> dict:
    """Run vllm bench throughput for a given mode.

    Returns dict with requests_per_second, tokens_per_second, etc.
    """
    mode = MODES[mode_name]

    env = os.environ.copy()
    env.update(mode["env"])

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "throughput",
        "--model", model,
        "--dataset-name", dataset_name,
        "--dataset-path", dataset_path,
        "--num-prompts", str(num_prompts),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--output-json", tmp.name,
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=900,  # 15 min timeout
        )

        if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
            with open(tmp.name) as f:
                data = json.load(f)
            return {
                "requests_per_second": data.get("requests_per_second", 0),
                "tokens_per_second": data.get("tokens_per_second", 0),
                "elapsed_time": data.get("elapsed_time", 0),
                "num_requests": data.get("num_requests", 0),
                "total_num_tokens": data.get("total_num_tokens", 0),
            }

        # Fallback: parse stdout
        match = re.search(
            r"Throughput:\s+([\d.]+)\s+requests/s,\s+([\d.]+)\s+total tokens/s",
            result.stdout,
        )
        if match:
            return {
                "requests_per_second": float(match.group(1)),
                "tokens_per_second": float(match.group(2)),
                "elapsed_time": 0,
                "num_requests": num_prompts,
                "total_num_tokens": 0,
            }

        print(f"    Warning: Could not parse output")
        if result.stderr:
            lines = result.stderr.strip().split("\n")
            for line in lines[-3:]:
                print(f"    stderr: {line[:150]}")
        return {"requests_per_second": 0, "tokens_per_second": 0,
                "elapsed_time": 0, "num_requests": 0, "total_num_tokens": 0}

    except subprocess.TimeoutExpired:
        print(f"    Warning: Timed out after 900s")
        return {"requests_per_second": 0, "tokens_per_second": 0,
                "elapsed_time": 0, "num_requests": 0, "total_num_tokens": 0}
    except Exception as e:
        print(f"    Error: {e}")
        return {"requests_per_second": 0, "tokens_per_second": 0,
                "elapsed_time": 0, "num_requests": 0, "total_num_tokens": 0}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def run_throughput_benchmark_suite(
    mode_names: list,
    models: list,
    dataset_name: str,
    dataset_path: str,
    num_prompts: int = 100,
    dtype: str = "half",
    max_model_len: int = 4096,
    enforce_eager: bool = True,
) -> dict:
    """Run throughput benchmark across all models and modes."""

    total = len(models) * len(mode_names)
    current = 0

    vllm_dir = os.path.dirname(os.path.abspath(__file__))
    git_info = get_git_info(vllm_dir)

    print("\n" + "=" * 80)
    print("Throughput Benchmark (AMD MI300X) - Real Dataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Branch:            {git_info['branch']} ({git_info['commit']})")
    print(f"  Models:            {models}")
    print(f"  Modes:             {mode_names}")
    print(f"  Dataset:           {dataset_name} ({dataset_path})")
    print(f"  Num prompts:       {num_prompts}")
    print(f"  Dtype:             {dtype}")
    print(f"  Max model len:     {max_model_len}")
    print(f"  Enforce eager:     {enforce_eager}")
    print(f"  Total benchmarks:  {total}")
    print("-" * 80)

    results = []

    for model in models:
        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = (f" (GQA={pval['gqa_ratio']}, "
                                f"{pval['query_heads']}q/{pval['kv_heads']}kv)")
                break

        print(f"\n{'=' * 80}")
        print(f"Model: {model}{profile_info}")
        print(f"{'=' * 80}")

        for mode_name in mode_names:
            current += 1
            mode = MODES[mode_name]

            print(f"  [{current}/{total}] {mode['short']:<12}",
                  end="", flush=True)

            res = run_single_throughput_benchmark(
                mode_name, model, dataset_name, dataset_path,
                num_prompts, dtype, max_model_len, enforce_eager,
            )

            rps = res["requests_per_second"]
            tps = res["tokens_per_second"]
            if rps > 0:
                print(f"  -> {rps:.2f} req/s, {tps:.1f} tok/s")
            else:
                print(f"  -> FAILED")

            results.append({
                "mode": mode_name,
                "mode_label": mode["short"],
                "model": model,
                "requests_per_second": round(rps, 2),
                "tokens_per_second": round(tps, 1),
                "elapsed_time": round(res["elapsed_time"], 2),
                "num_requests": res["num_requests"],
                "total_num_tokens": res["total_num_tokens"],
            })

    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark_type": "throughput",
            "git_branch": git_info["branch"],
            "git_commit": git_info["commit"],
            "models": models,
            "modes": mode_names,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "num_prompts": num_prompts,
            "dtype": dtype,
            "max_model_len": max_model_len,
            "enforce_eager": enforce_eager,
        },
        "results": results,
    }

    return data


def print_throughput_comparison_table(data: dict):
    """Print throughput comparison grouped by model."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        print("No results to display.")
        return

    # Preserve model order
    seen = []
    for r in results:
        if r["model"] not in seen:
            seen.append(r["model"])
    model_list = seen

    for model in model_list:
        model_results = [r for r in results if r["model"] == model]

        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = (f"  GQA={pval['gqa_ratio']} "
                                f"({pval['query_heads']}q/{pval['kv_heads']}kv)")
                break

        print("\n" + "=" * 80)
        print(f"THROUGHPUT COMPARISON")
        print("=" * 80)
        print(f"  Model: {model}{profile_info}")
        print(f"  Dataset: {data['metadata']['dataset_name']}")
        print(f"  Prompts: {data['metadata']['num_prompts']}")
        print(f"  Branch: {data['metadata'].get('git_branch', '?')} "
              f"({data['metadata'].get('git_commit', '?')})")

        print(f"\n{'Mode':<16} | {'req/s':>8} | {'tok/s':>10} | "
              f"{'Time (s)':>9} | {'vs Base':>8}")
        print("-" * 65)

        mode_data = {}
        for r in model_results:
            mode_data[r["mode"]] = r

        baseline_rps = mode_data.get("baseline", {}).get(
            "requests_per_second", 0)

        for m in mode_names:
            if m not in mode_data:
                continue
            r = mode_data[m]
            rps = r["requests_per_second"]
            tps = r["tokens_per_second"]
            elapsed = r["elapsed_time"]

            if rps > 0:
                if baseline_rps > 0:
                    speedup = f"{rps / baseline_rps:.2f}x"
                else:
                    speedup = "N/A"
                print(f"  {MODES[m]['short']:<14} | {rps:>8.2f} | {tps:>10.1f} | "
                      f"{elapsed:>9.2f} | {speedup:>8}")
            else:
                print(f"  {MODES[m]['short']:<14} | {'FAILED':>8} | "
                      f"{'':>10} | {'':>9} | {'':>8}")

        print("-" * 65)

        # Best mode
        valid = [r for r in model_results if r["requests_per_second"] > 0]
        if valid:
            best = max(valid, key=lambda r: r["requests_per_second"])
            print(f"  Best: {MODES[best['mode']]['short']} "
                  f"({best['requests_per_second']:.2f} req/s, "
                  f"{best['tokens_per_second']:.1f} tok/s)")


def print_throughput_markdown_table(data: dict):
    """Print throughput results in markdown format."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        return

    seen = []
    for r in results:
        if r["model"] not in seen:
            seen.append(r["model"])

    for model in seen:
        model_results = [r for r in results if r["model"] == model]

        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = f" (GQA={pval['gqa_ratio']})"
                break

        short_name = model.split("/")[-1]
        print("\n" + "=" * 70)
        print(f"MARKDOWN TABLE: {short_name}{profile_info} - Throughput")
        print("=" * 70 + "\n")

        print(f"| Mode | req/s | tok/s | Time (s) | vs Baseline |")
        print(f"|------|------:|------:|---------:|------------:|")

        mode_data = {}
        for r in model_results:
            mode_data[r["mode"]] = r

        baseline_rps = mode_data.get("baseline", {}).get(
            "requests_per_second", 0)

        for m in mode_names:
            if m not in mode_data:
                continue
            r = mode_data[m]
            rps = r["requests_per_second"]
            tps = r["tokens_per_second"]
            elapsed = r["elapsed_time"]

            if rps > 0:
                speedup = (f"{rps / baseline_rps:.2f}x"
                           if baseline_rps > 0 else "N/A")
                print(f"| {MODES[m]['short']} | {rps:.2f} | {tps:.1f} | "
                      f"{elapsed:.2f} | {speedup} |")
            else:
                print(f"| {MODES[m]['short']} | FAILED | | | |")


# ============================================================================
# Output
# ============================================================================

def print_comparison_table(data: dict):
    """Print a side-by-side comparison table of all modes."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        print("No results to display.")
        return

    from collections import defaultdict

    # Group by (num_kv_heads, seq_len, batch_size)
    kv_heads_set = sorted(set(r.get("num_kv_heads", 8) for r in results))
    num_query_heads = data["metadata"]["num_query_heads"]

    for nkv in kv_heads_set:
        kv_results = [r for r in results if r.get("num_kv_heads", 8) == nkv]
        grouped = defaultdict(dict)
        for r in kv_results:
            key = (r["seq_len"], r["batch_size"])
            grouped[key][r["mode"]] = r

        gqa = num_query_heads // nkv

        print("\n" + "=" * (42 + 18 * len(mode_names)))
        print(f"RESULTS COMPARISON  (GQA={gqa}, {num_query_heads}q/{nkv}kv)")
        print("=" * (42 + 18 * len(mode_names)))
        print(f"  Branch: {data['metadata'].get('git_branch', '?')} "
              f"({data['metadata'].get('git_commit', '?')})")

        # Header
        mode_headers = "".join(
            f" {'':>2}{MODES[m]['short']:>13}" for m in mode_names
        )
        print(f"\n{'Seq Len':>8} {'Batch':>6} |{mode_headers} | "
              f"{'Best':>10} {'vs Base':>8}")
        print("-" * (42 + 18 * len(mode_names)))

        for (seq_len, batch_size) in sorted(grouped.keys()):
            modes = grouped[(seq_len, batch_size)]

            latencies = {}
            for m in mode_names:
                if m in modes:
                    latencies[m] = modes[m]["latency_us"]

            if not latencies:
                continue

            baseline_us = latencies.get("baseline", 0)

            cols = ""
            for m in mode_names:
                if m in latencies:
                    val = latencies[m]
                    fused = modes[m].get("fused_active", False)
                    marker = "*" if fused else " "
                    cols += f" {marker}{val:>12.1f}us"
                else:
                    cols += f"  {'N/A':>12}  "

            best_mode = min(latencies, key=lambda k: latencies[k])
            best_us = latencies[best_mode]
            if baseline_us > 0:
                speedup = baseline_us / best_us
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{seq_len:>8} {batch_size:>6} |{cols} | "
                  f"{MODES[best_mode]['short']:>10} {speedup_str:>8}")

        print("-" * (42 + 18 * len(mode_names)))
        print("  * = fused (single-partition) path active")

        # Per-mode summary vs baseline
        if "baseline" in mode_names and len(mode_names) > 1:
            print(f"\nSpeedup summary vs baseline:")
            for m in mode_names:
                if m == "baseline":
                    continue
                speedups = []
                for key, modes in grouped.items():
                    if "baseline" in modes and m in modes:
                        b = modes["baseline"]["latency_us"]
                        o = modes[m]["latency_us"]
                        if b > 0 and o > 0:
                            speedups.append(b / o)
                if speedups:
                    avg_sp = sum(speedups) / len(speedups)
                    max_sp = max(speedups)
                    min_sp = min(speedups)
                    print(f"  {MODES[m]['short']:<14}: "
                          f"avg {avg_sp:.2f}x, "
                          f"best {max_sp:.2f}x, "
                          f"worst {min_sp:.2f}x")


def print_markdown_table(data: dict):
    """Print results in markdown format."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        return

    from collections import defaultdict

    kv_heads_set = sorted(set(r.get("num_kv_heads", 8) for r in results))
    num_query_heads = data["metadata"]["num_query_heads"]

    for nkv in kv_heads_set:
        kv_results = [r for r in results if r.get("num_kv_heads", 8) == nkv]
        grouped = defaultdict(dict)
        for r in kv_results:
            key = (r["seq_len"], r["batch_size"])
            grouped[key][r["mode"]] = r

        gqa = num_query_heads // nkv

        print("\n" + "=" * 70)
        print(f"MARKDOWN TABLE  (GQA={gqa}, {num_query_heads}q/{nkv}kv)")
        print("=" * 70 + "\n")

        mode_cols = " | ".join(f"{MODES[m]['short']} (us)" for m in mode_names)
        print(f"| Seq Len | Batch | {mode_cols} | Best | Speedup |")
        sep_cols = " | ".join("---:" for _ in mode_names)
        print(f"|---------|-------|{sep_cols}|------|---------|")

        for (seq_len, batch_size) in sorted(grouped.keys()):
            modes = grouped[(seq_len, batch_size)]
            latencies = {}
            for m in mode_names:
                if m in modes:
                    latencies[m] = modes[m]["latency_us"]

            cols = " | ".join(
                f"{latencies[m]:.1f}" if m in latencies else "N/A"
                for m in mode_names
            )

            baseline_us = latencies.get("baseline", 0)
            best_mode = min(latencies, key=lambda k: latencies[k]) if latencies else "N/A"
            best_us = latencies.get(best_mode, 0)
            speedup = (f"{baseline_us / best_us:.2f}x"
                       if baseline_us > 0 and best_us > 0 else "N/A")

            print(f"| {seq_len} | {batch_size} | {cols} | "
                  f"{MODES.get(best_mode, {}).get('short', 'N/A')} | {speedup} |")


def print_model_comparison_table(data: dict):
    """Print a side-by-side comparison table of model benchmark results."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        print("No results to display.")
        return

    from collections import defaultdict

    # Group by model
    model_list = sorted(set(r.get("model", data["metadata"].get("model", "?"))
                            for r in results),
                        key=lambda m: results[[r.get("model", "") for r in results].index(m)]["input_len"] if m in [r.get("model", "") for r in results] else 0)
    # Simpler: preserve insertion order
    seen = []
    for r in results:
        m = r.get("model", data["metadata"].get("model", "?"))
        if m not in seen:
            seen.append(m)
    model_list = seen

    for model in model_list:
        model_results = [r for r in results
                         if r.get("model", data["metadata"].get("model", "?")) == model]

        # Look up profile info
        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = (f"  GQA={pval['gqa_ratio']} "
                                f"({pval['query_heads']}q/{pval['kv_heads']}kv)")
                break

        grouped = defaultdict(dict)
        for r in model_results:
            key = (r["input_len"], r["output_len"], r["batch_size"])
            grouped[key][r["mode"]] = r

        print("\n" + "=" * (50 + 16 * len(mode_names)))
        print(f"MODEL LATENCY COMPARISON")
        print("=" * (50 + 16 * len(mode_names)))
        print(f"  Model: {model}{profile_info}")
        print(f"  Branch: {data['metadata'].get('git_branch', '?')} "
              f"({data['metadata'].get('git_commit', '?')})")

        # Header
        mode_headers = "".join(
            f" {'':>1}{MODES[m]['short']:>11}ms" for m in mode_names
        )
        print(f"\n{'In Len':>7} {'Out':>4} {'Batch':>6} |{mode_headers} | "
              f"{'Best':>10} {'vs Base':>8}")
        print("-" * (50 + 16 * len(mode_names)))

        for (input_len, output_len, batch_size) in sorted(grouped.keys()):
            modes = grouped[(input_len, output_len, batch_size)]

            latencies = {}
            for m in mode_names:
                if m in modes and modes[m]["avg_latency_ms"] > 0:
                    latencies[m] = modes[m]["avg_latency_ms"]

            if not latencies:
                continue

            baseline_ms = latencies.get("baseline", 0)

            cols = ""
            for m in mode_names:
                if m in latencies:
                    val = latencies[m]
                    fused = modes[m].get("fused_active", False)
                    marker = "*" if fused else " "
                    cols += f" {marker}{val:>10.1f}ms"
                else:
                    cols += f"  {'N/A':>10}  "

            best_mode = min(latencies, key=lambda k: latencies[k])
            best_ms = latencies[best_mode]
            if baseline_ms > 0:
                speedup = baseline_ms / best_ms
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            print(f"{input_len:>7} {output_len:>4} {batch_size:>6} |{cols} | "
                  f"{MODES[best_mode]['short']:>10} {speedup_str:>8}")

        print("-" * (50 + 16 * len(mode_names)))
        print("  * = fused (single-partition) path active")

        # Per-mode summary vs baseline
        if "baseline" in mode_names and len(mode_names) > 1:
            print(f"\nSpeedup summary vs baseline:")
            for m in mode_names:
                if m == "baseline":
                    continue
                speedups = []
                for key, modes in grouped.items():
                    if "baseline" in modes and m in modes:
                        b = modes["baseline"]["avg_latency_ms"]
                        o = modes[m]["avg_latency_ms"]
                        if b > 0 and o > 0:
                            speedups.append(b / o)
                if speedups:
                    avg_sp = sum(speedups) / len(speedups)
                    max_sp = max(speedups)
                    min_sp = min(speedups)
                    print(f"  {MODES[m]['short']:<14}: "
                          f"avg {avg_sp:.3f}x, "
                          f"best {max_sp:.3f}x, "
                          f"worst {min_sp:.3f}x")


def print_model_markdown_table(data: dict):
    """Print model benchmark results in markdown format."""

    results = data["results"]
    mode_names = data["metadata"]["modes"]
    if not results:
        return

    from collections import defaultdict

    # Preserve model order
    seen = []
    for r in results:
        m = r.get("model", data["metadata"].get("model", "?"))
        if m not in seen:
            seen.append(m)
    model_list = seen

    for model in model_list:
        model_results = [r for r in results
                         if r.get("model", data["metadata"].get("model", "?")) == model]

        profile_info = ""
        for pval in MODEL_PROFILES.values():
            if pval["name"] == model:
                profile_info = f" (GQA={pval['gqa_ratio']})"
                break

        grouped = defaultdict(dict)
        for r in model_results:
            key = (r["input_len"], r["output_len"], r["batch_size"])
            grouped[key][r["mode"]] = r

        short_name = model.split("/")[-1]
        print("\n" + "=" * 70)
        print(f"MARKDOWN TABLE: {short_name}{profile_info}")
        print("=" * 70 + "\n")

        mode_cols = " | ".join(f"{MODES[m]['short']} (ms)" for m in mode_names)
        print(f"| In Len | Out Len | Batch | {mode_cols} | Best | Speedup |")
        sep_cols = " | ".join("---:" for _ in mode_names)
        print(f"|--------|---------|-------|{sep_cols}|------|---------|")

        for (input_len, output_len, batch_size) in sorted(grouped.keys()):
            modes = grouped[(input_len, output_len, batch_size)]
            latencies = {}
            for m in mode_names:
                if m in modes and modes[m]["avg_latency_ms"] > 0:
                    latencies[m] = modes[m]["avg_latency_ms"]

            cols = " | ".join(
                f"{latencies[m]:.1f}" if m in latencies else "N/A"
                for m in mode_names
            )

            baseline_ms = latencies.get("baseline", 0)
            best_mode = min(latencies, key=lambda k: latencies[k]) if latencies else "N/A"
            best_ms = latencies.get(best_mode, 0)
            speedup = (f"{baseline_ms / best_ms:.2f}x"
                       if baseline_ms > 0 and best_ms > 0 else "N/A")

            print(f"| {input_len} | {output_len} | {batch_size} | {cols} | "
                  f"{MODES.get(best_mode, {}).get('short', 'N/A')} | {speedup} |")


def save_results(data: dict, output_dir: str):
    """Save results to JSON file outside the repo."""
    os.makedirs(output_dir, exist_ok=True)

    branch = data["metadata"].get("git_branch", "unknown")
    btype = data["metadata"].get("benchmark_type", "kernel")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_branch = branch.replace("/", "_").replace(" ", "_")
    filename = f"{btype}_{safe_branch}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {filepath}")
    return filepath


# ============================================================================
# Main
# ============================================================================

def add_common_args(parser):
    """Add arguments common to both kernel and model subcommands."""
    parser.add_argument(
        "--modes", type=str, nargs="+",
        default=ALL_MODE_NAMES,
        choices=ALL_MODE_NAMES,
        help=f"Modes to benchmark (default: all). "
             f"Choices: {ALL_MODE_NAMES}",
    )
    parser.add_argument(
        "--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
        help="Directory to save results (default: ../benchmark_results/)",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Also print markdown table for reports",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PagedAttention kernel optimizations on AMD MI300X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  kernel          Raw kernel latency benchmark (no model needed)
  model           End-to-end model latency benchmark

Modes (available for both subcommands):
  baseline        Original 2-kernel path (QKV + reduce always)
  fused           Skip reduce for short seqs (seq <= 256)
  mfma4_all       Use mfma4 for ALL GQA ratios + fused (seq <= 256)
  mfma4_all_512   mfma4 for all GQA + 512-token partition (seq <= 512)
  mfma4_all_1024  mfma4 for all GQA + 1024-token partition (seq <= 1024)

Examples:
  # Kernel benchmark (default if no subcommand)
  python benchmark_fused_short_seq.py kernel
  python benchmark_fused_short_seq.py kernel --modes baseline mfma4_all_512

  # Model benchmark
  python benchmark_fused_short_seq.py model --model meta-llama/Llama-3.1-8B-Instruct
  python benchmark_fused_short_seq.py model --model meta-llama/Llama-3.1-8B-Instruct \\
      --input-lens 64 128 256 512 --modes baseline mfma4_all_512

  # Load saved results
  python benchmark_fused_short_seq.py --load ../benchmark_results/results.json
        """,
    )

    parser.add_argument(
        "--load", type=str, default=None,
        help="Load and display results from a previous JSON file",
    )

    subparsers = parser.add_subparsers(dest="command")

    # --- kernel subcommand ---
    kernel_parser = subparsers.add_parser(
        "kernel",
        help="Raw kernel latency benchmark (no model needed)",
    )
    add_common_args(kernel_parser)
    kernel_parser.add_argument(
        "--seq-lengths", type=int, nargs="+",
        default=[64, 128, 256, 512, 1024],
        help="Sequence lengths to test (default: 64 128 256 512 1024)",
    )
    kernel_parser.add_argument(
        "--batch-sizes", type=int, nargs="+",
        default=[32],
        help="Batch sizes to test (default: 32)",
    )
    kernel_parser.add_argument(
        "--num-runs", type=int, default=3,
        help="Number of runs per configuration (default: 3)",
    )
    kernel_parser.add_argument(
        "--num-query-heads", type=int, default=64,
        help="Number of query heads (default: 64)",
    )
    kernel_parser.add_argument(
        "--num-kv-heads", type=int, nargs="+", default=[8],
        help="Number of KV heads to test (default: 8). "
             "Multiple values run separate sweeps, e.g. --num-kv-heads 4 8 12",
    )
    kernel_parser.add_argument(
        "--head-size", type=int, default=128,
        choices=[64, 80, 96, 112, 120, 128, 192, 256],
        help="Head size (default: 128)",
    )

    # --- model subcommand ---
    model_parser = subparsers.add_parser(
        "model",
        help="End-to-end model latency benchmark",
    )
    add_common_args(model_parser)

    model_group = model_parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model", type=str, nargs="+",
        help="HuggingFace model name(s) (e.g. Qwen/Qwen2.5-7B-Instruct)",
    )
    profile_keys = list(MODEL_PROFILES.keys())
    model_group.add_argument(
        "--preset", type=str, nargs="+",
        choices=profile_keys + ["all"],
        help=("Predefined model profile(s): "
              + ", ".join(f"{k} ({v['description']})"
                          for k, v in MODEL_PROFILES.items())
              + ", or 'all' for all profiles"),
    )

    model_parser.add_argument(
        "--input-lens", type=int, nargs="+",
        default=[64, 128, 256, 512],
        help="Input lengths to test (default: 64 128 256 512)",
    )
    model_parser.add_argument(
        "--output-len", type=int, default=64,
        help="Output length (default: 64)",
    )
    model_parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size (default: 32)",
    )
    model_parser.add_argument(
        "--num-iters", type=int, default=10,
        help="Number of iterations per config (default: 10)",
    )
    model_parser.add_argument(
        "--dtype", type=str, default="half",
        choices=["half", "bfloat16", "float"],
        help="Data type (default: half)",
    )
    model_parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Max model context length (default: 4096)",
    )
    model_parser.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Disable --enforce-eager (default: enforce-eager is ON)",
    )

    # --- throughput subcommand ---
    tp_parser = subparsers.add_parser(
        "throughput",
        help="Real-dataset throughput benchmark (req/s, tok/s)",
    )
    add_common_args(tp_parser)

    tp_model_group = tp_parser.add_mutually_exclusive_group(required=True)
    tp_model_group.add_argument(
        "--model", type=str, nargs="+",
        help="HuggingFace model name(s)",
    )
    tp_model_group.add_argument(
        "--preset", type=str, nargs="+",
        choices=profile_keys + ["all"],
        help="Predefined model profile(s), or 'all'",
    )

    tp_parser.add_argument(
        "--dataset-name", type=str, default="sharegpt",
        choices=["sharegpt", "sonnet"],
        help="Dataset type (default: sharegpt)",
    )
    tp_parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Path to dataset file "
             "(e.g. ShareGPT_V3_unfiltered_cleaned_split.json)",
    )
    tp_parser.add_argument(
        "--num-prompts", type=int, default=100,
        help="Number of prompts to process (default: 100)",
    )
    tp_parser.add_argument(
        "--dtype", type=str, default="half",
        choices=["half", "bfloat16", "float"],
        help="Data type (default: half)",
    )
    tp_parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="Max model context length (default: 4096)",
    )
    tp_parser.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Disable --enforce-eager (default: enforce-eager is ON)",
    )

    args = parser.parse_args()

    # Load mode: display saved results (works without subcommand)
    if args.load:
        with open(args.load) as f:
            data = json.load(f)
        print(f"Loaded results from: {args.load}")
        btype = data.get("metadata", {}).get("benchmark_type", "kernel")
        md = args.markdown if hasattr(args, "markdown") else False
        if btype == "model":
            print_model_comparison_table(data)
            if md:
                print_model_markdown_table(data)
        elif btype == "throughput":
            print_throughput_comparison_table(data)
            if md:
                print_throughput_markdown_table(data)
        else:
            print_comparison_table(data)
            if md:
                print_markdown_table(data)
        return

    # Default to kernel if no subcommand given
    if args.command is None:
        parser.print_help()
        print("\nHint: use 'kernel', 'model', or 'throughput' subcommand. Examples:")
        print("  python benchmark_fused_short_seq.py kernel")
        print("  python benchmark_fused_short_seq.py model --preset all")
        print("  python benchmark_fused_short_seq.py throughput --preset all "
              "--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json")
        return

    if args.command == "kernel":
        for nkv in args.num_kv_heads:
            if args.num_query_heads % nkv != 0:
                kernel_parser.error(
                    f"num_query_heads ({args.num_query_heads}) must be "
                    f"divisible by num_kv_heads ({nkv})")

        all_data = []
        for nkv in args.num_kv_heads:
            data = run_benchmark_suite(
                mode_names=args.modes,
                seq_lengths=args.seq_lengths,
                batch_sizes=args.batch_sizes,
                num_runs=args.num_runs,
                num_query_heads=args.num_query_heads,
                num_kv_heads=nkv,
                head_size=args.head_size,
            )
            all_data.append(data)
            print_comparison_table(data)
            if args.markdown:
                print_markdown_table(data)

        # Merge all KV head runs into one combined dataset
        if len(all_data) > 1:
            merged = {
                "metadata": {
                    **all_data[0]["metadata"],
                    "num_kv_heads": args.num_kv_heads,
                    "gqa_ratios": [args.num_query_heads // nkv
                                   for nkv in args.num_kv_heads],
                },
                "results": [],
            }
            for d in all_data:
                merged["results"].extend(d["results"])
            data = merged
        else:
            data = all_data[0]

    elif args.command == "model":
        # Resolve model list from --model or --preset
        if args.preset:
            if "all" in args.preset:
                presets = list(MODEL_PROFILES.keys())
            else:
                presets = args.preset
            models = [MODEL_PROFILES[p]["name"] for p in presets]
        else:
            models = args.model

        data = run_model_benchmark_suite(
            mode_names=args.modes,
            models=models,
            input_lens=args.input_lens,
            output_len=args.output_len,
            batch_size=args.batch_size,
            num_iters=args.num_iters,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enforce_eager=not args.no_enforce_eager,
        )

        print_model_comparison_table(data)
        if args.markdown:
            print_model_markdown_table(data)

    elif args.command == "throughput":
        # Resolve model list from --model or --preset
        if args.preset:
            if "all" in args.preset:
                presets = list(MODEL_PROFILES.keys())
            else:
                presets = args.preset
            models = [MODEL_PROFILES[p]["name"] for p in presets]
        else:
            models = args.model

        data = run_throughput_benchmark_suite(
            mode_names=args.modes,
            models=models,
            dataset_name=args.dataset_name,
            dataset_path=args.dataset_path,
            num_prompts=args.num_prompts,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            enforce_eager=not args.no_enforce_eager,
        )

        print_throughput_comparison_table(data)
        if args.markdown:
            print_throughput_markdown_table(data)

    else:
        return

    # Save results outside the repo
    filepath = save_results(data, args.results_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print(f"Re-display:  python benchmark_fused_short_seq.py --load {filepath}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
