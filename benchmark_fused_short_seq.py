#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark Runner for Fused Short-Seq Kernel Optimization (AMD MI300X)

Runs A/B comparisons of PagedAttention v2 with and without the fused
short-seq optimization. Outputs results as formatted text to console
and saves structured data to JSON for later graph generation.

Usage:
    # Basic run (prints text results + saves JSON)
    python benchmark_fused_short_seq.py

    # Custom sweep
    python benchmark_fused_short_seq.py \
        --seq-lengths 32 64 128 256 512 1024 \
        --batch-sizes 16 32 64 \
        --num-runs 5

    # Quick single-batch test
    python benchmark_fused_short_seq.py --seq-lengths 128 256 --batch-sizes 32

    # Specify output file
    python benchmark_fused_short_seq.py --output results.json

    # Include markdown table for report
    python benchmark_fused_short_seq.py --markdown

    # Test mfma4-for-all-GQA experiment
    python benchmark_fused_short_seq.py --mfma4-all

    # Test mfma4-for-all with 512-token partitions (fused up to seq_len=512)
    python benchmark_fused_short_seq.py --mfma4-all --partition-512

    # Compare all 3 kernel configs side by side
    python benchmark_fused_short_seq.py --compare-all

Graph generation is handled separately by benchmark_visualization.py:
    python benchmark_visualization.py --data-file results.json --graphs all
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

RESULTS_DIR_NAME = "benchmark_results"


# ============================================================================
# Git & Results Helpers
# ============================================================================

def get_git_branch(repo_dir: str = ".") -> str:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def get_git_commit(repo_dir: str = ".") -> str:
    """Get the current git short commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=repo_dir,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def save_to_results_dir(data: dict, vllm_dir: str):
    """
    Save benchmark results to the benchmark_results/ directory
    within the current repo.
    File is named: <branch>_<timestamp>.json
    """
    results_dir = os.path.join(vllm_dir, RESULTS_DIR_NAME)
    os.makedirs(results_dir, exist_ok=True)

    branch = data["metadata"].get("git_branch", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize branch name for filename
    safe_branch = branch.replace("/", "_").replace(" ", "_")
    filename = f"{safe_branch}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Results] Saved: {filepath}")


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_single_benchmark(
    seq_len: int,
    batch_size: int,
    num_runs: int = 3,
    optimized: bool = True,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
    mfma4_all: bool = False,
    partition_512: bool = False,
) -> float:
    """Run the paged attention benchmark and return average latency in microseconds."""

    env = os.environ.copy()
    env["VLLM_ROCM_FUSED_SHORT_SEQ"] = "1" if optimized else "0"
    env["VLLM_ROCM_MFMA4_ALL"] = "1" if mfma4_all else "0"
    env["VLLM_ROCM_PARTITION_512"] = "1" if partition_512 else "0"

    cmd = [
        sys.executable,
        "benchmark_paged_attention.py",
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
                cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "kernels"),
                timeout=120,
            )

            match = re.search(r"Kernel running time:\s+([\d.]+)\s*us", result.stdout)
            if match:
                latencies.append(float(match.group(1)))
            else:
                print(f"    Warning: Could not parse output (run {run+1})")
                if result.stderr:
                    # Print first line of stderr for diagnosis
                    first_line = result.stderr.strip().split("\n")[0]
                    print(f"    stderr: {first_line}")
        except subprocess.TimeoutExpired:
            print(f"    Warning: Run {run+1} timed out")
        except Exception as e:
            print(f"    Error: {e}")

    if not latencies:
        return 0.0

    return sum(latencies) / len(latencies)


def run_benchmark_suite(
    seq_lengths: List[int],
    batch_sizes: List[int],
    num_runs: int = 3,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
    mfma4_all: bool = False,
    partition_512: bool = False,
) -> dict:
    """
    Run full A/B benchmark suite.

    Returns a dict with metadata and results list, ready for JSON serialization.
    """

    total = len(seq_lengths) * len(batch_sizes)
    current = 0

    print("\n" + "=" * 70)
    print("Fused Short-Seq Kernel Optimization Benchmark")
    print("=" * 70)
    partition_size = 512 if partition_512 else 256
    kernel_mode = "mfma4 (all GQA)" if mfma4_all else "mixed (mfma4/mfma16)"

    print(f"\nConfiguration:")
    print(f"  Sequence lengths:  {seq_lengths}")
    print(f"  Batch sizes:       {batch_sizes}")
    print(f"  Runs per config:   {num_runs}")
    print(f"  Query heads:       {num_query_heads}")
    print(f"  KV heads:          {num_kv_heads}")
    print(f"  Head size:         {head_size}")
    print(f"  Partition size:    {partition_size} tokens")
    print(f"  Kernel mode:       {kernel_mode}")
    print(f"  Total configs:     {total}")
    print("-" * 70)

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            current += 1
            print(f"\n[{current}/{total}] seq_len={seq_len}, batch_size={batch_size}")

            # Run optimized (fused kernel for short seqs)
            print("  Optimized (FUSED_SHORT_SEQ=1)...", end=" ", flush=True)
            optimized_us = run_single_benchmark(
                seq_len, batch_size, num_runs, optimized=True,
                num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                head_size=head_size, mfma4_all=mfma4_all,
                partition_512=partition_512,
            )
            print(f"{optimized_us:.3f} us")

            # Run baseline (always 2-kernel path)
            print("  Baseline  (FUSED_SHORT_SEQ=0)...", end=" ", flush=True)
            baseline_us = run_single_benchmark(
                seq_len, batch_size, num_runs, optimized=False,
                num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                head_size=head_size, mfma4_all=mfma4_all,
                partition_512=partition_512,
            )
            print(f"{baseline_us:.3f} us")

            # Compute derived metrics
            improvement_us = baseline_us - optimized_us
            improvement_pct = ((improvement_us / baseline_us) * 100) if baseline_us > 0 else 0.0
            speedup = (baseline_us / optimized_us) if optimized_us > 0 else 1.0
            applies = seq_len <= partition_size
            num_partitions = (seq_len + partition_size - 1) // partition_size

            marker = "YES" if applies else "no"
            print(f"  -> Improvement: {improvement_pct:.1f}%, Speedup: {speedup:.2f}x, "
                  f"Single partition: {marker}")

            results.append({
                "seq_len": seq_len,
                "batch_size": batch_size,
                "optimized_us": round(optimized_us, 3),
                "baseline_us": round(baseline_us, 3),
                "improvement_us": round(improvement_us, 3),
                "improvement_pct": round(improvement_pct, 1),
                "speedup": round(speedup, 3),
                "applies_optimization": applies,
                "num_partitions": num_partitions,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "head_size": head_size,
            })

    vllm_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "git_branch": get_git_branch(vllm_dir),
            "git_commit": get_git_commit(vllm_dir),
            "seq_lengths": seq_lengths,
            "batch_sizes": batch_sizes,
            "num_runs": num_runs,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "partition_size": partition_size,
            "mfma4_all": mfma4_all,
            "partition_512": partition_512,
        },
        "results": results,
    }

    return data


# ============================================================================
# Text Output
# ============================================================================

def print_text_table(data: dict):
    """Print formatted text results table."""

    results = data["results"]
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    header = (f"{'Seq Len':>8} | {'Batch':>6} | {'Optimized':>12} | {'Baseline':>12} | "
              f"{'Improvement':>12} | {'Speedup':>8} | {'%':>7} | {'Fused':>5}")
    print(f"\n{header}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x["batch_size"], x["seq_len"])):
        fused = "YES" if r["applies_optimization"] else "no"
        print(f"{r['seq_len']:>8} | {r['batch_size']:>6} | "
              f"{r['optimized_us']:>10.3f}us | {r['baseline_us']:>10.3f}us | "
              f"{r['improvement_us']:>10.3f}us | {r['speedup']:>7.2f}x | "
              f"{r['improvement_pct']:>6.1f}% | {fused:>5}")

    print("-" * 90)

    # Summary for optimized cases
    opt = [r for r in results if r["applies_optimization"]]
    if opt:
        avg_pct = sum(r["improvement_pct"] for r in opt) / len(opt)
        avg_us = sum(r["improvement_us"] for r in opt) / len(opt)
        max_pct = max(r["improvement_pct"] for r in opt)
        avg_speedup = sum(r["speedup"] for r in opt) / len(opt)
        part_sz = data["metadata"].get("partition_size", 256)
        print(f"\nOptimization active (seq_len <= {part_sz}):")
        print(f"  Avg improvement:  {avg_pct:.1f}%")
        print(f"  Max improvement:  {max_pct:.1f}%")
        print(f"  Avg speedup:      {avg_speedup:.2f}x")
        print(f"  Avg time saved:   {avg_us:.3f} us/kernel call")

    # Check non-optimized (should show ~0 difference)
    non_opt = [r for r in results if not r["applies_optimization"]]
    if non_opt:
        print(f"\nBaseline verification (seq_len > {part_sz}, should be ~0 difference):")
        for r in non_opt:
            print(f"  seq_len={r['seq_len']}: {abs(r['improvement_us']):.3f} us "
                  f"({abs(r['improvement_pct']):.1f}%)")


def print_markdown_table(data: dict):
    """Print results in markdown format for reports."""

    results = data["results"]
    if not results:
        return

    print("\n" + "=" * 70)
    print("MARKDOWN TABLE (copy into report)")
    print("=" * 70 + "\n")

    print("| Seq Length | Batch | Partitions | Optimized (us) | Baseline (us) | "
          "Improvement | Speedup | Fused |")
    print("|------------|-------|------------|----------------|---------------|"
          "-------------|---------|-------|")

    for r in sorted(results, key=lambda x: (x["batch_size"], x["seq_len"])):
        fused = "Yes" if r["applies_optimization"] else "No"
        print(f"| {r['seq_len']} | {r['batch_size']} | {r['num_partitions']} | "
              f"{r['optimized_us']:.3f} | {r['baseline_us']:.3f} | "
              f"{r['improvement_pct']:.1f}% | {r['speedup']:.2f}x | {fused} |")


def save_results(data: dict, output_path: str):
    """Save results to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {path}")


# ============================================================================
# Compare-All Mode: 3-way side-by-side comparison
# ============================================================================

# The three kernel configurations to compare
CONFIGS = [
    {"name": "Original",   "short": "orig",    "mfma4_all": False, "partition_512": False},
    {"name": "mfma4-all",  "short": "mfma4",   "mfma4_all": True,  "partition_512": False},
    {"name": "mfma4+512",  "short": "m4+512",  "mfma4_all": True,  "partition_512": True},
]


def run_compare_all(
    seq_lengths: List[int],
    batch_sizes: List[int],
    num_runs: int = 3,
    num_query_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 128,
) -> dict:
    """
    Run all 3 kernel configs for each (seq_len, batch_size) combination.
    Returns structured data with results for each config.
    """
    total = len(seq_lengths) * len(batch_sizes)
    current = 0

    print("\n" + "=" * 70)
    print("3-Way Kernel Comparison Benchmark")
    print("=" * 70)
    print(f"\nConfigurations tested:")
    for cfg in CONFIGS:
        m4 = "ON" if cfg["mfma4_all"] else "OFF"
        p5 = "ON" if cfg["partition_512"] else "OFF"
        print(f"  {cfg['name']:12s}  MFMA4_ALL={m4}  PARTITION_512={p5}")
    print(f"\nSweep:")
    print(f"  Sequence lengths:  {seq_lengths}")
    print(f"  Batch sizes:       {batch_sizes}")
    print(f"  Runs per config:   {num_runs}")
    print(f"  Query heads:       {num_query_heads}")
    print(f"  KV heads:          {num_kv_heads}")
    print(f"  Head size:         {head_size}")
    print(f"  Total combos:      {total} x 3 configs = {total * 3} benchmarks")
    print("-" * 70)

    results = []

    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            current += 1
            print(f"\n[{current}/{total}] seq_len={seq_len}, batch_size={batch_size}")

            row = {"seq_len": seq_len, "batch_size": batch_size}

            for cfg in CONFIGS:
                label = cfg["name"]
                print(f"  {label:12s} ...", end=" ", flush=True)
                latency = run_single_benchmark(
                    seq_len, batch_size, num_runs, optimized=True,
                    num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                    head_size=head_size,
                    mfma4_all=cfg["mfma4_all"],
                    partition_512=cfg["partition_512"],
                )
                row[cfg["short"]] = round(latency, 3)
                print(f"{latency:.3f} us")

            # Compute speedups relative to Original
            orig = row["orig"]
            if orig > 0:
                row["mfma4_vs_orig"] = round(orig / row["mfma4"], 3) if row["mfma4"] > 0 else 0.0
                row["m4_512_vs_orig"] = round(orig / row["m4+512"], 3) if row["m4+512"] > 0 else 0.0
            else:
                row["mfma4_vs_orig"] = 0.0
                row["m4_512_vs_orig"] = 0.0

            results.append(row)

    vllm_dir = os.path.dirname(os.path.abspath(__file__))
    data = {
        "metadata": {
            "mode": "compare_all",
            "timestamp": datetime.now().isoformat(),
            "git_branch": get_git_branch(vllm_dir),
            "git_commit": get_git_commit(vllm_dir),
            "seq_lengths": seq_lengths,
            "batch_sizes": batch_sizes,
            "num_runs": num_runs,
            "num_query_heads": num_query_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
            "configs": [c["name"] for c in CONFIGS],
        },
        "results": results,
    }
    return data


def print_comparison_table(data: dict):
    """Print side-by-side comparison of all 3 kernel configs."""
    results = data["results"]
    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 110)
    print("3-WAY COMPARISON (all with FUSED_SHORT_SEQ=1)")
    print("=" * 110)

    header = (f"{'Seq':>6} | {'Batch':>5} | "
              f"{'Original':>11} | {'mfma4-all':>11} | {'mfma4+512':>11} | "
              f"{'m4 vs orig':>10} | {'m4+512 vs orig':>14}")
    print(f"\n{header}")
    print("-" * 110)

    for r in sorted(results, key=lambda x: (x["batch_size"], x["seq_len"])):
        m4_speedup = r["mfma4_vs_orig"]
        m4_512_speedup = r["m4_512_vs_orig"]
        # Color hints: mark speedups > 1.0
        m4_tag = "+" if m4_speedup > 1.005 else ("-" if m4_speedup < 0.995 else "=")
        m4_512_tag = "+" if m4_512_speedup > 1.005 else ("-" if m4_512_speedup < 0.995 else "=")

        print(f"{r['seq_len']:>6} | {r['batch_size']:>5} | "
              f"{r['orig']:>9.3f}us | {r['mfma4']:>9.3f}us | {r['m4+512']:>9.3f}us | "
              f"{m4_tag}{m4_speedup:>8.3f}x | {m4_512_tag}{m4_512_speedup:>12.3f}x")

    print("-" * 110)

    # Summary
    print("\n  Legend:  + = faster than original, - = slower, = = same")
    print(f"  Configs: Original = mixed mfma4/mfma16, partition 256")
    print(f"           mfma4-all = mfma4 for all GQA, partition 256")
    print(f"           mfma4+512 = mfma4 for all GQA, partition 512")


def print_comparison_markdown(data: dict):
    """Print comparison in markdown format."""
    results = data["results"]
    if not results:
        return

    print("\n" + "=" * 70)
    print("MARKDOWN TABLE (copy into report)")
    print("=" * 70 + "\n")

    print("| Seq Len | Batch | Original (us) | mfma4-all (us) | mfma4+512 (us) | "
          "m4 speedup | m4+512 speedup |")
    print("|---------|-------|---------------|----------------|----------------|"
          "------------|----------------|")

    for r in sorted(results, key=lambda x: (x["batch_size"], x["seq_len"])):
        print(f"| {r['seq_len']} | {r['batch_size']} | "
              f"{r['orig']:.3f} | {r['mfma4']:.3f} | {r['m4+512']:.3f} | "
              f"{r['mfma4_vs_orig']:.3f}x | {r['m4_512_vs_orig']:.3f}x |")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Runner for Fused Short-Seq Kernel Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run
  python benchmark_fused_short_seq.py

  # Custom sweep
  python benchmark_fused_short_seq.py \\
      --seq-lengths 32 64 128 256 512 1024 \\
      --batch-sizes 8 16 32 64

  # Quick test
  python benchmark_fused_short_seq.py --seq-lengths 128 256 --batch-sizes 32

  # Save to specific file
  python benchmark_fused_short_seq.py --output my_results.json

  # Then generate graphs separately:
  python benchmark_visualization.py --data-file results.json --graphs all
        """
    )

    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[64, 128, 256, 512, 1024],
                        help="Sequence lengths to test (default: 64 128 256 512 1024)")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[32],
                        help="Batch sizes to test (default: 32)")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs per configuration (default: 3)")
    parser.add_argument("--num-query-heads", type=int, default=64,
                        help="Number of query heads (default: 64)")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="Number of KV heads (default: 8)")
    parser.add_argument("--head-size", type=int, default=128,
                        choices=[64, 80, 96, 112, 120, 128, 192, 256],
                        help="Head size (default: 128)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Output JSON file path (default: results.json)")
    parser.add_argument("--markdown", action="store_true",
                        help="Also print markdown table for reports")
    parser.add_argument("--mfma4-all", action="store_true",
                        help="Use mfma4 kernel for ALL GQA ratios (VLLM_ROCM_MFMA4_ALL=1)")
    parser.add_argument("--partition-512", action="store_true",
                        help="Use 512-token partitions (VLLM_ROCM_PARTITION_512=1, requires --mfma4-all)")
    parser.add_argument("--compare-all", action="store_true",
                        help="Run all 3 configs side by side: original, mfma4-all, mfma4+512")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to benchmark_results/ directory")

    args = parser.parse_args()

    if args.partition_512 and not args.mfma4_all:
        parser.error("--partition-512 requires --mfma4-all")

    if args.compare_all:
        # ---- 3-way comparison mode ----
        data = run_compare_all(
            seq_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes,
            num_runs=args.num_runs,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
        )
        print_comparison_table(data)
        if args.markdown:
            print_comparison_markdown(data)
    else:
        # ---- Single-config A/B mode ----
        data = run_benchmark_suite(
            seq_lengths=args.seq_lengths,
            batch_sizes=args.batch_sizes,
            num_runs=args.num_runs,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            mfma4_all=args.mfma4_all,
            partition_512=args.partition_512,
        )
        print_text_table(data)
        if args.markdown:
            print_markdown_table(data)

    # Save JSON locally
    save_results(data, args.output)

    # Save to benchmark_results/ directory in the repo
    if not args.no_save:
        vllm_dir = os.path.dirname(os.path.abspath(__file__))
        save_to_results_dir(data, vllm_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print(f"Results saved to: {args.output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
