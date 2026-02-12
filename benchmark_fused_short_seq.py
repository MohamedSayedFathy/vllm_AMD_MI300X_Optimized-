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

RESULTS_REPO_URL = "https://github.com/MohamedSayedFathy/benchmark_results.git"
RESULTS_DIR_NAME = "benchmark_results"


# ============================================================================
# Git Helpers
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


def setup_results_repo(vllm_dir: str) -> str:
    """
    Ensure the benchmark_results repo exists as a sibling directory.
    Clones from GitHub if it doesn't exist, pulls if it does.
    Returns the path to the results repo.
    """
    parent_dir = os.path.dirname(os.path.abspath(vllm_dir))
    results_dir = os.path.join(parent_dir, RESULTS_DIR_NAME)

    if os.path.isdir(os.path.join(results_dir, ".git")):
        # Repo exists, pull latest
        print(f"\n[Results] Found existing repo at {results_dir}")
        subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=results_dir, capture_output=True,
        )
    else:
        # Try to clone
        print(f"\n[Results] Cloning {RESULTS_REPO_URL}...")
        result = subprocess.run(
            ["git", "clone", RESULTS_REPO_URL, results_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Repo doesn't exist on GitHub yet, init locally
            print(f"[Results] Clone failed, initializing new repo at {results_dir}")
            os.makedirs(results_dir, exist_ok=True)
            subprocess.run(["git", "init"], cwd=results_dir)
            subprocess.run(
                ["git", "remote", "add", "origin", RESULTS_REPO_URL],
                cwd=results_dir,
            )

    return results_dir


def push_results(data: dict, vllm_dir: str):
    """
    Save benchmark results to the benchmark_results repo and push.
    File is named: <branch>_<timestamp>.json
    """
    results_dir = setup_results_repo(vllm_dir)

    branch = data["metadata"].get("git_branch", "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize branch name for filename
    safe_branch = branch.replace("/", "_").replace(" ", "_")
    filename = f"{safe_branch}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[Results] Saved: {filepath}")

    # Git add, commit, push
    subprocess.run(["git", "add", filename], cwd=results_dir)
    commit_msg = f"benchmark: {branch} @ {timestamp}"
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=results_dir, capture_output=True,
    )

    result = subprocess.run(
        ["git", "push", "-u", "origin", "main"],
        cwd=results_dir, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"[Results] Pushed to {RESULTS_REPO_URL}")
    else:
        # Try pushing to master if main doesn't exist
        result2 = subprocess.run(
            ["git", "push", "-u", "origin", "master"],
            cwd=results_dir, capture_output=True, text=True,
        )
        if result2.returncode == 0:
            print(f"[Results] Pushed to {RESULTS_REPO_URL}")
        else:
            print(f"[Results] Push failed (you may need to create the repo on GitHub first)")
            print(f"[Results] Error: {result.stderr.strip()}")
            print(f"[Results] Results saved locally at: {filepath}")


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
) -> float:
    """Run the paged attention benchmark and return average latency in microseconds."""

    env = os.environ.copy()
    env["VLLM_ROCM_FUSED_SHORT_SEQ"] = "1" if optimized else "0"

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
    print(f"\nConfiguration:")
    print(f"  Sequence lengths:  {seq_lengths}")
    print(f"  Batch sizes:       {batch_sizes}")
    print(f"  Runs per config:   {num_runs}")
    print(f"  Query heads:       {num_query_heads}")
    print(f"  KV heads:          {num_kv_heads}")
    print(f"  Head size:         {head_size}")
    print(f"  Partition size:    256 tokens")
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
                head_size=head_size,
            )
            print(f"{optimized_us:.3f} us")

            # Run baseline (always 2-kernel path)
            print("  Baseline  (FUSED_SHORT_SEQ=0)...", end=" ", flush=True)
            baseline_us = run_single_benchmark(
                seq_len, batch_size, num_runs, optimized=False,
                num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
                head_size=head_size,
            )
            print(f"{baseline_us:.3f} us")

            # Compute derived metrics
            improvement_us = baseline_us - optimized_us
            improvement_pct = ((improvement_us / baseline_us) * 100) if baseline_us > 0 else 0.0
            speedup = (baseline_us / optimized_us) if optimized_us > 0 else 1.0
            applies = seq_len <= 256
            num_partitions = (seq_len + 255) // 256

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
            "partition_size": 256,
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
        print(f"\nOptimization active (seq_len <= 256):")
        print(f"  Avg improvement:  {avg_pct:.1f}%")
        print(f"  Max improvement:  {max_pct:.1f}%")
        print(f"  Avg speedup:      {avg_speedup:.2f}x")
        print(f"  Avg time saved:   {avg_us:.3f} us/kernel call")

    # Check non-optimized (should show ~0 difference)
    non_opt = [r for r in results if not r["applies_optimization"]]
    if non_opt:
        print(f"\nBaseline verification (seq_len > 256, should be ~0 difference):")
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
    parser.add_argument("--no-push", action="store_true",
                        help="Don't push results to benchmark_results repo")

    args = parser.parse_args()

    # Run benchmarks
    data = run_benchmark_suite(
        seq_lengths=args.seq_lengths,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
    )

    # Print text results
    print_text_table(data)

    if args.markdown:
        print_markdown_table(data)

    # Save JSON locally
    save_results(data, args.output)

    # Push to benchmark_results repo
    if not args.no_push:
        vllm_dir = os.path.dirname(os.path.abspath(__file__))
        push_results(data, vllm_dir)

    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print(f"Results saved to: {args.output}")
    print(f"Generate graphs:  python benchmark_visualization.py --data-file {args.output} --graphs all")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
