#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Graph Generator for Fused Short-Seq Kernel Optimization (AMD MI300X)

Reads benchmark results from a JSON file (produced by benchmark_fused_short_seq.py)
and generates publication-quality graphs using matplotlib/seaborn.

Usage:
    # Generate all graphs from benchmark results
    python benchmark_visualization.py --data-file results.json --graphs all

    # Generate specific graphs
    python benchmark_visualization.py --data-file results.json --graphs latency speedup

    # Test graphs with built-in demo data (no benchmark data needed)
    python benchmark_visualization.py --demo --graphs all

    # Custom output directory
    python benchmark_visualization.py --data-file results.json --graphs all --output-dir my_graphs

    # List available graph types
    python benchmark_visualization.py --list-graphs

Workflow:
    1. Run benchmarks:  python benchmark_fused_short_seq.py --output results.json
    2. Generate graphs: python benchmark_visualization.py --data-file results.json --graphs all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for saving files
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================================
# Data Loading
# ============================================================================

def load_results(path: str) -> dict:
    """Load benchmark results from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def get_demo_data() -> dict:
    """Generate realistic demo data for testing graphs without running benchmarks."""
    demo_results = []
    # Simulated data based on typical MI300X performance
    raw = [
        # (seq_len, batch_size, optimized_us, baseline_us)
        # batch_size=8
        (32, 8, 5.2, 7.1), (64, 8, 5.8, 7.9), (128, 8, 6.5, 9.0), (256, 8, 8.1, 11.2),
        (384, 8, 12.5, 12.4), (512, 8, 15.8, 15.9), (768, 8, 22.1, 22.0), (1024, 8, 28.5, 28.6),
        # batch_size=16
        (32, 16, 6.1, 8.3), (64, 16, 6.8, 9.3), (128, 16, 7.6, 10.5), (256, 16, 9.5, 13.1),
        (384, 16, 14.2, 14.1), (512, 16, 18.5, 18.6), (768, 16, 26.2, 26.1), (1024, 16, 34.0, 34.1),
        # batch_size=32
        (32, 32, 7.2, 9.8), (64, 32, 8.0, 11.0), (128, 32, 8.9, 12.3), (256, 32, 11.1, 15.3),
        (384, 32, 16.8, 16.7), (512, 32, 22.0, 22.1), (768, 32, 31.5, 31.4), (1024, 32, 41.2, 41.3),
        # batch_size=64
        (32, 64, 9.5, 13.0), (64, 64, 10.5, 14.5), (128, 64, 11.8, 16.2), (256, 64, 14.6, 20.1),
        (384, 64, 22.0, 21.9), (512, 64, 29.5, 29.6), (768, 64, 43.0, 42.9), (1024, 64, 57.2, 57.3),
    ]

    for seq_len, batch_size, opt_us, base_us in raw:
        improvement_us = base_us - opt_us
        improvement_pct = (improvement_us / base_us) * 100 if base_us > 0 else 0
        speedup = base_us / opt_us if opt_us > 0 else 1.0
        demo_results.append({
            "seq_len": seq_len,
            "batch_size": batch_size,
            "optimized_us": opt_us,
            "baseline_us": base_us,
            "improvement_us": round(improvement_us, 3),
            "improvement_pct": round(improvement_pct, 1),
            "speedup": round(speedup, 3),
            "applies_optimization": seq_len <= 256,
            "num_partitions": (seq_len + 255) // 256,
        })

    return {
        "metadata": {
            "type": "demo",
            "description": "Simulated data for testing graph generation",
        },
        "results": demo_results,
    }


# ============================================================================
# Plot Style
# ============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    if HAS_SEABORN:
        sns.set_theme(style="whitegrid", palette="husl")
        sns.set_context("paper", font_scale=1.2)

    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


# ============================================================================
# Individual Graph Functions
# ============================================================================

def plot_latency(results: List[dict], output_dir: Path) -> List[Path]:
    """
    Bar chart comparing optimized vs baseline latency.
    Generates one graph per batch size found in the data.
    """
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    paths = []

    for batch_size in batch_sizes:
        filtered = sorted([r for r in results if r["batch_size"] == batch_size],
                          key=lambda x: x["seq_len"])
        if not filtered:
            continue

        seq_lens = [r["seq_len"] for r in filtered]
        optimized = [r["optimized_us"] for r in filtered]
        baseline = [r["baseline_us"] for r in filtered]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(seq_lens))
        width = 0.35

        bars_base = ax.bar(x - width/2, baseline, width, label="Baseline (2 kernels)",
                           color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
        bars_opt = ax.bar(x + width/2, optimized, width, label="Optimized (Fused)",
                          color="#27ae60", alpha=0.8, edgecolor="black", linewidth=0.5)

        # Highlight optimization zone (seq <= 256)
        opt_indices = [i for i, s in enumerate(seq_lens) if s <= 256]
        if opt_indices:
            ax.axvspan(min(opt_indices) - 0.5, max(opt_indices) + 0.5,
                       alpha=0.1, color="green", label="Optimization Zone")

        # Boundary line
        boundary = len([s for s in seq_lens if s <= 256]) - 0.5
        if 0 <= boundary < len(seq_lens):
            ax.axvline(x=boundary + 0.5, color="gray", linestyle="--", linewidth=2)

        # Value labels on bars
        for bar, val in zip(bars_base, baseline):
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#c0392b")
        for bar, val in zip(bars_opt, optimized):
            ax.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color="#1e8449")

        ax.set_xlabel("Sequence Length (tokens)", fontweight="bold")
        ax.set_ylabel("Latency (us)", fontweight="bold")
        ax.set_title(f"PagedAttention Kernel Latency: Fused vs Baseline\n(Batch Size = {batch_size})",
                     fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_lens)
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"latency_batch{batch_size}.png"
        plt.savefig(path)
        plt.close()
        paths.append(path)

    return paths


def plot_speedup(results: List[dict], output_dir: Path) -> Optional[Path]:
    """Line plot showing speedup factor across sequence lengths, one line per batch size."""
    batch_sizes = sorted(set(r["batch_size"] for r in results))
    if not batch_sizes:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(batch_sizes)))

    for batch_size, color in zip(batch_sizes, colors):
        filtered = sorted([r for r in results if r["batch_size"] == batch_size],
                          key=lambda x: x["seq_len"])
        seq_lens = [r["seq_len"] for r in filtered]
        speedups = [r["speedup"] for r in filtered]

        ax.plot(seq_lens, speedups, "o-", color=color, linewidth=2, markersize=8,
                label=f"Batch Size = {batch_size}", alpha=0.8)

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(x=256, color="red", linestyle="--", linewidth=2, alpha=0.7,
               label="Partition Boundary (256)")
    ax.axvspan(0, 256, alpha=0.1, color="green")

    ax.set_xlabel("Sequence Length (tokens)", fontweight="bold")
    ax.set_ylabel("Speedup (Baseline / Optimized)", fontweight="bold")
    ax.set_title("Fused Short-Seq Kernel Speedup Factor", fontweight="bold", pad=20)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.9)

    plt.tight_layout()
    path = output_dir / "speedup_curve.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_heatmap(results: List[dict], output_dir: Path) -> Optional[Path]:
    """Heatmap of improvement % across batch_size x seq_len."""
    if not HAS_SEABORN:
        print("  (requires seaborn: pip install seaborn)")
        return None

    seq_lens = sorted(set(r["seq_len"] for r in results))
    batch_sizes = sorted(set(r["batch_size"] for r in results))

    if len(batch_sizes) < 2:
        print("  (need >= 2 batch sizes for heatmap)")
        return None

    data = np.zeros((len(batch_sizes), len(seq_lens)))
    for r in results:
        i = batch_sizes.index(r["batch_size"])
        j = seq_lens.index(r["seq_len"])
        data[i, j] = r["improvement_pct"]

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    sns.heatmap(data, annot=True, fmt=".1f", cmap=cmap,
                xticklabels=seq_lens, yticklabels=batch_sizes,
                center=0, vmin=-5, vmax=40,
                cbar_kws={"label": "Improvement (%)"}, ax=ax)

    boundary = len([s for s in seq_lens if s <= 256])
    ax.axvline(x=boundary, color="black", linewidth=3)

    ax.set_xlabel("Sequence Length (tokens)", fontweight="bold")
    ax.set_ylabel("Batch Size", fontweight="bold")
    ax.set_title("Performance Improvement Heatmap (%)", fontweight="bold", pad=20)
    plt.xticks(rotation=45)

    plt.tight_layout()
    path = output_dir / "improvement_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_memory(output_dir: Path) -> Optional[Path]:
    """Bar chart of memory bandwidth savings (theoretical analysis)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Memory savings per sequence per layer (HEAD_SIZE=128, NUM_HEADS=32, FP16)
    categories = ["tmp_out Write", "tmp_out Read", "exp_sums", "max_logits"]
    savings_bytes = [8192, 8192, 128, 128]
    savings_kb = [b / 1024 for b in savings_bytes]

    colors = ["#3498db", "#3498db", "#9b59b6", "#9b59b6"]
    ax1.bar(categories, savings_kb, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Memory Saved (KB)", fontweight="bold")
    ax1.set_title("Memory Bandwidth Savings\n(Per Sequence, Per Layer)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    total_kb = sum(savings_kb)
    ax1.annotate(f"Total: {total_kb:.1f} KB",
                 xy=(0.5, 0.95), xycoords="axes fraction", ha="center",
                 fontsize=12, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    # Kernel launch overhead
    ax2.bar(["Reduce Kernel\nLaunch"], [10], color="#e74c3c",
            edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Overhead Eliminated (us)", fontweight="bold")
    ax2.set_title("Kernel Launch Overhead Savings", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, 15)

    plt.tight_layout()
    path = output_dir / "memory_savings.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_architecture(output_dir: Path) -> Optional[Path]:
    """Diagram showing before/after kernel architecture."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # === Baseline (before) ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 2)
    ax1.axis("off")
    ax1.set_title("BASELINE: Two-Kernel PagedAttention v2", fontweight="bold",
                  fontsize=14, color="#c0392b", pad=10)

    boxes = [
        (1, "Query\nInput", "#ecf0f1"),
        (3, "QKV\nKernel", "#3498db"),
        (5, "tmp_out\n(16KB)", "#f39c12"),
        (7, "Reduce\nKernel", "#e74c3c"),
        (9, "final_out", "#2ecc71"),
    ]
    for xpos, text, color in boxes:
        ax1.text(xpos, 1, text, ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor=color, edgecolor="black"))

    for x1, x2 in [(1.5, 2.5), (3.5, 4.5), (5.5, 6.5), (7.5, 8.5)]:
        ax1.annotate("", xy=(x2, 1), xytext=(x1, 1),
                     arrowprops=dict(arrowstyle="->", color="black", lw=2))

    ax1.text(5, 0.3, "Memory Bottleneck (Write + Read)", ha="center",
             fontsize=9, color="#c0392b", style="italic")

    # === Optimized (after) ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2)
    ax2.axis("off")
    ax2.set_title("OPTIMIZED: Fused Single-Kernel (seq <= 256)", fontweight="bold",
                  fontsize=14, color="#27ae60", pad=10)

    ax2.text(2, 1, "Query\nInput", ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#ecf0f1", edgecolor="black"))
    ax2.text(5, 1, "Fused QKV\nKernel", ha="center", va="center", fontsize=10,
             fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="#27ae60", edgecolor="black", lw=3))
    ax2.text(8, 1, "final_out", ha="center", va="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="#2ecc71", edgecolor="black"))

    ax2.annotate("", xy=(4, 1), xytext=(3, 1),
                 arrowprops=dict(arrowstyle="->", color="black", lw=2))
    ax2.annotate("", xy=(7, 1), xytext=(6, 1),
                 arrowprops=dict(arrowstyle="->", color="#27ae60", lw=3))

    ax2.text(5, 0.3, "Direct Write (No Intermediate Buffer)", ha="center",
             fontsize=9, color="#27ae60", fontweight="bold")
    ax2.text(5, 1.7, "tmp_out eliminated    Reduce Kernel skipped",
             ha="center", fontsize=10, color="#95a5a6", style="italic")

    plt.tight_layout()
    path = output_dir / "architecture_comparison.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_dashboard(results: List[dict], output_dir: Path) -> Optional[Path]:
    """Summary dashboard with key metrics, mini-charts, and use cases."""
    if not results:
        return None

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, hspace=0.3, wspace=0.3)

    # Stats
    opt_results = [r for r in results if r["applies_optimization"]]
    if opt_results:
        avg_speedup = np.mean([r["speedup"] for r in opt_results])
        max_speedup = max(r["speedup"] for r in opt_results)
        avg_improvement = np.mean([r["improvement_pct"] for r in opt_results])
    else:
        avg_speedup = max_speedup = avg_improvement = 0

    # Panel 1: Key metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    ax1.set_title("Key Metrics", fontweight="bold", fontsize=14)
    metrics = [
        (f"{avg_speedup:.2f}x", "Avg Speedup"),
        (f"{max_speedup:.2f}x", "Max Speedup"),
        (f"{avg_improvement:.1f}%", "Avg Improvement"),
        ("~16.5 KB", "Memory Saved/seq"),
    ]
    for i, (value, label) in enumerate(metrics):
        y = 0.8 - i * 0.2
        ax1.text(0.5, y, value, ha="center", va="center", fontsize=20,
                 fontweight="bold", color="#27ae60")
        ax1.text(0.5, y - 0.08, label, ha="center", va="center",
                 fontsize=11, color="gray")

    # Panel 2: Speedup bars
    ax2 = fig.add_subplot(gs[0, 1])
    batch_sizes_list = [r["batch_size"] for r in results]
    target_batch = 32 if 32 in batch_sizes_list else max(set(batch_sizes_list), key=batch_sizes_list.count)
    batch_data = sorted([r for r in results if r["batch_size"] == target_batch],
                        key=lambda x: x["seq_len"])

    if batch_data:
        seq_lens = [r["seq_len"] for r in batch_data]
        speedups = [r["speedup"] for r in batch_data]
        bar_colors = ["#27ae60" if s <= 256 else "#95a5a6" for s in seq_lens]
        ax2.bar(range(len(seq_lens)), speedups, color=bar_colors, edgecolor="black")
        ax2.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)
        ax2.set_xticks(range(len(seq_lens)))
        ax2.set_xticklabels(seq_lens, rotation=45)
        ax2.set_xlabel("Sequence Length")
        ax2.set_ylabel("Speedup")
        ax2.set_title(f"Speedup by Seq Length\n(batch={target_batch})", fontweight="bold")

    # Panel 3: Latency comparison
    ax3 = fig.add_subplot(gs[0, 2])
    if batch_data:
        x = np.arange(len(seq_lens))
        width = 0.35
        ax3.bar(x - width/2, [r["baseline_us"] for r in batch_data], width,
                label="Baseline", color="#e74c3c", alpha=0.7)
        ax3.bar(x + width/2, [r["optimized_us"] for r in batch_data], width,
                label="Optimized", color="#27ae60", alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels(seq_lens, rotation=45)
        ax3.set_xlabel("Sequence Length")
        ax3.set_ylabel("Latency (us)")
        ax3.set_title(f"Latency Comparison\n(batch={target_batch})", fontweight="bold")
        ax3.legend(fontsize=9)

    # Panel 4: Use cases
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    ax4.set_title("Best Use Cases", fontweight="bold", fontsize=14)
    use_cases = [
        ("+ Chatbot responses", "#27ae60"),
        ("+ Code completion", "#27ae60"),
        ("+ Real-time streaming", "#27ae60"),
        ("+ Short context queries", "#27ae60"),
        ("", "black"),
        ("- Long document QA", "#e74c3c"),
        ("- Summarization (long input)", "#e74c3c"),
        ("- Multi-turn w/ long history", "#e74c3c"),
    ]
    for i, (text, color) in enumerate(use_cases):
        if text:
            ax4.text(0.1, 0.9 - i * 0.1, text, fontsize=10, color=color, va="top")

    # Panel 5: Heatmap (if multiple batch sizes)
    ax5 = fig.add_subplot(gs[1, 0:2])
    seq_lens_all = sorted(set(r["seq_len"] for r in results))
    batch_sizes_all = sorted(set(r["batch_size"] for r in results))

    if len(batch_sizes_all) >= 2:
        data = np.zeros((len(batch_sizes_all), len(seq_lens_all)))
        for r in results:
            i = batch_sizes_all.index(r["batch_size"])
            j = seq_lens_all.index(r["seq_len"])
            data[i, j] = r["improvement_pct"]

        im = ax5.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-5, vmax=40)
        ax5.set_xticks(range(len(seq_lens_all)))
        ax5.set_xticklabels(seq_lens_all)
        ax5.set_yticks(range(len(batch_sizes_all)))
        ax5.set_yticklabels(batch_sizes_all)
        ax5.set_xlabel("Sequence Length")
        ax5.set_ylabel("Batch Size")
        ax5.set_title("Improvement % Matrix", fontweight="bold")
        plt.colorbar(im, ax=ax5, label="Improvement %")

        boundary = len([s for s in seq_lens_all if s <= 256]) - 0.5
        ax5.axvline(x=boundary, color="black", linewidth=2)
    else:
        ax5.axis("off")
        ax5.set_title("(Run with multiple --batch-sizes for heatmap)", fontsize=10, color="gray")

    plt.suptitle("Fused Short-Seq Kernel Optimization - Summary Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    path = output_dir / "summary_dashboard.png"
    plt.savefig(path)
    plt.close()
    return path


# ============================================================================
# Graph Registry
# ============================================================================

AVAILABLE_GRAPHS = {
    "latency":       "Bar chart comparing optimized vs baseline latency (one per batch size)",
    "speedup":       "Line plot showing speedup factor by sequence length",
    "heatmap":       "2D heatmap of improvement % (batch x seq_len, needs >= 2 batch sizes)",
    "memory":        "Bar chart of memory bandwidth savings (theoretical, no data needed)",
    "architecture":  "Diagram showing before/after kernel flow (no data needed)",
    "dashboard":     "Summary dashboard with key metrics and mini-charts",
}


def generate_graphs(data: dict, graph_names: List[str], output_dir: Path) -> Dict[str, List[Path]]:
    """Generate requested graphs. Returns mapping of graph name to output paths."""

    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("ERROR: matplotlib and numpy are required for graph generation")
        print("Install with: pip install matplotlib numpy seaborn")
        return {}

    setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = data.get("results", [])
    generated = {}

    # Expand "all"
    if "all" in graph_names:
        graph_names = list(AVAILABLE_GRAPHS.keys())

    for name in graph_names:
        if name not in AVAILABLE_GRAPHS:
            print(f"  Unknown graph: '{name}' (use --list-graphs to see options)")
            continue

        print(f"  Generating: {name}...", end=" ", flush=True)

        try:
            if name == "latency":
                paths = plot_latency(results, output_dir)
                if paths:
                    generated[name] = paths
                    print(f"{len(paths)} graph(s)")
                else:
                    print("skipped (no data)")

            elif name == "speedup":
                path = plot_speedup(results, output_dir)
                if path:
                    generated[name] = [path]
                    print(f"saved")
                else:
                    print("skipped (no data)")

            elif name == "heatmap":
                path = plot_heatmap(results, output_dir)
                if path:
                    generated[name] = [path]
                    print(f"saved")
                else:
                    print("skipped")

            elif name == "memory":
                path = plot_memory(output_dir)
                if path:
                    generated[name] = [path]
                    print(f"saved")

            elif name == "architecture":
                path = plot_architecture(output_dir)
                if path:
                    generated[name] = [path]
                    print(f"saved")

            elif name == "dashboard":
                path = plot_dashboard(results, output_dir)
                if path:
                    generated[name] = [path]
                    print(f"saved")
                else:
                    print("skipped (no data)")

        except Exception as e:
            print(f"error: {e}")

    return generated


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Graph Generator for Fused Short-Seq Kernel Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all graphs from benchmark data
  python benchmark_visualization.py --data-file results.json --graphs all

  # Generate specific graphs
  python benchmark_visualization.py --data-file results.json --graphs latency speedup

  # Test with demo data (no GPU/benchmark needed)
  python benchmark_visualization.py --demo --graphs all

  # List available graphs
  python benchmark_visualization.py --list-graphs

Workflow:
  1. Run benchmarks:  python benchmark_fused_short_seq.py --output results.json
  2. Generate graphs: python benchmark_visualization.py --data-file results.json --graphs all
        """
    )

    parser.add_argument("--data-file", type=str,
                        help="Path to JSON results file (from benchmark_fused_short_seq.py)")
    parser.add_argument("--demo", action="store_true",
                        help="Use built-in demo data (no benchmark data needed)")
    parser.add_argument("--graphs", type=str, nargs="+",
                        help="Graphs to generate: " + ", ".join(AVAILABLE_GRAPHS.keys()) + ", all")
    parser.add_argument("--output-dir", type=str, default="benchmark_graphs",
                        help="Output directory for graph images (default: benchmark_graphs)")
    parser.add_argument("--list-graphs", action="store_true",
                        help="List available graph types and exit")

    args = parser.parse_args()

    # List graphs
    if args.list_graphs:
        print("\nAvailable graphs (use with --graphs):\n")
        for name, desc in AVAILABLE_GRAPHS.items():
            print(f"  {name:15s}  {desc}")
        print(f"\n  {'all':15s}  Generate all of the above")
        return

    # Load data
    if args.data_file:
        print(f"Loading data from: {args.data_file}")
        data = load_results(args.data_file)
    elif args.demo:
        print("Using demo data")
        data = get_demo_data()
    else:
        parser.print_help()
        print("\nError: Specify --data-file or --demo")
        sys.exit(1)

    # Must have --graphs
    if not args.graphs:
        parser.print_help()
        print("\nError: Specify --graphs (e.g., --graphs all)")
        sys.exit(1)

    # Generate
    print(f"\nOutput directory: {args.output_dir}")
    generated = generate_graphs(data, args.graphs, Path(args.output_dir))

    # Summary
    if generated:
        print("\n" + "=" * 60)
        print("GENERATED GRAPHS")
        print("=" * 60)
        for name, paths in generated.items():
            for p in paths:
                print(f"  {p}")
        total = sum(len(paths) for paths in generated.values())
        print(f"\n  Total: {total} graph(s) in {args.output_dir}/")
    else:
        print("\nNo graphs were generated.")

    print()


if __name__ == "__main__":
    main()
