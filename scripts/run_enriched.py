#!/usr/bin/env python3
"""Reproduce all enriched topology results from the paper.

Runs the paired ensemble simulation (basal + IL-1beta + TLR) and generates
all figures and statistics tables.

Usage
-----
    python scripts/run_enriched.py [--n-runs 200] [--output-dir results] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for servers

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnm.network import load_edge_list
from rnm.simulation import SimulationConfig, run_paired
from rnm.statistics import compute_node_stats, export_statistics_csv
from rnm.visualization import (
    plot_median_bars,
    plot_grouped_bars,
    plot_boxplots,
    plot_hist_kde_pages,
    plot_histogram_grid,
    plot_median_with_clusters,
)


def main():
    parser = argparse.ArgumentParser(description="Run enriched topology RNM simulation")
    parser.add_argument("--n-runs", type=int, default=200, help="Number of ensemble runs")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--data-file", type=str, default="data/enriched_topology.xlsx",
                        help="Path to enriched topology Excel file")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # ---- Load network ----
    print("=" * 60)
    print("Loading enriched network topology...")
    network = load_edge_list(args.data_file)
    print(network.summary())
    print(f"Node names: {network.node_names}")
    print("=" * 60)

    # ---- Configure simulation ----
    config = SimulationConfig(
        n_runs=args.n_runs,
        tspan=np.arange(0, 101, dtype=float),
        h=10.0,
        gamma=1.0,
        random_seed=args.seed,
    )

    # ---- Run paired simulation ----
    print(f"\nRunning paired simulation ({config.n_runs} runs)...")
    t0 = time.time()
    results = run_paired(network, config)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.1f}s")

    # ---- Compute statistics ----
    print("\nComputing statistics...")
    basal_stats = compute_node_stats(results.xfinal_basal)

    # ---- Export CSV ----
    csv_path = os.path.join(tbl_dir, "paired_statistics.csv")
    export_statistics_csv(results, csv_path)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("BASELINE STEADY STATE (median values)")
    print("=" * 60)
    for i, name in enumerate(network.node_names):
        print(f"  {name:20s}: {basal_stats.median[i]:.4f}")

    # ---- Generate all figures ----
    print(f"\nGenerating figures to {fig_dir}/...")
    import matplotlib.pyplot as plt

    # Fig 1: Median baseline bars (Fig. 6 style)
    plot_median_bars(
        network.node_names, basal_stats.median,
        title="Enriched Network: Median Baseline Steady State",
        output_path=os.path.join(fig_dir, "Bar_Basal_Median.png"),
    )
    plt.close("all")
    print("  Bar_Basal_Median.png")

    # Fig 2: IL-1beta median bars
    if results.xfinal_il1b is not None:
        il1b_stats = compute_node_stats(results.xfinal_il1b)
        plot_median_bars(
            network.node_names, il1b_stats.median,
            title="IL-1\u03b2 Stimulation: Median Steady State",
            output_path=os.path.join(fig_dir, "Bar_IL1b_Median.png"),
        )
        plt.close("all")
        print("  Bar_IL1b_Median.png")

    # Fig 3: TLR median bars
    if results.xfinal_tlr is not None:
        tlr_stats = compute_node_stats(results.xfinal_tlr)
        plot_median_bars(
            network.node_names, tlr_stats.median,
            title="TLR Stimulation: Median Steady State",
            output_path=os.path.join(fig_dir, "Bar_TLR_Median.png"),
        )
        plt.close("all")
        print("  Bar_TLR_Median.png")

    # Fig 4: Grouped bars (Basal vs IL-1beta vs TLR)
    plot_grouped_bars(
        network.node_names,
        basal_stats.median,
        il1b_median=il1b_stats.median if results.xfinal_il1b is not None else None,
        tlr_median=tlr_stats.median if results.xfinal_tlr is not None else None,
        output_path=os.path.join(fig_dir, "Bar_Grouped_Basal_IL1b_TLR.png"),
    )
    plt.close("all")
    print("  Bar_Grouped_Basal_IL1b_TLR.png")

    # Fig 5: Boxplots
    for label, xfinal in [
        ("Basal", results.xfinal_basal),
        ("IL1b", results.xfinal_il1b),
        ("TLR", results.xfinal_tlr),
    ]:
        if xfinal is not None:
            plot_boxplots(
                network.node_names, xfinal,
                title=f"{label}: Distribution of Final States",
                output_path=os.path.join(fig_dir, f"Boxplot_{label}.png"),
            )
            plt.close("all")
            print(f"  Boxplot_{label}.png")

    # Fig 6: Supplementary hist+KDE pages
    plot_hist_kde_pages(
        network.node_names,
        results.xfinal_basal,
        results.xfinal_il1b,
        results.xfinal_tlr,
        output_dir=fig_dir,
    )
    plt.close("all")
    print("  SuppDist_Page*.png")

    # Fig 7: Histogram grids (all nodes, one figure each)
    from rnm.visualization import COL_BASAL, COL_IL1B, COL_TLR
    for label, xfinal, col in [
        ("Basal", results.xfinal_basal, COL_BASAL),
        ("IL1b", results.xfinal_il1b, COL_IL1B),
        ("TLR", results.xfinal_tlr, COL_TLR),
    ]:
        if xfinal is not None:
            plot_histogram_grid(
                network.node_names, xfinal,
                title=f"{label} (All Nodes Histogram)",
                output_path=os.path.join(fig_dir, f"HistGrid_{label}.png"),
                color=col,
            )
            plt.close("all")
            print(f"  HistGrid_{label}.png")

    # Fig 8: Median + clusters
    if results.xfinal_il1b is not None:
        plot_median_with_clusters(
            network.node_names, basal_stats.median,
            il1b_stats.median, results.xfinal_il1b,
            title="IL-1\u03b2 Stimulation (cluster % shown)",
            output_path=os.path.join(fig_dir, "MedianClusters_IL1b.png"),
        )
        plt.close("all")
        print("  MedianClusters_IL1b.png")

    if results.xfinal_tlr is not None:
        plot_median_with_clusters(
            network.node_names, basal_stats.median,
            tlr_stats.median, results.xfinal_tlr,
            title="TLR Stimulation (cluster % shown)",
            output_path=os.path.join(fig_dir, "MedianClusters_TLR.png"),
        )
        plt.close("all")
        print("  MedianClusters_TLR.png")

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
