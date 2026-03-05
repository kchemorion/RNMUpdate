#!/usr/bin/env python3
"""Run basal-only simulation on the initial (66-node) topology.

Usage
-----
    python scripts/run_initial.py [--n-runs 200] [--output-dir results_initial] [--seed 42]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnm.network import load_adjacency_list
from rnm.simulation import SimulationConfig, run_basal_only
from rnm.statistics import compute_node_stats
from rnm.visualization import plot_median_bars, plot_boxplots, plot_histogram_grid


def main():
    parser = argparse.ArgumentParser(description="Run initial topology RNM simulation (basal only)")
    parser.add_argument("--n-runs", type=int, default=200, help="Number of ensemble runs")
    parser.add_argument("--output-dir", type=str, default="results_initial", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--data-file", type=str, default="data/initial_topology.xlsx",
                        help="Path to initial topology Excel file")
    args = parser.parse_args()

    fig_dir = os.path.join(args.output_dir, "figures")
    tbl_dir = os.path.join(args.output_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)

    # ---- Load network ----
    print("=" * 60)
    print("Loading initial network topology...")
    network = load_adjacency_list(args.data_file)
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

    # ---- Run basal-only simulation ----
    print(f"\nRunning basal-only simulation ({config.n_runs} runs)...")
    t0 = time.time()
    results = run_basal_only(network, config)
    elapsed = time.time() - t0
    print(f"Simulation completed in {elapsed:.1f}s")

    # ---- Compute statistics ----
    print("\nComputing statistics...")
    basal_stats = compute_node_stats(results.xfinal_basal)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("BASELINE STEADY STATE (median values)")
    print("=" * 60)
    for i, name in enumerate(network.node_names):
        print(f"  {name:20s}: {basal_stats.median[i]:.4f}")

    # ---- Generate figures ----
    import matplotlib.pyplot as plt

    plot_median_bars(
        network.node_names, basal_stats.median,
        title="Initial Network: Median Baseline Steady State",
        output_path=os.path.join(fig_dir, "Bar_Basal_Median.png"),
    )
    plt.close("all")
    print("  Bar_Basal_Median.png")

    plot_boxplots(
        network.node_names, results.xfinal_basal,
        title="Initial Network: Distribution of Basal Final States",
        output_path=os.path.join(fig_dir, "Boxplot_Basal.png"),
    )
    plt.close("all")
    print("  Boxplot_Basal.png")

    from rnm.visualization import COL_BASAL
    plot_histogram_grid(
        network.node_names, results.xfinal_basal,
        title="Initial Network: Basal (All Nodes Histogram)",
        output_path=os.path.join(fig_dir, "HistGrid_Basal.png"),
        color=COL_BASAL,
    )
    plt.close("all")
    print("  HistGrid_Basal.png")

    print(f"\nAll outputs saved to {args.output_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
