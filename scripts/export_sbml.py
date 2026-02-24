#!/usr/bin/env python3
"""Generate an SBML model from the enriched topology.

Creates an SBML Level 3 Version 2 file with rate rules encoding the
full Mendoza ODE system for each node.

Usage
-----
    python scripts/export_sbml.py [--output model.xml] [--n-runs 100]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rnm.network import load_edge_list
from rnm.simulation import SimulationConfig, run_basal_only
from rnm.sbml_export import export_sbml


def main():
    parser = argparse.ArgumentParser(description="Export SBML model")
    parser.add_argument("--output", type=str, default="model.xml",
                        help="Output SBML file path")
    parser.add_argument("--data-file", type=str, default="data/enriched_topology.xlsx",
                        help="Path to enriched topology Excel file")
    parser.add_argument("--n-runs", type=int, default=100,
                        help="Number of runs to compute baseline (for initial concentrations)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for baseline computation")
    args = parser.parse_args()

    # Load network
    print("Loading network...")
    network = load_edge_list(args.data_file)
    print(network.summary())

    # Compute baseline median for initial concentrations
    print(f"\nComputing baseline ({args.n_runs} runs) for initial concentrations...")
    config = SimulationConfig(
        n_runs=args.n_runs,
        tspan=np.arange(0, 101, dtype=float),
        h=10.0,
        gamma=1.0,
        random_seed=args.seed,
    )
    results = run_basal_only(network, config)
    baseline_median = np.median(results.xfinal_basal, axis=0)

    # Export SBML
    print(f"\nExporting SBML to {args.output}...")
    success = export_sbml(
        network=network,
        baseline_values=baseline_median,
        filepath=args.output,
        h=config.h,
        gamma=config.gamma,
    )

    if success:
        print(f"\nSBML model successfully exported to {args.output}")
        file_size = Path(args.output).stat().st_size / 1024
        print(f"File size: {file_size:.1f} KB")
    else:
        print("SBML export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
