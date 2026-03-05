# Regulatory Network Model for IVD NP Cell Signaling

**A Network-Based Approach to Understanding Key Signaling Pathways in Intervertebral Disc Biology**

Tseranidou S.<sup>1</sup>, Workineh Z.G.<sup>1</sup>, Segarra-Queralt M.<sup>1</sup>, Chemorion F.K.<sup>1</sup>, Bermudez-Lekerika P.<sup>2,3</sup>, Kanelis E.<sup>4</sup>, Crump K.<sup>2,3</sup>, Gantenbein B.<sup>2,5</sup>, Alexopoulos L.G.<sup>4,6</sup>, Le Maitre C.L.<sup>7</sup>, Pinero J.<sup>8</sup>, Noailly J.<sup>1</sup>

<sup>1</sup>Universitat Pompeu Fabra, ES; <sup>2</sup>University of Bern, CH; <sup>3</sup>GCB, University of Bern, CH; <sup>4</sup>Protavio Ltd, GR; <sup>5</sup>Inselspital, University of Bern, CH; <sup>6</sup>NTUA, GR; <sup>7</sup>University of Sheffield, UK; <sup>8</sup>Hospital del Mar Research Institute, ES

---

## Overview

This repository contains the computational implementation of a **literature-curated regulatory network model (RNM)** for human **nucleus pulposus (NP) cells** in the intervertebral disc (IVD). The model captures the key signaling pathways involved in IVD homeostasis and degeneration.

### Network at a glance

| Property | Value |
|----------|-------|
| Proteins (nodes) | 82 |
| Directed interactions (edges) | 199 |
| Activation edges | 147 |
| Inhibition edges | 52 |
| NP-specific corpus | 41.4% |

The model integrates data from PubMed (45 + 90 articles), STRING, KEGG, and R&D Systems pathway databases.

---

## Mathematical Framework

### Mendoza ODE System

The static knowledge-based network is transformed into a **semi-quantitative dynamical system** using the method of [Mendoza & Xenarios (2006)](https://doi.org/10.1186/1742-4682-3-13). Each node's activation evolves according to an ordinary differential equation (ODE) that integrates upstream activating and inhibiting signals through a sigmoidal transfer function.

#### State equation (Eq. 3)

For each protein node $x_n$, the rate of change is:

$$\frac{dx_n}{dt} = \frac{-e^{0.5 h_n} + e^{-h_n(\omega_n - 0.5)}}{(1 - e^{0.5 h_n})(1 + e^{-h_n(\omega_n - 0.5)})} - \gamma_n x_n$$

where:
- $h_n$ is the **gain** (steepness) parameter controlling how fast the sigmoid transitions between 0 and 1. High gain values push the system toward Boolean (ON/OFF) behavior.
- $\omega_n$ is the **aggregated regulatory input** combining all upstream activators and inhibitors.
- $\gamma_n$ is the **linear decay** constant.
- The first term is a **sigmoid activation function** that maps $\omega_n \in [0,1]$ to an activation level in $[0,1]$.

#### Aggregated regulatory input $\omega_n$ (Eq. 4)

The net regulatory input $\omega_n$ is computed differently depending on which types of regulators are present:

**Case (i) — Only activators** ($\{x_{nk}^a\}$ is non-empty, no inhibitors):

$$\omega_n = \left(\frac{1 + \sum_k \alpha_{nk}}{\sum_k \alpha_{nk}}\right) \cdot \frac{\sum_k \alpha_{nk} x_{nk}^a}{1 + \sum_k \alpha_{nk} x_{nk}^a}$$

**Case (ii) — Only inhibitors** ($\{x_{nl}^i\}$ is non-empty, no activators):

$$\omega_n = 1 - \left(\frac{1 + \sum_l \beta_{nl}}{\sum_l \beta_{nl}}\right) \cdot \frac{\sum_l \beta_{nl} x_{nl}^i}{1 + \sum_l \beta_{nl} x_{nl}^i}$$

**Case (iii) — Both activators and inhibitors:**

$$\omega_n = \left(\frac{1 + \sum_k \alpha_{nk}}{\sum_k \alpha_{nk}}\right) \cdot \frac{\sum_k \alpha_{nk} x_{nk}^a}{1 + \sum_k \alpha_{nk} x_{nk}^a} \cdot \left(1 - \left(\frac{1 + \sum_l \beta_{nl}}{\sum_l \beta_{nl}}\right) \cdot \frac{\sum_l \beta_{nl} x_{nl}^i}{1 + \sum_l \beta_{nl} x_{nl}^i}\right)$$

where:
- $\alpha_{nk}$ are the activation weights (influence of activator $k$ on node $n$)
- $\beta_{nl}$ are the inhibition weights (influence of inhibitor $l$ on node $n$)
- $x_{nk}^a$ is the current state of activator $k$
- $x_{nl}^i$ is the current state of inhibitor $l$

The $\omega_n$ formulation ensures values remain in $[0,1]$, preserving the Boolean asymptotic behavior under high gain.

#### Default parameter values

| Parameter | Symbol | Value | Rationale |
|-----------|--------|-------|-----------|
| Gain | $h$ | 10 | Intermediate sigmoid steepness |
| Activation weights | $\alpha$ | 1 | Uniform (no sensitivity data) |
| Inhibition weights | $\beta$ | 1 | Uniform (no sensitivity data) |
| Decay constant | $\gamma$ | 1 | Consistent degradation rate |

### Simulation protocol

1. **Baseline (basal)**: 200 independent runs from random initial conditions $x_0 \sim \text{Uniform}(0,1)^N$, integrated over $t \in [0, 100]$. The steady state (SS) is taken as $x(t=100)$. The median across runs defines the baseline expression.

2. **Perturbation (paired design)**: For each basal run, the final state seeds two perturbation runs:
   - **IL-1$\beta$ stimulation**: Clamp IL-1$\beta$ = 1 (constant input), integrate from basal SS.
   - **TLR stimulation**: Clamp TLR = 1 (Pam2CSK4 analog), integrate from basal SS.

3. **Statistical tests** (per node, per comparison):
   - Paired t-test + Wilcoxon signed-rank test
   - Benjamini-Hochberg FDR correction (q-values)
   - Distribution diagnostics: skewness, KDE-based mode counting

### IkBa interpretation

In the network topology, IkBa is modeled as an **inhibitory node upstream of p65 NF-kB**. The IKK complex inhibits IkBa, which in turn inhibits p65. This double-negative path reproduces net NF-kB activation upon IL-1$\beta$ stimulation. Importantly, **measured increases in IkBa phosphorylation** (experimental) correspond to **decreased IkBa node activation** (model), because phosphorylation marks IkBa for degradation.

---

## Repository Structure

```
RNM/
├── README.md                           # This file
├── pyproject.toml                      # Python package configuration
├── requirements.txt                    # Dependencies
├── .gitignore
│
├── data/
│   ├── enriched_topology.xlsx          # Enriched network (82 nodes, 199 edges)
│   └── initial_topology.xlsx           # Initial network (66 nodes, 86 edges)
│
├── rnm/                                # Core Python package
│   ├── __init__.py                     # Package exports
│   ├── network.py                      # Load Excel topology -> matrices
│   ├── ode.py                          # Mendoza ODE system implementation
│   ├── simulation.py                   # Ensemble simulation engine
│   ├── statistics.py                   # Statistical analysis (t-test, FDR, etc.)
│   ├── visualization.py               # Publication-quality plots
│   └── sbml_export.py                  # SBML Level 3 model export
│
├── scripts/
│   ├── run_enriched.py                 # Reproduce enriched topology results
│   ├── run_initial.py                  # Run initial topology (basal only)
│   └── export_sbml.py                  # Generate SBML model file
│
├── model.xml                           # Pre-generated SBML model
│
├── results/                            # Generated outputs
│   ├── figures/                        # All plots (generated by run_enriched.py)
│   └── tables/                         # Statistics CSV files
│
└── legacy/                             # Previous version (32-node model)
    ├── diffsolvemendoza.py
    ├── sbmlgenerator.py
    └── ...
```

---

## Installation

### Requirements

- Python >= 3.10
- NumPy, SciPy, Pandas, Matplotlib, openpyxl, python-libsbml

### Setup

```bash
git clone https://github.com/kchemorion/RNMUpdate.git
cd RNMUpdate
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

---

## Usage

### Reproduce all paper results

```bash
python scripts/run_enriched.py --n-runs 200 --output-dir results
```

This runs the full paired simulation and generates:
- **Figures**: Median bars, grouped bars, boxplots, hist+KDE pages, histogram grids, median+cluster plots
- **Tables**: `paired_statistics.csv` with per-node mean, median, SD, skewness, modes, p-values, and q-values

### Generate SBML model

```bash
python scripts/export_sbml.py --output model.xml
```

Produces an SBML Level 3 Version 2 file with **rate rules** encoding the full Mendoza ODE for each node. The SBML model can be loaded in COPASI, libRoadRunner, or any SBML-compliant simulator.

### Use as a Python library

```python
from rnm import load_edge_list, SimulationConfig, run_paired

# Load network
network = load_edge_list("data/enriched_topology.xlsx")
print(network.summary())

# Run simulation
config = SimulationConfig(n_runs=200, h=10.0, gamma=1.0)
results = run_paired(network, config)

# Analyze
import numpy as np
basal_median = np.median(results.xfinal_basal, axis=0)
for name, val in zip(network.node_names, basal_median):
    print(f"{name}: {val:.3f}")
```

---

## SBML Model

The SBML export (`model.xml`) encodes the complete Mendoza ODE system as **rate rules** rather than individual reactions. Each species has a single rate rule:

```
d[species]/dt = sigmoid(omega(activators, inhibitors); h) - gamma * [species]
```

where `omega` aggregates all upstream activators and inhibitors according to Eq. 4. This is the mathematically correct representation, replacing the legacy per-reaction kinetic law approach.

---

## Data Format

### Enriched topology (`data/enriched_topology.xlsx`)

The network is defined as an **adjacency list** with one row per node:

| Nodes | Activators | Inhibitors | Stimuli |
|-------|-----------|------------|---------|
| ACAN | Sox9, SMAD2/3 | TLR, β-catenin | ACAN |
| COL1A | RUNX2 | NOTHING | COL1A |

Each row defines a node and its comma-separated upstream activators and inhibitors. "NOTHING" indicates no regulators of that type.

### Initial topology (`data/initial_topology.xlsx`)

Same adjacency-list format with 66 nodes and 86 edges (the pre-enrichment network).

---

## Acknowledgments

Financial support was received from the European Commission (Marie Sklodowska-Curie Innovative Training Network Disc4all, grant agreement 955735) and from the European Research Council (ERC Consolidator Grant O-Health, grant agreement 101044828).

---

## Authors

| Author | ORCID |
|--------|-------|
| Sofia Tseranidou | [0000-0003-1459-5650](https://orcid.org/0000-0003-1459-5650) |
| Zerihun G. Workineh | [0000-0002-6191-7854](https://orcid.org/0000-0002-6191-7854) |
| Maria Segarra-Queralt | [0000-0001-9332-0764](https://orcid.org/0000-0001-9332-0764) |
| Francis K. Chemorion | [0000-0002-2099-0035](https://orcid.org/0000-0002-2099-0035) |
| Paola Bermudez-Lekerika | [0000-0002-6858-8213](https://orcid.org/0000-0002-6858-8213) |
| Exarchos Kanelis | [0000-0002-2059-1480](https://orcid.org/0000-0002-2059-1480) |
| Katherine Crump | [0000-0001-5328-667X](https://orcid.org/0000-0001-5328-667X) |
| Benjamin Gantenbein | [0000-0002-9005-0655](https://orcid.org/0000-0002-9005-0655) |
| Leonidas G. Alexopoulos | [0000-0003-0425-166X](https://orcid.org/0000-0003-0425-166X) |
| Christine L. Le Maitre | [0000-0003-4489-7107](https://orcid.org/0000-0003-4489-7107) |
| Janet Pinero | [0000-0003-1244-7654](https://orcid.org/0000-0003-1244-7654) |
| Jerome Noailly | [0000-0003-3446-7621](https://orcid.org/0000-0003-3446-7621) |

---

## Citation

If you use this code or model, please cite:

> Tseranidou, S., Workineh, Z.G., Segarra-Queralt, M., Chemorion, F.K., et al. (2025). A Network-Based Approach to Understanding Key Signaling Pathways in Intervertebral Disc Biology. *[Journal]*.

Previous version (32-node model):
> Tseranidou, S. et al. (2025). Nucleus pulposus cell network modelling in the intervertebral disc. *npj Systems Biology and Applications*, 11, 13.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
