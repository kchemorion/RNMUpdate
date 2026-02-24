"""Network topology loading and matrix construction.

Supports two Excel formats:
  - Edge-list: rows of (STIMULI, RELATION, RESPONSE) — new enriched topology
  - Adjacency-list: rows of (Node, Activators, Inhibitors, Stimuli) — legacy SMENR1.xlsx
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Network:
    """Regulatory network with activation and inhibition matrices.

    Attributes
    ----------
    node_names : list[str]
        Ordered list of node (protein) names.
    mact : np.ndarray
        Activation matrix of shape (N, N). mact[i, j] = 1 means node j
        activates node i.
    minh : np.ndarray
        Inhibition matrix of shape (N, N). minh[i, j] = 1 means node j
        inhibits node i.
    num_nodes : int
        Number of nodes in the network.
    """

    node_names: list[str]
    mact: np.ndarray
    minh: np.ndarray
    num_nodes: int

    def activators_of(self, node: str) -> list[str]:
        """Return names of all activators of a given node."""
        idx = self.node_names.index(node)
        return [self.node_names[j] for j in range(self.num_nodes) if self.mact[idx, j] > 0]

    def inhibitors_of(self, node: str) -> list[str]:
        """Return names of all inhibitors of a given node."""
        idx = self.node_names.index(node)
        return [self.node_names[j] for j in range(self.num_nodes) if self.minh[idx, j] > 0]

    def node_index(self, name: str) -> int:
        """Return the index of a node by name."""
        try:
            return self.node_names.index(name)
        except ValueError:
            raise ValueError(
                f"Node '{name}' not found. Available nodes: {self.node_names}"
            )

    def summary(self) -> str:
        """Return a summary string of the network."""
        n_act = int(self.mact.sum())
        n_inh = int(self.minh.sum())
        return (
            f"Network: {self.num_nodes} nodes, "
            f"{n_act} activation edges, {n_inh} inhibition edges, "
            f"{n_act + n_inh} total edges"
        )


# ---------------------------------------------------------------------------
# Name harmonization for the enriched topology Excel
# ---------------------------------------------------------------------------

_NAME_MAP: dict[str, str] = {
    "IL-1beta": "IL-1\u03b2",
    "TGF-beta": "TGF-\u03b2",
    "IFN-y": "IFN-\u03b3",
    "IFN-yr": "IFN-\u03b3R",
    "IFN-yR": "IFN-\u03b3R",
    "IFN-\u03b3r": "IFN-\u03b3R",  # case-insensitive variant
    "b-catenin": "\u03b2-catenin",
    "beta-catenin": "\u03b2-catenin",
    "COL2A1": "COL2A",
    "\u0399L-1R1": "IL-1R1",  # Greek iota (Ι) -> Latin I
}


def _harmonize_name(name: str) -> str:
    """Normalize protein names to resolve duplicates/variants."""
    name = name.strip()
    return _NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# Edge-list loader (new enriched topology)
# ---------------------------------------------------------------------------


def load_edge_list(
    filepath: str,
    sheet_name: str = "Topology for NW30",
) -> Network:
    """Load a regulatory network from an edge-list Excel file.

    Each row encodes one directed interaction:
        STIMULI --(activation|inhibition)--> RESPONSE

    Parameters
    ----------
    filepath : str
        Path to the Excel file.
    sheet_name : str
        Name of the sheet containing the topology.

    Returns
    -------
    Network
        The loaded network with activation/inhibition matrices.
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    required = {"STIMULI", "RELATION", "RESPONSE"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Excel must have columns {required}, found {set(df.columns)}")

    # Drop rows with missing essential data
    df = df.dropna(subset=["STIMULI", "RELATION", "RESPONSE"])

    # Harmonize names
    df["STIMULI"] = df["STIMULI"].astype(str).apply(_harmonize_name)
    df["RESPONSE"] = df["RESPONSE"].astype(str).apply(_harmonize_name)
    df["RELATION"] = df["RELATION"].astype(str).str.strip().str.lower()

    # Collect unique nodes (sorted for reproducibility)
    all_nodes = sorted(set(df["STIMULI"].tolist() + df["RESPONSE"].tolist()))
    num_nodes = len(all_nodes)
    node_idx = {name: i for i, name in enumerate(all_nodes)}

    mact = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    minh = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for _, row in df.iterrows():
        src = row["STIMULI"]
        tgt = row["RESPONSE"]
        rel = row["RELATION"]
        j = node_idx[src]  # source (activator/inhibitor)
        i = node_idx[tgt]  # target (the node being regulated)

        if rel == "activation":
            mact[i, j] = 1.0
        elif rel == "inhibition":
            minh[i, j] = 1.0
        else:
            raise ValueError(f"Unknown relation '{rel}' for edge {src} -> {tgt}")

    return Network(
        node_names=all_nodes,
        mact=mact,
        minh=minh,
        num_nodes=num_nodes,
    )


# ---------------------------------------------------------------------------
# Adjacency-list loader (legacy SMENR1.xlsx format)
# ---------------------------------------------------------------------------


def load_adjacency_list(filepath: str) -> Network:
    """Load a regulatory network from the legacy adjacency-list format.

    Each row is a node with comma-separated Activators and Inhibitors.

    Parameters
    ----------
    filepath : str
        Path to the Excel file (e.g., SMENR1.xlsx).

    Returns
    -------
    Network
        The loaded network.
    """
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()

    # The legacy file has "Nodes " with trailing space
    nodes_col = [c for c in df.columns if c.lower().startswith("node")][0]
    node_names = df[nodes_col].astype(str).str.strip().tolist()
    num_nodes = len(node_names)
    node_idx = {name: i for i, name in enumerate(node_names)}

    # Stimuli column maps to same node list for matrix indexing
    stimuli_col = "Stimuli" if "Stimuli" in df.columns else nodes_col
    stimuli_names = df[stimuli_col].astype(str).str.strip().tolist()
    stimuli_idx = {name: i for i, name in enumerate(stimuli_names)}

    mact = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    minh = np.zeros((num_nodes, num_nodes), dtype=np.float64)

    for i in range(num_nodes):
        # Parse comma-separated activator/inhibitor lists
        act_str = str(df["Activators"].iloc[i]).strip()
        inh_str = str(df["Inhibitors"].iloc[i]).strip()

        if act_str and act_str != "nan":
            for a in act_str.split(","):
                a = a.strip()
                if a in stimuli_idx:
                    mact[i, stimuli_idx[a]] = 1.0

        if inh_str and inh_str != "nan":
            for b in inh_str.split(","):
                b = b.strip()
                if b in stimuli_idx:
                    minh[i, stimuli_idx[b]] = 1.0

    return Network(
        node_names=node_names,
        mact=mact,
        minh=minh,
        num_nodes=num_nodes,
    )
