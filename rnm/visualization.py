"""Publication-quality visualization for regulatory network simulations.

Generates bar plots, boxplots, histogram+KDE distributions, histogram grids,
and median+cluster composition plots matching the paper figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COL_BASAL = (0.00, 0.50, 0.00)  # green
COL_IL1B = (0.00, 0.00, 1.00)   # blue
COL_TLR = (1.00, 0.00, 0.00)    # red
COL_ANABOLIC = (0.20, 0.66, 0.33)
COL_CATABOLIC = (0.80, 0.30, 0.30)
COL_NEUTRAL = (0.55, 0.55, 0.55)

# Known anabolic and catabolic nodes for color-coding
_ANABOLIC_NODES = {
    "ACAN", "COL2A", "Sox9", "TGF-\u03b2", "GDF5", "IGF1", "BMP",
    "SMAD1/5/8", "SMAD2/3", "SMAD2", "SMAD3", "akt1",
    "TGFBRI", "BMPR", "ALK6", "IGF1R",
    "IL-10", "IL-4", "IL-1Ra", "TIMP1/2", "TIMP3",
    "IL-10R", "IL-10R1", "IL-4R",
    "SIRT1", "SIRT2", "TSP-1", "CCN2", "PGRN",
}

_CATABOLIC_NODES = {
    "IL-1\u03b2", "TNF", "IL-6", "IL-17", "IL-18", "IL-12",
    "IFN-\u03b3", "IL-8", "CSF2", "CGRP",
    "MMP1", "MMP2", "MMP3", "MMP9", "MMP13",
    "ADAMTS4/5", "VEGF", "COL1A", "COL10A1",
    "p65", "p38", "JNK",
    "\u03b2-catenin", "wnt3a", "wnt5a",
    "TLR", "TNFR1", "IL-1R1", "IL-6R", "IFN-\u03b3R",
    "IL-17RA", "IL-18RA", "CXCR1", "IL-12R",
    "TRAF6", "Ikk", "IkBa",
    "iNos", "ROS", "FOXO", "RUNX2",
    "CCL3", "CCL3/4", "CCL4",
}


def _node_color(name: str) -> tuple:
    """Return color for a node based on its anabolic/catabolic role."""
    if name in _ANABOLIC_NODES:
        return COL_ANABOLIC
    elif name in _CATABOLIC_NODES:
        return COL_CATABOLIC
    return COL_NEUTRAL


def _apply_node_ticks(ax, node_names, rotation=45, fontsize=8):
    """Configure x-axis with node name labels."""
    ax.set_xticks(range(len(node_names)))
    ax.set_xticklabels(node_names, rotation=rotation, fontsize=fontsize, ha="right")


def _ensure_dir(path: str | Path):
    """Create parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Plot 1: Median bar chart (Fig. 6 style)
# ---------------------------------------------------------------------------


def plot_median_bars(
    node_names: list[str],
    median_values: np.ndarray,
    title: str = "Median Baseline Steady State",
    output_path: str | None = None,
    figsize: tuple = (18, 6),
) -> plt.Figure:
    """Horizontal bar chart of median steady-state values, color-coded by role."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = [_node_color(n) for n in node_names]

    bars = ax.bar(range(len(node_names)), median_values, 0.8, color=colors)
    _apply_node_ticks(ax, node_names)
    ax.set_ylabel("Activation Level", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.5, len(node_names) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COL_ANABOLIC, label="Pro-anabolic"),
        Patch(facecolor=COL_CATABOLIC, label="Pro-catabolic"),
        Patch(facecolor=COL_NEUTRAL, label="Dual/uncertain"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Grouped bars (Basal vs IL-1beta vs TLR)
# ---------------------------------------------------------------------------


def plot_grouped_bars(
    node_names: list[str],
    basal_median: np.ndarray,
    il1b_median: np.ndarray | None = None,
    tlr_median: np.ndarray | None = None,
    title: str = "Basal vs IL-1\u03b2 vs TLR (median final states)",
    output_path: str | None = None,
    figsize: tuple = (20, 6),
) -> plt.Figure:
    """Grouped bar chart comparing conditions."""
    fig, ax = plt.subplots(figsize=figsize)
    n = len(node_names)
    x = np.arange(n)

    conditions = [("Basal", basal_median, COL_BASAL)]
    if il1b_median is not None:
        conditions.append(("IL-1\u03b2", il1b_median, COL_IL1B))
    if tlr_median is not None:
        conditions.append(("TLR", tlr_median, COL_TLR))

    n_cond = len(conditions)
    width = 0.8 / n_cond
    offsets = np.linspace(-(n_cond - 1) * width / 2, (n_cond - 1) * width / 2, n_cond)

    for (label, values, color), offset in zip(conditions, offsets):
        ax.bar(x + offset, values, width, color=color, label=label)

    _apply_node_ticks(ax, node_names)
    ax.set_ylabel("Activation Level", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Boxplots
# ---------------------------------------------------------------------------


def plot_boxplots(
    node_names: list[str],
    xfinal: np.ndarray,
    title: str = "Distribution of Final States",
    output_path: str | None = None,
    figsize: tuple = (20, 6),
) -> plt.Figure:
    """Boxplot of steady-state distributions across runs."""
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        xfinal,
        widths=0.6,
        patch_artist=True,
        flierprops=dict(marker="o", markerfacecolor="red", markersize=4, alpha=0.5),
        medianprops=dict(color="black", linewidth=2),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor((0.7, 0.85, 1.0))
        patch.set_edgecolor("black")

    _apply_node_ticks(ax, node_names)
    ax.set_ylabel("Activation Level", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot 4: Histogram + KDE pages (supplementary)
# ---------------------------------------------------------------------------


def plot_hist_kde_pages(
    node_names: list[str],
    xfinal_basal: np.ndarray,
    xfinal_il1b: np.ndarray | None = None,
    xfinal_tlr: np.ndarray | None = None,
    output_dir: str = "results/figures",
    panels_per_page: int = 16,
    layout: tuple = (4, 4),
    hist_bin_width: float = 0.05,
) -> list[plt.Figure]:
    """Generate paged histogram + KDE distribution plots for all nodes."""
    _ensure_dir(f"{output_dir}/placeholder")
    num_nodes = len(node_names)
    n_pages = int(np.ceil(num_nodes / panels_per_page))
    xi = np.linspace(0, 1, 256)
    figs = []

    for page in range(n_pages):
        fig, axes = plt.subplots(*layout, figsize=(16, 12))
        axes = axes.ravel()
        start = page * panels_per_page
        end = min(start + panels_per_page, num_nodes)

        for panel_idx, node_idx in enumerate(range(start, end)):
            ax = axes[panel_idx]
            bins = np.arange(0, 1 + hist_bin_width, hist_bin_width)

            # Basal
            ax.hist(xfinal_basal[:, node_idx], bins=bins, density=True,
                    color=COL_BASAL, alpha=0.2, edgecolor="none")
            try:
                kde = stats.gaussian_kde(xfinal_basal[:, node_idx])
                ax.plot(xi, kde(xi), color=COL_BASAL, linewidth=1.5)
            except Exception:
                pass

            # IL-1beta
            if xfinal_il1b is not None:
                ax.hist(xfinal_il1b[:, node_idx], bins=bins, density=True,
                        color=COL_IL1B, alpha=0.2, edgecolor="none")
                try:
                    kde = stats.gaussian_kde(xfinal_il1b[:, node_idx])
                    ax.plot(xi, kde(xi), color=COL_IL1B, linewidth=1.5)
                except Exception:
                    pass

            # TLR
            if xfinal_tlr is not None:
                ax.hist(xfinal_tlr[:, node_idx], bins=bins, density=True,
                        color=COL_TLR, alpha=0.2, edgecolor="none")
                try:
                    kde = stats.gaussian_kde(xfinal_tlr[:, node_idx])
                    ax.plot(xi, kde(xi), color=COL_TLR, linewidth=1.5)
                except Exception:
                    pass

            ax.set_title(node_names[node_idx], fontweight="bold", fontsize=9)
            ax.set_xlim(0, 1)
            ax.grid(alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Hide unused panels
        for panel_idx in range(end - start, len(axes)):
            axes[panel_idx].set_visible(False)

        fig.suptitle(
            f"Supplementary SS Distributions (Page {page + 1}/{n_pages})",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        outpath = f"{output_dir}/SuppDist_Page{page + 1:03d}.png"
        fig.savefig(outpath, dpi=300, bbox_inches="tight")
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Plot 5: Histogram grid (all nodes in one figure)
# ---------------------------------------------------------------------------


def plot_histogram_grid(
    node_names: list[str],
    xfinal: np.ndarray,
    title: str = "All Nodes Histogram",
    output_path: str | None = None,
    grid: tuple = (7, 12),
    hist_bin_width: float = 0.05,
    color: tuple = COL_BASAL,
) -> plt.Figure:
    """Single figure with small histogram panels for all nodes."""
    n_rows, n_cols = grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 14))
    axes_flat = axes.ravel()
    bins = np.arange(0, 1 + hist_bin_width, hist_bin_width)
    num_nodes = len(node_names)

    for k in range(n_rows * n_cols):
        ax = axes_flat[k]
        if k < num_nodes:
            ax.hist(xfinal[:, k], bins=bins, color=color, alpha=0.85, edgecolor="none")
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticklabels([])
            ax.set_title(node_names[k], fontweight="bold", fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        else:
            ax.set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot 6: Median bars + cluster composition (dots + %)
# ---------------------------------------------------------------------------


def plot_median_with_clusters(
    node_names: list[str],
    basal_median: np.ndarray,
    stim_median: np.ndarray,
    xfinal_stim: np.ndarray,
    title: str = "Stimuli (cluster % shown)",
    output_path: str | None = None,
    cluster_edges: np.ndarray | None = None,
    min_pct_to_show: float = 5.0,
    figsize: tuple = (10, 18),
) -> plt.Figure:
    """Horizontal median bars with cluster composition dots and percentages."""
    if cluster_edges is None:
        cluster_edges = np.array([0, 0.125, 0.375, 0.625, 0.875, 1.00001])

    centers = (cluster_edges[:-1] + cluster_edges[1:]) / 2
    num_nodes = len(node_names)
    y = np.arange(num_nodes)

    fig, ax = plt.subplots(figsize=figsize)

    # Grouped horizontal bars
    bar_height = 0.35
    bars_basal = ax.barh(y - bar_height / 2, basal_median, bar_height,
                         color=(0.12, 0.47, 0.96), label="Baseline (median)")
    bars_stim = ax.barh(y + bar_height / 2, stim_median, bar_height,
                        color=(0.98, 0.55, 0.25), label="Stimulated (median)")

    # Format axes
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(node_names, fontsize=8)
    ax.set_xlabel("Expression Level", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.15)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9)

    # Overlay cluster dots + percentage labels
    for i in range(num_nodes):
        xi = xfinal_stim[:, i]
        xi = xi[~np.isnan(xi)]
        if len(xi) == 0:
            continue

        counts, _ = np.histogram(xi, bins=cluster_edges)
        total = counts.sum()
        if total == 0:
            continue
        pct = 100 * counts / total

        for b in range(len(centers)):
            if counts[b] == 0 or pct[b] < min_pct_to_show:
                continue
            ax.scatter(centers[b], y[i] + bar_height / 2, s=18, c="black", zorder=5)
            ax.text(
                centers[b] + 0.02, y[i] + bar_height / 2,
                f"{pct[b]:.0f}%", fontsize=7, va="center",
            )

    fig.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
