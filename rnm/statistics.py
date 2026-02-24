"""Statistical analysis for ensemble simulation results.

Implements paired t-tests, Wilcoxon signed-rank tests, Benjamini-Hochberg
FDR correction, and distribution diagnostics (skewness, KDE mode counting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks


@dataclass
class NodeStatistics:
    """Descriptive statistics for a set of steady-state values.

    Each array has shape (num_nodes,).
    """

    mean: np.ndarray
    median: np.ndarray
    std: np.ndarray
    skewness: np.ndarray
    n_modes: np.ndarray


@dataclass
class ComparisonStatistics:
    """Paired comparison statistics between two conditions.

    Each array has shape (num_nodes,).
    """

    p_ttest: np.ndarray
    q_ttest: np.ndarray  # BH-FDR corrected
    t_stat: np.ndarray
    p_wilcoxon: np.ndarray
    q_wilcoxon: np.ndarray  # BH-FDR corrected
    z_stat: np.ndarray


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction to p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Raw p-values (may contain NaN).

    Returns
    -------
    np.ndarray
        FDR-corrected q-values.
    """
    p = p_values.copy().ravel()
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    n = len(pv)

    if n == 0:
        return q.reshape(p_values.shape)

    sorted_idx = np.argsort(pv)
    sorted_p = pv[sorted_idx]
    ranks = np.arange(1, n + 1)

    q_sorted = sorted_p * n / ranks

    # Enforce monotonicity (step-down)
    for k in range(n - 2, -1, -1):
        q_sorted[k] = min(q_sorted[k], q_sorted[k + 1])

    q_sorted = np.minimum(q_sorted, 1.0)

    qv = np.empty(n)
    qv[sorted_idx] = q_sorted
    q[valid] = qv

    return q.reshape(p_values.shape)


def count_modes_kde(
    x: np.ndarray,
    grid_n: int = 256,
    min_prom_frac: float = 0.02,
) -> int:
    """Count the number of modes in a distribution using KDE peak detection.

    Parameters
    ----------
    x : np.ndarray
        1D array of values.
    grid_n : int
        Number of grid points for KDE evaluation.
    min_prom_frac : float
        Minimum peak prominence as a fraction of max KDE density.

    Returns
    -------
    int
        Number of detected modes (>=1).
    """
    x = x[~np.isnan(x)]
    if len(x) < 10 or np.all(np.abs(x - x[0]) < 1e-12):
        return 1

    xi = np.linspace(0, 1, grid_n)
    try:
        kde = stats.gaussian_kde(x, bw_method="silverman")
        f = kde(xi)
        prom = min_prom_frac * np.max(f)
        if prom <= 0 or not np.isfinite(prom):
            return 1
        peaks, _ = find_peaks(f, prominence=prom)
        n_modes = len(peaks)
        return max(n_modes, 1)
    except Exception:
        return 1


def compute_node_stats(xfinal: np.ndarray) -> NodeStatistics:
    """Compute descriptive statistics for steady-state values.

    Parameters
    ----------
    xfinal : np.ndarray
        Steady-state values, shape (n_runs, num_nodes).

    Returns
    -------
    NodeStatistics
        Per-node descriptive statistics.
    """
    num_nodes = xfinal.shape[1]
    n_modes = np.array([
        count_modes_kde(xfinal[:, i]) for i in range(num_nodes)
    ])

    return NodeStatistics(
        mean=np.mean(xfinal, axis=0),
        median=np.median(xfinal, axis=0),
        std=np.std(xfinal, axis=0, ddof=0),
        skewness=stats.skew(xfinal, axis=0, bias=False),
        n_modes=n_modes,
    )


def compare_paired(
    xfinal_a: np.ndarray,
    xfinal_b: np.ndarray,
) -> ComparisonStatistics:
    """Run paired statistical tests between two conditions.

    For each node, performs:
      - Paired t-test
      - Wilcoxon signed-rank test

    Both corrected with Benjamini-Hochberg FDR.

    Parameters
    ----------
    xfinal_a : np.ndarray
        Condition A steady states, shape (n_runs, num_nodes).
    xfinal_b : np.ndarray
        Condition B steady states, shape (n_runs, num_nodes).

    Returns
    -------
    ComparisonStatistics
        Per-node comparison statistics.
    """
    num_nodes = xfinal_a.shape[1]
    p_t = np.full(num_nodes, np.nan)
    t_s = np.full(num_nodes, np.nan)
    p_w = np.full(num_nodes, np.nan)
    z_s = np.full(num_nodes, np.nan)

    for i in range(num_nodes):
        a = xfinal_a[:, i]
        b = xfinal_b[:, i]

        # Paired t-test
        try:
            result = stats.ttest_rel(a, b)
            t_s[i] = result.statistic
            p_t[i] = result.pvalue
        except Exception:
            pass

        # Wilcoxon signed-rank
        try:
            diff = a - b
            if np.all(np.abs(diff) < 1e-15):
                p_w[i] = 1.0
                z_s[i] = 0.0
            else:
                result = stats.wilcoxon(a, b, alternative="two-sided")
                p_w[i] = result.pvalue
                # Approximate z-statistic
                n = len(diff[diff != 0])
                if n > 0:
                    z_s[i] = stats.norm.isf(result.pvalue / 2)
                    if np.median(diff) < 0:
                        z_s[i] = -z_s[i]
        except Exception:
            pass

    return ComparisonStatistics(
        p_ttest=p_t,
        q_ttest=benjamini_hochberg(p_t),
        t_stat=t_s,
        p_wilcoxon=p_w,
        q_wilcoxon=benjamini_hochberg(p_w),
        z_stat=z_s,
    )


def export_statistics_csv(
    results,  # SimulationResults (avoid circular import)
    filepath: str,
) -> pd.DataFrame:
    """Export all statistics to a CSV file.

    Parameters
    ----------
    results : SimulationResults
        Ensemble simulation results.
    filepath : str
        Output CSV path.

    Returns
    -------
    pd.DataFrame
        The exported statistics table.
    """
    basal_stats = compute_node_stats(results.xfinal_basal)

    data = {
        "Node": results.node_names,
        "Basal_Mean": basal_stats.mean,
        "Basal_Median": basal_stats.median,
        "Basal_SD": basal_stats.std,
        "Skew_Basal": basal_stats.skewness,
        "Modes_Basal": basal_stats.n_modes,
    }

    if results.xfinal_il1b is not None:
        il1b_stats = compute_node_stats(results.xfinal_il1b)
        data.update({
            "IL1b_Mean": il1b_stats.mean,
            "IL1b_Median": il1b_stats.median,
            "IL1b_SD": il1b_stats.std,
            "Skew_IL1b": il1b_stats.skewness,
            "Modes_IL1b": il1b_stats.n_modes,
        })

        comp_a = compare_paired(results.xfinal_basal, results.xfinal_il1b)
        data.update({
            "p_t_Basal_vs_IL1b": comp_a.p_ttest,
            "q_t_Basal_vs_IL1b": comp_a.q_ttest,
            "tstat_Basal_vs_IL1b": comp_a.t_stat,
            "p_w_Basal_vs_IL1b": comp_a.p_wilcoxon,
            "q_w_Basal_vs_IL1b": comp_a.q_wilcoxon,
            "z_Basal_vs_IL1b": comp_a.z_stat,
        })

    if results.xfinal_tlr is not None:
        tlr_stats = compute_node_stats(results.xfinal_tlr)
        data.update({
            "TLR_Mean": tlr_stats.mean,
            "TLR_Median": tlr_stats.median,
            "TLR_SD": tlr_stats.std,
            "Skew_TLR": tlr_stats.skewness,
            "Modes_TLR": tlr_stats.n_modes,
        })

        comp_b = compare_paired(results.xfinal_basal, results.xfinal_tlr)
        data.update({
            "p_t_Basal_vs_TLR": comp_b.p_ttest,
            "q_t_Basal_vs_TLR": comp_b.q_ttest,
            "tstat_Basal_vs_TLR": comp_b.t_stat,
            "p_w_Basal_vs_TLR": comp_b.p_wilcoxon,
            "q_w_Basal_vs_TLR": comp_b.q_wilcoxon,
            "z_Basal_vs_TLR": comp_b.z_stat,
        })

    if results.xfinal_il1b is not None and results.xfinal_tlr is not None:
        comp_c = compare_paired(results.xfinal_il1b, results.xfinal_tlr)
        data.update({
            "p_t_IL1b_vs_TLR": comp_c.p_ttest,
            "q_t_IL1b_vs_TLR": comp_c.q_ttest,
            "tstat_IL1b_vs_TLR": comp_c.t_stat,
            "p_w_IL1b_vs_TLR": comp_c.p_wilcoxon,
            "q_w_IL1b_vs_TLR": comp_c.q_wilcoxon,
            "z_IL1b_vs_TLR": comp_c.z_stat,
        })

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Statistics saved to {filepath}")
    return df
