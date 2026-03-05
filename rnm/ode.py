"""Mendoza ODE system for regulatory network dynamics.

Implements the semi-quantitative dynamical system from Mendoza & Xenarios (2006),
as adapted for IVD NP cell signaling. Each node's rate of change is governed by
a sigmoidal activation function modulated by upstream activators and inhibitors,
with linear decay.

This module provides both a vectorized (fast) implementation for production use
and the original per-node loop for reference/validation.

Reference
---------
Mendoza, L. & Xenarios, I. (2006). A method for the generation of standardized
qualitative dynamical systems of regulatory networks. Theor Biol Med Model, 3, 1-18.
"""

from __future__ import annotations

import numpy as np


def _precompute_masks(mact: np.ndarray, minh: np.ndarray):
    """Precompute boolean masks and weight sums for vectorized omega.

    Returns a dict of arrays that can be passed to mendoza_ode_vectorized
    to avoid recomputation at every ODE evaluation.
    """
    has_act = np.any(mact > 0, axis=1)  # (N,) bool
    has_inh = np.any(minh > 0, axis=1)  # (N,) bool
    sum_alpha = np.sum(mact, axis=1)    # (N,)
    sum_beta = np.sum(minh, axis=1)     # (N,)

    # Precompute (1 + sum) / sum, safe for zero sums
    with np.errstate(divide="ignore", invalid="ignore"):
        act_coeff = np.where(sum_alpha > 0, (1.0 + sum_alpha) / sum_alpha, 0.0)
        inh_coeff = np.where(sum_beta > 0, (1.0 + sum_beta) / sum_beta, 0.0)

    return {
        "has_act": has_act,
        "has_inh": has_inh,
        "act_coeff": act_coeff,
        "inh_coeff": inh_coeff,
    }


def mendoza_ode(
    x: np.ndarray,
    t: float,
    num_nodes: int,
    gamma: np.ndarray,
    h: float,
    mact: np.ndarray,
    minh: np.ndarray,
    clamped: np.ndarray,
    clamped_values: np.ndarray,
    _cache: dict | None = None,
) -> np.ndarray:
    """Vectorized Mendoza ODE system for regulatory network dynamics.

    Computes dx/dt for all nodes simultaneously using matrix operations:
        dx_n/dt = sigmoid(omega_n; h) - gamma_n * x_n

    Clamped nodes have dx/dt = 0 (held at their clamped value).

    Parameters
    ----------
    x : np.ndarray
        Current state vector (num_nodes,).
    t : float
        Current time (unused -- autonomous system).
    num_nodes : int
        Number of network nodes.
    gamma : np.ndarray
        Decay constants for each node (num_nodes,).
    h : float
        Hill gain parameter (steepness of sigmoid).
    mact : np.ndarray
        Activation matrix (num_nodes, num_nodes).
    minh : np.ndarray
        Inhibition matrix (num_nodes, num_nodes).
    clamped : np.ndarray
        Binary vector (num_nodes,). 1 = node is clamped.
    clamped_values : np.ndarray
        Values for clamped nodes (num_nodes,).
    _cache : dict | None
        Precomputed masks from _precompute_masks(). If None, computed on the fly.

    Returns
    -------
    np.ndarray
        Time derivatives dx/dt for each node (num_nodes,).
    """
    # --- Vectorized omega computation (Eq. 4) ---
    # Matrix-vector products: sum_alpha_x[i] = sum_j mact[i,j] * x[j]
    sum_alpha_x = mact @ x  # (N,)
    sum_beta_x = minh @ x   # (N,)

    if _cache is not None:
        has_act = _cache["has_act"]
        has_inh = _cache["has_inh"]
        act_coeff = _cache["act_coeff"]
        inh_coeff = _cache["inh_coeff"]
    else:
        has_act = np.any(mact > 0, axis=1)
        has_inh = np.any(minh > 0, axis=1)
        sum_alpha = np.sum(mact, axis=1)
        sum_beta = np.sum(minh, axis=1)
        act_coeff = np.where(sum_alpha > 0, (1.0 + sum_alpha) / sum_alpha, 0.0)
        inh_coeff = np.where(sum_beta > 0, (1.0 + sum_beta) / sum_beta, 0.0)

    # Activator term: ((1+sum_a)/sum_a) * (sum_a_x / (1 + sum_a_x))
    act_term = act_coeff * (sum_alpha_x / (1.0 + sum_alpha_x))

    # Inhibitor term: ((1+sum_b)/sum_b) * (sum_b_x / (1 + sum_b_x))
    inh_term = inh_coeff * (sum_beta_x / (1.0 + sum_beta_x))

    # Combine per Eq. 4 (three cases handled via masking)
    omega = np.zeros(num_nodes, dtype=np.float64)

    # Case (i): only activators
    mask_act_only = has_act & ~has_inh
    omega[mask_act_only] = act_term[mask_act_only]

    # Case (ii): only inhibitors
    mask_inh_only = has_inh & ~has_act
    omega[mask_inh_only] = 1.0 - inh_term[mask_inh_only]

    # Case (iii): both
    mask_both = has_act & has_inh
    omega[mask_both] = act_term[mask_both] * (1.0 - inh_term[mask_both])

    # Case (iv): no regulators -> omega stays 0

    # --- Vectorized sigmoid (Eq. 3) ---
    exp_half_h = np.exp(0.5 * h)
    exp_neg = np.exp(-h * (omega - 0.5))
    sigmoid = (-exp_half_h + exp_neg) / ((1.0 - exp_half_h) * (1.0 + exp_neg))

    # --- dx/dt = sigmoid(omega) - gamma * x ---
    dxdt = sigmoid - gamma * x

    # --- Clamp: zero derivative for clamped nodes ---
    dxdt[clamped > 0] = 0.0

    return dxdt
