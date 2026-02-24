"""Mendoza ODE system for regulatory network dynamics.

Implements the semi-quantitative dynamical system from Mendoza & Xenarios (2006),
as adapted for IVD NP cell signaling. Each node's rate of change is governed by
a sigmoidal activation function modulated by upstream activators and inhibitors,
with linear decay.

Reference
---------
Mendoza, L. & Xenarios, I. (2006). A method for the generation of standardized
qualitative dynamical systems of regulatory networks. Theor Biol Med Model, 3, 1-18.
"""

from __future__ import annotations

import numpy as np


def _compute_omega(
    x: np.ndarray,
    ract: np.ndarray,
    rinh: np.ndarray,
) -> float:
    """Compute the aggregated regulatory input omega for a single node.

    Implements Eq. 4 from the paper with three cases:
      (i)   Only activators
      (ii)  Only inhibitors
      (iii) Both activators and inhibitors

    Parameters
    ----------
    x : np.ndarray
        Current state vector (num_nodes,).
    ract : np.ndarray
        Row of the activation matrix for this node (num_nodes,).
    rinh : np.ndarray
        Row of the inhibition matrix for this node (num_nodes,).

    Returns
    -------
    float
        Aggregated input omega in [0, 1].
    """
    has_act = np.any(ract > 0)
    has_inh = np.any(rinh > 0)

    if has_act and not has_inh:
        # Case (i): only activators
        sum_alpha = np.sum(ract)
        sum_alpha_x = np.dot(ract, x)
        return ((1.0 + sum_alpha) / sum_alpha) * (sum_alpha_x / (1.0 + sum_alpha_x))

    elif has_inh and not has_act:
        # Case (ii): only inhibitors
        sum_beta = np.sum(rinh)
        sum_beta_x = np.dot(rinh, x)
        return 1.0 - ((1.0 + sum_beta) / sum_beta) * (sum_beta_x / (1.0 + sum_beta_x))

    elif has_act and has_inh:
        # Case (iii): both activators and inhibitors
        sum_alpha = np.sum(ract)
        sum_alpha_x = np.dot(ract, x)
        act_term = ((1.0 + sum_alpha) / sum_alpha) * (sum_alpha_x / (1.0 + sum_alpha_x))

        sum_beta = np.sum(rinh)
        sum_beta_x = np.dot(rinh, x)
        inh_term = ((1.0 + sum_beta) / sum_beta) * (sum_beta_x / (1.0 + sum_beta_x))

        return act_term * (1.0 - inh_term)

    else:
        # No regulators
        return 0.0


def _sigmoid(omega: float, h: float) -> float:
    """Sigmoidal activation function (Eq. 3, production term).

    Parameters
    ----------
    omega : float
        Aggregated regulatory input in [0, 1].
    h : float
        Gain (steepness) parameter.

    Returns
    -------
    float
        Activation value in [0, 1].
    """
    exp_half_h = np.exp(0.5 * h)
    exp_neg = np.exp(-h * (omega - 0.5))
    numerator = -exp_half_h + exp_neg
    denominator = (1.0 - exp_half_h) * (1.0 + exp_neg)
    return numerator / denominator


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
) -> np.ndarray:
    """Mendoza ODE system for regulatory network dynamics.

    Computes dx/dt for all nodes according to:
        dx_n/dt = sigmoid(omega_n; h) - gamma_n * x_n

    Clamped nodes have dx/dt = 0 (held at their clamped value).

    Parameters
    ----------
    x : np.ndarray
        Current state vector (num_nodes,).
    t : float
        Current time (unused — autonomous system).
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

    Returns
    -------
    np.ndarray
        Time derivatives dx/dt for each node (num_nodes,).
    """
    dxdt = np.zeros(num_nodes, dtype=np.float64)

    for i in range(num_nodes):
        if clamped[i]:
            dxdt[i] = 0.0
            continue

        omega = _compute_omega(x, mact[i, :], minh[i, :])
        dxdt[i] = _sigmoid(omega, h) - gamma[i] * x[i]

    return dxdt
