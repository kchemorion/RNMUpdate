"""Ensemble simulation engine with paired design.

Supports two modes:
  - basal_only: N runs from random initial conditions
  - paired: N runs where each basal steady state seeds IL-1beta and TLR perturbations
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import odeint

from rnm.network import Network
from rnm.ode import mendoza_ode, _precompute_masks


@dataclass
class SimulationConfig:
    """Configuration for ensemble simulations.

    Attributes
    ----------
    n_runs : int
        Number of independent simulation runs.
    tspan : np.ndarray
        Time points for ODE integration.
    h : float
        Hill gain parameter (sigmoid steepness).
    gamma : float
        Uniform decay constant for all nodes.
    random_seed : int | None
        Random seed for reproducibility. None = non-deterministic.
    """

    n_runs: int = 200
    tspan: np.ndarray = field(default_factory=lambda: np.arange(0, 101, dtype=float))
    h: float = 10.0
    gamma: float = 1.0
    random_seed: int | None = None


@dataclass
class SimulationResults:
    """Results from an ensemble simulation.

    Attributes
    ----------
    node_names : list[str]
        Ordered list of node names matching array columns.
    config : SimulationConfig
        Configuration used for the simulation.
    xfinal_basal : np.ndarray
        Basal steady states, shape (n_runs, num_nodes).
    xtraj_basal : np.ndarray
        Full basal trajectories, shape (n_runs, n_timepoints, num_nodes).
    xfinal_il1b : np.ndarray | None
        IL-1beta-stimulated steady states (paired mode only).
    xtraj_il1b : np.ndarray | None
        IL-1beta trajectories (paired mode only).
    xfinal_tlr : np.ndarray | None
        TLR-stimulated steady states (paired mode only).
    xtraj_tlr : np.ndarray | None
        TLR trajectories (paired mode only).
    """

    node_names: list[str]
    config: SimulationConfig
    xfinal_basal: np.ndarray
    xtraj_basal: np.ndarray
    xfinal_il1b: np.ndarray | None = None
    xtraj_il1b: np.ndarray | None = None
    xfinal_tlr: np.ndarray | None = None
    xtraj_tlr: np.ndarray | None = None

    @property
    def num_nodes(self) -> int:
        return self.xfinal_basal.shape[1]

    @property
    def n_runs(self) -> int:
        return self.xfinal_basal.shape[0]

    @property
    def is_paired(self) -> bool:
        return self.xfinal_il1b is not None or self.xfinal_tlr is not None


def _solve_ode(
    x0: np.ndarray,
    tspan: np.ndarray,
    num_nodes: int,
    gamma: np.ndarray,
    h: float,
    mact: np.ndarray,
    minh: np.ndarray,
    clamped: np.ndarray,
    clamped_values: np.ndarray,
    cache: dict | None = None,
) -> np.ndarray:
    """Integrate the Mendoza ODE system from initial condition x0."""
    xout = odeint(
        mendoza_ode,
        x0,
        tspan,
        args=(num_nodes, gamma, h, mact, minh, clamped, clamped_values, cache),
        full_output=False,
    )
    return xout


def run_basal_only(
    network: Network,
    config: SimulationConfig,
) -> SimulationResults:
    """Run basal-only ensemble simulation.

    Each of n_runs starts from a random initial condition in [0, 1]^N
    and integrates to steady state with no perturbation.

    Parameters
    ----------
    network : Network
        The regulatory network.
    config : SimulationConfig
        Simulation parameters.

    Returns
    -------
    SimulationResults
        Results with basal data only (il1b/tlr fields are None).
    """
    N = network.num_nodes
    n_t = len(config.tspan)
    rng = np.random.default_rng(config.random_seed)
    gamma = np.full(N, config.gamma, dtype=np.float64)
    clamped = np.zeros(N, dtype=np.float64)
    clamped_values = np.zeros(N, dtype=np.float64)
    cache = _precompute_masks(network.mact, network.minh)

    xfinal_basal = np.zeros((config.n_runs, N), dtype=np.float64)
    xtraj_basal = np.zeros((config.n_runs, n_t, N), dtype=np.float64)

    for r in range(config.n_runs):
        x0 = rng.random(N)
        xout = _solve_ode(
            x0, config.tspan, N, gamma, config.h,
            network.mact, network.minh, clamped, clamped_values, cache,
        )
        xtraj_basal[r] = xout
        xfinal_basal[r] = xout[-1]

        if (r + 1) % 50 == 0 or r == 0:
            print(f"  Basal run {r + 1}/{config.n_runs}")

    return SimulationResults(
        node_names=network.node_names,
        config=config,
        xfinal_basal=xfinal_basal,
        xtraj_basal=xtraj_basal,
    )


def run_paired(
    network: Network,
    config: SimulationConfig,
    il1b_node: str = "IL-1\u03b2",
    tlr_node: str = "TLR",
) -> SimulationResults:
    """Run paired ensemble simulation: basal -> IL-1beta + TLR perturbations.

    For each run:
      1. Solve basal from random IC to steady state.
      2. Use basal SS as IC, clamp IL-1beta=1, solve to new SS.
      3. Use basal SS as IC, clamp TLR=1, solve to new SS.

    Parameters
    ----------
    network : Network
        The regulatory network.
    config : SimulationConfig
        Simulation parameters.
    il1b_node : str
        Name of the IL-1beta node to clamp.
    tlr_node : str
        Name of the TLR node to clamp.

    Returns
    -------
    SimulationResults
        Results with basal, IL-1beta, and TLR data.
    """
    N = network.num_nodes
    n_t = len(config.tspan)
    rng = np.random.default_rng(config.random_seed)
    gamma = np.full(N, config.gamma, dtype=np.float64)
    cache = _precompute_masks(network.mact, network.minh)

    # Find stimulus node indices
    idx_il1b = network.node_index(il1b_node)
    idx_tlr = network.node_index(tlr_node)

    # Preallocate
    xfinal_basal = np.zeros((config.n_runs, N), dtype=np.float64)
    xtraj_basal = np.zeros((config.n_runs, n_t, N), dtype=np.float64)
    xfinal_il1b = np.zeros((config.n_runs, N), dtype=np.float64)
    xtraj_il1b = np.zeros((config.n_runs, n_t, N), dtype=np.float64)
    xfinal_tlr = np.zeros((config.n_runs, N), dtype=np.float64)
    xtraj_tlr = np.zeros((config.n_runs, n_t, N), dtype=np.float64)

    for r in range(config.n_runs):
        # (A) Basal: random IC, no clamping
        clamped_off = np.zeros(N, dtype=np.float64)
        clamped_vals_off = np.zeros(N, dtype=np.float64)

        x0 = rng.random(N)
        xout_b = _solve_ode(
            x0, config.tspan, N, gamma, config.h,
            network.mact, network.minh, clamped_off, clamped_vals_off, cache,
        )
        xtraj_basal[r] = xout_b
        xb_final = xout_b[-1].copy()
        xfinal_basal[r] = xb_final

        # (B) IL-1beta: IC = basal final, clamp IL-1beta = 1
        clamped_il1b = np.zeros(N, dtype=np.float64)
        clamped_il1b[idx_il1b] = 1.0
        clamped_vals_il1b = np.zeros(N, dtype=np.float64)
        clamped_vals_il1b[idx_il1b] = 1.0

        x0_il1b = xb_final.copy()
        x0_il1b[idx_il1b] = 1.0
        xout_il1b = _solve_ode(
            x0_il1b, config.tspan, N, gamma, config.h,
            network.mact, network.minh, clamped_il1b, clamped_vals_il1b, cache,
        )
        xtraj_il1b[r] = xout_il1b
        xfinal_il1b[r] = xout_il1b[-1]

        # (C) TLR: IC = basal final, clamp TLR = 1
        clamped_tlr = np.zeros(N, dtype=np.float64)
        clamped_tlr[idx_tlr] = 1.0
        clamped_vals_tlr = np.zeros(N, dtype=np.float64)
        clamped_vals_tlr[idx_tlr] = 1.0

        x0_tlr = xb_final.copy()
        x0_tlr[idx_tlr] = 1.0
        xout_tlr = _solve_ode(
            x0_tlr, config.tspan, N, gamma, config.h,
            network.mact, network.minh, clamped_tlr, clamped_vals_tlr, cache,
        )
        xtraj_tlr[r] = xout_tlr
        xfinal_tlr[r] = xout_tlr[-1]

        if (r + 1) % 50 == 0 or r == 0:
            print(f"  Paired run {r + 1}/{config.n_runs}")

    return SimulationResults(
        node_names=network.node_names,
        config=config,
        xfinal_basal=xfinal_basal,
        xtraj_basal=xtraj_basal,
        xfinal_il1b=xfinal_il1b,
        xtraj_il1b=xtraj_il1b,
        xfinal_tlr=xfinal_tlr,
        xtraj_tlr=xtraj_tlr,
    )
