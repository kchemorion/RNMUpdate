"""SBML export with correct Mendoza ODE rate rules.

Generates an SBML Level 3 Version 2 model where each species has a RateRule
encoding the full Mendoza ODE (Eq. 3-4), including the aggregated omega
function over all activators and inhibitors.

This replaces the legacy per-reaction approach (sbmlgenerator.py) which
incorrectly decomposed the node-level ODE into individual reaction kinetics.
"""

from __future__ import annotations

import numpy as np

try:
    import libsbml
except ImportError:
    libsbml = None

from rnm.network import Network


def _sanitize_id(name: str) -> str:
    """Convert protein names to valid SBML SId identifiers."""
    s = name
    replacements = {
        " ": "_", "-": "_", "/": "_", "(": "", ")": "",
        "\u03b2": "beta", "\u03b3": "gamma", "\u03b1": "alpha",
        "+": "plus",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    # SId must start with a letter or underscore
    if s and s[0].isdigit():
        s = "_" + s
    return s


def _build_omega_formula(
    node_sid: str,
    activator_sids: list[str],
    inhibitor_sids: list[str],
) -> str:
    """Build the MathML-compatible infix formula for omega_n.

    Implements Eq. 4 from the paper with three cases.
    All alpha and beta weights are 1 (uniform weights).
    """
    has_act = len(activator_sids) > 0
    has_inh = len(inhibitor_sids) > 0

    if has_act:
        n_act = len(activator_sids)
        sum_alpha_x = " + ".join(activator_sids)
        act_term = f"(({n_act} + 1) / {n_act}) * (({sum_alpha_x}) / (1 + ({sum_alpha_x})))"

    if has_inh:
        n_inh = len(inhibitor_sids)
        sum_beta_x = " + ".join(inhibitor_sids)
        inh_term = f"(({n_inh} + 1) / {n_inh}) * (({sum_beta_x}) / (1 + ({sum_beta_x})))"

    if has_act and not has_inh:
        return act_term
    elif has_inh and not has_act:
        return f"(1 - {inh_term})"
    elif has_act and has_inh:
        return f"({act_term}) * (1 - {inh_term})"
    else:
        return "0"


def _build_dxdt_formula(
    node_sid: str,
    omega_formula: str,
    h: float = 10.0,
    gamma: float = 1.0,
) -> str:
    """Build the full dx/dt infix formula for a node's rate rule.

    dx/dt = sigmoid(omega; h) - gamma * x
    sigmoid = (-exp(0.5*h) + exp(-h*(omega - 0.5))) / ((1 - exp(0.5*h)) * (1 + exp(-h*(omega - 0.5))))
    """
    w = omega_formula
    return (
        f"((-exp(0.5 * h) + exp(-1 * h * (({w}) - 0.5))) / "
        f"((1 - exp(0.5 * h)) * (1 + exp(-1 * h * (({w}) - 0.5))))) - "
        f"gamma * {node_sid}"
    )


def export_sbml(
    network: Network,
    baseline_values: np.ndarray,
    filepath: str,
    h: float = 10.0,
    gamma: float = 1.0,
) -> bool:
    """Export the regulatory network as an SBML Level 3 Version 2 model.

    Each species gets a RateRule encoding the complete Mendoza ODE, with
    omega aggregating all activators and inhibitors for that node.

    Parameters
    ----------
    network : Network
        The regulatory network.
    baseline_values : np.ndarray
        Initial concentrations (median baseline), shape (num_nodes,).
    filepath : str
        Output SBML file path.
    h : float
        Hill gain parameter.
    gamma : float
        Decay constant.

    Returns
    -------
    bool
        True if export succeeded without errors.
    """
    if libsbml is None:
        raise ImportError(
            "python-libsbml is required for SBML export. "
            "Install with: pip install python-libsbml"
        )

    # Create SBML document (Level 3 Version 2)
    document = libsbml.SBMLDocument(3, 2)
    model = document.createModel()
    model.setId("IVD_NP_RNM_enriched")
    model.setName("IVD NP Cell Signaling Regulatory Network Model (Enriched)")

    # --- Time units ---
    time_ud = model.createUnitDefinition()
    time_ud.setId("time_unit")
    u = time_ud.createUnit()
    u.setKind(libsbml.UNIT_KIND_SECOND)
    u.setExponent(1)
    u.setScale(0)
    u.setMultiplier(1)
    model.setTimeUnits("time_unit")

    # --- Dimensionless units ---
    dimless_ud = model.createUnitDefinition()
    dimless_ud.setId("arbitrary_units")
    u = dimless_ud.createUnit()
    u.setKind(libsbml.UNIT_KIND_DIMENSIONLESS)
    u.setExponent(1)
    u.setScale(0)
    u.setMultiplier(1)

    # --- Compartment ---
    compartment = model.createCompartment()
    compartment.setId("cell")
    compartment.setName("NP Cell")
    compartment.setConstant(True)
    compartment.setSize(1.0)
    compartment.setSpatialDimensions(3)

    # --- Global parameters ---
    p_h = model.createParameter()
    p_h.setId("h")
    p_h.setName("Hill gain (sigmoid steepness)")
    p_h.setValue(h)
    p_h.setConstant(True)

    p_gamma = model.createParameter()
    p_gamma.setId("gamma")
    p_gamma.setName("Decay constant")
    p_gamma.setValue(gamma)
    p_gamma.setConstant(True)

    # --- Species (one per node) ---
    sanitized_ids = [_sanitize_id(name) for name in network.node_names]

    # Ensure unique IDs
    seen = {}
    for idx, sid in enumerate(sanitized_ids):
        if sid in seen:
            sanitized_ids[idx] = f"{sid}_{idx}"
        seen[sanitized_ids[idx]] = True

    for i, (name, sid) in enumerate(zip(network.node_names, sanitized_ids)):
        sp = model.createSpecies()
        sp.setId(sid)
        sp.setName(name)
        sp.setCompartment("cell")
        sp.setInitialConcentration(float(baseline_values[i]))
        sp.setHasOnlySubstanceUnits(False)
        sp.setBoundaryCondition(False)
        sp.setConstant(False)

    # --- Rate rules (one per species) ---
    for i in range(network.num_nodes):
        node_sid = sanitized_ids[i]

        # Get activator/inhibitor SIds
        act_indices = np.where(network.mact[i, :] > 0)[0]
        inh_indices = np.where(network.minh[i, :] > 0)[0]
        act_sids = [sanitized_ids[j] for j in act_indices]
        inh_sids = [sanitized_ids[j] for j in inh_indices]

        # Build omega and full dxdt formula
        omega_formula = _build_omega_formula(node_sid, act_sids, inh_sids)
        dxdt_formula = _build_dxdt_formula(node_sid, omega_formula, h, gamma)

        rule = model.createRateRule()
        rule.setVariable(node_sid)
        math_ast = libsbml.parseL3Formula(dxdt_formula)
        if math_ast is None:
            print(f"WARNING: Failed to parse formula for {node_sid}: {dxdt_formula}")
            continue
        rule.setMath(math_ast)

    # --- Model annotations ---
    notes = (
        '<body xmlns="http://www.w3.org/1999/xhtml">'
        "<h1>IVD NP Cell Signaling Regulatory Network Model</h1>"
        "<p>A Network-Based Approach to Understanding Key Signaling "
        "Pathways in Intervertebral Disc Biology.</p>"
        f"<p>Network: {network.num_nodes} proteins, "
        f"{int(network.mact.sum())} activation edges, "
        f"{int(network.minh.sum())} inhibition edges.</p>"
        "<p>Mathematical framework: Semi-quantitative Mendoza ODE system "
        "(Mendoza &amp; Xenarios, 2006). Each species rate rule encodes "
        "dx/dt = sigmoid(omega; h=10) - gamma*x, where omega aggregates "
        "all upstream activator and inhibitor inputs.</p>"
        "<p>Parameters: h=10 (sigmoid steepness), gamma=1 (decay), "
        "alpha=beta=1 (uniform interaction weights).</p>"
        "</body>"
    )
    model.setNotes(notes)

    # --- Model history ---
    history = libsbml.ModelHistory()

    creators = [
        ("Sofia", "Tseranidou", "sofiatseranidou@upf.edu", "Universitat Pompeu Fabra"),
        ("Zerihun G.", "Workineh", "zerihungetahun.workineh@upf.edu", "Universitat Pompeu Fabra"),
        ("Maria", "Segarra-Queralt", "maria.segarra@upf.edu", "Universitat Pompeu Fabra"),
        ("Francis K.", "Chemorion", "francis.chemorion@upf.edu", "Universitat Pompeu Fabra"),
    ]
    for given, family, email, org in creators:
        c = libsbml.ModelCreator()
        c.setGivenName(given)
        c.setFamilyName(family)
        c.setEmail(email)
        c.setOrganization(org)
        history.addCreator(c)

    date = libsbml.Date(2025, 1, 1, 0, 0, 0, 0, 0, 0)
    history.setCreatedDate(date)
    history.setModifiedDate(date)
    model.setModelHistory(history)

    # --- Validate ---
    errors = document.checkConsistency()
    if errors > 0:
        n_real_errors = 0
        for i in range(document.getNumErrors()):
            err = document.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                print(f"SBML Error: {err.getMessage()}")
                n_real_errors += 1
        if n_real_errors > 0:
            print(f"SBML validation: {n_real_errors} error(s) found")
    else:
        print("SBML validation: document is consistent")

    # --- Write ---
    success = libsbml.writeSBMLToFile(document, filepath)
    if success:
        print(f"SBML model written to {filepath}")
    else:
        print(f"Failed to write SBML to {filepath}")
    return bool(success)
