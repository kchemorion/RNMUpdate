import numpy as np
import pandas as pd
import libsbml

def sanitize_id(name):
    """ Sanitize identifiers to be valid SBML IDs """
    return name.replace(' ', '_').replace('-', '_').replace('/', '_').replace('β', 'beta').replace('γ', 'gamma').replace('α', 'alpha')

def create_matrices(filename):
    """ Create activation and inhibition matrices from an Excel file """
    df = pd.read_excel(filename)
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces from column names
    df.columns = [col.strip() for col in df.columns]  # Ensure all columns are stripped

    num_of_nodes = len(df['Nodes'])
    num_of_stimuli = len(df['Stimuli'])
    stimuli_names = df['Stimuli'].tolist()
    node_names = df['Nodes'].tolist()
    
    mact = np.zeros((num_of_nodes, num_of_stimuli))
    minh = np.zeros((num_of_nodes, num_of_stimuli))
    
    for i in range(num_of_nodes):
        act_in_line = str(df['Activators'][i]).split(',')
        inh_in_line = str(df['Inhibitors'][i]).split(',')
        
        for j in range(num_of_stimuli):
            if stimuli_names[j] in act_in_line:
                mact[i, j] = 1
            if stimuli_names[j] in inh_in_line:
                minh[i, j] = 1
    
    return mact, minh, node_names, num_of_nodes, stimuli_names

def define_units(model):
    """ Define custom units for the model """
    # Define unit for concentration
    unit_def = model.createUnitDefinition()
    unit_def.setId('concentration_unit')
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_MOLE)
    unit.setScale(-3)
    unit.setExponent(1)
    unit.setMultiplier(1)
    
    # Define unit for time
    unit_def = model.createUnitDefinition()
    unit_def.setId('time_unit')
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_SECOND)
    unit.setScale(0)
    unit.setExponent(1)
    unit.setMultiplier(1)
    
    # Define unit for rate constants (per second)
    unit_def = model.createUnitDefinition()
    unit_def.setId('rate_constant_unit')
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_SECOND)
    unit.setScale(0)
    unit.setExponent(-1)
    unit.setMultiplier(1)
    
    # Define unit for dimensionless parameters with a different id
    unit_def = model.createUnitDefinition()
    unit_def.setId('dimensionless_unit')
    unit = unit_def.createUnit()
    unit.setKind(libsbml.UNIT_KIND_DIMENSIONLESS)
    unit.setScale(0)
    unit.setExponent(1)
    unit.setMultiplier(1)

def add_model_annotation(model):
    """ Add annotations to the SBML model """
    model_history = libsbml.ModelHistory()

    creator_1 = libsbml.ModelCreator()
    creator_1.setGivenName("Sofia")
    creator_1.setFamilyName("Tseranidou")
    creator_1.setEmail("sofias@example.com")
    creator_1.setOrganization("Institution")

    creator_2 = libsbml.ModelCreator()
    creator_2.setGivenName("Francis")
    creator_2.setFamilyName("Chemorion")
    creator_2.setEmail("francis@example.com")
    creator_2.setOrganization("Institution")

    model_history.addCreator(creator_1)
    model_history.addCreator(creator_2)
    
    model.setModelHistory(model_history)

    # Add additional metadata
    model.setName("stseranidou2024")
    model.setId("stseranidou2024")

    notes = ("<body xmlns='http://www.w3.org/1999/xhtml'>"
             "<p>ABSTRACT:</p>"
             "<p>Intervertebral disc degeneration (IDD) arises from an intricate imbalance between the anabolic and catabolic processes governing the extracellular matrix (ECM) within the disc. "
             "Biochemical processes are complex, redundant and feedback-looped, and improved integration of knowledge is needed. To addess this, a literature-based regulatory network model (RNM) for nucleus pulposus cells (NPC) is proposed, representing the normal state of the intervertebral disc (IVD), "
             "in which proteins are represented by nodes that interact among each other through activation and/or inhibition edges. This model includes 32 different proteins and 150 edges by incorporating critical biochemical interactions in IVD regulation, "
             "tested in vivo or vitro in humans’ and animals’ NPC, alongside non tissue specific protein-protein interactions. We used the network to calculate the dynamic regulation of each node through a semi-quantitative method. "
             "The basal steady state successfully represented the activity of a normal NPC, and the model was assessed through the published literature, by replicating two independent experimental studies in human normal NPC. "
             "Pro-catabolic or pro-anabolic shifts of the network activated by nodal perturbations could be predicted. Sensitivity analysis underscores the significant influence of transforming growth factor beta (TGF-β) and interleukin-1 receptor antagonist (IL-1Ra) "
             "on the regulation of structural proteins and degrading enzymes within the system. Given the ongoing challenge of elucidating the mechanisms driving ECM degradation in IDD, this unique IVD RNM holds promise as a tool for exploring and predicting IDD progression, "
             "shedding light on IVD phenotypes and guiding experimental research efforts.</p>"
             "</body>")
    
    model.setNotes(notes)

def export_to_sbml_with_params(filename, node_names, mact, minh, xinit, clamped, gamma, h):
    sbml_document = libsbml.SBMLDocument(2, 1)
    model = sbml_document.createModel()
    model.setId('stseranidou2024')
    model.setName('stseranidou2024')
    
    # Define units
    define_units(model)

    compartment = model.createCompartment()
    compartment.setId('default')
    compartment.setConstant(True)
    compartment.setSize(1.0)
    compartment.setUnits('dimensionless_unit')
    compartment.setName('Default Compartment')

    decay_param = model.createParameter()
    decay_param.setId('gamma')
    decay_param.setValue(gamma)
    decay_param.setConstant(True)
    decay_param.setUnits('rate_constant_unit')
    decay_param.setName('Decay Rate')

    steepness_param = model.createParameter()
    steepness_param.setId('h')
    steepness_param.setValue(h)
    steepness_param.setConstant(True)
    steepness_param.setUnits('dimensionless_unit')
    steepness_param.setName('Steepness')

    sanitized_node_ids = [sanitize_id(node) for node in node_names]
    for i, node in enumerate(sanitized_node_ids):
        species = model.createSpecies()
        species.setId(node)
        species.setCompartment('default')
        species.setInitialConcentration(xinit[i])
        species.setBoundaryCondition(bool(clamped[i]))
        species.setConstant(False)
        species.setHasOnlySubstanceUnits(False)
        species.setUnits('concentration_unit')
        species.setName(node_names[i])

    # Add reactions
    for i, target_node in enumerate(sanitized_node_ids):
        for j, activator_node in enumerate(sanitized_node_ids):
            if mact[j, i]:
                reaction = model.createReaction()
                reaction.setId(f"{activator_node}_activates_{target_node}")
                reaction.setReversible(False)
                reaction.setFast(False)
                reactant = reaction.createReactant()
                reactant.setSpecies(activator_node)
                reactant.setStoichiometry(1)
                reactant.setConstant(True)
                product = reaction.createProduct()
                product.setSpecies(target_node)
                product.setStoichiometry(1)
                product.setConstant(True)
                kinetic_law = reaction.createKineticLaw()
                formula = f"1 / (1 + exp(-h * ({activator_node} - 0.5))) - gamma * {target_node}"
                math_ast = libsbml.parseFormula(formula)
                kinetic_law.setMath(math_ast)
                reaction.setKineticLaw(kinetic_law)

        for j, inhibitor_node in enumerate(sanitized_node_ids):
            if minh[j, i]:
                reaction = model.createReaction()
                reaction.setId(f"{inhibitor_node}_inhibits_{target_node}")
                reaction.setReversible(False)
                reaction.setFast(False)
                reactant = reaction.createReactant()
                reactant.setSpecies(inhibitor_node)
                reactant.setStoichiometry(1)
                reactant.setConstant(True)
                product = reaction.createProduct()
                product.setSpecies(target_node)
                product.setStoichiometry(1)
                product.setConstant(True)
                kinetic_law = reaction.createKineticLaw()
                formula = f"1 / (1 + exp(-h * ({inhibitor_node} - 0.5))) - gamma * {target_node}"
                math_ast = libsbml.parseFormula(formula)
                kinetic_law.setMath(math_ast)
                reaction.setKineticLaw(kinetic_law)

    # Add model annotations
    add_model_annotation(model)

    # Validate SBML document
    if sbml_document.checkConsistency():
        for i in range(sbml_document.getNumErrors()):
            print(sbml_document.getError(i).getMessage())
    else:
        print("SBML document is consistent")

    # Write SBML file
    libsbml.writeSBMLToFile(sbml_document, filename)

# Example usage
filename = 'SMENR1.xlsx'
mact, minh, node_names, num_of_nodes, stimuli_names = create_matrices(filename)
xinit = np.random.rand(num_of_nodes)
clamped = np.zeros(num_of_nodes)  # Example: No nodes are clamped
gamma = 1.0  # Example decay rate for all nodes
h = 10.0  # Example steepness for all nodes
export_to_sbml_with_params('model.xml', node_names, mact, minh, xinit, clamped, gamma, h)
