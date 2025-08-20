import starsim as ss
import pandas as pd
import tbsim as mtb
import numpy as np
from scripts.interventions.constants import DEFAULT_SPARS, DEFAULT_TBPARS, AGE_DATA, START, STOP

def create_sample_households(n_agents=500):
    """Create sample household structure for the simulation."""
    # Create simple household structure
    n_households = n_agents // 4  # Average 4 people per household
    households = []
    
    for i in range(n_households):
        household_size = np.random.randint(1, 8)  # 1-7 people per household
        household_members = list(range(i * household_size, min((i + 1) * household_size, n_agents)))
        if household_members:  # Only add non-empty households
            households.append(household_members)
    
    return households


def build_sim(scenario=None, spars=None):
    scenario = scenario or {}
    
    # Merge parameters
    spars = {**DEFAULT_SPARS, **(spars or {})}
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}
    
    # Create interventions list
    interventions = []
    
    # Add BCG interventions (can be single or multiple)
    bcg_params = scenario.get('bcgintervention')
    if bcg_params:
        if isinstance(bcg_params, dict):
            # Single BCG intervention
            interventions.append(mtb.BCGProtection(pars=bcg_params))
        elif isinstance(bcg_params, list):
            # Multiple BCG interventions
            for i, params in enumerate(bcg_params):
                params['name'] = f'BCG_{i}'  # Give unique name
                interventions.append(mtb.BCGProtection(pars=params))
    
    # Add TPT interventions (can be single or multiple)
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        if isinstance(tpt_params, dict):
            # Single TPT intervention
            interventions.append(mtb.TPTInitiation(pars=tpt_params))
        elif isinstance(tpt_params, list):
            # Multiple TPT interventions
            for i, params in enumerate(tpt_params):
                params['name'] = f'TPT_{i}'  # Give unique name
                interventions.append(mtb.TPTInitiation(pars=params))
    
    # Add Beta interventions (can be single or multiple)
    beta_params = scenario.get('betabyyear')
    if beta_params:
        if isinstance(beta_params, dict):
            # Single Beta intervention
            interventions.append(mtb.BetaByYear(pars=beta_params))
        elif isinstance(beta_params, list):
            # Multiple Beta interventions
            for i, params in enumerate(beta_params):
                params['name'] = f'Beta_{i}'  # Give unique name
                interventions.append(mtb.BetaByYear(pars=params))
    
    # Create simulation components
    pop = ss.People(n_agents=500, age_data=AGE_DATA, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    
    # Create household structure for HouseholdNetGeneric
    households = create_sample_households(500)
    
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        mtb.HouseholdNet(),
        mtb.HouseholdNetGeneric(hhs=households, pars={'add_newborns': True})
    ]
    
    # Create and return simulation
    return ss.Sim(
        people=pop,
        networks=networks,
        interventions=interventions,
        diseases=[tb],
        pars=spars,
    )

