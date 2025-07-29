import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import pprint as pprint
import pandas as pd
import numpy as np
from tbsim.utils.plots import plot_household_structure, plot_household_network_analysis

# Simple default parameters
DEFAULT_SPARS = dict(
    unit='day',
    dt=7,
    start=sc.date('1975-01-01'),
    stop=sc.date('2030-12-31'),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    beta=ss.rate_prob(0.0025),
    init_prev=ss.bernoulli(p=0.25),
    unit='day',
    dt=7,      
    start=sc.date('1975-02-01'),
    stop=sc.date('2030-12-31'),
)

# Simple age distribution
age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def create_sample_households(n_agents=500):
    """Create sample household structure for testing."""
    households = []
    current_uid = 0
    while current_uid < n_agents:
        household_size = min(np.random.randint(3, 8), n_agents - current_uid)
        if household_size < 2:
            break
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households

def build_sim(scenario=None, spars=None, show_household_plot=False, household_plot_type='basic'):
    """
    Build and return a complete Starsim-based simulation instance for TB modeling,
    incorporating optional interventions and user-defined parameters.

    Args:
        scenario (dict, optional): A dictionary defining scenario-specific components,
            such as intervention parameters and TB simulation settings. Expected keys:
                - 'tbpars' (dict): TB-specific simulation parameters.
                - 'tptintervention' (dict, optional): Parameters for TPT intervention.
                - 'bcgintervention' (dict, optional): Parameters for BCG intervention.
        spars (dict, optional): General simulation parameters (e.g., timestep, duration).
            These override values in the DEFAULT_SPARS global dictionary.

    Returns:
        ss.Sim: A fully initialized simulation object containing:
            - A population (`People`) with TB-related extra states.
            - A TB disease module initialized with merged parameters.
            - A list of social and household network layers.
            - Optional interventions (TPT, BCG or Beta) as defined by the scenario.
            - Demographic processes like births and deaths.
            - Core simulation parameters merged from defaults and user inputs.

    Notes:
        - If no parameters are provided, it will use the default values of the participating
          simulation components.
    
    Example:
        sim = build_sim(scenario=my_scenario, spars={'n_steps': 200})
        sim.run()
    """
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
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    
    # Create household structure for HouseholdNetGeneric
    households = create_sample_households(500)
    
    # Show household plot if requested
    if show_household_plot:
        # Initialize people object for plotting
        temp_sim = ss.Sim(
            people=pop,
            networks=[],
            diseases=[],
            pars=spars,
        )
        temp_sim.init()
        
        print(f"\nGenerating household plot (type: {household_plot_type})...")
        if household_plot_type == 'basic':
            plot_household_structure(
                households=households,
                people=pop,
                title="HouseholdNetGeneric Structure",
                show_household_ids=True,
                show_agent_ids=False,
                max_households_to_show=30,
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        elif household_plot_type == 'analysis':
            plot_household_network_analysis(
                households=households,
                people=pop,
                figsize=(15, 10),
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        elif household_plot_type == 'both':
            # Show both plots
            plot_household_structure(
                households=households,
                people=pop,
                title="HouseholdNetGeneric Structure",
                show_household_ids=True,
                show_agent_ids=False,
                max_households_to_show=30,
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
            plot_household_network_analysis(
                households=households,
                people=pop,
                figsize=(15, 10),
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        print("Household plot generation completed!")
    
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

def get_scenarios():
    """ HELP
    Define a set of simulation scenarios for evaluating TB interventions.

    Returns:
        dict: A dictionary where each key is the name of a scenario and the value is 
        a dictionary of simulation parameters. Each scenario may include:
            - 'name' (str): A human-readable scenario name.
            - 'tbpars' (dict, optional): Parameters controlling the simulation timeframe.
            - 'bcgintervention' (dict, optional): BCG vaccine intervention settings.
            - 'tptintervention' (dict, optional): Tuberculosis Preventive Therapy settings.
            - 'betabyyear' : (dict, optional): For changing the value of beta during the same simulation period.
    
    Scenarios included:
        - 'Baseline': No intervention, default simulation window.
        - 'BCG': BCG vaccination with 90% coverage.
        - 'TPT': TPT with full eligibility, conditional on HIV status.
    """
    
    return {
        'Baseline': {
            'name': 'No interventions',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        },
        'TPT with Household Network': {
            'name': 'TPT intervention with optimized household network',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
            'tptintervention': dict(
                p_tpt=0.8,
                age_range=[0, 100],
                hiv_status_threshold=False,
                tpt_treatment_duration=ss.peryear(0.25),  # 3 months
                tpt_protection_duration=ss.peryear(2.0),  # 2 years
                start=sc.date('1980-01-01'),
                stop=sc.date('2020-12-31'),
            ),
        },
        # 'Baseline and BetaByYear': {
        #     'name': 'No interventions',
        #     'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        #     'betabyyear':dict(years=[1990, 2000], x_beta=[0.5, 1.4])
        # },
        'Single BCG': {
            'name': 'Single BCG intervention',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
            'bcgintervention': dict(
                coverage=0.8,
                start=sc.date('1980-01-01'),
                stop=sc.date('2020-12-31'),
                age_range=[1, 5],
            ),
        },
        
        # 'Multiple BCG': {
        #     'name': 'Multiple BCG interventions',
        #     'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        #     'bcgintervention': [
        #         dict(
        #             coverage=0.9,
        #             start=sc.date('1980-01-01'),
        #             stop=sc.date('2020-12-31'),
        #             age_range=[0, 2],           # For children
        #         ),
        #         dict(
        #             coverage=0.3,
        #             start=sc.date('1985-01-01'),
        #             stop=sc.date('2015-12-31'),
        #             age_range=[25, 40],         # For adults
        #         ),
        #     ],
        # },
    }

def run_scenarios(plot=True, show_household_plot=False, household_plot_type='basic'):
    """Run all scenarios and optionally plot results."""
    import tbsim.utils.plots as pl
    
    scenarios = get_scenarios()
    results = {}
    
    for name, scenario in scenarios.items():
        print(f"\nRunning: {name}")
        sim = build_sim(scenario=scenario, show_household_plot=show_household_plot, household_plot_type=household_plot_type)
        sim.run()
        results[name] = sim.results.flatten()
    
    if plot:
        pl.plot_combined(results, 
                        heightfold=2, outdir='results/interventions')
                        
                        # filter=mtb.FILTERS.important_metrics)
        plt.show()
    
    return results


def test_household_plots_only():
    """Test only the household plotting functionality without running full simulations."""
    print("Testing HouseholdNetGeneric Plotting Functionality")
    print("=" * 50)
    
    # Create a simple simulation just for testing household plots
    sim = build_sim(show_household_plot=True, household_plot_type='both')
    
    print("\nHousehold plot testing completed!")
    print("Check the 'results/household_plots' directory for saved figures.")


if __name__ == '__main__':
    # Run all scenarios with household plot option
    # Options for household_plot_type: 'basic', 'analysis', 'both', or None
    # Set show_household_plot=False to disable household plotting
    results = run_scenarios(plot=True, show_household_plot=False, household_plot_type='analysis')
