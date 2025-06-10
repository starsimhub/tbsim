import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import pprint as pp
import pandas as pd
# Default simulation parameters
DEFAULT_SPARS = dict(
    unit='day',
    dt=7,
    start=sc.date('1965-01-01'),
    stop=sc.date('2035-12-31'),
    rand_seed=123,
    verbose =0,
)
DEFAULT_TBPARS = dict(
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        unit = 'day',
        dt=7,      
        start=sc.date('1975-02-01'),
        stop=sc.date('2030-12-31'),
    )
age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def build_sim(scenario=None, spars=None):
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
            - Optional interventions (TPT or BCG) as defined by the scenario.
            - Demographic processes like births and deaths.
            - Core simulation parameters merged from defaults and user inputs.

    Notes:
        - If no scenario is provided, defaults are used.
        - Intervention objects (`TPTInitiation`, `BCGProtection`) are conditionally added
          based on the scenario dictionary contents.
        - The population size is fixed at 100 agents for simplicity.
        - This function is typically used to instantiate simulations for batch execution,
          comparison, or visualization.
    
    Example:
        sim = build_sim(scenario=my_scenario, spars={'n_steps': 200})
        sim.run()
    """
    scenario = scenario or {}
    # merge and override default parameters
    
    spars = {**DEFAULT_SPARS, **(spars or {})}  # Merge user spars with default
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})} 
    pp.pp(spars)
    pp.pp(tbpars)
    
    # Set up interventions safely
    inv = []
    for key, cls in [('tptintervention', mtb.TPTInitiation), 
                     ('bcgintervention', mtb.BCGProtection)]:
        params = scenario.get(key)
        if params:
            inv.append(cls(pars=params))

    # Core sim components
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    networks = [ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
                mtb.HouseholdNet(),
                ss.MaternalNet()]
    
    demographics = [ss.Births(pars={'birth_rate': 15}),
                    ss.Deaths(pars={'death_rate': 15})]

    # Create and return simulation
    return ss.Sim(
        people=pop,
        networks=networks,
        interventions=inv,
        diseases=tb,
        demographics=demographics,
        pars=spars,
    )


def get_scenarios():
    """
    Define a set of simulation scenarios for evaluating TB interventions.

    Returns:
        dict: A dictionary where each key is the name of a scenario and the value is 
        a dictionary of simulation parameters. Each scenario may include:
            - 'name' (str): A human-readable scenario name.
            - 'tbpars' (dict, optional): Parameters controlling the simulation timeframe.
            - 'bcgintervention' (dict, optional): BCG vaccine intervention settings.
            - 'tptintervention' (dict, optional): Tuberculosis Preventive Therapy settings.
    
    Scenarios included:
        - 'Baseline': No intervention, default simulation window.
        - 'BCG': BCG vaccination with 90% coverage.
        - 'TPT': TPT with full eligibility, conditional on HIV status.
    """
    
    return {
        'Baseline': {
            'name': 'BASELINE',
            'tbpars': dict(start=sc.date('1975-02-07'), 
                stop=sc.date('2030-12-31')),
        },
        'BCG': {
            'name': 'BCG PROTECTION',
            'tbpars': dict(start=sc.date('1975-02-15'), 
                           stop=sc.date('2030-12-31')),
            'bcgintervention': dict(
                coverage=0.90,
            ),
        },
        'TPT': {
            'name': 'TPT INITIATION',
            'tptintervention': dict(
                p_tpt=ss.bernoulli(1.0),
                tpt_duration=2.0,
                max_age=25,
                hiv_status_threshold=True,
                p_3HP=0.8,
                start=sc.date('1970-01-01'),
            ),
        },
    }


def run_scenarios(plot=True):
    """
    Execute all defined TB simulation scenarios and optionally visualize results.

    Args:
        plot (bool, optional): If True (default), generates comparative plots of 
        scenario outcomes using a built-in plotting module.

    Returns:
        None: Results are stored locally within the function and plotted if requested.

    Workflow:
        1. Retrieves all predefined scenarios using get_scenarios().
        2. Runs a simulation for each scenario using build_sim().
        3. Collects and flattens the results for each simulation.
        4. Optionally plots results in a grid layout with custom styling.
    
    Example:
        >>> run_scenarios(True)
        
    
    NOTE:  
    -----
    This line:
        >>> results[name] = sim.results.flatten()
         
    Converts the simulation's time series outputs into a flat dictionary or DataFrame.
    Makes results easier to compare across scenarios (e.g., plotting incidence over time).
    The results dictionary now maps scenario names to their flattened outputs:
    {
        'BCG': <results>,
        'TPT': <results>,
        ...
    }
    """

    import tbsim.utils.plots as pl

    results = {}
    for name, scenario in get_scenarios().items():
        print(f"\nRunning scenario: {name}")
        sim = build_sim(scenario=scenario)
        sim.run()
        
        # 
        results[name] = sim.results.flatten()     

    if plot:
        pl.plot_results(results, n_cols=5, dark=True, cmap='viridis', heightfold=2, outdir='results/interventions',)
        plt.show()


if __name__ == '__main__':
    run_scenarios()
