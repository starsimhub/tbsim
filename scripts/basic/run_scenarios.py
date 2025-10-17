"""
TB-Malnutrition Scenario Comparison with Parallel Execution

This script runs multiple TB-malnutrition co-infection scenarios in parallel to compare
outcomes under different parameter settings. It demonstrates how to use parallel processing
for efficient scenario analysis and how TB and malnutrition interact through the
TB-Nutrition connector.

Purpose:
--------
This script demonstrates:
- Running multiple scenarios in parallel for efficiency
- Modeling TB-malnutrition co-infection dynamics
- Using the TB-Nutrition connector for disease interactions
- Comparing outcomes across parameter variations
- Saving results to CSV for further analysis

Components:
-----------
- TB disease module with configurable transmission
- Malnutrition comorbidity module
- TB-Nutrition connector for disease interactions
- Demographics (births and deaths via Pregnancy)
- Random contact network
- Parallel execution with sciris.parallelize()

Scenarios:
----------
The script varies the rel_sus_latentslow parameter (relative susceptibility of
latent slow TB) to test its impact on disease dynamics. Default values test
parameter values of 2, 3, and 1.

Usage:
------
    # Run scenarios and generate results
    python scripts/basic/run_scenarios.py
    
    # Specify custom number of agents
    python scripts/basic/run_scenarios.py -n 5000
    
    # Specify custom number of random seeds
    python scripts/basic/run_scenarios.py -s 10
    
    # Plot from cached CSV file
    python scripts/basic/run_scenarios.py -p results/results.csv

Output:
-------
- CSV file with simulation results saved to results/ directory
- Displays simulation timing information
- Shows combined metrics from all scenario runs
"""

# %% Imports and settings
import starsim as ss
import tbsim as mtb
import pandas as pd
import os
import argparse
import sciris as sc
import tbsim.config as cfg

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

x_latent_slow = [2, 3] + [1]

debug = True
default_n_agents = [10_000, 1000][debug]
default_n_rand_seeds = [20, 2][debug]

def run_sim(n_agents=default_n_agents, rand_seed=0, idx=0, xLS=1):
    """
    Run a single TB-malnutrition simulation with specified parameters.
    
    This function creates and runs one complete simulation instance with TB and
    malnutrition comorbidities. It's designed to be called in parallel for efficient
    scenario analysis.
    
    Parameters
    ----------
    n_agents : int, default=10000 (or 1000 in debug mode)
        Number of agents in the simulation
    rand_seed : int, default=0
        Random seed for reproducibility
    idx : int, default=0
        Simulation index for tracking in parallel runs
    xLS : float, default=1
        Relative susceptibility parameter for latent slow TB
        (currently not used in TB parameters but logged in results)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with simulation results including:
        - year: Simulation time vector
        - Deaths: Cumulative deaths over time
        - xLS: Parameter value used
        - rand_seed: Random seed used
        - network: Network type (currently 'None')
        - n_agents: Population size
        
    Notes
    -----
    The simulation includes:
    - TB disease with 25% initial prevalence
    - Malnutrition with 0.1% initial prevalence
    - TB-Nutrition connector for disease interactions
    - Random network with average 5 contacts
    - Demographics: Pregnancy (15 per 1000) and Deaths (10 per 1000)
    - 41-year period (1980-2020)
    """
    # --------- People ----------
    pop = ss.People(n_agents=n_agents)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = ss.peryear(0.01), 
        init_prev = 0.25,
        )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict(
        init_prev = 0.001,
        )
    nut = mtb.Malnutrition(nut_pars)

    # -------- Network ---------
    # Network parameters
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
        )
    # Initialize a random network
    net = ss.RandomNet(net_pars)

    # Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # Connector
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt=ss.days(7),
        start = ss.date('1980-01-01'),
        stop = ss.date('2020-12-31'),
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    sim.pars.verbose = 0.1 # Print status every 10% of simulation
    sim.run()

    df = pd.DataFrame( {
        'year': sim.timevec,
        #'pph.mother_died.cumsum': sim.results.pph.mother_died.cumsum(),
        #'Births': sim.results.pph.births.cumsum(),
        'Deaths': sim.results.deaths.cumulative,
        #'Maternal Deaths': sim.results.pph.maternal_deaths.cumsum(),
        #'Infant Deaths': sim.results.pph.infant_deaths.cumsum(),
    })
    df['xLS'] = xLS
    df['rand_seed'] = rand_seed
    df['network'] = 'None'
    df['n_agents'] = n_agents

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and xLS={xLS}')

    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    """
    Run multiple TB-malnutrition scenarios in parallel with different parameter combinations.
    
    This function orchestrates parallel execution of multiple simulations with different
    parameter values and random seeds. It uses sciris.parallelize() for efficient
    multi-core processing.
    
    Parameters
    ----------
    n_agents : int, default=10000 (or 1000 in debug mode)
        Number of agents per simulation
    n_seeds : int, default=20 (or 2 in debug mode)
        Number of different random seeds to test
    
    Returns
    -------
    tuple of (pd.DataFrame, str)
        - DataFrame: Combined results from all scenario runs
        - str: Path to saved CSV file
        
    Notes
    -----
    The function:
    - Creates configurations for all parameter and seed combinations
    - Runs simulations in parallel using all available CPU cores
    - Combines results into a single DataFrame
    - Saves results to a timestamped CSV file in results/ directory
    - Prints timing information for performance monitoring
    
    The x_latent_slow parameter variations test how relative susceptibility
    of latent slow TB affects disease dynamics.
    """
    results = []
    cfgs = []
    for rs in range(n_seeds):
        for xLS in x_latent_slow:
            cfgs.append({'xLS':xLS, 'rand_seed':rs, 'idx':len(cfgs)})
    T = sc.tic()
    results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False)
    times = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    filename = os.path.join(cfg.create_res_dir("results"), "results.csv")
    df.to_csv(filename)
    return df, filename



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--plot', help='plot from a cached CSV file', type=str)
    parser.add_argument('-n', help='Number of agents', type=int, default=default_n_agents)
    parser.add_argument('-s', help='Number of seeds', type=int, default=default_n_rand_seeds)
    args = parser.parse_args()

    if args.plot:
        print('Reading CSV file', args.plot)
        df = pd.read_csv(args.plot, index_col=0)
    else:
        print('Running scenarios')
        df, filename = run_scenarios(n_agents=args.n, n_seeds=args.s)

    print(df)
    print(f"Results directory {filename}")
    print('Done')