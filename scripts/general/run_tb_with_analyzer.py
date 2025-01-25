import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

TBS = mtb.TBS

def make_tb(sim_pars=None):
    params = dict(
        unit='day',
        dt=30,
        start=sc.date('1940-01-01'),
        stop=sc.date('2016-12-31'),
        rand_seed=123,
    )
    if sim_pars is not None:
        params.update(sim_pars)

    np.random.seed()
    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        unit='day',
        rel_sus_latentslow=0.5,
    ))
    dwell_analyzer = mtb.DwtAnalyzer()

    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=params,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = sim.pars.dt / 365

    return sim


def calculate_expected_distributions(start, stop):
    duration_days = (stop - start).days
    min_value = 150  # Minimum dwell time for all states
    max_value = duration_days  # Maximum dwell time is the total duration of the simulation

    # Scale parameters for each state
    scales = {
        TBS.NONE: 126, 
        TBS.LATENT_SLOW: 365,
        TBS.LATENT_FAST: 200,
        TBS.ACTIVE_PRESYMP: 100,
        TBS.ACTIVE_SMPOS: 150,
        TBS.ACTIVE_SMNEG: 300,
        TBS.ACTIVE_EXPTB: 250,
        TBS.DEAD: 400,
    }

    # Generate truncated exponential distributions for all states
    return {
        state: lambda x, scale=scale: stats.truncexpon(
            b=(max_value - min_value) / scale, loc=min_value, scale=scale
        ).cdf(x)
        for state, scale in scales.items()
    }

def run_simulation():
    import pandas as pd
    # Create and run the simulation
    sim_tb = make_tb()
    sim_tb.run()

    # Define start and stop times
    start = sim_tb.pars.start
    stop = sim_tb.pars.stop
    # Calculate expected distributions
    expected_distributions = calculate_expected_distributions(start, stop)
    transitions_dict = {
        'None': ['Latent Slow', 'Latent Fast'],
        'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
    }

    # Extract the analyzer
    ana_dwt = sim_tb.analyzers[0]    #shortuct to the dwell time analyzer
    # ana_dwt.validate_dwell_time_distributions(expected_distributions=expected_distributions)  # Optional validation
    # ana_dwt.histogram_with_kde()  # Optional plotting


    # Plotting
    ana_dwt.plot_state_transition_lengths_custom(transitions_dict=transitions_dict)
    # ana_dwt.graph_state_transitions(states=['None', 'Latent Slow', 'Latent Fast', 'Active Presymp', 'Active Smpos', 'Active Smneg'], layout=0)
    # ana_dwt.plot_dwell_time_validation()
    # ana_dwt.plot_dwell_time_validation_interactive()
    # ana_dwt.graph_compartments_transitions(layout=0)
    # ana_dwt.interactive_all_state_transitions()
    # ana_dwt.stacked_bars_states_per_agent_static()
    # ana_dwt.interactive_stacked_bar_charts_dt_by_state()
    # ana_dwt.plot_binned_stacked_bars_state_transitions(bin_size=50, num_bins=50)
    # ana_dwt.plot_binned_by_compartment(num_bins=50)
    # ana_dwt.sankey()

    # Perform validation and plotting

    # Create a sample DataFrame
    # file = '/Users/mine/git/tbsim/results/dwell_time_logger_20250124160857.csv'   # Option #1:  MANUALLY PASS THE FILE PATH
    file = ana_dwt.file_path                                                        # Option #2:  Get the file path from the analyzer   

    # Initialize the DwtPlotter
    plotter = mtb.DwtPlotter(file_path=file)

    plotter.histogram_with_kde(num_bins=10, bin_size=30)
    plotter.plot_state_transition_lengths_custom(transitions_dict=transitions_dict)
    plotter.graph_state_transitions()
    plotter.plot_dwell_time_validation()
    plotter.plot_dwell_time_validation_interactive()
    plotter.graph_compartments_transitions(layout=0)
    plotter.interactive_all_state_transitions()
    plotter.stacked_bars_states_per_agent_static()
    plotter.interactive_stacked_bar_charts_dt_by_state()
    plotter.plot_binned_stacked_bars_state_transitions(bin_size=50, num_bins=50)
    plotter.plot_binned_by_compartment(num_bins=50)
    plotter.sankey()

if __name__ == '__main__':
    run_simulation()