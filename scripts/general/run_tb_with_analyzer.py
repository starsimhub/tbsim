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
        dt=23,
        start=sc.date('1990-01-01'),
        stop=sc.date('2016-12-31'),
    )
    if sim_pars is not None:
        params.update(sim_pars)

    np.random.seed()
    pop = ss.People(n_agents=200)
    tb = mtb.TB(dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        unit='day',
        rel_sus_latentslow=0.5,
    ))
    dwell_analyzer = mtb.DTAn()

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

    # Extract the analyzer
    dwelltime_an = sim_tb.analyzers[0]
    dwelltime_an.graph_state_transitions()
    dwelltime_an.interactive_all_state_transitions()

    # # Perform validation and plotting
    dwelltime_an.validate_dwell_time_distributions(expected_distributions=expected_distributions)  # Optional validation
    # dwelltime_an.plot_dwell_time_validation_interactive()
    # dwelltime_an.graph_agent_dynamics()
    # dwelltime_an.plot_dwell_time_validation()

    # External plotting
    file_name = dwelltime_an.file_name

    transitions_dict = {
        'None': ['Latent Slow', 'Latent Fast'],
        'Active Presymp': ['Active Smpos', 'Active Smneg', 'Active Exptb'],
        'Active Smpos': ['Dead', 'None'],
    }
    mtb.sankey(file_path=file_name)
    mtb.state_transition_matrix(file_path=file_name)
    
    # dwell_time_logger = pd.read_csv(file_name, na_values=[], keep_default_na=False)
    
    # mtb.interactive_stacked_bar_charts_dt_by_state(dwell_time_logger=dwell_time_logger, bin_size=50)
    # mtb.plot_state_transition_lengths_custom(dwell_time_logger=dwell_time_logger, transitions_dict=transitions_dict)
    # mtb.graph_state_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Latent Slow', 'Latent Fast', 'Active Presymp', 'Active Smpos', 'Active Smneg', 'Active Exptb'], pos=0 )
    # mtb.graph_compartments_transitions(dwell_time_logger=dwell_time_logger, states=['None', 'Active Presymp']) 
    # mtb.plot_binned_by_compartment(dwell_time_logger=dwell_time_logger,  bin_size=50, num_bins=8)
    # mtb.plot_binned_stacked_bars_state_transitions(dwell_time_logger=dwell_time_logger, bin_size=50, num_bins=8)

if __name__ == '__main__':
    run_simulation()
