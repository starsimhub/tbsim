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
        dt=1,
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

    # Create and run the simulation
    sim_tb = make_tb()
    sim_tb.run()

    # Define start and stop times
    start = sim_tb.pars.start
    stop = sim_tb.pars.stop

    # Calculate expected distributions
    expected_distributions = calculate_expected_distributions(start, stop)

    print("Expected distributions:", expected_distributions)
    # Extract the analyzer
    an = sim_tb.analyzers[0]
    # an.plot_state_transition_graph_interactive()
    # an.plot_state_transition_graph_static()
    
    # an.plot_combined_rates_area(num_bins=50)
    # an.plot_binned_stacked_bars_state_transitions(num_bins=20) #DONE
    # an.plot_binned_by_compartment(num_bins=40)  #DONE
    # an.graph_state_transitions()    #DONE
    an.graph_compartments_transitions() 

    # # Perform validation and plotting
    # an.validate_dwell_time_distributions(expected_distributions=expected_distributions)  # Optional validation
    # an.plot_dwell_time_validation_interactive()
    # an.graph_agent_dynamics()
    # an.plot_dwell_time_validation()

    # # External plotting
    # file_dwt = an.file_name
    # mtb.stacked_bars_states_per_agent_clean(file_dwt)
    # mtb.plot_dwell_time_lines_for_each_agent(file_dwt)
    # mtb.parallel_coordinates(file_dwt)
    # mtb.parallel_categories(file_dwt)


if __name__ == '__main__':
    run_simulation()
