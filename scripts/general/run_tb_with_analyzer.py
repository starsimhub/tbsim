import tbsim as mtb
import starsim as ss
import sciris as sc 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

TBS = mtb.TBS

def make_tb(sim_pars=None):
    sim_params = dict(
        start=sc.date('1940-01-01'),
        stop=sc.date('2025-12-31'),
        rand_seed=123,
        unit='days',
        dt=30,
    )
    if sim_pars is not None:
        sim_params.update(sim_pars)

    np.random.seed()
    pop = ss.People(n_agents=1000)

    tb_params = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
    )
    tb = mtb.TB(tb_params)
    
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))

    dwell_analyzer = mtb.DwtAnalyzer(adjust_to_unit=True, unit=1.0, scenario_name='run_TB_Dwell_analyzer') # ANALYZER

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=sim_params,
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

if __name__ == '__main__':
    sim_tb = make_tb()
    sim_tb.run()
    start = sim_tb.pars.start
    stop = sim_tb.pars.stop

    # Calculate expected distributions
    expected_distributions = calculate_expected_distributions(start, stop)

    # Extract the analyzer
    ana : mtb.DwtAnalyzer = sim_tb.analyzers[0] #shortcut to the dwell time analyzer
    ana.graph_state_transitions()


    # file = '/Users/mine/git/tbsim/results/dwell_time_logger_20250127151951.csv'   # Option #1:  MANUALLY PASS THE FILE PATH
    # file = ana_dwt.file_path                                                        # Option #2:  Get the file path from the analyzer   

    # # # # Initialize the DwtPlotter
    # plotter = mtb.DwtPlotter(file_path=file)
    # #  plotter.histogram_with_kde()
    # # plotter.plot_state_transition_lengths_custom(transitions_dict=transitions_dict)
    # plotter.graph_state_transitions()
    # # plotter.plot_dwell_time_validation()

