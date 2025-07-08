import tbsim as mtb
import starsim as ss
import sciris as sc 
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

TBS = mtb.TBS

def build_tbsim(sim_pars=None):
    sim_params = dict(
        start = sc.date('2013-01-01'),      
        stop = sc.date('2016-12-31'), 
        rand_seed=123,
        unit='day',
        dt=7,
    )
    if sim_pars is not None:
        sim_params.update(sim_pars)

    pop = ss.People(n_agents=1000)

    tb_params = dict(
        beta=ss.rate_prob(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
        unit='day'
    )
    tb = mtb.TB(tb_params)
    
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    dwell_analyzer = mtb.DwtAnalyzer(adjust_to_unit=True, unit=1.0, scenario_name='run_TB_Dwell_analyzer') # ANALYZER

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        # demographics=[deaths, births],
        pars=sim_params,
        analyzers=dwell_analyzer,
    )
    sim.pars.verbose = 30/365

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
    sim_tb = build_tbsim()
    sim_tb.run()
    start = sim_tb.pars.start
    stop = sim_tb.pars.stop

    # Calculate expected distributions
    expected_distributions = calculate_expected_distributions(start, stop)

    # # Extract the analyzer
    ana : mtb.DwtAnalyzer = sim_tb.analyzers[0] #shortcut to the dwell time analyzer
    ana.graph_state_transitions()
    ana.sankey_agents_by_age_subplots(bins = [0,5,200])


    # Sample using directly from the generated file(s)
    file = ana.file_path        # (uses the file from the analyzer)
    plotter = mtb.DwtPlotter(file_path=file)
    plotter.histogram_with_kde()