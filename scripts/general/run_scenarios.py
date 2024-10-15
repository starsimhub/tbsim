"""
TB-Malnutrition scenarios
"""

# %% Imports and settings
import starsim as ss
import tbsim as mtb
import pandas as pd
import os
import argparse
import sciris as sc
import tbsim.config as cfg
from functools import partial
import numpy as np

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

# Weight percentile rate ratios
wprr = [2, 3] + [1] # 1 is the baseline

debug = True
default_n_agents = [10_000, 1000][debug]
default_n_rand_seeds = [20, 2][debug]

def weightpercentile_rr(tb, mn, uids, threshold=0.2, rate_ratio=3):
    # Agents with a weight percentile below the threshold get an elevated rate ratio
    rr = np.ones_like(uids)
    rr[mn.weight_percentile[uids] < threshold] = rate_ratio
    return rr

def run_sim(n_agents=default_n_agents, rand_seed=0, idx=0, rr=1):
    # --------- People ----------
    pop = ss.People(n_agents=n_agents)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = ss.beta(0.1), 
        init_prev = ss.bernoulli(0.25),
    )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # -------- Network ---------
    # Network parameters
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after each timestep
    )
    # Initialize a random network
    net = ss.RandomNet(net_pars)

    # Add demographics
    dems = [
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 women 15-49
    ]

    # Connector
    cn_pars = dict(
        rr_activation_func = partial(weightpercentile_rr, threshold=0.2, rate_ratio=rr)
    )
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        start = 1980,
        stop = 2020,
    )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps
    sim.run()

    df = pd.DataFrame( {
        'year': sim.timevec,
        #'pph.mother_died.cumsum': sim.results.pph.mother_died.cumsum(),
        #'Births': sim.results.pph.births.cumsum(),
        'Deaths': sim.results.deaths.cumulative,
        #'Maternal Deaths': sim.results.pph.maternal_deaths.cumsum(),
        #'Infant Deaths': sim.results.pph.infant_deaths.cumsum(),
    })
    df['rr'] = rr
    df['rand_seed'] = rand_seed

    print(f'Finishing sim {idx} with rand_seed={rand_seed} and rr={rr}')

    return df


def run_scenarios(n_agents=default_n_agents, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for rs in range(n_seeds):
        for rr in wprr:
            cfgs.append({'rr':rr, 'rand_seed':rs, 'idx':len(cfgs)})
    T = sc.tic()
    results += sc.parallelize(run_sim, kwargs={'n_agents': n_agents}, iterkwargs=cfgs, die=False, serial=False)
    times = sc.toc(T, output=True)

    print('Timings:', times)

    df = pd.concat(results)
    filename = os.path.join(cfg.create_res_dir('scenarios'), "results.csv")
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

    # Plotting code would go here

    print(df)
    print(f"Results directory {filename}")
    print('Done')
