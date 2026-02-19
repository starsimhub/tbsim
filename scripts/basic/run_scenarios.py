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

# Suppress warning from seaborn
import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

x_latent_slow = [2, 3] + [1]

debug = True
default_n_agents = [10_000, 1000][debug]
default_n_rand_seeds = [20, 2][debug]

def run_sim(n_agents=default_n_agents, rand_seed=0, idx=0, xLS=1):
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
    filename = sc.makefilepath(filename='results.csv', folder='results', makedirs=True)
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