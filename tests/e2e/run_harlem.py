import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import pandas as pd
import os 
import tbsim.config as cfg
import datetime as dt

import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False
default_n_rand_seeds = [100, 1][debug]

def run_harlem(rand_seed=0):

    np.random.seed(rand_seed)

    # --------- Harlem ----------
    harlem = mtb.Harlem()

    # --------- People ----------
    pop = harlem.people()

    # -------- Network ---------
    harlemnet = harlem.net()

    # Network parameters
    randnet_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
    )
    # Initialize a random network
    randnet = ss.RandomNet(randnet_pars)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = dict(harlem=0.1, random=0.001),
        init_prev = 0, # Infections seeded by Harlem class
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Nutrition --------
    nut_pars = dict()
    nut = mtb.Nutrition(nut_pars)

    # Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # -------- Connector -------
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------
    vs = mtb.VitaminSupplementation(year=[1942, 1943], rate=[2.0, 0.25])
    m = mtb.MacroNutrients
    lsff0 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.25, 0], from_state=m.UNSATISFACTORY, to_state=m.MARGINAL)
    lsff1 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.75, 0], from_state=m.MARGINAL, to_state=m.SLIGHTLY_BELOW_STANDARD)
    lsff2 = mtb.LargeScaleFoodFortification(year=[1942, 1944], rate=[1.75, 0], from_state=m.SLIGHTLY_BELOW_STANDARD, to_state=m.STANDARD_OR_ABOVE)
    intvs = [vs, lsff0, lsff1, lsff2]

    # -------- Analyzer -------
    az = mtb.HarlemAnalyzer()

    # -------- Simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        start = 1940, # 2y burn-in
        #start = 1942,
        end = 1947,
        rand_seed = rand_seed,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=[harlemnet, randnet], diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn, interventions=intvs, analyzers=az)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    sim.initialize()

    harlem.set_states(sim)
    seed_uids = harlem.choose_seed_infections(sim, p_hh=0.835) #83% or 84%
    tb.set_prognoses(sim, seed_uids)

    sim.run() # Actually run the sim

    df = sim.analyzers['harlemanalyzer'].df

    '''
    df = pd.DataFrame( {
        'year': sim.yearvec,
        #'pph.mother_died.cumsum': sim.results.pph.mother_died.cumsum(),
        'Births': sim.results.pph.births.cumsum(),
        'Deaths': sim.results.deaths.cumulative,
        'Maternal Deaths': sim.results.pph.maternal_deaths.cumsum(),
        'Infant Deaths': sim.results.pph.infant_deaths.cumsum(),
    })
    '''
    df['rand_seed'] = rand_seed

    print(f'Finishing sim with rand_seed={rand_seed} ')

    return df


def run_sims(n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for rs in range(n_seeds):
        cfgs.append({'rand_seed':rs})
    T = sc.tic()
    results += sc.parallelize(run_harlem, iterkwargs=cfgs, die=False, serial=debug)

    print('Timings:', sc.toc(T, output=True))

    df = pd.concat(results)
    df.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"result_{cfg.FILE_POSTFIX}.csv"))

    return df

def plot(df):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.dates as mdates
    import sciris as sc
    import tbsim.config as cfg

    first_year = int(df['year'].iloc[0])
    assert df['year'].iloc[0] == first_year
    df['date'] = pd.to_datetime(365 * (df['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    d = pd.melt(df.drop(['rand_seed', 'year'], axis=1), id_vars=['date', 'arm'], var_name='channel', value_name='Value')
    g = sns.relplot(data=d, kind='line', x='date', hue='arm', col='channel', y='Value', palette='Set1', facet_kws={'sharey':False}, col_wrap=4)

    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels('Date')
    for ax in g.axes.flat:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    sc.savefig(f"result_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)
    return

if __name__ == '__main__':
    df = run_sims()
    if debug:
        sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
        sim.diseases['tb'].plot()
        sim.plot()
        sim.analyzers['harlemanalyzer'].plot()
        plt.show()

    plot(df)

    print(f"Results directory {cfg.RESULTS_DIRECTORY}\nThis run: {cfg.FILE_POSTFIX}")
    print('Done')
