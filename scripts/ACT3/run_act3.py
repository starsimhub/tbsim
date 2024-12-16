import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os

from run_act3_calibration import make_sim, build_sim

do_run = True
debug = False #NOTE: Debug runs in serial

# Each scenario will be run n_seeds times for each of intervention and control.
n_seeds = [30, 3][debug] # Results bootstrapped over K=60 seeds (of intv and ctrl)
n_reps = 1 # Default, leave as 1 for now, results will be combined by bootstrapping across seeds

# Check if the results directory exists, if not, create it
resdir = os.path.join('results', 'ACT3')
os.makedirs(resdir, exist_ok=True)

def run_ACF(base_sim, skey, scen, rand_seed=0):
    """
    Run n_reps of control and intervention simulations in a single multisim for seed rand_seed
    """

    sim = base_sim.copy()
    # MODIFY THE SIMULATION OBJECT BASED ON THE SCENARIO HERE
    # skey, scen

    sim.pars.rand_seed = rand_seed # This is the base seed that build_sim will increment from for n_reps
    #########################################################

    ms = build_sim(sim, calib_pars=scen['CalibPars'], n_reps=n_reps)
    ms.run()

    tb_res = []
    acf_res = []
    for s in ms.sims:
        df = pd.DataFrame({
            'time_year': s.results.timevec,
            'on_treatment': s.results.tb.n_on_treatment, 
            'prevalence': s.results.tb.prevalence,
            'active_presymp': s.results.tb.n_active_presymp,
            'active_smpos': s.results.tb.n_active_smpos,
            'active_exptb': s.results.tb.n_active_exptb,
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['rand_seed'] = s.pars.rand_seed
        tb_res.append(df)

        act3_dates = [ss.date(t) for t in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']]
        inds = np.searchsorted(s.results.timevec, act3_dates, side='left')
        df = pd.DataFrame({
            'time_year': s.results.timevec[inds],
            'n_elig': s.results['ACT3 Active Case Finding'].n_elig[inds],
            'n_tested': s.results['ACT3 Active Case Finding'].n_tested[inds],
            'n_positive': s.results['ACT3 Active Case Finding'].n_positive[inds],
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['rand_seed'] = s.pars.rand_seed
        acf_res.append(df)

    tb_res = pd.concat(tb_res)
    acf_res = pd.concat(acf_res)

    return {'TB': tb_res, 'ACT3': acf_res}


def run_scenarios(scens, n_seeds=n_seeds):
    results = []
    cfgs = []

    seeds = np.random.randint(0, 1e6, n_seeds)

    # Iterate over scenarios and random seeds
    for skey, scen in scens.items():
        for seed in seeds:
            # Append configuration for parallel execution
            seed = np.random.randint(0, 1e6) # Use a random seed because the multisim will increment from this and we don't want to reuse
            cfgs.append({'skey': skey, 'scen': scen, 'rand_seed': seed})

    # Run simulations in parallel
    T = sc.tic()

    sim = make_sim()
    results += sc.parallelize(run_ACF, iterkwargs=cfgs, kwargs=dict(base_sim=sim), die=True, serial=debug)

    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # separate the results for each component of the simulation (TB and ACT3)
    dfs = {}
    for k in results[0].keys():
        df_list = [r.get(k) 
                   for r in results 
                   if r.get(k) is not None
                   ]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))
    return dfs


if __name__ == '__main__':

    scens = {
        'Basic ACT3': {
            # default has been set to basic ACT3
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                #{'beta': 0.445499764760726, 'beta_change': 0.6880249223150746, 'beta_change_year': 1986, 'xpcf': 0.08450198158889916, 'rand_seed': 925220}. Best is trial 1747 with value: 103.93212613894507.
                beta = dict(low=0.01, high=0.70, value=0.445499764760726, suggest_type='suggest_float', log=False), # Log scale and no "path", will be handled by build_sim (above)
                beta_change = dict(low=0.25, high=1, value=0.6880249223150746),
                beta_change_year = dict(low=1950, high=1986, value=2001, suggest_type='suggest_int'),
                xpcf = dict(low=0, high=1.0, value=0.08450198158889916),
            )
        },
    }
    # Other calib pars:
    '''
    'Other calib pars': {
        # default has been set to basic ACT3
        'ACT3': None,
        'TB': None,
        'Simulation': None,
        'CalibPars': dict(
            # {'beta': 0.4016615352455662, 'beta_change': 0.7075221406739112, 'beta_change_year': 2001, 'xpcf': 0.14812009041601049, 'rand_seed': 172264}. Best is trial 88 with value: 11479.499964475886.
            beta = dict(low=0.01, high=0.70, value=0.4016615352455662, suggest_type='suggest_float', log=False), # Log scale and no "path", will be handled by build_sim (above)
            beta_change = dict(low=0.25, high=1, value=0.7075221406739112),
            beta_change_year = dict(low=1950, high=2014, value=2001, suggest_type='suggest_int'),
            xpcf = dict(low=0, high=1.0, value=0.14812009041601049),
        )
    },
    '''

    if do_run:
        df_result = run_scenarios(scens)
    else:
        try:
            df_result = {}
            for k in ['TB', 'ACT3']:
                df_result[k] = pd.read_csv(os.path.join(resdir, f'{k}.csv'))
        except FileNotFoundError:
            print('No results found, please set do_run to True')
            raise

    # plot the results
    df_result.get('ACT3')


    # MOVE TO aplt:

    # TB time series# ###################################################
    import seaborn as sns
    ret = df_result.get('TB').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'timeseries.png'), dpi=600)

    # ACT3 time series ##################################################
    ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'act3.png'), dpi=600)

    # ACT3 cases found, scaled to trial #################################
    df = df_result.get('ACT3')
    df.set_index('rand_seed', inplace=True)
    df['time_year'] = pd.to_datetime(df['time_year'])
    seeds = df.index.unique()
    K = 60
    n_boots = 1000
    dfs = []
    expected = pd.DataFrame({
        'x': [169, 136, 78, 53],           # Number of individuals found to be infectious
        'n': [43425, 44082, 44311, 42150], # Number of individuals sampled
    ####}, index=pd.Index([ss.date(d) for d in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']], name='t')) # On these dates
    }, index=pd.Index([ss.date(d) for d in ['2014-06-16', '2015-06-11', '2016-06-05', '2017-06-30']], name='t')) # On these dates
    for bi in range(n_boots):
        boot_seeds = np.random.choice(seeds, K)
        dfb = df.loc[boot_seeds].groupby(['scenario', 'arm', 'time_year']).sum()
        combined = pd.merge(dfb.reset_index(), expected.reset_index(), left_on='time_year', right_on='t')

        #combined['scaled_positive'] = combined['n_positive'] * combined['n']  / combined['n_tested']
        alpha = combined['n_positive'] + 1
        beta = combined['n_tested'] - combined['n_positive'] + 1
        n = combined['n']
        combined['scaled_positive'] = n * alpha / (alpha + beta)
        combined.loc[combined['n_tested'] == 0, 'scaled_positive'] = np.nan

        combined['bi'] = bi
        dfs.append(combined)
    df = pd.concat(dfs)

    def plot_observed(data, **kwargs):
        #df = expected.reset_index()
        #sns.scatterplot(data=df, x='t', y='x')
        import matplotlib.pyplot as plt
        plt.scatter(np.arange(len(expected)), expected['x'].values, color='orange', edgecolors='black', s=100)

        # Control
        plt.plot(3, 94, color='blue', marker='o', mec='black', ms=10)

    #ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    d = df[['scenario', 'arm', 'time_year', 'scaled_positive', 'bi']]
    d = d.dropna(axis=0)
    g = sns.catplot(kind='strip', data=d, x='time_year', order=expected.index, y='scaled_positive', hue='arm', col='scenario', estimator=None, units='bi', size=1) # SD for speed, units='rand_seed'
    g.map_dataframe(plot_observed)
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'act3_acf.png'), dpi=600)



    # Historical incidence ##################################################
    df = df_result.get('TB')
    df.set_index('rand_seed', inplace=True)
    df['time_year'] = pd.to_datetime(df['time_year'])
    seeds = df.index.unique()
    K = 60
    n_boots = 1000
    dfs = []
    expected = pd.DataFrame({
        'x': [169, 136, 78, 53],           # Number of individuals found to be infectious
        'n': [43425, 44082, 44311, 42150], # Number of individuals sampled
    ####}, index=pd.Index([ss.date(d) for d in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']], name='t')) # On these dates
    }, index=pd.Index([ss.date(d) for d in ['2014-06-16', '2015-06-11', '2016-06-05', '2017-06-30']], name='t')) # On these dates
    for bi in range(n_boots):
        boot_seeds = np.random.choice(seeds, K)
        dfb = df.loc[boot_seeds].groupby(['scenario', 'arm', 'time_year']).sum()
        combined = pd.merge(dfb.reset_index(), expected.reset_index(), left_on='time_year', right_on='t')

        #combined['scaled_positive'] = combined['n_positive'] * combined['n']  / combined['n_tested']
        alpha = combined['n_positive'] + 1
        beta = combined['n_tested'] - combined['n_positive'] + 1
        n = combined['n']
        combined['scaled_positive'] = n * alpha / (alpha + beta)
        combined.loc[combined['n_tested'] == 0, 'scaled_positive'] = np.nan

        combined['bi'] = bi
        dfs.append(combined)
    df = pd.concat(dfs)

    def plot_observed(data, **kwargs):
        #df = expected.reset_index()
        #sns.scatterplot(data=df, x='t', y='x')
        import matplotlib.pyplot as plt
        plt.scatter(np.arange(len(expected)), expected['x'].values, color='orange', edgecolors='black', s=100)

        # Control
        plt.plot(3, 94, color='blue', marker='o', mec='black', ms=10)

    #ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    d = df[['scenario', 'arm', 'time_year', 'scaled_positive', 'bi']]
    d = d.dropna(axis=0)
    g = sns.catplot(kind='strip', data=d, x='time_year', order=expected.index, y='scaled_positive', hue='arm', col='scenario', estimator=None, units='bi', size=1) # SD for speed, units='rand_seed'
    g.map_dataframe(plot_observed)
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'act3_acf.png'), dpi=600)






    # APLT PLOTS #######################################################
    aplt.plot_scenarios(results=df_result.get('TB'))