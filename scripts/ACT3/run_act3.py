import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os

from run_act3_calibration import make_sim, build_sim

# TEMP, move to aplt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.stats as sps

do_run = True
debug = False #NOTE: Debug runs in serial

# Each scenario will be run n_seeds times for each of intervention and control.
n_seeds = [60*5, 2][debug]
n_reps = 1 # Default, leave as 1 for now, results will be combined by bootstrapping across seeds

# Check if the results directory exists, if not, create it
resdir = os.path.join('results', 'ACT3')
os.makedirs(resdir, exist_ok=True)

class prev_by_age(ss.Intervention):
    def __init__(self, year, **kwargs):
        self.year = year
        super().__init__(**kwargs)
        return
    
    def step(self):
        if self.year >= self.t.now('year') and  self.year < self.t.now('year')+self.t.dt_year:
            self.age_bins = np.arange(0, 101, 5)
            self.n, _        = np.histogram(self.sim.people.age, bins=self.age_bins)
            self.ever, _     = np.histogram(self.sim.people.age[self.sim.diseases.tb.ever_infected], bins=self.age_bins)
            self.infected, _ = np.histogram(self.sim.people.age[self.sim.diseases.tb.infected], bins=self.age_bins)
            self.active, _   = np.histogram(self.sim.people.age[self.sim.diseases.tb.infectious], bins=self.age_bins)
        return

def run_ACF(base_sim, skey, scen, rand_seed=0):
    """
    Run n_reps of control and intervention simulations in a single multisim for seed rand_seed
    """

    sim = base_sim.copy()
    # MODIFY THE SIMULATION OBJECT BASED ON THE SCENARIO HERE
    # skey, scen

    sim.pars.interventions += [prev_by_age(year=2013)]

    sim.pars.rand_seed = rand_seed # This is the base seed that build_sim will increment from for n_reps
    #########################################################

    ms = build_sim(sim, calib_pars=scen['CalibPars'], n_reps=n_reps)
    ms.run()

    ### EVAL LL
    prevalence_ctrl = ss.Binomial(
        name = 'Prevalence Active (Control)',
        include_fn = lambda sim: sim.label == 'Control' and np.any(sim.results.tb.n_infected[sim.timevec >= ss.date('2013-01-01')] > 0),
        weight = 1,
        conform = 'step_containing',

        expected = pd.DataFrame({
            'x': [94],      # Number of individuals found to be infectious
            'n': [41680], # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-06-01']], name='t')), # On these dates

        extract_fn = lambda sim: pd.DataFrame({
            # sim.results.tb.n_active / sim.results.n_alive,
            'p': (sim.results['ACT3 Active Case Finding'].n_positive +1) / (sim.results['ACT3 Active Case Finding'].n_tested + 2)
        }, index=pd.Index(sim.results.timevec, name='t')),
    )
    nLL = prevalence_ctrl(ms)

    '''
    fig = prevalence_ctrl.plot()
    plt.subplots_adjust(top=0.9)
    plt.title(f'LL: {LL:.2f}')
    plt.savefig(os.path.join(resdir, 'figs', f'll_prevalencectrl_{skey}_{rand_seed}.png'), dpi=600)

    fig = prevalence_ctrl.plot(bootstrap=True)
    plt.subplots_adjust(top=0.9)
    plt.title(f'LL: {LL:.2f}')
    plt.savefig(os.path.join(resdir, 'figs', f'll_prevalencectrl_{skey}_{rand_seed}_bootstrap.png'), dpi=600)
    '''
    ####################################################################

    tb_res = []
    acf_res = []
    pba_res = []
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
        df['include'] = np.any(s.results.tb.n_infected[s.timevec >= ss.date('2013-01-01')] > 0)
        tb_res.append(df)

        act3_dates = [ss.date(t) for t in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']]
        inds = np.searchsorted(s.results.timevec, act3_dates, side='left')
        df = pd.DataFrame({
            'time_year': s.results.timevec[inds],
            'n_elig': s.results['ACT3 Active Case Finding'].n_elig[inds],
            'n_tested': s.results['ACT3 Active Case Finding'].n_tested[inds],
            'n_positive': s.results['ACT3 Active Case Finding'].n_positive[inds],

            'n_positive_presymp': s.results['ACT3 Active Case Finding'].n_positive_presymp[inds],
            'n_positive_smpos': s.results['ACT3 Active Case Finding'].n_positive_smpos[inds],
            'n_positive_smneg': s.results['ACT3 Active Case Finding'].n_positive_smneg[inds],
            'n_positive_exp': s.results['ACT3 Active Case Finding'].n_positive_exp[inds],

            'n_positive_via_LF': s.results['ACT3 Active Case Finding'].n_positive_via_LF[inds],
            'n_positive_via_LS': s.results['ACT3 Active Case Finding'].n_positive_via_LS[inds],
            'n_positive_via_LF_dur': s.results['ACT3 Active Case Finding'].n_positive_via_LF_dur[inds],
            'n_positive_via_LS_dur': s.results['ACT3 Active Case Finding'].n_positive_via_LS_dur[inds],
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['nLL'] = nLL
        # TEMP
        df['p'] = prevalence_ctrl.actual['p'].values[0]
        ######
        df['include'] = np.any(s.results.tb.n_infected[s.timevec >= ss.date('2013-01-01')] > 0)
        df['rand_seed'] = s.pars.rand_seed
        acf_res.append(df)

        intv = s.interventions.get('prev_by_age', None)
        if intv is None:
            continue
        df = pd.DataFrame({
            'n': intv.n,
            'ever': intv.ever,
            'infected': intv.infected,
            'active': intv.active,
        }, index = pd.Index([f'{b}-{e}' for b,e in zip(intv.age_bins[:-1], intv.age_bins[1:])], name='age bin'))
        df['year'] = ss.date(intv.year)
        df['scenario'] = skey
        df['arm'] = s.label
        df['rand_seed'] = s.pars.rand_seed
        pba_res.append(df)

    tb_res = pd.concat(tb_res)
    acf_res = pd.concat(acf_res)
    pba_res = pd.concat(pba_res)

    return {'TB': tb_res, 'ACT3': acf_res, 'PBA': pba_res}


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
            'CalibPars': 
                {k:dict(value=v) for k,v in
                #'rand_seed': 850549
                #{'beta': 0.9067522769987235, 'x_pcf1': 0.3231894161330828, 'x_pcf2': 0.5695811456744782, 'beta_x_final': 0.779008573903516, 'beta_dur': 24.30430256638232, 'beta_mid': 1973.2357310549744, 'x_acf_cov': 0.8736961778986055, 'p_fast': 0.17403427420948536}
                #'rand_seed': 470796
                #{'beta': 0.910396235207257, 'beta_x_final': 0.05737434233165578, 'beta_dur': 24.686880757985698, 'beta_mid': 2015.7462785352063, 'x_acf_cov': 0.8746741743886838, 'p_fast': 0.20701467084246836} \
                # 'rand_seed': 233965
                #{'beta': 0.3639402812775401, 'x_pcf1': 0.17422127142544253, 'x_pcf2': 0.8635517496019752, 'beta_x_final': 0.2266611894479731, 'beta_dur': 16.96261135468262, 'beta_mid': 1979.4790117285563, 'x_acf_cov': 0.5672782978680914, 'p_fast': 0.421811292136281} \
                #{'beta': 0.50, 'x_pcf1': 0.17422127142544253, 'x_pcf2': 0.8635517496019752, 'beta_x_final': 0.2266611894479731, 'beta_dur': 16.96261135468262, 'beta_mid': 1979.4790117285563, 'x_acf_cov': 0.5672782978680914, 'p_fast': 0.421811292136281} \
                # 'rand_seed': 102211
                #{'beta': 0.43529159300902504, 'x_pcf1': 0.5156170226355246, 'x_pcf2': 0.9049310957856859, 'beta_x_final': 0.5113042336658371, 'beta_dur': 18.392517993256075, 'beta_mid': 1981.9180527400858, 'x_acf_cov': 0.7155236261871701, 'p_fast': 0.6144963029973003} \
                # , 'rand_seed': 479997
                #{'beta': 0.2515466183818931, 'x_pcf1': 0.4422889810210864, 'x_pcf2': 0.8090931524610825, 'beta_x_final': 0.9722331124476845, 'beta_dur': 18.65710753236865, 'beta_mid': 1980.945342039754, 'x_acf_cov': 0.6110592364870079, 'p_fast': 0.7682229846113128} \
                # , 'rand_seed': 179507
                {'beta': 0.31013128138223345, 'x_pcf1': 0.238131788244131, 'x_pcf2': 0.9992504136800656, 'beta_x_final': 0.8834600066122873, 'beta_dur': 19.79941839318105, 'beta_mid': 1975.6750295037912, 'x_acf_cov': 0.29885149828207896, 'p_fast': 0.6437272056839243}
                .items() },
        }
    }

    if do_run:
        df_result = run_scenarios(scens)
    else:
        try:
            df_result = {}
            for k in ['TB', 'ACT3', 'PBA']:
                df_result[k] = pd.read_csv(os.path.join(resdir, f'{k}.csv'), index_col=0)
                if 'time_year' in df_result[k]:
                    df_result[k]['time_year'] = pd.to_datetime(df_result[k]['time_year'])
        except FileNotFoundError:
            print('No results found, please set do_run to True')
            raise

    # plot the results
    df_result.get('ACT3')


    # MOVE TO aplt:
    '''
    ret = df_result.get('ACT3').groupby('rand_seed')[['nLL', 'x', 'n']].mean()
    for seed, row in ret.groupby('rand_seed'):
        #a_n, a_x = row['n'], row['x']
        a_p = row['p']
        e_n, e_x = 41680, 94
        q = sps.binom(n=e_n, p=p)
        print(f'LL: {q.logpmf(e_x)[0]:.2f} vs {row["nLL"].values[0]:.2f}')
    
    ret['e_n'] = 41680
    ret['e_x'] = 94

    q = sps.binom(n=e_n, p=p)
    ret['LL2'] = q.logpmf(e_x)
    ret['mean'] = q.mean()
    ret['median'] = q.median()
    intv95 = q.interval(0.95)
    ret['2.5%'] = intv95[0]
    ret['97.5%'] = intv95[1]

    b = sps.binom(n=e_n, p=(ret['x']+1)/(ret['n']+2)) # Smoothed
    ret['bLL'] = b.logpmf(e_x)
    ret['bmean'] = b.mean()
    ret['bmedian'] = b.median()
    intv95 = b.interval(0.95)
    ret['b2.5%'] = intv95[0]
    ret['b97.5%'] = intv95[1]
    print(ret)
    '''

    # TB by age ###################################################
    ret = df_result.get('PBA')
    ret['prev ever'] = ret['ever'] / ret['n']
    ret['prev infected'] = ret['infected'] / ret['n']
    ret['prev active'] = ret['active'] / ret['n']
    df = ret.reset_index().melt(id_vars=['age bin', 'scenario', 'arm', 'year', 'rand_seed'], value_vars=['prev ever', 'prev infected', 'prev active'], value_name='value', var_name='variable')
    g = sns.catplot(kind='bar', data=df.reset_index(), x='age bin', hue='variable', y='value', col='scenario')
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'age.png'), dpi=600)

    # TB time series ###################################################
    ret = df_result.get('TB').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'timeseries.png'), dpi=600)

    # ACT3 time series ##################################################
    ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', col_wrap=4, kind='line', style='scenario', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    for ax in g.axes.flat:
        #locator = mdates.AutoDateLocator()
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    g.fig.savefig(os.path.join(resdir, 'figs', 'act3.png'), dpi=600)

    # ACT3 cases found, scaled to trial #################################
    df = df_result.get('ACT3')
    df.set_index('rand_seed', inplace=True)
    df['time_year'] = pd.to_datetime(df['time_year'])
    seeds = df.index.unique()

    K = min(60, len(seeds))

    # Seed filtering
    seeds = seeds[df.groupby('rand_seed')['include'].mean().loc[seeds].values.astype(bool)] # Ugly

    n_boots = 1000
    dfs = []

    expected = pd.DataFrame({
        'x': [169, 136, 78, 53],           # Number of individuals found to be infectious
        'n': [43425, 44082, 44311, 42150], # Number of individuals sampled
    }, index=pd.Index([ss.date(d) for d in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']], name='t')) # On these dates

    for bi in range(n_boots):
        boot_seeds = np.random.choice(seeds, K)
        dfb = df.loc[boot_seeds].groupby(['scenario', 'arm', 'time_year']).sum()
        # Very hacky to get date alignment
        combined = dfb.copy()
        for keys, val in combined.groupby(['scenario', 'arm']):
            combined.loc[keys, 'x'] = expected['x'].values
            combined.loc[keys, 'n'] = expected['n'].values
            combined.loc[keys, 't'] = expected.index.values
        #combined = pd.merge(dfb.reset_index(), expected.reset_index(), left_on='time_year', right_on='t')

        #combined['scaled_positive'] = combined['n_positive'] * combined['n']  / combined['n_tested']
        #alpha = combined['n_positive'] + 1
        #beta = combined['n_tested'] - combined['n_positive'] + 1
        p = (combined['n_positive'] + 1) / (combined['n_tested'] + 2)
        n = combined['n']
        #combined['scaled_positive'] = n * alpha / (alpha + beta)
        combined['scaled_positive'] = n * p
        combined.loc[combined['n_tested'] == 0, 'scaled_positive'] = np.nan

        combined['bi'] = bi
        dfs.append(combined)
    df = pd.concat(dfs).reset_index()

    def plot_observed(data, **kwargs):
        #df = expected.reset_index()
        #sns.scatterplot(data=df, x='t', y='x')
        plt.scatter(np.arange(len(expected)), expected['x'].values, color='orange', edgecolors='black', s=100)

        # Control
        plt.plot(3, 94, color='blue', marker='o', mec='black', ms=10)

    #ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    d = df[['scenario', 'arm', 't', 'scaled_positive', 'bi']]
    d = d.dropna(axis=0)
    g = sns.catplot(kind='strip', data=d, x='t', order=expected.index, y='scaled_positive', hue='arm', col='scenario', estimator=None, units='bi', size=1) # SD for speed, units='rand_seed'
    g.map_dataframe(plot_observed)
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'act3_acf.png'), dpi=600)




    # APLT PLOTS #######################################################
    aplt.plot_scenarios(results=df_result.get('TB'))