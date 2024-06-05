import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import os 
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False
default_n_rand_seeds = [1000, 10][debug]
run_scens = ['Base', 'NoSecular', 'SecularMicro', 'RelSus5', 'LSProgAlt']

if debug:
    run_scens = [r for i, r in enumerate(run_scens) if i in [1, 2]]


def compute_rel_LS_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE)         & (micro == mtb.MicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD)   & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == mtb.MacroNutrients.MARGINAL)                  & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY)            & (micro == mtb.MicroNutrients.DEFICIENT)] = 3.0
    return ret


def compute_rel_LS_prog_alternate(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE)         & (micro == mtb.MicroNutrients.DEFICIENT)] = 2
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD)   & (micro == mtb.MicroNutrients.DEFICIENT)] = 5
    ret[(macro == mtb.MacroNutrients.MARGINAL)                  & (micro == mtb.MicroNutrients.DEFICIENT)] = 10
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY)            & (micro == mtb.MicroNutrients.DEFICIENT)] = 20
    return ret

scens = {
    #'COMBINED': {
    #    'p_control': 0.5,
    #    'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
    #    'relsus_microdeficient': 1,
    #    'beta': 0.75,
    #    'n_hhs': 194,
    #},
    'Base CONTROL': {
        'p_control': 1,
        'vitamin_year_rate': None,
        'relsus_microdeficient': 1,
        'beta': 0.12,
        'n_hhs': 194/2,
    },
    'Base VITAMIN': {
        'p_control': 0,
        'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
        'relsus_microdeficient': 1,
        'beta': 0.12,
        'n_hhs': 194/2,
        'ref': 'Base CONTROL',
    },
    'NoSecular CONTROL': {
        'p_control': 1,
        'vitamin_year_rate': None,
        'relsus_microdeficient': 1,
        'beta': 0.11,
        'n_hhs': 194/2,
        'secular_trend': False,
    },
    'NoSecular VITAMIN': {
        'p_control': 0,
        'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
        'relsus_microdeficient': 1,
        'beta': 0.11,
        'n_hhs': 194/2,
        'secular_trend': False,
        'ref': 'NoSecular CONTROL',
    },
    'SecularMicro CONTROL': {
        'p_control': 1,
        'vitamin_year_rate': None,
        'relsus_microdeficient': 1,
        'beta': 0.13,
        'n_hhs': 194/2,
        'p_new_micro': 0.5,
    },
    'SecularMicro VITAMIN': {
        'p_control': 0,
        'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
        'relsus_microdeficient': 1,
        'beta': 0.13,
        'n_hhs': 194/2,
        'p_new_micro': 0.5,
        'ref': 'SecularMicro CONTROL',
    },
    'RelSus5 CONTROL': {
        'p_control': 1,
        'vitamin_year_rate': None,
        'relsus_microdeficient': 5,
        'beta': 0.04,
        'n_hhs': 194/2,
    },
    'RelSus5 VITAMIN': {
        'p_control': 0,
        'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
        'relsus_microdeficient': 5,
        'beta': 0.04,
        'n_hhs': 194/2,
        'ref': 'RelSus5 CONTROL',
    },

    'LSProgAlt CONTROL': {
        'p_control': 1,
        'vitamin_year_rate': None,
        'relsus_microdeficient': 1,
        'beta': 0.08,
        'n_hhs': 194/2,
        'rel_LS_prog_func': compute_rel_LS_prog_alternate,
    },
    'LSProgAlt VITAMIN': {
        'p_control': 0,
        'vitamin_year_rate': [(1942, 10.0), (1943, 3.0)],
        'relsus_microdeficient': 1,
        'beta': 0.08,
        'n_hhs': 194/2,
        'rel_LS_prog_func': compute_rel_LS_prog_alternate,
        'ref': 'LSProgAlt CONTROL',
    },
}

def run_harlem(scen, rand_seed=0, idx=0, n_hhs=194, p_control=0.5, vitamin_year_rate=None, relsus_microdeficient=1,
               beta=0.1, secular_trend=True, p_new_micro=0.0, rel_LS_prog_func=compute_rel_LS_prog, **kwargs):
    # vitamin_year_rate is a list of tuples like [(1942, 10.0), (1943, 3.0)] or None if CONTROL
    lbl = f'sim {idx}: {scen} with rand_seed={rand_seed}, p_control={p_control}, vitamin_year_rate={vitamin_year_rate}'
    print(f'Starting {lbl}')

    np.random.seed(rand_seed)

    # --------- Harlem ----------
    harlem_pars = dict(n_hhs=n_hhs, p_control=p_control)
    harlem = mtb.Harlem(harlem_pars)

    # --------- People ----------
    pop = harlem.people()

    # -------- Network ---------
    harlemnet = harlem.net()

    # Network parameters
    matnet = ss.MaternalNet() # To track newborn --> household
    nets = [harlemnet, matnet] # randnet, 

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = dict(harlem=beta, maternal=0.0),
        init_prev = 0, # Infections seeded by Harlem class
        rate_LS_to_presym = 3e-5, # Slow down LS-->Presym as this is now the rate for healthy individuals

        # Relative transmissibility by TB state
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # Add demographics
    dems = [
        mtb.HarlemPregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]

    # -------- Connector -------
    cn_pars = dict(
        rel_LS_prog_func = rel_LS_prog_func,
        relsus_microdeficient = relsus_microdeficient # Increased susceptibilty of those with micronutrient deficiency (could make more complex function like LS_prog)
    )
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------
    m = mtb.MicroNutrients
    M = mtb.MacroNutrients
    intvs = []
    if secular_trend:
        # Rates hand picked to match Appendix Table 7
        nc0 = mtb.NutritionChange(year=[1942, 1944], rate=[1.25, 0], from_state=M.UNSATISFACTORY, to_state=M.MARGINAL,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        nc1 = mtb.NutritionChange(year=[1942, 1944], rate=[2.50, 0], from_state=M.MARGINAL, to_state=M.SLIGHTLY_BELOW_STANDARD,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        nc2 = mtb.NutritionChange(year=[1942, 1944], rate=[1.25, 0], from_state=M.SLIGHTLY_BELOW_STANDARD, to_state=M.STANDARD_OR_ABOVE,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        intvs = [nc0, nc1, nc2]

    vs = []
    if vitamin_year_rate is not None:
        years, rates = zip(*vitamin_year_rate)
        vs = mtb.VitaminSupplementation(year=years, rate=rates) # Need coverage, V1 vs V2
        intvs += [vs]


    # -------- Analyzer -------
    azs = [
        mtb.HarlemAnalyzer(),
        mtb.HHAnalyzer(),
        mtb.NutritionAnalyzer(),
    ]

    # -------- Simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        #start = 1935, # Start early to burn-in
        start = 1941,
        end = 1947,
        rand_seed = rand_seed,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=nets, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn, interventions=intvs, analyzers=azs)
    #sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps
    sim.pars.verbose = 0 # Don't print

    sim.initialize()

    harlem.set_states(sim)
    seed_uids = harlem.choose_seed_infections()
    tb = sim.diseases['tb']
    tb.set_prognoses(seed_uids)

    # After set_prognoses, seed_uids will be in latent slow or fast
    # Change to ACTIVE_PRESYMP and set time of activation to current time step
    tb.state[seed_uids] = mtb.TBS.ACTIVE_PRESYMP
    tb.ti_active[seed_uids] = sim.ti

    # In Harlem, 83% or 84% have SmPos active infections, let's fix that now
    desired_n_active = 0.835 * len(seed_uids)
    cur_n_active = np.count_nonzero(tb.active_tb_state[seed_uids]==mtb.TBS.ACTIVE_SMPOS)
    add_n_active = desired_n_active - cur_n_active
    non_smpos_uids = seed_uids[tb.active_tb_state[seed_uids]!=mtb.TBS.ACTIVE_SMPOS]
    p = add_n_active / ( len(seed_uids) - cur_n_active)
    change_to_smpos = np.random.rand(len(non_smpos_uids)) < p
    tb.active_tb_state[non_smpos_uids[change_to_smpos]] = mtb.TBS.ACTIVE_SMPOS

    sim.run() # Actually run the sim

    df = sim.analyzers['harlemanalyzer'].df
    # Could us df.attrs here?
    df['p_control'] = p_control
    df['rand_seed'] = rand_seed
    df['Scenario'] = scen

    dfhh = sim.analyzers['hhanalyzer'].df
    dfhh['rand_seed'] = rand_seed
    dfhh['Scenario'] = scen

    dfn = sim.analyzers['nutritionanalyzer'].df
    dfn['rand_seed'] = rand_seed
    dfn['Scenario'] = scen

    #print(f'Finishing {lbl}')

    return df, dfhh, dfn


def run_scenarios(n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []

    for skey, scen in scens.items():
        if skey.split(' ')[0] not in run_scens: # Limit scenarios
            continue
        for rs in range(n_seeds):
            cfgs.append({'scen': skey,'rand_seed':rs, 'idx':len(cfgs)} | scen) # Merge dicts with pipe operators
    T = sc.tic()
    results += sc.parallelize(run_harlem, iterkwargs=cfgs, die=False, serial=debug) # , kwargs={'n_hhs':194/2}
    epi, hh, nut = zip(*results)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    df_epi = pd.concat(epi)
    df_epi.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"result_{cfg.FILE_POSTFIX}.csv"))

    df_hh = pd.concat(hh)
    df_hh.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"hhsizes_{cfg.FILE_POSTFIX}.csv"))

    df_nut = pd.concat(nut)
    df_nut.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"nutrition_{cfg.FILE_POSTFIX}.csv"))

    return df_epi, df_hh, df_nut

def plot_epi(df):
    # Sum over arms
    dfs = df.drop(['arm', 'p_control'], axis=1).groupby(['rand_seed', 'year', 'Scenario']).sum().reset_index()

    first_year = int(dfs['year'].iloc[0])
    assert dfs['year'].iloc[0] == first_year
    dfs['date'] = pd.to_datetime(365 * (dfs['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    d = pd.melt(dfs.drop(['rand_seed', 'year'], axis=1), id_vars=['date', 'Scenario'], var_name='channel', value_name='Value')
    g = sns.relplot(data=d, kind='line', x='date', hue='Scenario', col='channel', y='Value', palette='Set1',
        facet_kws={'sharey':False}, col_wrap=3, lw=2, errorbar='sd') # Can change errorbar to None for bootstrapped bars, but it is slow

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


def plot_hh(df):
    dfm = df.reset_index().drop(['rand_seed', 'Scenario'], axis=1).melt(id_vars='HH Size', var_name='Year', value_name='Frequency')
    # By year, aggregating over Scenario and Arm
    dfm['Per Cent'] = df['Scenario'].nunique() * df['rand_seed'].nunique() * dfm.groupby('Year')['Frequency'].transform(lambda x: x / x.sum())
    dfm['Year'] = dfm['Year'].astype(str)
    #g = sns.barplot(dfm, x='HH Size', y='Frequency', hue='Year')
    g = sns.FacetGrid(data=dfm, height=4, aspect=1.5)
    g.map_dataframe(sns.barplot, x='HH Size', y='Per Cent', hue='Year', palette='Set1')

    def hh_data(data, color, **kwargs):
        data = np.array([3, 17, 24, 20, 13, 9, 7, 4, 3]) / 100
        ax = plt.gca()
        ax.scatter(range(len(data)), data, 150, marker='+', lw=2, color='black', label='Downes data (1942)')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
        return
    g.map_dataframe(hh_data)

    g.add_legend()

    sc.savefig(f"hhsizedist_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)
    return


def stackedbar(data, color, **kwargs):

    # Mean over scenarios
    dfs = data.drop(['Scenario'], axis=1).groupby(['Arm', 'Micro', 'Macro', 'Year']).mean().reset_index()
    
    #categories = ['STANDARD_OR_ABOVE', 'SLIGHTLY_BELOW_STANDARD', 'MARGINAL', 'UNSATISFACTORY']
    Mcats = mtb.MacroNutrients.__dict__['_member_names_']
    dfs['Macro'] = pd.Categorical(dfs['Macro'], categories=Mcats)

    mcats = mtb.MicroNutrients.__dict__['_member_names_']
    dfs['Micro'] = pd.Categorical(dfs['Micro'], categories=mcats)

    df = dfs.set_index(['Arm', 'Macro', 'Micro'])
    #base = pd.DataFrame(np.zeros((len(Mcats), len(mcats))), index=pd.Index(Mcats, name='Macro'), columns=mcats)

    vit_data = pd.DataFrame({
        1942: [29.2, 30.3, 28.1, 12.4],
        1944: [71.2, 18.8, 5.0, 5.0],
        1947: [50.0, 37.2, 10.2, 2.6]
    }, index=pd.Index(Mcats, name='Macro')) #.melt(var_name='Year', value_name='Per Cent')
    ctl_data = pd.DataFrame({
        1942: [21.1, 28.9, 38.9, 11.1],
        1944: [70.9, 22.8, 3.8, 2.5],
        1947: [71.6, 21.6, 4.1, 2.7]
    }, index=pd.Index(Mcats, name='Macro')) # .melt(var_name='Year', value_name='Per Cent')

    year = int(float(df.iloc[0]['Year']))
    vit_data = vit_data[year] / 100
    ctl_data = ctl_data[year] / 100

    Nvit = df.loc['VITAMIN']['Frequency'].sum()
    Nctl = df.loc['CONTROL']['Frequency'].sum()

    # Calculate the micronutrient status
    vit = df.loc['VITAMIN'].drop('Year', axis=1).unstack('Micro')['Frequency'].fillna(0).stack().astype(int) / Nvit
    ctl = df.loc['CONTROL'].drop('Year', axis=1).unstack('Micro')['Frequency'].fillna(0).stack().astype(int) / Nctl

    vitamin_deficient = vit.loc[slice(None), 'DEFICIENT'].values
    vitamin_sufficient = vit.loc[slice(None), 'NORMAL'].values
    control_deficient = ctl.loc[slice(None), 'DEFICIENT'].values
    control_sufficient = ctl.loc[slice(None), 'NORMAL'].values

    # Create the figure and axis
    ax = plt.gca()

    # Plotting the bars with updated colors and order
    bar_width = 0.35
    index = np.arange(len(Mcats))

    bars1 = ax.barh(index, vitamin_deficient, bar_width, color='steelblue', edgecolor='steelblue', label='Vitamin Group, Micronutrient-Deficient')
    bars2 = ax.barh(index, vitamin_sufficient, bar_width, color='lightsteelblue', edgecolor='steelblue', left=vitamin_deficient, label='Vitamin Group, Micronutrient-Sufficient')
    ax.scatter(vit_data.values, index, 100, c='black', marker='+', lw=2)#, label='Downes Appendix Table 7')
    bars3 = ax.barh(index + bar_width, control_deficient, bar_width, color='goldenrod', edgecolor='goldenrod', label='Control Group, Micronutrient-Deficient')
    bars4 = ax.barh(index + bar_width, control_sufficient, bar_width, color='wheat', edgecolor='goldenrod', left=control_deficient, label='Control Group, Micronutrient-Sufficient')
    ax.scatter(ctl_data.values, index+bar_width, 100, c='black', marker='+', lw=2)

    # Add labels and title
    ax.set_xlabel('Per Cent')
    ax.set_ylabel('Categories')
    #ax.set_title('Distribution of Families According to Food Habits (1942)')
    ax.set_yticks(index + bar_width / 2)
    ax.set_yticklabels(Mcats)

    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))

    # Combine legends for better readability
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')
    return

def plot_nut(df):
    # Sum over reps
    # 'Scenario' was 'Arm'
    dfs = df.drop('rand_seed', axis=1).groupby(['Scenario', 'Arm', 'Macro', 'Micro']).sum()
    dfm = dfs.reset_index().melt(id_vars=['Scenario', 'Arm', 'Micro', 'Macro'], var_name='Year', value_name='Frequency')
    g = sns.FacetGrid(data=dfm, col='Year', height=4, aspect=1.5) # , row='Arm'
    g.map_dataframe(stackedbar)
    plt.subplots_adjust(bottom=0.3)

    plt.legend(loc='upper center', bbox_to_anchor=(0,-0.2), ncol=2)
    sc.savefig(f"nutrition_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)
    return

def diff(data, baseline, counterfactual, label, channel='cum_active_infections'):

    final_year = data['year'].max()

    bl = data.loc[(data['Scenario'] == baseline) & (data['year']==final_year)]
    cf = data.loc[(data['Scenario'] == counterfactual) & (data['year']==final_year)]

    # Sum over arm for this analysis
    blm = bl.groupby(['rand_seed', 'year'])[channel].sum()
    blm.name = baseline
    cfm = cf.groupby(['rand_seed', 'year'])[channel].sum()
    cfm.name = counterfactual

    # Concat
    df = pd.concat([cfm, blm], axis=1)
    df[label] = df[counterfactual] - df[baseline]
    df.index = df.index.droplevel('year')

    return df

def plot_diff(data, channel='cum_active_infections'):
    scenarios = data['Scenario'].unique()
    diffs = []
    for scen in scenarios:
        if 'ref' in scens[scen]:
            label = scen.split(' ')[0]
            d = diff(data, scens[scen]['ref'], scen, label=label, channel=channel)
            diffs.append(d[label])

    df = pd.concat(diffs, axis=1)
    df = df * -1
    dfm = pd.melt(df, var_name='Scenario', value_name='Active infections averted')

    g = sns.displot(kind='kde', data=dfm, hue='Scenario', x='Active infections averted', rug=True, fill=True, bw_adjust=2)
    #g = sns.displot(kind='hist', data=dfm, hue='Scenario', x='Active infections averted', stat='density', common_norm=False, multiple='dodge', discrete=True)
    sc.savefig(f'diff_{channel}_{cfg.FILE_POSTFIX}.png', folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return g.figure

def plot_calib(data, channel='cum_active_infections'):
    scenarios = data['Scenario'].unique()
    calibs = []
    for scen in scenarios:
        if 'ref' not in scens[scen]:
            label = scen.split(' ')[0]
            
            trial_start = 1942
            years = data['year'].unique()
            trial_start = years[np.argmax(years >= trial_start)]

            final_year = data['year'].max()
            df = data.loc[(data['Scenario'] == scen) & (data['year'].isin([trial_start, final_year]))]

            # Sum over arm for this analysis
            dfs = df.groupby(['rand_seed', 'year'])[[channel]].sum().reset_index()
            dfm = pd.pivot(data=dfs, index='rand_seed', columns='year', values=channel)
            dfm[label] = dfm[final_year] - dfm[trial_start]
            dfm = dfm[[label]]

            #dfs.index = dfs.index.droplevel('year')
            #dfs.name = label
            calibs.append(dfm)

    # Concat
    df = pd.concat(calibs, axis=1)
    dfm = pd.melt(df, var_name='Scenario', value_name=channel)

    g = sns.displot(kind='kde', data=dfm, hue='Scenario', x=channel, rug=True, fill=True)
    sc.savefig(f'calib_{channel}_{cfg.FILE_POSTFIX}.png', folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return g.figure

def plot_active_infections(data):
    df = data.groupby(['Scenario', 'rand_seed', 'year'])[['cum_active_infections']].sum().sort_index() # Sum over arms
    trial_start = 1942
    df.index.get_level_values('year')
    years = df.index.get_level_values('year').unique()
    trial_start = years[np.argmax(years >= trial_start)]
    df = df.loc[slice(None), slice(None), trial_start:]

    df['Incident Cases'] = df.groupby(['Scenario', 'rand_seed'])['cum_active_infections'].transform(lambda x: x - x.iloc[0]) 

    g = sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scenario', errorbar='ci', palette='Paired')
    sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scenario', errorbar='se', palette='Paired', legend=False)
    #sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scenario', estimator=None, units='rand_seed', alpha=0.1, lw=0.1, legend=False)
    g.set_xlabel('Year')

    sc.savefig(f"incidence_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return


if __name__ == '__main__':
    if False:
        df_epi, df_hh, df_nut = run_scenarios()
    else:
        resdir = os.path.join('figs', 'TB')
        df_epi = pd.read_csv(os.path.join(resdir, 'result_06-04_21-05-49.csv'))

    '''
    if debug:
        sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
        sim.diseases['tb'].plot()
        sim.plot()
        sim.analyzers['harlemanalyzer'].plot()
        plt.show()
    '''

    plot_calib(df_epi, channel='cum_active_infections')
    plot_diff(df_epi, channel='cum_active_infections')
    plot_active_infections(df_epi)
    plot_epi(df_epi)
    plot_hh(df_hh)
    plot_nut(df_nut)

    print(f"Results directory {cfg.RESULTS_DIRECTORY}\nThis run: {cfg.FILE_POSTFIX}")
    print('Done')
