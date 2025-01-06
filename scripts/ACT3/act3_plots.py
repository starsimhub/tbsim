import os
import seaborn as sns
import pandas as pd
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt

# TEMP, move to aplt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import scipy.stats as sps



# if pltdir does not exist, create it
figs_path = os.path.join('results', 'ACT3', 'figs')
os.makedirs(figs_path, exist_ok=True)


def plot_scenarios(results, path=figs_path):
    # process results to make ready for plotting
    # maybe have a different function to process the data? 
    tb_results_agg = (
      results
        .melt(id_vars=['time_year', 'scenario', 'rand_seed'], 
                  value_vars=['on_treatment', 'prevalence', 'active_presymp', 'active_smpos', 'active_exptb'], 
                  var_name='state',
                  value_name='count')
        .drop(columns='rand_seed')
        .groupby(['time_year', 'scenario', 'state'])
        .agg(
              median=('count', 'median'), 
              q11=('count', lambda x: np.percentile(x, 11)),
              q89=('count', lambda x: np.percentile(x, 89))
              )
        .reset_index()
        )

    # plot the results
    sns.set_context("notebook")
    sns.set_style("whitegrid")
    g = sns.FacetGrid(tb_results_agg, col="state", 
                      margin_titles=True, aspect=1, 
                      sharey=False, hue="scenario", 
                      palette="tab10", col_wrap=3
                      )
    g.map_dataframe(sns.lineplot, x="time_year", y="median")
    g.map_dataframe(plt.fill_between, "time_year", "q11", "q89", alpha=0.3)

    g.set_axis_labels("Year", "Count")
    g.add_legend()

    g.tight_layout()

    sc.savefig('act3_scene.png', folder=path)

def plot_TODO(results, path=figs_path):
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