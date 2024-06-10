"""
Plotting scripts
"""

import tbsim as mtb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import sciris as sc
import tbsim.config as cfg
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import datetime as dt

def plot_epi(dfs):
    # Sum over arms
    #dfs = df.drop(['arm', 'p_control'], axis=1).groupby(['rand_seed', 'year', 'Scen', 'Arm']).sum().reset_index()

    first_year = int(dfs['year'].iloc[0])
    assert dfs['year'].iloc[0] == first_year
    dfs['date'] = pd.to_datetime(365 * (dfs['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    d = pd.melt(dfs.drop(['rand_seed', 'year', 'p_control', 'Scenario'], axis=1), id_vars=['date', 'Scen', 'arm'], var_name='channel', value_name='Value')
    g = sns.relplot(data=d, kind='line', x='date', hue='Scen', style='arm', col='channel', y='Value', palette='tab10',
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
    g.map_dataframe(sns.barplot, x='HH Size', y='Per Cent', hue='Year', palette='tab10')

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

def plot_nut(df, scenarios=None, lbl=None):
    # Sum over reps
    # 'Scenario' was 'Arm'
    dfs = df.drop('rand_seed', axis=1).groupby(['Scenario', 'Arm', 'Macro', 'Micro']).sum()
    dfm = dfs.reset_index().melt(id_vars=['Scenario', 'Arm', 'Micro', 'Macro'], var_name='Year', value_name='Frequency')

    if scenarios is not None:
        dfm = dfm.set_index('Scenario').loc[scenarios].reset_index()

    g = sns.FacetGrid(data=dfm, col='Year', height=4, aspect=1.5, palette='tab10') # , row='Arm'
    g.map_dataframe(stackedbar)
    plt.subplots_adjust(bottom=0.3)

    plt.legend(loc='upper center', bbox_to_anchor=(0,-0.2), ncol=2)
    fn = f'nutrition_{cfg.FILE_POSTFIX}'
    if lbl is not None:
        fn += f'_{lbl}'
    sc.savefig(f'{fn}.png', folder=cfg.RESULTS_DIRECTORY)
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

def plot_diff(data, scens, channel='cum_active_infections'):
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

    #g = sns.displot(kind='kde', data=dfm, hue='Scenario', x='Active infections averted', rug=True, fill=True, bw_adjust=2, palette='tab10')
    #g = sns.displot(kind='hist', data=dfm, hue='Scenario', x='Active infections averted', stat='density', common_norm=False, multiple='dodge', discrete=True)

    g = sns.FacetGrid(data=dfm, hue='Scenario', palette='tab10', height=5)
    g.map_dataframe(sns.kdeplot, x='Active infections averted', fill=True) # rug=True, , bw_adjust=2
    def mean_line(data, color, ch, **kwargs):
        plt.axvline(data[ch].mean(), color=color, lw=1, ls='-')
    g.map_dataframe(mean_line, ch='Active infections averted')
    g.add_legend()
    sc.savefig(f'diff_{channel}_{cfg.FILE_POSTFIX}.png', folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    g = sns.boxenplot(data=dfm, y='Scenario', x='Active infections averted', orient='h')
    g.figure.tight_layout()
    sc.savefig(f'diffbox_{channel}_{cfg.FILE_POSTFIX}.png', folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return g.figure

def plot_calib(data, scens, channel='cum_active_infections'):
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

            calibs.append(dfm)

    # Concat
    df = pd.concat(calibs, axis=1)
    dfm = pd.melt(df, var_name='Scenario', value_name=channel)

    g = sns.FacetGrid(data=dfm, hue='Scenario', palette='tab10', height=5)
    g.map_dataframe(sns.kdeplot, x=channel, fill=True) # rug=True, 

    def mean_line(data, color, ch, **kwargs):
        plt.axvline(data[ch].mean(), color=color, lw=1, ls='-')
    g.map_dataframe(mean_line, ch=channel)
    g.add_legend()

    sc.savefig(f'calib_{channel}_{cfg.FILE_POSTFIX}.png', folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return g.figure

def plot_active_infections(data):
    #df = data.groupby(['Scen', 'arm', 'rand_seed', 'year'])[['cum_active_infections']].sum().sort_index() # Sum over arms
    df = data.set_index(['Scen', 'arm', 'rand_seed', 'year'])[['cum_active_infections']].sort_index()

    trial_start = 1942
    df.index.get_level_values('year')
    years = df.index.get_level_values('year').unique()
    trial_start = years[np.argmax(years >= trial_start)]
    df = df.loc[slice(None), slice(None), slice(None), trial_start:]

    df['Incident Cases'] = df.groupby(['Scen', 'arm', 'rand_seed'])['cum_active_infections'].transform(lambda x: x - x.iloc[0]) 

    g = sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scen', style='arm', errorbar=('se', 2), palette='tab20')
    #sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scenario', errorbar=('sd', 2), palette='tab20', legend=False)
    #sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='Scenario', estimator=None, units='rand_seed', alpha=0.1, lw=0.1, legend=False)
    g.set_xlabel('Year')

    sc.savefig(f"incidence_{cfg.FILE_POSTFIX}.pdf", folder=cfg.RESULTS_DIRECTORY)
    sc.savefig(f"incidence_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)

    return

## Old code below here

def plot_scenarios(df):
    g = sns.relplot(kind='line', data=df, x='year', y='Deaths', hue='xLS', 
                palette='Set1', estimator=None, units='rand_seed', lw=0.5)
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    #g.figure.suptitle('MultiRNG' if ms else 'SingleRNG')
    #g.figure.subplots_adjust(top=0.88)
    g.set_xlabels('Date')
    for ax in g.axes.flat:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    sc.savefig(f"result_{cfg.FILE_POSTFIX}.png", folder=cfg.RESULTS_DIRECTORY)
    plt.close(g.figure)
    return

def plot_sim(sim, title = ' Sim Results Plot ' ):
    plt.rcParams['font.size'] = 8 
    title = ' - '.join([str(disease) for disease in sim.diseases]) + title
    with sc.options.with_style('fancy'):
        flat = sc.flattendict(sim.results, sep=': ')
        fig, axs = sc.getrowscols(len(flat), make=True)
        fig.set_size_inches(12, 8)
        colors = cm.rainbow(np.linspace(0, 1, len(flat))) 
        for ax, (k, v), color in zip(axs.flatten(), flat.items(), colors):
            ax.plot(sim.yearvec, v, linewidth=2, color = color)  # Increase line width
            ax.set_title(k)  
            ax.set_xlabel('Year')
            ax.grid(True)  # Add grid
            ax.spines['top'].set_visible(False)  # Remove top border
            ax.spines['right'].set_visible(False)  # Remove right border
    fig.suptitle(title, fontsize=13)
    return fig
