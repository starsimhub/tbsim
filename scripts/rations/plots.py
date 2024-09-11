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

def plot_rations(resdir, df):
    first_year = int(df['Year'].iloc[0])
    assert df['Year'].iloc[0] == first_year
    df['date'] = pd.to_datetime(365 * (df['Year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    #months = sc.date(['2019-08-31', '2019-09-30', '2019-10-31', '2019-11-30', '2019-12-31', '2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30', '2020-12-31', '2021-01-31'])
    #enrolled = np.array([105, 215, 244, 284, 248, 263, 265, 184, 63, 69, 122, 104, 54, 107, 112, 115, 186, 60]).cumsum()
    #axv[0].plot(months, enrolled, label='RATIONS Trial')
    g = sns.relplot(kind='line', data=df, x='date', col='Channel', y='Values', hue='Scenario', style='Arm', errorbar='sd', facet_kws={'sharey': False, 'sharex': True}) # Hoping errorbar ci makes things faster

    for ax in g.axes.flat:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

    sc.savefig('rations.png', folder=resdir)
    plt.close(g.figure)

    fig, axv = plt.subplots(1,2, sharey=True, figsize=(10,5))

    dfg = df.groupby(['Channel', 'Arm', 'Scenario', 'Seed'])['Values'] \
        .sum() \
        .loc[['Incident Cases', 'Person Years']] \
        .unstack('Channel')

    dfg['Incidence Rate per 1,000'] = 1000 * dfg['Incident Cases'] / dfg['Person Years']

    # Calibration
    for ax, arm, data in zip(axv, ['Control', 'Intervention'], [90/9557, 62/12208]):
        sns.barplot(data=dfg.loc[arm], y='Scenario', x='Incidence Rate per 1,000', hue='Scenario', ax=ax)
        ax.axvline(x=1000 * data, color='r', lw=2, label='RATIONS data')
        ax.set_title(arm)

    fig.tight_layout()

    sc.savefig('calib.png', folder=resdir)
    plt.close(fig)

    return

def plot_epi(resdir, df):
    first_year = int(df['year'].iloc[0])
    assert df['year'].iloc[0] == first_year
    df['date'] = pd.to_datetime(365 * (df['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    d = pd.melt(df.drop(['rand_seed', 'year'], axis=1), id_vars=['date', 'arm'], var_name='channel', value_name='Value')
    g = sns.relplot(data=d, kind='line', x='date', hue='arm', col='channel', y='Value', palette='Set1',
        facet_kws={'sharey':False}, col_wrap=3, lw=2, errorbar='sd')

    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.set_xlabels('Date')
    for ax in g.axes.flat:
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
    sc.savefig('epi.png', folder=resdir)
    plt.close(g.figure)
    return

def plot_hh(resdir, df):
    dfm = df.reset_index().drop('rand_seed', axis=1).melt(id_vars='HH Size', var_name='Year', value_name='Frequency')
    dfm['Per Cent'] = (df['rand_seed'].max()+1) * dfm.groupby('Year')['Frequency'].transform(lambda x: x / x.sum())
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

    sc.savefig('hhsizedist.png', folder=resdir)
    plt.close(g.figure)
    return

def stackedbar(data, color, **kwargs):
    
    #categories = ['STANDARD_OR_ABOVE', 'SLIGHTLY_BELOW_STANDARD', 'MARGINAL', 'UNSATISFACTORY']
    Mcats = mtb.MacroNutrients.__dict__['_member_names_']
    data['Macro'] = pd.Categorical(data['Macro'], categories=Mcats)

    mcats = mtb.MicroNutrients.__dict__['_member_names_']
    data['Micro'] = pd.Categorical(data['Micro'], categories=mcats)

    df = data.set_index(['Arm', 'Macro', 'Micro'])
    #base = pd.DataFrame(np.zeros((len(Mcats), len(mcats))), index=pd.Index(Mcats, name='Macro'), columns=mcats)
    
    
    # --------------------------  PUBLICATION DATA ------------------------------------
    # Use data with exact values from the publication
    vitamin = [29.2, 30.3, 28.1, 12.4]
    control = [21.1, 28.9, 38.9, 11.1]

    vit_data = pd.DataFrame({
        2017: [29.2, 30.3, 28.1, 12.4],
        2018: [71.2, 18.8, 5.0, 5.0],
        2021: [50.0, 37.2, 10.2, 2.6]
    }, index=pd.Index(Mcats, name='Macro')) #.melt(var_name='Year', value_name='Per Cent')
    ctl_data = pd.DataFrame({
        2017: [21.1, 28.9, 38.9, 11.1],
        2018: [70.9, 22.8, 3.8, 2.5],
        2021: [71.6, 21.6, 4.1, 2.7]
    }, index=pd.Index(Mcats, name='Macro')) # .melt(var_name='Year', value_name='Per Cent')
    # ----------------------------------------------------------------------------------
    
    
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

def plot_nut(resdir, df):
    # Sum over reps
    dfs = df.drop('rand_seed', axis=1).groupby(['Arm', 'Macro', 'Micro']).sum()
    dfm = dfs.reset_index().melt(id_vars=['Arm', 'Micro', 'Macro'], var_name='Year', value_name='Frequency')
    g = sns.FacetGrid(data=dfm, col='Year', height=4, aspect=1.5) # , row='Arm'
    g.map_dataframe(stackedbar)
    plt.subplots_adjust(bottom=0.3)

    plt.legend(loc='upper center', bbox_to_anchor=(0,-0.2), ncol=2)
    sc.savefig('nutrition.png', folder=resdir)
    plt.close(g.figure)
    return

def plot_active_infections(resdir, data):
    df = data.set_index(['arm', 'rand_seed', 'year'])[['cum_active_infections']].sort_index()
    trial_start = 1942
    df.index.get_level_values('year')
    years = df.index.get_level_values('year').unique()
    trial_start = years[np.argmax(years > trial_start)]
    df = df.loc[slice(None), slice(None), trial_start:]

    df['Incident Cases'] = df.groupby(['arm', 'rand_seed'])['cum_active_infections'].transform(lambda x: x - x.iloc[0]) 

    g = sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='arm', errorbar='se')
    sns.lineplot(data=df.reset_index(), x='year', y='Incident Cases', hue='arm', estimator=None, units='rand_seed', alpha=0.1, lw=0.1, legend=False)

    g.set_xlabel('Year')
    #locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    #formatter = mdates.ConciseDateFormatter(locator)
    sc.savefig('incidence.png', folder=resdir)
    plt.close(g.figure)

    return

