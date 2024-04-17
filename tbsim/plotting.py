"""
Plotting scripts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

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
    g.figure.savefig(os.path.join(figdir, f'result.png'), bbox_inches='tight', dpi=300)
    plt.close(g.figure)
    return
