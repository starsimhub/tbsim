"""
Plotting scripts
"""

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import sciris as sc
import tbsim.config as cfg
import matplotlib.cm as cm
import numpy as np
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
