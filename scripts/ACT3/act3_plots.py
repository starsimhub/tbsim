import os
import seaborn as sns
import pandas as pd
import numpy as np
import sciris as sc
import matplotlib.pyplot as plt


# if pltdir does not exist, create it
figs_path = os.path.join('results', 'ACT3', 'figs')
os.makedirs(figs_path, exist_ok=True)


def plot_scenarios(results, path=figs_path):
    # process results to make ready for plotting
    # maybe have a different function to process the data? 
    tb_results_agg = (
      results
        .drop(columns='time')
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

