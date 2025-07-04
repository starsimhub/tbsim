import datetime
import starsim as ss 
import tbsim as mtb 
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import os
import re

#- - - - - - MAKE INTERVENTIONS - - - - - -
def make_hiv_interventions(include:bool=True, pars=None):
    if not include: return None
    if pars is None:
        pars=dict(
                mode='both',
                prevalence=0.30,            # Maintain 30 percent of the alive population infected
                percent_on_ART=0.50,        # Maintain 50 percent of the % infected population on ART
                min_age=15, max_age=60,     # Min and Max age of agents that can be hit with the intervention
                start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'),   # Intervention's start and stop dates
        )
        
    return [mtb.HivInterventions(pars=pars),]
    
# - - - - - - MAKE HIV - - - - - -
def make_hiv(include:bool=True, hiv_pars=None):
    if hiv_pars is None:
        hiv_pars = dict(
            init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
            init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
        )
    return mtb.HIV(pars=hiv_pars)

#- - - - - - - - MAKE TB - - - - - - - -
def make_tb(include:bool=True, tb_pars=None):
    if tb_pars is None:
        pars = dict(
            beta=ss.beta(0.1),
            init_prev=ss.bernoulli(p=0.25),
            rel_sus_latentslow=0.1,
        )
    return mtb.TB(pars=pars)

# - - - - - - - MAKE TB-HIV CONNECTOR - - - - - -
def make_tb_hiv_connector(include:bool=True, pars=None):
    if not include: 
        return None
    return mtb.TB_HIV_Connector(pars=pars)  

# - - - - - -  MAKE DEMOGRAPHICS - - - - - -
def make_demographics(include:bool=False):
    if not include: return None
    return [ss.Births(pars=dict(birth_rate=8.4)),
            ss.Deaths(pars=dict(death_rate=8.4)),]


def plot_results( flat_results, keywords=None, exclude=('15',), n_cols=5,
    dark=True, cmap='tab20', heightfold=3, style='default'):
    """
    Parameters
    ----------
    flat_results : dict[str, dict[str, Result]]  -  Mapping scenario→{metric→Result(timevec, values)}.
    keywords :  list[str], optional - Only plot metrics containing any of these substrings.
    exclude :   tuple[str], optional - Skip metrics whose name contains any of these substrings.
    n_cols :    int, optional -  Number of columns in the subplot grid.
    dark :      If True use greyish dark mode; otherwise default style.
    cmap :      str, optional -  Name of the Matplotlib colormap to use.
    """
    try:
        plt.style.use(style)
    except Exception:
        print(f"Warning: {style} style not found. Using default style.")
        plt.style.use('default')

    # collect & filter metric names
    all_metrics = {m for flat in flat_results.values() for m in flat}
    if keywords is not None:
        all_metrics = {m for m in all_metrics if any(kw in m for kw in keywords)}
    metrics = sorted(m for m in all_metrics if not any(ex in m for ex in exclude))
    if not metrics:
        print("No metrics to plot.")
        return

    # plot layout and colors
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, heightfold*n_rows))
    axs = np.array(axs).flatten()

    if dark:
        fig.patch.set_facecolor('lightgray')  # figure background
        for ax in axs:
            ax.set_facecolor('darkgray')
    palette = plt.cm.get_cmap(cmap, len(flat_results))

    # plot each metric
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, (scen, flat) in enumerate(flat_results.items()):
            if metric in flat:
                r = flat[metric]
                ax.plot(r.timevec, r.values, lw=0.8, label=scen, color=palette(j))
        ax.set_title(metric, fontsize=10)
        vmax = max(flat.get(metric, r).values)
        if vmax < 1.001:
            ax.set_ylim(0, max(0.5, vmax))
            ax.set_ylabel('%')
        else:
            ax.set_ylabel('Value')
        ax.set_xlabel('Time')

        # grid lines
        ax.grid(True, color='white' if dark else 'gray', alpha=0.3)
        leg = ax.legend(fontsize=6 if len(flat_results)>5 else 7)
        if leg: leg.get_frame().set_alpha(0.3)

    # remove unused axes
    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    plt.tight_layout()
    # save figure
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')
    try:
        out = os.path.join(sc.thisdir(), f'scenarios_{timestamp}.png')
    except Exception:
        out = f'scenarios_{timestamp}.png'
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.show()


def uncertanty_plot():
    import matplotlib.pyplot as plt
    import numpy as np
    from datetime import datetime

    # Example data generation
    np.random.seed(0)
    timesteps = np.arange(8)  # Number of timepoints

    # Simulate multiple runs for two groups (blue and orange)
    n_runs = 5

    # Create fake datasets
    blue_data = [np.random.rand(len(timesteps)) + np.linspace(1, 2, len(timesteps)) for _ in range(n_runs)]
    orange_data = [np.random.rand(len(timesteps)) + np.linspace(0.5, 1.5, len(timesteps)) for _ in range(n_runs)]

    # Organize data
    variables = ['n_positive_smpos', 'n_positive_smneg', 'n_positive_via_LS', 'n_positive_via_LF_dur']

    # Setup plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    linestyles = ['-.', ':', '--']  # Various styles

    for idx, var in enumerate(variables):
        ax = axs[idx // 2, idx % 2]

        # Plot blue group
        for run_idx, run in enumerate(blue_data):
            ax.plot(timesteps, run + np.random.normal(0, 0.2, size=len(timesteps)), 
                    linestyle=linestyles[run_idx % len(linestyles)], 
                    color='steelblue', alpha=0.9)
            ax.fill_between(timesteps,
                            run - 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            run + 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            color='steelblue', alpha=0.3)

        # Plot orange group
        for run_idx, run in enumerate(orange_data):
            ax.plot(timesteps, run + np.random.normal(0, 0.2, size=len(timesteps)),
                    linestyle=linestyles[run_idx % len(linestyles)], 
                    color='darkorange', alpha=0.9)
            ax.fill_between(timesteps,
                            run - 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            run + 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            color='darkorange', alpha=0.3)

        ax.set_title(var, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(timesteps[0], timesteps[-1])
        ax.set_ylim(0, None)

    plt.tight_layout()
    plt.show()

