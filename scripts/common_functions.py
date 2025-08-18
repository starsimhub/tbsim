import starsim as ss 
import starsim_examples as sse
import sciris as sc
import tbsim as mtb 
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
import re

def make_tb(pars = None):
    """
    Set up the TB simulation with default parameters.
    """
    # Define simulation parameters
    if pars is None: 
        pars = dict(
            beta = ss.peryear(0.0025),
            init_prev = ss.bernoulli(p=0.25)
        )
    return mtb.TB(pars=pars)


def make_pop(pars = None, n_agents=500):
    """
    Set up the population with default parameters.
    """
    # Define population parameters
    if pars is None: 
        pars = dict(
            n_agents=n_agents,  # Number of agents in the population
            age_date=load_age_data('default'),  # Load age data from a CSV file or use default data
        )
        
    pop = ss.People(
        n_agents = n_agents,
        age_data = load_age_data,
    )
    return pop

def make_net(pars = None):
    """
    Set up the network with default parameters.
    """
    # Define network parameters
    if pars is None: 
        pars = dict(
            n_contacts=ss.poisson(lam=5),  # Number of contacts per agent
            dur=0,  # Duration of contacts
            # Add any other default parameters here
        )
    return sse.RandomNet(pars=pars)


def make_births(pars = None):
    """
    Set up the births demographic with default parameters.
    """
    # Define births parameters
    if pars is None: 
        pars = dict(
            birth_rate=20,  # Birth rate
            # Add any other default parameters here
        )
    return ss.Births(pars=pars)


def make_deaths(pars = None):
    """
    Set up the deaths demographic with default parameters.
    """
    # Define deaths parameters
    if pars is None: 
        pars = dict(
            death_rate=15,  # Death rate
            # Add any other default parameters here
        )
    return ss.Deaths(pars=pars)

def make_intervention(pars = None):
    """
    Set up the intervention with default parameters.
    """
    # Define intervention parameters
    if pars is None: 
        pars = dict(    )
    return ss.Intervention(pars=pars)  # Placeholder for the actual intervention class

def make_hiv(pars = None):
    """
    Set up the HIV intervention with default parameters.
    """
    # Define HIV parameters
    if pars is None: 
        pars = dict(
            # Add any default parameters here
        )
    return sse.HIV(pars=pars)  # Placeholder for the actual HIV class

def make_cnn(pars = None):
    """
    Set up the CNN intervention with default parameters.
    """
    # Define CNN parameters
    if pars is None: 
        pars = dict(
            # Add any default parameters here
        )
    return ss.CNN(pars=pars)  # Placeholder for the actual CNN class

def load_age_data(source='default', file_path=''):
    """
    Load population data from a CSV file or use default data.
    """
    if source == 'default':
        # Default population data
        # Gathered from WPP, https://population.un.org/wpp/Download/Standard/MostUsed/
        age_data = pd.DataFrame({ 
            'age': np.arange(0, 101, 5),
            'value': [5791, 4446, 3130, 2361, 2279, 2375, 2032, 1896, 1635, 1547, 1309, 1234, 927, 693, 460, 258, 116, 36, 5, 1, 0]  # 1960
        })
    elif source == 'json':
        if not file_path:
            raise ValueError("file_path must be provided when source is 'json'.")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file at {file_path} does not exist.")
        data = pd.read_json(file_path)
        age_data = pd.DataFrame(data)
    else:
        raise ValueError("Invalid source. Use 'default' or 'json'.")
    return age_data

def make_hiv_interventions(include:bool=True, pars=None):
    """
    Set up HIV interventions with default parameters.
    
    Parameters
    ----------
    include : bool, optional
        Whether to include HIV interventions. Default is True.
    pars : dict, optional
        Custom parameters for HIV interventions. If None, uses defaults.
        
    Returns
    -------
    list or None
        List containing HIV intervention object, or None if not included.
    """
    if not include: 
        return None
    if pars is None:
        pars=dict(
                mode='both',
                prevalence=0.30,            # Maintain 30 percent of the alive population infected
                percent_on_ART=0.50,        # Maintain 50 percent of the % infected population on ART
                min_age=15, max_age=60,     # Min and Max age of agents that can be hit with the intervention
                start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'),   # Intervention's start and stop dates
        )
        
    return [mtb.HivInterventions(pars=pars),]

def make_hiv_comorbidity(include:bool=True, hiv_pars=None):
    """
    Set up HIV comorbidity with default parameters.
    
    Parameters
    ----------
    include : bool, optional
        Whether to include HIV. Default is True.
    hiv_pars : dict, optional
        Custom parameters for HIV. If None, uses defaults.
        
    Returns
    -------
    tbsim.HIV or None
        HIV object, or None if not included.
    """
    if not include:
        return None
    if hiv_pars is None:
        hiv_pars = dict(
            init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
            init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
        )
    return mtb.HIV(pars=hiv_pars)

def make_tb_comorbidity(include:bool=True, tb_pars=None):
    """
    Set up TB comorbidity with default parameters.
    
    Parameters
    ----------
    include : bool, optional
        Whether to include TB. Default is True.
    tb_pars : dict, optional
        Custom parameters for TB. If None, uses defaults.
        
    Returns
    -------
    tbsim.TB or None
        TB object, or None if not included.
    """
    if not include:
        return None
    if tb_pars is None:
        pars = dict(
            beta=ss.peryear(0.0025),
            init_prev=ss.bernoulli(p=0.25),
            rel_sus_latentslow=0.1,
        )
    else:
        pars = tb_pars
    return mtb.TB(pars=pars)

def make_tb_hiv_connector(include:bool=True, pars=None):
    """
    Set up TB-HIV connector with default parameters.
    
    Parameters
    ----------
    include : bool, optional
        Whether to include TB-HIV connector. Default is True.
    pars : dict, optional
        Custom parameters for TB-HIV connector. If None, uses defaults.
        
    Returns
    -------
    tbsim.TB_HIV_Connector or None
        TB-HIV connector object, or None if not included.
    """
    if not include: 
        return None
    return mtb.TB_HIV_Connector(pars=pars)

def make_demographics(include:bool=False):
    """
    Set up demographics with default parameters.
    
    Parameters
    ----------
    include : bool, optional
        Whether to include demographics. Default is False.
        
    Returns
    -------
    list or None
        List containing birth and death objects, or None if not included.
    """
    if not include: 
        return None
    return [ss.Births(pars=dict(birth_rate=8.4)),
            ss.Deaths(pars=dict(death_rate=8.4)),]

def plot_results(flat_results, keywords=None, exclude=('15',), n_cols=5,
    dark=True, cmap='tab20', heightfold=3, style='default'):
    """
    Plot simulation results across multiple scenarios.
    
    Parameters
    ----------
    flat_results : dict[str, dict[str, Result]]
        Mapping scenario→{metric→Result(timevec, values)}.
    keywords : list[str], optional
        Only plot metrics containing any of these substrings.
    exclude : tuple[str], optional
        Skip metrics whose name contains any of these substrings.
    n_cols : int, optional
        Number of columns in the subplot grid.
    dark : bool, optional
        If True use greyish dark mode; otherwise default style.
    cmap : str, optional
        Name of the Matplotlib colormap to use.
    heightfold : int, optional
        Height factor for subplot rows.
    style : str, optional
        Matplotlib style to use.
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

def uncertainty_plot():
    """
    Create an uncertainty plot with example data.
    This is a demonstration function showing how to create uncertainty plots.
    """
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