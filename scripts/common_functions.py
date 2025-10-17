"""
Common Functions for TB Simulation Scripts

This module provides a centralized collection of reusable functions for setting up
and running TB simulations. It includes factory functions for creating simulation
components (diseases, populations, networks, interventions) with sensible defaults,
as well as plotting and analysis utilities.

Main Components:
----------------
- Disease modules: TB, HIV, and comorbidities
- Population setup: Age distributions and demographic initialization
- Network setup: Random and household contact networks
- Interventions: HIV interventions, demographics (births/deaths)
- Connectors: TB-HIV disease interaction modeling
- Plotting: Comparative scenario plotting with multiple metrics

Usage:
------
Import this module in simulation scripts to access reusable components:

    from common_functions import make_tb, make_pop, plot_results
    
    # Create TB disease with defaults
    tb = make_tb()
    
    # Create population
    pop = make_pop(n_agents=1000)
    
    # Plot scenario comparison
    plot_results(results_dict, keywords=['prevalence', 'incidence'])
"""

import starsim as ss 
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
    Set up the TB disease module with default parameters.
    
    Creates a TB disease module suitable for use in Starsim simulations.
    If no parameters are provided, uses sensible defaults for transmission
    rate and initial prevalence.
    
    Parameters
    ----------
    pars : dict, optional
        TB-specific parameters. If None, uses defaults:
        - beta: 0.0025 per year (transmission rate)
        - init_prev: 25% Bernoulli (initial prevalence)
    
    Returns
    -------
    mtb.TB
        Configured TB disease module ready to add to a simulation
        
    Examples
    --------
    Create TB with default parameters:
    
        >>> tb = make_tb()
    
    Create TB with custom transmission rate:
    
        >>> tb = make_tb(pars={'beta': ss.peryear(0.005), 'init_prev': ss.bernoulli(p=0.1)})
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
    Set up the population with default parameters and age distribution.
    
    Creates a population (People object) for use in Starsim simulations.
    The population can be initialized with a default age distribution based
    on UN World Population Prospects data from 1960.
    
    Parameters
    ----------
    pars : dict, optional
        Population-specific parameters. Currently not used but reserved for
        future extensions. Age data is loaded from load_age_data() function.
    n_agents : int, default=500
        Number of agents (individuals) in the population
    
    Returns
    -------
    ss.People
        Configured population object ready to add to a simulation
        
    Examples
    --------
    Create population with default size:
    
        >>> pop = make_pop()
    
    Create larger population:
    
        >>> pop = make_pop(n_agents=10000)
    
    Notes
    -----
    The age distribution is loaded from the load_age_data() function which
    provides UN WPP 1960 data by default.
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
    Set up a random contact network with default parameters.
    
    Creates a RandomNet network where agents form random contacts with each other.
    The number of contacts per agent follows a Poisson distribution, and contacts
    can have optional durations.
    
    Parameters
    ----------
    pars : dict, optional
        Network-specific parameters. If None, uses defaults:
        - n_contacts: Poisson(λ=5) - average 5 contacts per agent per timestep
        - dur: 0 - instantaneous contacts (end after one timestep)
    
    Returns
    -------
    ss.RandomNet
        Configured random network ready to add to a simulation
        
    Examples
    --------
    Create network with default parameters:
    
        >>> net = make_net()
    
    Create network with more contacts:
    
        >>> net = make_net(pars={'n_contacts': ss.poisson(lam=10), 'dur': 5})
    
    Notes
    -----
    The dur=0 setting means contacts are instantaneous and reset each timestep,
    which is appropriate for modeling transient social interactions.
    """
    # Define network parameters
    if pars is None: 
        pars = dict(
            n_contacts=ss.poisson(lam=5),  # Number of contacts per agent
            dur=0,  # Duration of contacts
            # Add any other default parameters here
        )
    return ss.RandomNet(pars=pars)


def make_births(pars = None):
    """
    Set up the births demographic process with default parameters.
    
    Creates a Births module that adds new agents to the population over time.
    The birth rate controls how many new individuals enter the population.
    
    Parameters
    ----------
    pars : dict, optional
        Birth process parameters. If None, uses defaults:
        - birth_rate: 20 per 1000 population per year
    
    Returns
    -------
    ss.Births
        Configured births module ready to add to a simulation
        
    Examples
    --------
    Create births with default rate:
    
        >>> births = make_births()
    
    Create births with custom rate:
    
        >>> births = make_births(pars={'birth_rate': 25})
    
    Notes
    -----
    The birth_rate is typically specified per 1000 population per year.
    A rate of 20 means 20 births per 1000 people annually.
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
    Set up the deaths demographic process with default parameters.
    
    Creates a Deaths module that removes agents from the population over time.
    The death rate controls background mortality independent of disease-specific
    mortality.
    
    Parameters
    ----------
    pars : dict, optional
        Death process parameters. If None, uses defaults:
        - death_rate: 15 per 1000 population per year
    
    Returns
    -------
    ss.Deaths
        Configured deaths module ready to add to a simulation
        
    Examples
    --------
    Create deaths with default rate:
    
        >>> deaths = make_deaths()
    
    Create deaths with custom rate:
    
        >>> deaths = make_deaths(pars={'death_rate': 10})
    
    Notes
    -----
    The death_rate is typically specified per 1000 population per year.
    A rate of 15 means 15 deaths per 1000 people annually (1.5% annual mortality).
    This is background mortality; TB-specific mortality is handled by the TB module.
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
    return ss.HIV(pars=pars)  # Placeholder for the actual HIV class

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
    Load population age distribution data from various sources.
    
    This function provides age distribution data for population initialization.
    It can either return default UN WPP 1960 data or load custom data from a JSON file.
    
    Parameters
    ----------
    source : str, default='default'
        Source of age data:
        - 'default': Use built-in UN WPP 1960 age distribution
        - 'json': Load from a JSON file (requires file_path)
    file_path : str, optional
        Path to JSON file when source='json'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'age': Age groups (0, 5, 10, ..., 100)
        - 'value': Population counts in each age group
    
    Raises
    ------
    ValueError
        If source is 'json' but file_path is not provided
    FileNotFoundError
        If the specified file_path does not exist
    ValueError
        If source is neither 'default' nor 'json'
        
    Examples
    --------
    Load default age distribution:
    
        >>> age_data = load_age_data()
    
    Load custom age distribution from file:
    
        >>> age_data = load_age_data(source='json', file_path='my_age_data.json')
    
    Notes
    -----
    The default age distribution is from UN World Population Prospects (WPP) 1960 data,
    representing a demographic structure typical of developing countries at that time.
    Source: https://population.un.org/wpp/Download/Standard/MostUsed/
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
    Create an uncertainty plot with example data and confidence bands.
    
    This demonstration function shows how to create publication-quality uncertainty
    plots for TB simulation results. It visualizes multiple simulation runs with
    confidence bands, different line styles, and proper statistical representation.
    
    The plot includes:
    - Multiple simulation runs shown as individual lines
    - Confidence bands (shaded regions) representing uncertainty
    - Different groups (e.g., different scenarios) in different colors
    - Grid layout for multiple metrics
    
    Returns
    -------
    None
        Displays the plot directly
        
    Examples
    --------
    Generate example uncertainty plot:
    
        >>> uncertainty_plot()
    
    Notes
    -----
    This is a demonstration function using synthetic data. In practice, you would
    replace the fake data generation with actual simulation results from multiple
    runs with different random seeds.
    
    The plot shows 4 example TB metrics:
    - n_positive_smpos: Smear-positive TB cases
    - n_positive_smneg: Smear-negative TB cases
    - n_positive_via_LS: Positive cases via latent slow progression
    - n_positive_via_LF_dur: Positive cases via latent fast progression
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