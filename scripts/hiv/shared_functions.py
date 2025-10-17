"""
Shared Utility Functions for HIV and TB-HIV Scripts

This module provides reusable functions for building HIV/TB-HIV simulations, including
disease module creation, connector setup, demographic configuration, and specialized
plotting functions. It centralizes common functionality to avoid code duplication
across multiple HIV-related scripts.

Purpose:
--------
- Provide standard configurations for HIV and TB disease modules
- Centralize intervention and connector creation
- Standardize demographic components (births/deaths)
- Offer flexible plotting utilities for scenario comparison
- Enable consistent setup across different HIV/TB-HIV scripts

Functions:
----------
Disease and Network Setup:
- make_hiv_interventions: Create HIV prevalence/ART control interventions
- make_hiv: Create HIV disease module
- make_tb: Create TB disease module
- make_tb_hiv_connector: Create TB-HIV disease interaction connector
- make_demographics: Create births and deaths modules

Plotting:
- plot_results: Flexible multi-panel scenario comparison plots
- uncertanty_plot: Example uncertainty visualization (template)

Usage:
------
Import in HIV scripts:

    from shared_functions import (
        make_hiv, make_tb, make_hiv_interventions,
        make_tb_hiv_connector, make_demographics, plot_results
    )

Then use in simulation setup:

    tb = make_tb()
    hiv = make_hiv()
    interventions = make_hiv_interventions()
    connector = make_tb_hiv_connector()
    demographics = make_demographics(include=True)

Notes:
------
This module is specific to the scripts/hiv/ directory. For similar functionality
in other directories, see scripts/common_functions.py which provides broader
utility functions used across the entire project.
"""

import datetime
import starsim as ss 
import tbsim as mtb 
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import os

def make_hiv_interventions(include:bool=True, pars=None):
    """
    Create HIV interventions for prevalence and ART coverage control.
    
    Constructs HivInterventions module that maintains target HIV prevalence
    and ART coverage levels in the population. This is useful for modeling
    scenarios with controlled HIV epidemics or testing intervention strategies.
    
    Parameters
    ----------
    include : bool, default=True
        Whether to include HIV interventions. If False, returns None.
    pars : dict, optional
        Override intervention parameters. If None, uses sensible defaults.
        Expected keys:
        - mode : str, 'both' for prevalence + ART control
        - prevalence : float, target HIV prevalence (fraction)
        - percent_on_ART : float, target ART coverage (fraction of HIV+)
        - min_age, max_age : int, age range for intervention eligibility
        - start, stop : ss.date, intervention active period
    
    Returns
    -------
    list of mtb.HivInterventions or None
        List containing configured HivInterventions module, or None if include=False
        
    Default Parameters:
    -------------------
    - mode: 'both' (control both prevalence and ART)
    - prevalence: 0.30 (30% of population HIV+)
    - percent_on_ART: 0.50 (50% of HIV+ on ART)
    - min_age: 15, max_age: 60
    - start: 2000-01-01, stop: 2035-12-31
    
    Examples
    --------
    Default HIV interventions:
    
        >>> interventions = make_hiv_interventions()
    
    No interventions:
    
        >>> interventions = make_hiv_interventions(include=False)
    
    Custom intervention parameters:
    
        >>> pars = dict(
        ...     mode='both',
        ...     prevalence=0.20,
        ...     percent_on_ART=0.70,
        ...     min_age=18, max_age=65,
        ...     start=ss.date('2010-01-01'),
        ...     stop=ss.date('2030-12-31')
        ... )
        >>> interventions = make_hiv_interventions(pars=pars)
    
    Notes
    -----
    The intervention operates in 'both' mode by default, controlling both
    HIV prevalence and ART coverage simultaneously. This creates a controlled
    HIV epidemic useful for studying TB-HIV interactions under stable HIV conditions.
    """
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
    
def make_hiv(include:bool=True, hiv_pars=None):
    """
    Create an HIV disease module for simulation.
    
    Constructs an HIV disease module with configurable initial prevalence and
    ART coverage. By default, creates an HIV-free population suitable for
    testing interventions that introduce HIV.
    
    Parameters
    ----------
    include : bool, default=True
        Whether to include HIV module. Currently not used but reserved for
        future functionality (function always returns HIV module).
    hiv_pars : dict, optional
        Override HIV parameters. If None, uses defaults with zero initial
        prevalence and ART coverage.
        Expected keys:
        - init_prev : ss.Dist, initial HIV prevalence distribution
        - init_onart : ss.Dist, initial ART coverage distribution
    
    Returns
    -------
    mtb.HIV
        Configured HIV disease module ready to add to simulation
        
    Default Parameters:
    -------------------
    - init_prev: ss.bernoulli(p=0.00) - No initial HIV infections
    - init_onart: ss.bernoulli(p=0.00) - No initial ART coverage
    
    Examples
    --------
    HIV-free population (default):
    
        >>> hiv = make_hiv()
    
    Population with 20% HIV prevalence, 60% on ART:
    
        >>> hiv_pars = dict(
        ...     init_prev=ss.bernoulli(p=0.20),
        ...     init_onart=ss.bernoulli(p=0.60)
        ... )
        >>> hiv = make_hiv(hiv_pars=hiv_pars)
    
    Notes
    -----
    The default HIV-free population is useful when using HivInterventions
    to introduce and control HIV dynamically. For baseline HIV prevalence,
    provide custom hiv_pars.
    
    The init_onart parameter specifies the fraction of HIV+ individuals on ART,
    not the fraction of the total population.
    """
    if hiv_pars is None:
        hiv_pars = dict(
            init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
            init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
        )
    return mtb.HIV(pars=hiv_pars)

def make_tb(include:bool=True, tb_pars=None):
    """
    Create a TB disease module for simulation.
    
    Constructs a TB disease module with standard parameters suitable for
    HIV-TB co-infection modeling. Uses moderate transmission rate and
    initial prevalence by default.
    
    Parameters
    ----------
    include : bool, default=True
        Whether to include TB module. Currently not used but reserved for
        future functionality (function always returns TB module).
    tb_pars : dict, optional
        Override TB parameters. If None, uses sensible defaults.
        Expected keys:
        - beta : ss.Dist, transmission rate (per year)
        - init_prev : ss.Dist, initial TB prevalence distribution
        - rel_sus_latentslow : float, relative susceptibility for latent slow TB
    
    Returns
    -------
    mtb.TB
        Configured TB disease module ready to add to simulation
        
    Default Parameters:
    -------------------
    - beta: ss.peryear(0.025) - Transmission rate of 0.025 per year
    - init_prev: ss.bernoulli(p=0.25) - 25% initial TB prevalence
    - rel_sus_latentslow: 0.1 - 10% relative susceptibility for latent slow
    
    Examples
    --------
    TB with default parameters:
    
        >>> tb = make_tb()
    
    TB with custom parameters:
    
        >>> tb_pars = dict(
        ...     beta=ss.peryear(0.030),
        ...     init_prev=ss.bernoulli(p=0.15),
        ...     rel_sus_latentslow=0.15
        ... )
        >>> tb = make_tb(tb_pars=tb_pars)
    
    Notes
    -----
    The default parameters are calibrated for typical TB-HIV scenarios.
    The rel_sus_latentslow parameter controls reinfection susceptibility
    for individuals with latent slow TB, typically set low (0.1) to
    represent partial immunity from existing infection.
    """
    if tb_pars is None:
        pars = dict(
            beta=ss.peryear(0.025),
            init_prev=ss.bernoulli(p=0.25),
            rel_sus_latentslow=0.1,             # 10% reduction in susceptibility to latent slow TB
        )
    else:
        pars = tb_pars
    return mtb.TB(pars=pars)

def make_tb_hiv_connector(include:bool=True, pars=None):
    """
    Create a TB-HIV connector for disease interaction modeling.
    
    Constructs a TB_HIV_Connector module that modifies TB progression rates
    based on HIV and ART status. The connector implements epidemiological
    evidence that HIV increases TB activation risk, with effects varying
    by HIV disease stage.
    
    Parameters
    ----------
    include : bool, default=True
        Whether to include the connector. If False, returns None.
    pars : dict, optional
        Override connector parameters. If None, uses defaults.
        Possible keys (see mtb.TB_HIV_Connector documentation):
        - acute_multiplier : float, TB activation multiplier during acute HIV
        - latent_multiplier : float, TB activation multiplier during chronic HIV
        - aids_multiplier : float, TB activation multiplier during AIDS
        - art_effect : float, reduction in multipliers when on ART
    
    Returns
    -------
    mtb.TB_HIV_Connector or None
        Configured TB-HIV connector module, or None if include=False
        
    Default Behavior:
    -----------------
    Uses default TB_HIV_Connector parameters which typically:
    - Increase TB activation risk for HIV+ individuals
    - Vary multipliers by HIV disease stage
    - Reduce (but don't eliminate) HIV effects when on ART
    
    Examples
    --------
    Default TB-HIV connector:
    
        >>> connector = make_tb_hiv_connector()
    
    No connector (TB and HIV independent):
    
        >>> connector = make_tb_hiv_connector(include=False)
    
    Custom connector parameters:
    
        >>> pars = dict(
        ...     acute_multiplier=2.0,
        ...     latent_multiplier=3.0,
        ...     aids_multiplier=5.0
        ... )
        >>> connector = make_tb_hiv_connector(pars=pars)
    
    Notes
    -----
    The TB-HIV connector is essential for realistic TB-HIV co-infection
    modeling. Without it, HIV and TB are independent diseases. The connector
    implements well-established epidemiological relationships:
    - HIV increases TB activation from latent TB
    - Effect varies by HIV stage (acute, chronic, AIDS)
    - ART reduces but doesn't eliminate the increased TB risk
    """
    if not include: 
        return None
    return mtb.TB_HIV_Connector(pars=pars)  

def make_demographics(include:bool=False):
    """
    Create demographic modules (births and deaths) for simulation.
    
    Constructs balanced births and deaths modules to maintain stable
    population size. The default rate of 8.4 per 1000 per year is typical
    for many low-to-middle income countries.
    
    Parameters
    ----------
    include : bool, default=False
        Whether to include demographics. If False, returns None.
    
    Returns
    -------
    list or None
        List containing [Births, Deaths] modules, or None if include=False
        
    Default Configuration:
    ----------------------
    - Birth rate: 8.4 per 1000 per year
    - Death rate: 8.4 per 1000 per year
    - Balanced rates maintain stable population size
    
    Examples
    --------
    Include demographics:
    
        >>> demographics = make_demographics(include=True)
    
    No demographics (closed population):
    
        >>> demographics = make_demographics(include=False)
    
    Use in simulation:
    
        >>> sim = ss.Sim(
        ...     diseases=[tb, hiv],
        ...     demographics=make_demographics(include=True)
        ... )
    
    Notes
    -----
    Balanced birth and death rates (both 8.4) create a stable population
    with no net growth. This is useful for focusing on disease dynamics
    without confounding from population growth.
    
    For realistic demographic scenarios, consider using age-specific rates
    from scripts/data/ or adjusting rates to match specific countries.
    
    The rates are per 1000 per year, so 8.4 means 0.84% annual rate.
    """
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
    timestamp = sc.now(tostring=True)
    try:
        out = os.path.join(sc.thisdir(), f'scenarios_{timestamp}.png')
    except Exception:
        out = f'scenarios_{timestamp}.png'
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.show()


def uncertanty_plot():
    """
    Create an example uncertainty plot with confidence bands (demonstration only).
    
    This function demonstrates how to create plots showing uncertainty across
    multiple simulation runs. It generates synthetic data to illustrate the
    visualization approach but is not connected to actual simulation results.
    
    Plot Features:
    --------------
    - 2×2 subplot grid for multiple variables
    - Multiple runs per scenario with different line styles
    - Confidence bands (fill_between) showing uncertainty
    - Two scenario groups (blue and orange)
    - Grid lines for better readability
    
    Variables Demonstrated:
    -----------------------
    - n_positive_smpos: Smear-positive TB cases
    - n_positive_smneg: Smear-negative TB cases
    - n_positive_via_LS: TB cases via latent slow progression
    - n_positive_via_LF_dur: TB cases via latent fast progression
    
    Returns
    -------
    None
        Displays plot interactively but doesn't return anything
        
    Notes
    -----
    This is a TEMPLATE/EXAMPLE function showing uncertainty visualization
    techniques. It uses randomly generated data, not real simulation output.
    
    To use this approach with actual simulation data:
    1. Run multiple simulations with different random seeds
    2. Collect results for each variable
    3. Calculate mean and confidence intervals across runs
    4. Plot means as lines, confidence intervals as fill_between
    
    The function demonstrates best practices for uncertainty visualization:
    - Multiple runs shown with varied line styles
    - Shaded confidence bands for visual uncertainty
    - Clear variable labels and grid lines
    - Distinct colors for scenario groups
    """
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