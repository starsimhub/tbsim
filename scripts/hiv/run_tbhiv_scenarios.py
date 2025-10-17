"""
TB-HIV Scenarios with Variable HIV Intervention Timing and Coverage

This script demonstrates TB-HIV co-infection modeling with HIV interventions that
vary in timing and coverage. It shows how different intervention strategies affect
TB-HIV epidemic trajectories.

Purpose:
--------
- Compare baseline (no intervention) vs. intervention scenarios
- Test effects of early vs. late intervention timing
- Evaluate low vs. high coverage interventions
- Demonstrate how HIV control affects TB burden
- Provide framework for intervention optimization

Scenarios:
----------
1. **Baseline**: No HIV interventions, natural epidemic dynamics
2. **Early Low Coverage**: 1990-2000, 10% prevalence, 10% ART coverage, ages 15-49
3. **Mid Coverage Mid Years**: 2000-2010, 20% prevalence, 40% ART coverage, ages 20-60

Key Insights:
-------------
- Early intervention can prevent epidemic growth
- Higher ART coverage reduces TB activation from latent TB
- Intervention timing matters for long-term outcomes
- Age targeting affects population-level impact

Components:
-----------
- TB disease module (25% initial prevalence, beta=0.025)
- HIV disease module (10% initial prevalence, 50% on ART initially)
- TB-HIV Connector for disease interactions
- HIV Interventions with variable timing and coverage
- Random network (Poisson(2) contacts)

Usage:
------
    python scripts/hiv/run_tbhiv_scenarios.py

Output:
-------
- Multi-panel plot comparing all scenarios
- Saved as 'tbhiv_scenarios.png' with timestamp
- Shows TB and HIV metrics over 56 years (1980-2035)
- Light gray background for better contrast

Notes:
------
This script is useful for:
- Policy analysis and intervention planning
- Understanding intervention timing effects
- Cost-effectiveness comparisons
- Optimal coverage level determination
"""

import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import tbsim as mtb
import starsim as ss


def build_tbhiv_sim(simpars=None, tbpars=None, hivinv_pars=None) -> ss.Sim:
    """
    Build a TB-HIV simulation with configurable intervention parameters.
    
    Creates a TB-HIV co-infection model with flexible HIV intervention configuration.
    The simulation is smaller and faster than other TB-HIV scripts (500 agents vs.
    1000+) to enable rapid scenario comparison.
    
    Parameters
    ----------
    simpars : dict, optional
        Override simulation parameters (dt, start, stop, rand_seed, verbose)
    tbpars : dict, optional
        Override TB disease parameters (beta, init_prev, rel_sus_latentslow)
        Currently not used but reserved for future extensions
    hivinv_pars : dict, optional
        HIV intervention parameters. If None, no interventions are applied.
        Expected keys:
        - mode : str, 'both' for prevalence + ART control
        - prevalence : float, target HIV prevalence fraction
        - percent_on_ART : float, target ART coverage fraction
        - min_age, max_age : int, age range for intervention
        - start, stop : ss.date, intervention period
    
    Returns
    -------
    ss.Sim
        Configured TB-HIV simulation ready to run
        
    Simulation Configuration:
    -------------------------
    - Period: 1980-2035 (56 years)
    - Population: 500 agents (smaller for speed)
    - Time step: 7 days (weekly)
    - TB: beta=0.025/year, 25% initial prevalence, rel_sus_latentslow=0.1
    - HIV: 10% initial prevalence, 50% on ART initially
    - Network: Random with Poisson(2) contacts
    - Connector: TB-HIV interactions
    
    Examples
    --------
    Baseline simulation without interventions:
    
        >>> sim = build_tbhiv_sim()
        >>> sim.run()
    
    Simulation with low coverage intervention:
    
        >>> intv_pars = dict(
        ...     mode='both',
        ...     prevalence=0.10,
        ...     percent_on_ART=0.10,
        ...     min_age=15, max_age=49,
        ...     start=ss.date('1990-01-01'),
        ...     stop=ss.date('2000-12-31')
        ... )
        >>> sim = build_tbhiv_sim(hivinv_pars=intv_pars)
        >>> sim.run()
    
    Notes
    -----
    The smaller population (500 vs. 1000+ agents) provides faster execution
    for scenario comparison while maintaining reasonable statistical properties.
    Random seed (123) ensures reproducibility across runs.
    """

    # --- Simulation Parameters ---
    default_simpars = dict(
        dt=ss.days(7),
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
        verbose=0,
    )
    if simpars:
        default_simpars.update(simpars)

    # --- Population ---
    n_agents = 500
    people = ss.People(n_agents=n_agents)

    # --- TB Model ---
    pars = dict(
        beta=ss.peryear(0.025),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
    )
    tb = mtb.TB(pars=pars)

    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.10),
        init_onart=ss.bernoulli(p=0.50),
    )
    hiv = mtb.HIV(pars=hiv_pars)
    
    # --- Network ---
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    # --- Connector ---
    connector = mtb.TB_HIV_Connector()
    
    # --- HIV Intervention ---
    hiv_intervention = None
    if hivinv_pars is not None:
        hiv_intervention = mtb.HivInterventions(pars=hivinv_pars)

    # --- Assemble Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        interventions=None if hiv_intervention is None else [hiv_intervention],
        networks=network,
        connectors=[connector],
        pars=default_simpars,
    )

    return sim


def run_scenarios():
    """
    Run all TB-HIV intervention scenarios and collect results.
    
    Executes three scenarios with different HIV intervention strategies and collects
    flattened results for comparison plotting. Each scenario tests a different approach
    to HIV control and its effects on TB-HIV co-infection dynamics.
    
    Scenarios:
    ----------
    1. **baseline**: No HIV interventions
       - Natural epidemic dynamics
       - HIV and TB follow intrinsic transmission patterns
       - Provides reference for intervention effects
    
    2. **early_low_coverage**: Early intervention (1990-2000), low coverage
       - 10% prevalence target, 10% ART coverage
       - Ages 15-49 (reproductive and economically active)
       - Tests early intervention with limited resources
    
    3. **mid_coverage_mid_years**: Mid-period intervention (2000-2010), moderate coverage
       - 20% prevalence target, 40% ART coverage
       - Ages 20-60 (broader age range)
       - Tests delayed but more intensive intervention
    
    Returns
    -------
    dict
        Dictionary mapping scenario names to flattened results:
        {scenario_name: {metric_name: Result(timevec, values), ...}, ...}
        
    Examples
    --------
    Run scenarios and analyze results:
    
        >>> results = run_scenarios()
        >>> for name, metrics in results.items():
        ...     print(f"{name}: {list(metrics.keys())}")
    
    Plot specific metrics:
    
        >>> results = run_scenarios()
        >>> plot_results(results, keywords=['hiv', 'tb'])
    
    Notes
    -----
    All scenarios share the same baseline conditions:
    - Same population size and network structure
    - Same initial TB and HIV prevalence
    - Same random seed for comparability
    - Only HIV intervention parameters vary
    
    The flattened results format enables easy comparison across scenarios
    and flexible plotting of specific metrics.
    """
    scenarios = {
        'baseline': None,
            
        'early_low_coverage': dict(
            mode='both',
            prevalence=0.10,
            percent_on_ART=0.10,
            min_age=15,
            max_age=49,
            start=ss.date('1990-01-01'),
            stop=ss.date('2000-12-31'),
        ),
        'mid_coverage_mid_years': dict(
            mode='both',
            prevalence=0.20,
            percent_on_ART=0.40,
            min_age=20,
            max_age=60,
            start=ss.date('2000-01-01'),
            stop=ss.date('2010-12-31'),
        ),
    }
    flat_results = {}

    for name, hivinv_pars in scenarios.items():
        print(f'Running scenario: {name}')
        sim = build_tbhiv_sim(hivinv_pars=hivinv_pars)
        sim.run()
        flat_results[name] = sim.results.flatten()

    return flat_results

def plot_results(flat_results, keywords=None, exclude=['15']):
    """
    Plot comparison of TB-HIV simulation results across multiple scenarios.
    
    Creates a multi-panel figure showing selected metrics across all scenarios.
    Each panel shows time series for one metric with all scenarios overlaid.
    The function automatically identifies relevant metrics and creates an
    organized grid layout.
    
    Parameters
    ----------
    flat_results : dict
        Nested dictionary: {scenario_name: {metric_name: Result(timevec, values)}}
        Output from run_scenarios() or similar functions
    keywords : list of str, optional
        Only plot metrics containing any of these substrings.
        If None, plots all metrics.
        Example: ['hiv', 'tb', 'art'] to focus on key outcomes
    exclude : list of str, default=['15']
        Skip metrics containing any of these substrings.
        Default excludes age-15 specific metrics which can clutter plots.
    
    Output Files
    ------------
    tbhiv_scenarios.png : High-resolution plot saved to script directory
    
    Plot Features:
    --------------
    - Grid layout: 5 columns, as many rows as needed
    - Color-coded scenarios (tab10 colormap)
    - Light gray background (#f0f0f0) for better contrast
    - Auto-scaled y-axis: percentage scale (<1) or absolute scale (≥1)
    - Grid lines for easier reading
    - Small legends (reduced font for crowded plots)
    - Transparent legend backgrounds
    
    Examples
    --------
    Plot all metrics:
    
        >>> results = run_scenarios()
        >>> plot_results(results)
    
    Plot only HIV-related metrics:
    
        >>> plot_results(results, keywords=['hiv'])
    
    Plot TB and HIV, exclude age-specific:
    
        >>> plot_results(results, keywords=['tb', 'hiv'], exclude=['15', '65'])
    
    Notes
    -----
    The function assumes Result objects have:
    - timevec: Time vector for x-axis
    - values: Metric values for y-axis
    
    Metrics with max values < 1 are assumed to be fractions/percentages
    and are plotted with percentage y-axis labels and appropriate scale.
    """
    # Automatically identify all unique metrics across all scenarios
    metrics = []
    if keywords is None:
        metrics = sorted({key for flat in flat_results.values() for key in flat.keys()}, reverse=True)
        
    else:
        metrics = sorted({
            k for flat in flat_results.values() for k in flat
            if any(kw in k for kw in keywords) 
        })
        # Exclude specified metrics
    
    metrics = [m for m in metrics if not any(excl in m for excl in exclude)]
        
    n_metrics = len(metrics)
    if n_metrics > 0:
        # If there are more than 5 metrics, use a grid of 5 columns
        n_cols = 5
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*2))
        axs = axs.flatten()
        
    cmap = plt.cm.get_cmap('tab10', len(flat_results))

    for i, metric in enumerate(metrics):
        ax = axs[i] if n_metrics > 1 else axs
        for j, (scenario, flat) in enumerate(flat_results.items()):
            if metric in flat:
                result = flat[metric]
                ax.plot(result.timevec, result.values, label=scenario, color=cmap(j))
        ax.set_title(metric)
        if max(result.values) < 1:
            # identify the max value of result.values
            v = max(result.values)
            ax.set_ylim(0, max(0.5, v)) 
            ax.set_ylabel('%')
        else:
            ax.set_ylabel('Value')
        ax.set_xlabel('Time')
        
        ax.grid(True)
        ax.legend()
        # reduce the legend font size if there are many scenarios
        if len(flat_results) > 5:
            leg = ax.legend(loc='upper right', fontsize=5)
        else:
            leg = ax.legend(loc='upper right', fontsize=6)
            
        # Handle legend positioning for crowded plots
        if leg:
            leg.get_frame().set_alpha(0.5)
    plt.tight_layout()
    
    # add an option to change the background color of the plot for better visibility
    for ax in axs:
        ax.set_facecolor('#f0f0f0')  # Light gray background for better contrast

    dirname = sc.thisdir()
    plt.savefig(f'{dirname}/tbhiv_scenarios.png', dpi=300)
    # Show the plot
    plt.show()

if __name__ == '__main__':
    flat_results = run_scenarios()
    plot_results(flat_results)