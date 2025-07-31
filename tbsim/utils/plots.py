import os
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from typing import Dict, List, Tuple
import starsim as ss
import datetime
import sys


def plot_results(flat_results=None, results=None, sim=None, age_bins=None, keywords=None, exclude=('None',), n_cols=5,
                 dark=True, cmap='tab20', heightfold=2, 
                 style='default', savefig=True, outdir=None, metric_filter=None):
    """
    Visualize simulation outputs from multiple scenarios in a structured grid layout.

    Args:
        flat_results (dict, optional): Nested dictionary of the form:
            {
                'Scenario A': {'metric1': Result, 'metric2': Result, ...},
                'Scenario B': {'metric1': Result, 'metric2': Result, ...},
                ...
            }
            Each Result must have `timevec` and `values` attributes representing
            time series data for a given metric.

        results (object, optional): A simulation results object (e.g., sim.results) that has a .flatten() method.
            If provided along with age_bins, will generate age-stratified series.
            
        sim (object, optional): A simulation object (e.g., sim) that contains people and disease data.
            Required for age stratification when using age_bins parameter.
            
        age_bins (list, optional): Age bin boundaries for stratifying results. 
            Example: [0, 5, 15, 30, 50, 200] creates bins [0-5), [5-15), [15-30), [30-50), [50-200).
            If provided with results and sim, will generate age-specific metrics for each bin.
            
        keywords (list[str], optional): If provided, only plot metrics containing at least one of these substrings.
        exclude (tuple[str], optional): Substrings of metric names to skip. Default is ('15',).
        n_cols (int, optional): Number of columns in the plot grid. Default is 5.
        dark (bool, optional): If True (default), uses a gray-on-dark theme for improved contrast.
        cmap (str, optional): Name of a matplotlib colormap (e.g., 'viridis', 'tab10'). Default is 'tab20'.
        heightfold (int, optional): Height multiplier per row of subplots. Default is 3.
        style (str, optional): Matplotlib style to apply. Defaults to 'default'. Falls back to 'default' if not found.
        savefig (bool, optional): If True (default), saves the figure as a PNG file with a timestamped filename.
        outdir (str, optional): Directory to save the figure. If None, saves in the current script's directory under 'results'.
        metric_filter (list[str], optional): List of metric names to plot. If provided, only these metrics will be plotted.
    
    Returns:
        None: The figure is displayed and also saved as a PNG with a timestamped filename.

    Workflow:
        1. If results and age_bins are provided, generates age-stratified series from the results object.
        2. Collects all metric names across scenarios.
        3. Filters metrics based on `keywords` and `exclude`.
        4. Lays out subplots based on the number of metrics and specified `n_cols`.
        5. Iterates over each metric and plots it across all scenarios.
        6. Adjusts appearance (background, style, gridlines, labels).
        7. Saves the figure as 'scenarios_<timestamp>.png'.

    Example:
        >>> # Using flat_results (existing functionality)
        >>> results = {
        ...     'BCG': {
        ...         'incidence': Result(timevec=[0, 1, 2], values=[0.1, 0.2, 0.3]),
        ...         'mortality': Result(timevec=[0, 1, 2], values=[0.05, 0.07, 0.1])
        ...     },
        ...     'TPT': {
        ...         'incidence': Result(timevec=[0, 1, 2], values=[0.08, 0.15, 0.25]),
        ...         'mortality': Result(timevec=[0, 1, 2], values=[0.03, 0.05, 0.08])
        ...     }
        ... }
        >>> plot_results(flat_results=results, keywords=['incidence'], n_cols=2, dark=False, cmap='viridis')
        
        >>> # Using results object with age bins (new functionality)
        >>> sim.run()
        >>> plot_results(results=sim.results, sim=sim, age_bins=[0, 5, 15, 30, 50, 200], 
        ...              keywords=['incidence'], n_cols=3)

    NOTE:
    -----
    This plotting utility can work with either:
    1. Pre-flattened results (flat_results parameter) - existing functionality
    2. Results object with age bins (results + age_bins parameters) - new functionality
    
    When using results + age_bins, the function will:
    - Flatten the results object
    - Generate age-stratified versions of metrics for each age bin
    - Create scenario names based on age bin ranges
    
    FLATTENING RESULTS:
    ---------------
    This line:
        >>> results['Any_Name'] = sim.results.flatten()       <-- The name will be used for the series name
         
    Converts the simulation's time series outputs into a flat dictionary or DataFrame.
    Makes results easier to compare across scenarios (e.g., plotting incidence over time).
    The results dictionary now maps scenario names to their flattened outputs:
    {
        'BCG': <results>,
        'TPT': <results>,
        ...
    }
    
    AGE STRATIFICATION:
    ---------------
    When age_bins is provided, the function creates age-specific metrics:
    - For each metric in the results, creates age-stratified versions
    - Age bin [0, 5, 15, 30, 50, 200] creates scenarios: "0-5", "5-15", "15-30", "30-50", "50-200"
    - Each scenario contains metrics filtered for agents in that age range
    
    """
    
    # Handle results object with age bins
    if results is not None and age_bins is not None:
        if flat_results is not None:
            print("Warning: Both flat_results and results provided. Using results with age_bins.")
        
        if sim is None:
            raise ValueError("Simulation object (sim) is required for age stratification. Please provide the sim parameter.")
        
        # Flatten the results object
        base_results = results.flatten()
        
        # Generate age-stratified results
        flat_results = _generate_age_stratified_results(sim, base_results, age_bins)
    
    elif results is not None and age_bins is None:
        # Just flatten the results object
        if flat_results is not None:
            print("Warning: Both flat_results and results provided. Using flat_results.")
        else:
            flat_results = {'Results': results.flatten()}
    
    elif flat_results is None:
        raise ValueError("Either flat_results or results must be provided.")

    try:
        plt.style.use(style)
    except Exception:
        print(f"Warning: {style} style not found. Using default style.")
        plt.style.use('default')

    # collect & filter metric names
    all_metrics = {m for flat in flat_results.values() for m in flat}
    if keywords is not None:
        all_metrics = {m for m in all_metrics if any(kw in m for kw in keywords)}
    if metric_filter is not None:
        metrics = metric_filter
    else:
        metrics = sorted(m for m in all_metrics if not any(ex in m for ex in exclude))
    if not metrics:
        print("No metrics to plot.")
        return

    # plot layout and colors
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, heightfold*n_rows))
    axs = np.array(axs).flatten()

    # Calculate global X-axis range across all metrics
    all_x_min = None
    all_x_max = None
    for flat in flat_results.values():
        for metric in metrics:
            if metric in flat:
                r = flat[metric]
                if all_x_min is None:
                    all_x_min = min(r.timevec)
                    all_x_max = max(r.timevec)
                else:
                    all_x_min = min(all_x_min, min(r.timevec))
                    all_x_max = max(all_x_max, max(r.timevec))

    if dark:
        fig.patch.set_facecolor('#606060')  # medium dark gray background
        for ax in axs:
            ax.set_facecolor('#404040')  # dark gray for axes
    palette = plt.cm.get_cmap(cmap, len(flat_results))

    # plot each metric
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, (scen, flat) in enumerate(flat_results.items()):
            if metric in flat:
                r = flat[metric]
                ax.plot(r.timevec, r.values, lw=0.8, label=scen, color=palette(j))
        ax.set_title(metric, fontsize=10, color='white' if dark else 'black')
        vmax = max(flat.get(metric, r).values)
        if vmax < 1.001:
            ax.set_ylim(0, max(0.5, vmax))
            ax.set_ylabel('%', color='white' if dark else 'black')
        else:
            ax.set_ylabel('Value', color='white' if dark else 'black')
        ax.set_xlabel('Time', color='white' if dark else 'black')
        ax.tick_params(axis='both', colors='white' if dark else 'black')
        # Set consistent X-axis range for all plots
        ax.set_xlim(all_x_min, all_x_max)

        # grid lines
        ax.grid(True, color='white' if dark else 'gray', alpha=0.3)
        leg = ax.legend(fontsize=6 if len(flat_results)>5 else 7)
        if leg: leg.get_frame().set_alpha(0.3)

    # remove unused axes
    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    plt.tight_layout(pad=2.0)

    if savefig:
        out = out_to(outdir)
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved figure to {out}")
    plt.show()


def _generate_age_stratified_results(sim, base_results, age_bins):
    """
    Generate age-stratified results from a simulation object.
    
    Args:
        sim: Simulation object with people and disease data
        base_results: Flattened base results dictionary
        age_bins: List of age bin boundaries
        
    Returns:
        dict: Age-stratified results dictionary
    """
    stratified_results = {}
    
    # Get people and disease data
    people = sim.people
    
    # Extract actual disease objects from sim.diseases
    diseases = []
    if hasattr(sim.diseases, 'keys'):
        for key, disease in sim.diseases.items():
            # Check if it's a disease object by looking for common disease attributes
            if (hasattr(disease, 'state') and hasattr(disease, 'sim') and 
                hasattr(disease, '__class__') and 'TB' in str(disease.__class__)):
                diseases.append(disease)
    else:
        # Fallback: try to get diseases directly
        diseases = sim.diseases if isinstance(sim.diseases, list) else []
    
    # Create age bin labels
    age_labels = []
    for i in range(len(age_bins) - 1):
        if age_bins[i+1] == float('inf'):
            age_labels.append(f"{int(age_bins[i])}+")
        else:
            age_labels.append(f"{int(age_bins[i])}-{int(age_bins[i+1])}")
    
    # Generate stratified results for each age bin
    for i in range(len(age_bins) - 1):
        age_min, age_max = age_bins[i], age_bins[i+1]
        age_label = age_labels[i]
        
        # Create age mask
        if age_max == float('inf'):
            age_mask = people.age >= age_min
        else:
            age_mask = (people.age >= age_min) & (people.age < age_max)
        
        # Create stratified results for this age bin
        age_results = _create_age_stratified_metrics(
            base_results, people, diseases, age_mask, age_label
        )
        stratified_results[age_label] = age_results
    
    return stratified_results


def _create_age_stratified_metrics(base_results, people, diseases, age_mask, age_label):
    """
    Create age-stratified metrics for a specific age range.
    
    Args:
        base_results: Base flattened results dictionary
        people: People object with age and alive attributes
        diseases: List of disease objects
        age_mask: Boolean mask for the age range
        age_label: Label for the age range
        
    Returns:
        dict: Age-stratified metrics
    """
    stratified_metrics = {}
    
    # Get time vector from base results
    timevec = None
    for metric_name, metric_data in base_results.items():
        if hasattr(metric_data, 'timevec'):
            timevec = metric_data.timevec
            break
    
    if timevec is None:
        print(f"Warning: Could not find time vector in results for age bin {age_label}")
        return {}
    
    # Calculate age-stratified metrics
    for metric_name, metric_data in base_results.items():
        if not hasattr(metric_data, 'values'):
            continue
            
        # Create age-stratified version of the metric
        stratified_values = _calculate_age_stratified_metric(
            metric_name, metric_data, people, diseases, age_mask, timevec
        )
        
        if stratified_values is not None:
            # Create a Result-like object
            stratified_metrics[metric_name] = type('Result', (), {
                'timevec': timevec,
                'values': stratified_values
            })()
    
    return stratified_metrics


def _calculate_age_stratified_metric(metric_name, metric_data, people, diseases, age_mask, timevec):
    """
    Calculate age-stratified values for a specific metric.
    
    Args:
        metric_name: Name of the metric
        metric_data: Original metric data
        people: People object
        diseases: List of disease objects
        age_mask: Boolean mask for the age range
        timevec: Time vector
        
    Returns:
        array: Age-stratified values or None if not supported
    """
    # Get TB disease if available
    tb_disease = None
    for disease in diseases:
        if hasattr(disease, 'name') and disease.name == 'tb':
            tb_disease = disease
            break
        elif hasattr(disease, '__class__') and 'TB' in disease.__class__.__name__:
            tb_disease = disease
            break
    
    if tb_disease is None:
        # If no TB disease found, return None for this metric
        return None
    
    # Import TB states if available
    try:
        from tbsim.tb import TBS
        tb_states_available = True
    except ImportError:
        tb_states_available = False
    
    # Calculate age-stratified values based on metric type
    if 'prevalence' in metric_name.lower():
        # Prevalence metrics: count of people with condition in age group / total alive in age group
        if 'active' in metric_name.lower():
            # Active TB prevalence
            if hasattr(tb_disease, 'state') and tb_states_available:
                # Define active TB states
                active_states = [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]
                active_mask = np.isin(tb_disease.state, active_states)
                age_active_mask = age_mask & active_mask
                age_alive_mask = age_mask & people.alive
                
                stratified_values = []
                for t in range(len(timevec)):
                    if np.sum(age_alive_mask) > 0:
                        prevalence = np.sum(age_active_mask) / np.sum(age_alive_mask)
                    else:
                        prevalence = 0.0
                    stratified_values.append(prevalence)
                return np.array(stratified_values)
    
    elif 'incidence' in metric_name.lower():
        # Incidence metrics: new cases per time period in age group
        if hasattr(tb_disease, 'ti_infected'):
            stratified_values = []
            for t in range(len(timevec)):
                # Count new infections in this age group at this time
                new_infections = np.sum((tb_disease.ti_infected == t) & age_mask)
                stratified_values.append(new_infections)
            return np.array(stratified_values)
    
    elif 'deaths' in metric_name.lower():
        # Death metrics: deaths in age group
        if hasattr(tb_disease, 'new_deaths'):
            stratified_values = []
            for t in range(len(timevec)):
                # Count deaths in this age group at this time
                deaths = np.sum((tb_disease.new_deaths == t) & age_mask)
                stratified_values.append(deaths)
            return np.array(stratified_values)
    
    elif 'latent' in metric_name.lower():
        # Latent TB metrics
        if hasattr(tb_disease, 'state') and tb_states_available:
            # Define latent TB states
            latent_states = [TBS.LATENT_SLOW, TBS.LATENT_FAST]
            latent_mask = np.isin(tb_disease.state, latent_states)
            age_latent_mask = age_mask & latent_mask
            
            stratified_values = []
            for t in range(len(timevec)):
                count = np.sum(age_latent_mask)
                stratified_values.append(count)
            return np.array(stratified_values)
    
    # For other metrics, return None (not supported for age stratification)
    return None


def plot_combined(flat_results, keywords=None, exclude=('None',), n_cols=7,
                 dark=True, cmap='plasma', heightfold=2, 
                 style='default', savefig=True, outdir=None, plot_type='line',
                 marker_styles=None, alpha=0.85, grid_alpha=0.4, title_fontsize=10, legend_fontsize=6, 
                 line_width=0.3, marker_size=2, markeredgewidth=0.2, grid_linewidth=0.5, 
                 spine_linewidth=0.5, label_fontsize=6, tick_fontsize=6, filter=None):
    """
    Visualize simulation outputs from multiple scenarios in a structured grid layout.

    Args:
        flat_results (dict): Nested dictionary of the form:
            {
                'Scenario A': {'metric1': Result, 'metric2': Result, ...},
                'Scenario B': {'metric1': Result, 'metric2': Result, ...},
                ...
            }
            Each Result must have `timevec` and `values` attributes representing
            time series data for a given metric.

        keywords (list[str], optional): If provided, only plot metrics containing at least one of these substrings.
        exclude (tuple[str], optional): Substrings of metric names to skip. Default is ('15',).
        n_cols (int, optional): Number of columns in the plot grid. Default is 5.
        dark (bool, optional): If True (default), uses a gray-on-dark theme for improved contrast.
        cmap (str, optional): Name of a matplotlib colormap (e.g., 'viridis', 'tab10', 'plasma'). Default is 'plasma'.
        heightfold (int, optional): Height multiplier per row of subplots. Default is 3.
        style (str, optional): Matplotlib style to apply. Defaults to 'default'. Falls back to 'default' if not found.
        savefig (bool, optional): If True (default), saves the figure as a PNG file with a timestamped filename.
        outdir (str, optional): Directory to save the figure. If None, saves in the current script's directory under 'results'.
        plot_type (str, optional): Type of plot to use for each metric ('line' or 'scatter'). Default is 'line'.
        marker_styles (list[str], optional): List of marker styles for scatter plots. Defaults to a preset list.
        alpha (float, optional): Transparency for lines/markers. Default is 0.85.
        grid_alpha (float, optional): Transparency for grid lines. Default is 0.4.
        title_fontsize (int, optional): Font size for subplot titles. Default is 10.
        legend_fontsize (int, optional): Font size for legends. Default is 6.
        line_width (float, optional): Line width for line plots. Default is 0.2.
        marker_size (int, optional): Marker size for scatter and line plots. Default is 3.
        markeredgewidth (float, optional): Marker edge width. Default is 0.5.
        grid_linewidth (float, optional): Grid line width. Default is 0.5.
        spine_linewidth (float, optional): Axis spine line width. Default is 0.5.
        label_fontsize (int, optional): Font size for axis labels. Default is 9.
        tick_fontsize (int, optional): Font size for axis tick labels. Default is 7.
    
    Returns:
        None: The figure is displayed and also saved as a PNG with a timestamped filename.

    Workflow:
        1. Collects all metric names across scenarios.
        2. Filters metrics based on `keywords` and `exclude`.
        3. Lays out subplots based on the number of metrics and specified `n_cols`.
        4. Iterates over each metric and plots it across all scenarios.
        5. Adjusts appearance (background, style, gridlines, labels).
        6. Saves the figure as 'scenarios_<timestamp>.png'.

    Example:
        >>> results = {
        ...     'BCG': {
        ...         'incidence': Result(timevec=[0, 1, 2], values=[0.1, 0.2, 0.3]),
        ...         'mortality': Result(timevec=[0, 1, 2], values=[0.05, 0.07, 0.1])
        ...     },
        ...     'TPT': {
        ...         'incidence': Result(timevec=[0, 1, 2], values=[0.08, 0.15, 0.25]),
        ...         'mortality': Result(timevec=[0, 1, 2], values=[0.03, 0.05, 0.08])
        ...     }
        ... }
        >>> plot_results(results, keywords=['incidence'], n_cols=2, dark=False, cmap='viridis', plot_type='scatter')

    NOTE:
    -----
    This plotting utility assumes results have already been flattened, such that
    each scenario maps to a dictionary of named time series outputs. This structure
    enables clean side-by-side comparisons of metrics like incidence or mortality
    across scenarios in a single visual layout.
    
    FLATTENING RESULTS:
    ---------------
    This line:
        >>> results['Any_Name'] = sim.results.flatten()       <-- The name will be used for the series name
         
    Converts the simulation's time series outputs into a flat dictionary or DataFrame.
    Makes results easier to compare across scenarios (e.g., plotting incidence over time).
    The results dictionary now maps scenario names to their flattened outputs:
    {
        'BCG': <results>,
        'TPT': <results>,
        ...
    }
    
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
    if filter is not None:
        metrics = filter
        n_cols = len(metrics)
        
    # plot layout and colors
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, heightfold*n_rows+1))
    axs = np.array(axs).flatten()

    # Calculate global X-axis range across all metrics
    all_x_min = None
    all_x_max = None
    for flat in flat_results.values():
        for metric in metrics:
            if metric in flat:
                r = flat[metric]
                if all_x_min is None:
                    all_x_min = min(r.timevec)
                    all_x_max = max(r.timevec)
                else:
                    all_x_min = min(all_x_min, min(r.timevec))
                    all_x_max = max(all_x_max, max(r.timevec))

    # Fancy background gradient
    if dark:
        # fig.patch.set_facecolor("#606060")  # medium dark gray background
        fig.patch.set_facecolor('lightgray') 
        for ax in axs:
            #ax.set_facecolor('#404040')  # dark gray for axes
            ax.set_facecolor('darkgray')
            ax.set_axisbelow(True)
            ax.tick_params(colors='#222', which='both', width=spine_linewidth)
            for spine in ax.spines.values():
                spine.set_color('#888')
                spine.set_linewidth(spine_linewidth)
    else:
        fig.patch.set_facecolor("#f7f7f7")
        for ax in axs:
            ax.set_facecolor('#f0f0f0')
            ax.set_axisbelow(True)
            ax.tick_params(colors='#222', which='both', width=spine_linewidth)
            for spine in ax.spines.values():
                spine.set_color('#888')
                spine.set_linewidth(spine_linewidth)
    
    palette = plt.cm.get_cmap(cmap, len(flat_results))
    if marker_styles is None:
        marker_styles = ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', '8', 'p', '<', '>', 'H', 'd']

    # plot each metric
    for i, metric in enumerate(metrics):
        ax = axs[i]
        for j, (scen, flat) in enumerate(flat_results.items()):
            if metric in flat:
                r = flat[metric]
                color = palette(j)
                marker = marker_styles[j % len(marker_styles)]
                if plot_type == 'scatter':
                    ax.scatter(r.timevec, r.values, label=scen, color=color, 
                               s=marker_size, marker=marker, alpha=alpha, 
                               edgecolor='darkgray', linewidths=markeredgewidth)
                else:
                    ax.plot(r.timevec, r.values, lw=line_width, label=scen, color=color, 
                            alpha=alpha, marker=marker, markersize=marker_size, 
                            markeredgewidth=markeredgewidth, markeredgecolor='darkgrey')
                    
        ax.set_title(metric, fontsize=title_fontsize, fontweight='light', color='navy' if dark else 'black')
        ax.set_xlabel('Time', fontsize=label_fontsize, color='black' if dark else 'black', fontweight='light')
        ax.tick_params(axis='both', labelsize=tick_fontsize, colors='black')
        ax.grid(True, color='white' if dark else 'gray', alpha=grid_alpha, linestyle='--', linewidth=grid_linewidth)
        # Set consistent X-axis range for all plots
        ax.set_xlim(all_x_min, all_x_max)
        leg = ax.legend(fontsize=legend_fontsize, frameon=True, loc='best')
        if leg: 
            leg.get_frame().set_alpha(0.7)
            leg.get_frame().set_facecolor('#eee')
            for text in leg.get_texts():
                text.set_color('#222')

    # remove unused axes
    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    plt.tight_layout(pad=2.0)

    if savefig:
        out = out_to(outdir)
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved figure to {out}")
    plt.show()
    

def plot_household_structure(households, people=None, figsize=(12, 8), 
                           dark=True, savefig=True, outdir=None, 
                           title="Household Structure Visualization",
                           show_household_ids=True, show_agent_ids=False,
                           max_households_to_show=50):
    """
    Visualize household structure and connections in a network-like layout.
    
    This function creates a visual representation of households where each household
    is shown as a cluster of connected nodes, with complete graphs within households
    and optional connections between households.
    
    Args:
        households (list): List of household lists, where each household contains agent UIDs.
        people (ss.People, optional): People object containing additional agent attributes.
        figsize (tuple): Figure size (width, height). Default is (12, 8).
        dark (bool): If True, uses dark theme. Default is True.
        savefig (bool): If True, saves the figure. Default is True.
        outdir (str, optional): Directory to save the figure. If None, saves in 'results'.
        title (str): Title for the plot. Default is "Household Structure Visualization".
        show_household_ids (bool): If True, shows household IDs. Default is True.
        show_agent_ids (bool): If True, shows individual agent IDs. Default is False.
        max_households_to_show (int): Maximum number of households to display. Default is 50.
    
    Returns:
        None: The figure is displayed and optionally saved.
    
    Example:
        >>> households = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        >>> plot_household_structure(households, title="My Household Network")
    """
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    
    # Limit households for visualization if too many
    if len(households) > max_households_to_show:
        print(f"Warning: Showing only first {max_households_to_show} households out of {len(households)}")
        households = households[:max_households_to_show]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set theme
    if dark:
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#3b3b3b')
        text_color = 'white'
        edge_color = '#666666'
        household_colors = plt.cm.Set3(np.linspace(0, 1, len(households)))
    else:
        fig.patch.set_facecolor('#f8f9fa')
        ax.set_facecolor('#ffffff')
        text_color = 'black'
        edge_color = '#333333'
        household_colors = plt.cm.tab20(np.linspace(0, 1, len(households)))
    
    # Calculate layout parameters
    n_households = len(households)
    if n_households == 0:
        ax.text(0.5, 0.5, 'No households to display', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=14, color=text_color)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        plt.tight_layout()
        if savefig:
            out = out_to(outdir)
            fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
            print(f"Saved figure to {out}")
        plt.show()
        return
    
    # Calculate grid layout
    cols = int(np.ceil(np.sqrt(n_households)))
    rows = int(np.ceil(n_households / cols))
    
    # Household spacing
    household_spacing = 2.0
    node_spacing = 0.3
    
    # Track all agent positions for edge drawing
    agent_positions = {}
    
    # Draw each household
    for hh_idx, household in enumerate(households):
        if len(household) == 0:
            continue
            
        # Calculate household position in grid
        row = hh_idx // cols
        col = hh_idx % cols
        hh_center_x = col * household_spacing
        hh_center_y = (rows - 1 - row) * household_spacing  # Flip Y for better layout
        
        # Calculate node positions within household
        n_agents = len(household)
        if n_agents == 1:
            # Single agent
            node_positions = [(hh_center_x, hh_center_y)]
        elif n_agents == 2:
            # Two agents side by side
            node_positions = [
                (hh_center_x - node_spacing/2, hh_center_y),
                (hh_center_x + node_spacing/2, hh_center_y)
            ]
        else:
            # Multiple agents in a circle
            angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
            radius = node_spacing * 0.8
            node_positions = [
                (hh_center_x + radius * np.cos(angle), 
                 hh_center_y + radius * np.sin(angle))
                for angle in angles
            ]
        
        # Draw household boundary
        household_color = household_colors[hh_idx]
        if n_agents > 1:
            # Calculate bounding box for household
            x_coords = [pos[0] for pos in node_positions]
            y_coords = [pos[1] for pos in node_positions]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            # Add padding
            padding = node_spacing * 0.5
            x_min -= padding
            x_max += padding
            y_min -= padding
            y_max += padding
            
            # Draw household boundary
            household_box = FancyBboxPatch(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                boxstyle="round,pad=0.1",
                facecolor=household_color,
                alpha=0.2,
                edgecolor=household_color,
                linewidth=2
            )
            ax.add_patch(household_box)
        
        # Draw nodes (agents)
        for agent_idx, (agent_id, pos) in enumerate(zip(household, node_positions)):
            agent_positions[agent_id] = pos
            
            # Node color based on household
            node_color = household_color
            
            # Draw agent node
            circle = plt.Circle(pos, 0.1, facecolor=node_color, 
                              edgecolor=edge_color, linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
            
            # Add agent ID label if requested
            if show_agent_ids:
                ax.text(pos[0], pos[1] - 0.15, str(agent_id), 
                       ha='center', va='top', fontsize=8, color=text_color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Draw edges within household (complete graph)
        if n_agents > 1:
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    pos1 = node_positions[i]
                    pos2 = node_positions[j]
                    ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                           color=edge_color, linewidth=1, alpha=0.6)
        
        # Add household ID label
        if show_household_ids:
            ax.text(hh_center_x, hh_center_y + household_spacing/2, 
                   f'HH {hh_idx}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', color=text_color,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=household_color, alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(-household_spacing/2, (cols-0.5) * household_spacing)
    ax.set_ylim(-household_spacing/2, (rows-0.5) * household_spacing)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide axes
    
    # Add title
    ax.text(0.5, 1.02, title, ha='center', va='bottom', 
            transform=ax.transAxes, fontsize=16, fontweight='bold', color=text_color)
    
    # Add legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='gray', markersize=10, 
                                     label='Agent'))
    legend_elements.append(plt.Line2D([0], [0], color=edge_color, linewidth=2, 
                                     label='Household Connection'))
    
    # Try to add age distribution info if available and accessible
    try:
        if people is not None and hasattr(people, 'age') and len(people.age) > 0:
            # Add age distribution info
            all_agent_ids = [agent_id for household in households for agent_id in household]
            valid_agent_ids = [agent_id for agent_id in all_agent_ids if agent_id < len(people.age)]
            if valid_agent_ids:
                ages = [people.age[agent_id] for agent_id in valid_agent_ids]
                if ages:
                    avg_age = np.mean(ages)
                    legend_elements.append(plt.Line2D([0], [0], marker='', color='none', 
                                                    label=f'Avg Age: {avg_age:.1f}'))
    except (IndexError, AttributeError):
        # If age data is not accessible, skip it
        pass
    
    ax.legend(handles=legend_elements, loc='upper right', 
             facecolor='white' if not dark else '#404040', 
             edgecolor=edge_color, framealpha=0.8)
    
    # Add statistics
    total_agents = sum(len(hh) for hh in households)
    total_connections = sum(len(hh) * (len(hh) - 1) // 2 for hh in households if len(hh) > 1)
    
    stats_text = f'Total Agents: {total_agents}\nTotal Households: {n_households}\nTotal Connections: {total_connections}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, color=text_color, va='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white' if not dark else '#404040', alpha=0.8))
    
    plt.tight_layout()
    
    if savefig:
        out = out_to(outdir)
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved figure to {out}")
    
    plt.show()


def plot_household_network_analysis(households, people=None, figsize=(15, 10),
                                  dark=True, savefig=True, outdir=None):
    """
    Create a comprehensive analysis plot of household network structure.
    
    This function creates multiple subplots showing different aspects of the household network:
    1. Household size distribution
    2. Network connectivity analysis
    3. Agent age distribution (if available)
    4. Household structure visualization
    
    Args:
        households (list): List of household lists, where each household contains agent UIDs.
        people (ss.People, optional): People object containing additional agent attributes.
        figsize (tuple): Figure size (width, height). Default is (15, 10).
        dark (bool): If True, uses dark theme. Default is True.
        savefig (bool): If True, saves the figure. Default is True.
        outdir (str, optional): Directory to save the figure. If None, saves in 'results'.
    
    Returns:
        None: The figure is displayed and optionally saved.
    """
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Set theme
    if dark:
        fig.patch.set_facecolor('#2b2b2b')
        text_color = 'white'
        grid_color = '#444444'
    else:
        fig.patch.set_facecolor('#f8f9fa')
        text_color = 'black'
        grid_color = '#dddddd'
    
    # Create subplots
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # 1. Household size distribution
    ax1 = fig.add_subplot(gs[0, 0])
    household_sizes = [len(hh) for hh in households]
    if household_sizes:
        ax1.hist(household_sizes, bins=range(1, max(household_sizes) + 2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Household Size', color=text_color)
        ax1.set_ylabel('Number of Households', color=text_color)
        ax1.set_title('Household Size Distribution', color=text_color, fontweight='bold')
        ax1.grid(True, alpha=0.3, color=grid_color)
        ax1.tick_params(colors=text_color)
    
    # 2. Network connectivity analysis
    ax2 = fig.add_subplot(gs[0, 1])
    connections_per_household = [len(hh) * (len(hh) - 1) // 2 for hh in households if len(hh) > 1]
    if connections_per_household:
        ax2.hist(connections_per_household, bins=range(0, max(connections_per_household) + 2), 
                alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Connections per Household', color=text_color)
        ax2.set_ylabel('Number of Households', color=text_color)
        ax2.set_title('Network Connectivity', color=text_color, fontweight='bold')
        ax2.grid(True, alpha=0.3, color=grid_color)
        ax2.tick_params(colors=text_color)
    
    # 3. Agent age distribution (if available)
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        if people is not None and hasattr(people, 'age') and len(people.age) > 0:
            all_agent_ids = [agent_id for hh in households for agent_id in hh]
            # Filter agent IDs that are within bounds
            valid_agent_ids = [agent_id for agent_id in all_agent_ids if agent_id < len(people.age)]
            if valid_agent_ids:
                ages = [people.age[agent_id] for agent_id in valid_agent_ids]
                if ages:
                    ax3.hist(ages, bins=20, alpha=0.7, color='salmon', edgecolor='black')
                    ax3.set_xlabel('Age', color=text_color)
                    ax3.set_ylabel('Number of Agents', color=text_color)
                    ax3.set_title('Agent Age Distribution', color=text_color, fontweight='bold')
                    ax3.grid(True, alpha=0.3, color=grid_color)
                    ax3.tick_params(colors=text_color)
                else:
                    ax3.text(0.5, 0.5, 'No valid age data', ha='center', va='center', 
                            transform=ax3.transAxes, color=text_color)
                    ax3.set_title('Agent Age Distribution', color=text_color, fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'No valid agent IDs', ha='center', va='center', 
                        transform=ax3.transAxes, color=text_color)
                ax3.set_title('Agent Age Distribution', color=text_color, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No people object or age data', ha='center', va='center', 
                    transform=ax3.transAxes, color=text_color)
            ax3.set_title('Agent Age Distribution', color=text_color, fontweight='bold')
    except (IndexError, AttributeError):
        ax3.text(0.5, 0.5, 'Age data not accessible', ha='center', va='center', 
                transform=ax3.transAxes, color=text_color)
        ax3.set_title('Agent Age Distribution', color=text_color, fontweight='bold')
    
    # 4. Household structure visualization (spans bottom row)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Limit households for visualization
    max_households_to_show = 20
    households_to_show = households[:max_households_to_show]
    
    # Calculate layout for visualization
    n_households = len(households_to_show)
    if n_households > 0:
        cols = int(np.ceil(np.sqrt(n_households)))
        rows = int(np.ceil(n_households / cols))
        
        household_spacing = 1.5
        node_spacing = 0.2
        
        # Track agent positions
        agent_positions = {}
        
        # Draw households
        for hh_idx, household in enumerate(households_to_show):
            if len(household) == 0:
                continue
                
            # Calculate position
            row = hh_idx // cols
            col = hh_idx % cols
            hh_center_x = col * household_spacing
            hh_center_y = (rows - 1 - row) * household_spacing
            
            # Node positions
            n_agents = len(household)
            if n_agents == 1:
                node_positions = [(hh_center_x, hh_center_y)]
            elif n_agents == 2:
                node_positions = [
                    (hh_center_x - node_spacing/2, hh_center_y),
                    (hh_center_x + node_spacing/2, hh_center_y)
                ]
            else:
                angles = np.linspace(0, 2*np.pi, n_agents, endpoint=False)
                radius = node_spacing * 0.6
                node_positions = [
                    (hh_center_x + radius * np.cos(angle), 
                     hh_center_y + radius * np.sin(angle))
                    for angle in angles
                ]
            
            # Draw nodes
            for agent_id, pos in zip(household, node_positions):
                agent_positions[agent_id] = pos
                circle = plt.Circle(pos, 0.05, facecolor='skyblue', 
                                  edgecolor='black', linewidth=1)
                ax4.add_patch(circle)
            
            # Draw edges
            if n_agents > 1:
                for i in range(n_agents):
                    for j in range(i + 1, n_agents):
                        pos1 = node_positions[i]
                        pos2 = node_positions[j]
                        ax4.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               color='gray', linewidth=0.5, alpha=0.6)
            
            # Add household label
            ax4.text(hh_center_x, hh_center_y + household_spacing/2, 
                    f'HH {hh_idx}', ha='center', va='bottom', 
                    fontsize=8, color=text_color)
        
        ax4.set_xlim(-household_spacing/2, (cols-0.5) * household_spacing)
        ax4.set_ylim(-household_spacing/2, (rows-0.5) * household_spacing)
        ax4.set_aspect('equal')
        ax4.axis('off')
        ax4.set_title('Household Network Structure (First 20 Households)', 
                     color=text_color, fontweight='bold')
    
    # Add overall statistics
    total_agents = sum(len(hh) for hh in households)
    total_connections = sum(len(hh) * (len(hh) - 1) // 2 for hh in households if len(hh) > 1)
    avg_household_size = total_agents / len(households) if households else 0
    
    stats_text = f'Total Agents: {total_agents}\nTotal Households: {len(households)}\n'
    stats_text += f'Total Connections: {total_connections}\nAvg Household Size: {avg_household_size:.1f}'
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, color=text_color,
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white' if not dark else '#404040', alpha=0.8))
    
    plt.tight_layout()
    
    if savefig:
        out = out_to(outdir)
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved figure to {out}")
    
    plt.show()


def out_to(outdir):
           # save figure
        timestamp = sc.now(dateformat='%Y%m%d_%H%M%S') 
        # Determine script directory in a cross-platform way
        if hasattr(sys.modules['__main__'], '__file__'):
            script_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        else:
            script_dir = os.getcwd()

        outdir = 'results' if outdir is None else outdir
        outdir = os.path.join(script_dir, outdir)
        os.makedirs(outdir, exist_ok=True)   
        out = os.path.join(outdir, f'scenarios_{timestamp}.png')
        return out
    
class FILTERS(): 
    important_metrics = [
        'tb_prevalence_active',
        'tb_prevalence', 
        'cum_deaths' ]