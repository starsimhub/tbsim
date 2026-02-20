import os
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from typing import Dict, List, Tuple
import starsim as ss
import datetime
import sys


def plot_results(flat_results, keywords=None, exclude=('None',), n_cols=5,
                 dark=True, cmap='tab20', heightfold=2, 
                 style='default', savefig=True, outdir=None, metric_filter=None, title='',
                 shared_legend=True, legend_position='upper right'):
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
        cmap (str, optional): Name of a matplotlib colormap (e.g., 'viridis', 'tab10'). Default is 'tab20'.
        heightfold (int, optional): Height multiplier per row of subplots. Default is 3.
        style (str, optional): Matplotlib style to apply. Defaults to 'default'. Falls back to 'default' if not found.
        savefig (bool, optional): If True (default), saves the figure as a PNG file with a timestamped filename.
        outdir (str, optional): Directory to save the figure. If None, saves in the current script's directory under 'results'.
        metric_filter (list[str], optional): List of metric names to plot. If provided, only these metrics will be plotted.
        shared_legend (bool, optional): If True, creates a single shared legend for all subplots. Default is True.
        legend_position (str, optional): Position for the shared legend. Options: 'upper right', 'upper left', 'lower right', 'lower left', 'center', etc. Default is 'upper right'.
    
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
        >>> plot_results(results, keywords=['incidence'], n_cols=2, dark=False, cmap='viridis')

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
        if vmax < 1.010:
            ax.set_ylim(0, max(0.5, vmax))
            ax.set_ylabel('%', color='white' if dark else 'black')
        else:
            ax.set_ylabel('Value', color='white' if dark else 'black')
        ax.set_xlabel('Time', color='white' if dark else 'black')
        ax.tick_params(axis='both', colors='white' if dark else 'black', labelsize=6)
        # Set consistent X-axis range for all plots
        ax.set_xlim(all_x_min, all_x_max)

        # grid lines
        ax.grid(True, color='white' if dark else 'gray', alpha=0.3)
        
        # Only add individual legends if shared_legend is False
        if not shared_legend:
            leg = ax.legend(fontsize=6 if len(flat_results)>5 else 7)
            if leg: leg.get_frame().set_alpha(0.3)

    # remove unused axes
    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    # Create shared legend if requested
    if shared_legend:
        # Get handles and labels from the first subplot that has data
        handles, labels = None, None
        for ax in axs[:len(metrics)]:
            if ax.get_legend_handles_labels()[0]:  # Check if there are handles
                handles, labels = ax.get_legend_handles_labels()
                break
        
        if handles and labels:
            # Create shared legend
            fig.legend(handles, labels, loc=legend_position, fontsize=6 if len(flat_results)>5 else 7, 
                      frameon=True, fancybox=True, shadow=True)
            # Style the shared legend
            legend = fig.legends[-1]
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_facecolor('#f0f0f0')
            legend.get_frame().set_edgecolor('#888')
            for text in legend.get_texts():
                text.set_color('#222')

    if title:
        fig.suptitle(title, fontsize=12, color='white' if dark else 'black')

    plt.tight_layout(pad=2.0)

    if savefig:
        out = out_to(outdir)
        fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
        print(f"Saved figure to {out}")
    plt.show()


def plot_combined(flat_results, keywords=None, exclude=('None',), n_cols=7,
                 dark=True, cmap='plasma', heightfold=2, 
                 style='default', savefig=True, outdir=None, plot_type='line',
                 marker_styles=None, alpha=0.85, grid_alpha=0.4, title_fontsize=10, legend_fontsize=7, 
                 line_width=0.3, marker_size=2, markeredgewidth=0.2, grid_linewidth=0.5, 
                 spine_linewidth=0.5, label_fontsize=6, tick_fontsize=6, filter=None, title='',
                 shared_legend=True, legend_position='upper left'):
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
        tick_fontsize (int, optional): Font size for axis tick labels. Default is 5.
        shared_legend (bool, optional): If True, creates a single shared legend for all subplots. Default is True.
        legend_position (str, optional): Position for the shared legend. Options: 'upper right', 'upper left', 'lower right', 'lower left', 'center', etc. Default is 'upper right'.
    
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
    
    cmap:
    ---------------
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Grays_r', 'Greens', 'Greens_r',
    'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r',
    'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r',
    'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r',
    'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r',
    'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu',
    'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'alpine', 'autumn',
    'autumn_r', 'banded', 'berlin', 'berlin_r', 'bi', 'binary', 'binary_r', 'bone', 'bone_r', 'brg',
    'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper',
    'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth',
    'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_grey_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow',
    'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gist_yerg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r',
    'gnuplot_r', 'gray', 'gray_r', 'grey', 'grey_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire',
    'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'managua',
    'managua_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'orangeblue', 'parula', 'pink', 'pink_r', 'plasma',
    'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'sciris-alpine', 'sciris-banded', 'sciris-bi',
    'sciris-orangeblue', 'sciris-parula', 'sciris-turbo', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10',
    'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo',
    'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'vanimo', 'vanimo_r', 'viridis', 'viridis_r', 'vlag',
    'vlag_r', 'winter', 'winter_r'

    """


    try:
        plt.style.use(style)
    except Exception:
        print(f"Warning: {style} style not found. Using default style.")
        plt.style.use('default')

    # normalize all keys to strings so numeric/other types from flatten() become string column names
    flat_results = {
        label: {str(k): v for k, v in flat.items()}
        for label, flat in flat_results.items()
    }
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
        
        # Only add individual legends if shared_legend is False
        if not shared_legend:
            leg = ax.legend(fontsize=legend_fontsize, frameon=True, loc='best')
            if leg: 
                leg.get_frame().set_alpha(0.7)
                leg.get_frame().set_facecolor('#eee')
                for text in leg.get_texts():
                    text.set_color('#222')

    # remove unused axes
    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    # Create shared legend if requested
    if shared_legend:
        # Collect all unique handles and labels from all subplots to get maximum coverage
        all_handles = []
        all_labels = []
        seen_labels = set()
        
        for ax in axs[:len(metrics)]:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
                    seen_labels.add(label)
        
        if all_handles and all_labels:
            # Create shared legend with all unique entries
            fig.legend(all_handles, all_labels, loc=legend_position, fontsize=legend_fontsize, 
                      frameon=True, fancybox=True, shadow=True)
            # Style the shared legend
            legend = fig.legends[-1]
            legend.get_frame().set_alpha(0.9)
            legend.get_frame().set_facecolor('#f0f0f0')
            legend.get_frame().set_edgecolor('#888')
            for text in legend.get_texts():
                text.set_color('#222')

    if title:
        fig.suptitle(title, fontsize=12, color='white' if dark else 'black')

    plt.tight_layout(pad=2.0)

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