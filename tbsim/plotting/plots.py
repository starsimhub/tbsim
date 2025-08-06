import os
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from typing import Dict, List, Tuple
import starsim as ss
import datetime
import sys
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D


def plot_results(flat_results, keywords=None, exclude=('None',), n_cols=5,
                 dark=True, cmap='tab20', heightfold=2, 
                 style='default', savefig=True, outdir=None, metric_filter=None):
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

class CalibrationPlotter:
    """
    A centralized class for creating calibration plots and visualizations.
    
    This class provides methods for creating various types of plots used in
    model calibration, including violin plots, parameter sweeps, and comparison plots.
    """
    
    def __init__(self, style='default', figsize=(12, 8)):
        """
        Initialize the calibration plotter.
        
        Args:
            style (str): Matplotlib style to use
            figsize (tuple): Default figure size
        """
        self.style = style
        self.default_figsize = figsize
        self._setup_style()
    
    def _setup_style(self):
        """Set up the plotting style."""
        try:
            plt.style.use(self.style)
        except Exception:
            print(f"Warning: {self.style} style not found. Using default style.")
            plt.style.use('default')
    
    def plot_calibration_comparison(self, sim, calibration_data, timestamp, save_path=None):
        """
        Create comprehensive calibration comparison plots.
        
        Args:
            sim: Simulation object
            calibration_data: CalibrationData object
            timestamp: Timestamp for file naming
            save_path: Optional path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        from ..calibration.functions import compute_case_notifications, compute_age_stratified_prevalence
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Case notification comparison
        notifications = compute_case_notifications(sim)
        years = list(notifications.keys())
        model_rates = [notifications[year]['rate_per_100k'] for year in years]
        data_rates = calibration_data.case_notifications['rate_per_100k'].values
        
        ax1.plot(years, model_rates, 'bo-', label='Model Output', linewidth=2, markersize=8)
        ax1.plot(years, data_rates, 'ro-', label='South Africa Data', linewidth=2, markersize=8)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Case Notification Rate (per 100,000)')
        ax1.set_title('TB Case Notifications: Model vs Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add percentage difference
        for i, year in enumerate(years):
            if data_rates[i] > 0:
                pct_diff = ((model_rates[i] - data_rates[i]) / data_rates[i]) * 100
                ax1.annotate(f'{pct_diff:.1f}%', 
                            xy=(year, model_rates[i]), 
                            xytext=(0, 10), 
                            textcoords='offset points',
                            ha='center', fontsize=8)
        
        # 2. Age-stratified prevalence comparison
        age_prevalence = compute_age_stratified_prevalence(sim)
        age_groups = list(age_prevalence.keys())
        model_prevalence = [age_prevalence[group]['prevalence_per_100k'] for group in age_groups]
        data_prevalence = calibration_data.age_prevalence['prevalence_per_100k'].values
        
        x_pos = np.arange(len(age_groups))
        width = 0.35
        
        ax2.bar(x_pos - width/2, model_prevalence, width, label='Model Output', alpha=0.8)
        ax2.bar(x_pos + width/2, data_prevalence, width, label='South Africa Data', alpha=0.8)
        ax2.set_xlabel('Age Group')
        ax2.set_ylabel('Active TB Prevalence (per 100,000)')
        ax2.set_title('Age-Stratified TB Prevalence: Model vs Data (2018)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(age_groups)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add percentage differences
        for i, (model_val, data_val) in enumerate(zip(model_prevalence, data_prevalence)):
            if data_val > 0:
                pct_diff = ((model_val - data_val) / data_val) * 100
                ax2.annotate(f'{pct_diff:.1f}%', 
                            xy=(i, max(model_val, data_val)), 
                            xytext=(0, 5), 
                            textcoords='offset points',
                            ha='center', fontsize=8)
        
        # 3. Overall prevalence over time
        time_years = np.array([d.year for d in sim.results['timevec']])
        active_prev = sim.results['tb']['prevalence_active']
        
        ax3.plot(time_years, active_prev * 100, 'b-', linewidth=2, label='Model Active TB Prevalence')
        
        # Get target from calibration data
        target_overall_prev = None
        for target_name, target in calibration_data.targets.items():
            if 'prevalence' in target_name.lower() and target.year == 2018:
                target_overall_prev = target.value
                break
        
        if target_overall_prev is None:
            target_overall_prev = 0.852  # Default fallback
        
        ax3.axhline(target_overall_prev, color='r', linestyle='--', 
                    label=f"Target: {target_overall_prev:.3f}%")
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Active TB Prevalence (%)')
        ax3.set_title('Overall TB Prevalence Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Diagnostic and treatment cascade
        if 'tbdiagnostic' in sim.results and 'tbtreatment' in sim.results:
            tbdiag = sim.results['tbdiagnostic']
            tbtx = sim.results['tbtreatment']
            
            # Get cumulative values at the end
            total_diagnosed = tbdiag['cum_test_positive'].values[-1]
            total_treated = tbtx['cum_treatment_success'].values[-1]
            total_failures = tbtx['cum_treatment_failure'].values[-1]
        else:
            # Fallback values if interventions not present
            total_diagnosed = 0
            total_treated = 0
            total_failures = 0
        
        # Create cascade plot
        cascade_data = [total_diagnosed, total_treated, total_failures]
        cascade_labels = ['Diagnosed', 'Successfully Treated', 'Treatment Failures']
        colors = ['skyblue', 'lightgreen', 'lightcoral']
        
        bars = ax4.bar(cascade_labels, cascade_data, color=colors, alpha=0.8)
        ax4.set_ylabel('Number of People')
        ax4.set_title('TB Care Cascade (Cumulative)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, cascade_data):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cascade_data)*0.01,
                    f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle(f'TB Model Calibration: {calibration_data.country} Data Comparison', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sweep_results(self, results_df, timestamp, save_path=None):
        """
        Create comprehensive sweep results plots including violin plots.
        
        Args:
            results_df: DataFrame with sweep results
            timestamp: Timestamp for file naming
            save_path: Optional path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Score distribution histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(results_df['composite_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Composite Calibration Score')
        ax1.set_ylabel('Number of Simulations')
        ax1.set_title('Distribution of Calibration Scores')
        ax1.grid(True, alpha=0.3)
        
        # 2. Beta vs Score scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(results_df['beta'], results_df['composite_score'], alpha=0.6, s=30)
        ax2.set_xlabel('Beta (Transmission Rate)')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Beta vs Calibration Score')
        ax2.grid(True, alpha=0.3)
        
        # 3. Rel Sus vs Score scatter
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(results_df['rel_sus_latentslow'], results_df['composite_score'], alpha=0.6, s=30)
        ax3.set_xlabel('Relative Susceptibility (Latent)')
        ax3.set_ylabel('Composite Score')
        ax3.set_title('Relative Susceptibility vs Calibration Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. TB Mortality vs Score scatter
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.scatter(results_df['tb_mortality'], results_df['composite_score'], alpha=0.6, s=30)
        ax4.set_xlabel('TB Mortality Rate')
        ax4.set_ylabel('Composite Score')
        ax4.set_title('TB Mortality vs Calibration Score')
        ax4.grid(True, alpha=0.3)
        
        # 5. Violin plot for Beta
        ax5 = fig.add_subplot(gs[1, 0])
        beta_bins = pd.cut(results_df['beta'], bins=5, labels=False)
        violin_data = []
        violin_labels = []
        for i in range(5):
            mask = beta_bins == i
            if mask.any():
                violin_data.append(results_df.loc[mask, 'composite_score'].values)
                beta_range = results_df.loc[mask, 'beta']
                violin_labels.append(f'{beta_range.min():.3f}-{beta_range.max():.3f}')
        
        if violin_data:
            parts = ax5.violinplot(violin_data, positions=range(len(violin_data)))
            ax5.set_xticks(range(len(violin_labels)))
            ax5.set_xticklabels(violin_labels, rotation=45)
            ax5.set_xlabel('Beta Range')
            ax5.set_ylabel('Composite Score')
            ax5.set_title('Score Distribution by Beta Range')
            ax5.grid(True, alpha=0.3)
        
        # 6. Violin plot for Relative Susceptibility
        ax6 = fig.add_subplot(gs[1, 1])
        rel_sus_bins = pd.cut(results_df['rel_sus_latentslow'], bins=5, labels=False)
        violin_data = []
        violin_labels = []
        for i in range(5):
            mask = rel_sus_bins == i
            if mask.any():
                violin_data.append(results_df.loc[mask, 'composite_score'].values)
                rel_sus_range = results_df.loc[mask, 'rel_sus_latentslow']
                violin_labels.append(f'{rel_sus_range.min():.2f}-{rel_sus_range.max():.2f}')
        
        if violin_data:
            parts = ax6.violinplot(violin_data, positions=range(len(violin_data)))
            ax6.set_xticks(range(len(violin_labels)))
            ax6.set_xticklabels(violin_labels, rotation=45)
            ax6.set_xlabel('Rel Sus Range')
            ax6.set_ylabel('Composite Score')
            ax6.set_title('Score Distribution by Rel Sus Range')
            ax6.grid(True, alpha=0.3)
        
        # 7. Violin plot for TB Mortality
        ax7 = fig.add_subplot(gs[1, 2])
        tb_mort_bins = pd.cut(results_df['tb_mortality'], bins=5, labels=False)
        violin_data = []
        violin_labels = []
        for i in range(5):
            mask = tb_mort_bins == i
            if mask.any():
                violin_data.append(results_df.loc[mask, 'composite_score'].values)
                tb_mort_range = results_df.loc[mask, 'tb_mortality']
                violin_labels.append(f'{tb_mort_range.min():.1e}-{tb_mort_range.max():.1e}')
        
        if violin_data:
            parts = ax7.violinplot(violin_data, positions=range(len(violin_data)))
            ax7.set_xticks(range(len(violin_labels)))
            ax7.set_xticklabels(violin_labels, rotation=45)
            ax7.set_xlabel('TB Mortality Range')
            ax7.set_ylabel('Composite Score')
            ax7.set_title('Score Distribution by TB Mortality Range')
            ax7.grid(True, alpha=0.3)
        
        # 8. 3D scatter plot (Beta vs Rel Sus vs Score)
        ax8 = fig.add_subplot(gs[1, 3], projection='3d')
        scatter = ax8.scatter(results_df['beta'], results_df['rel_sus_latentslow'], 
                             results_df['composite_score'], 
                             c=results_df['composite_score'], cmap='viridis', alpha=0.6)
        ax8.set_xlabel('Beta')
        ax8.set_ylabel('Rel Sus')
        ax8.set_zlabel('Composite Score')
        ax8.set_title('3D: Beta vs Rel Sus vs Score')
        plt.colorbar(scatter, ax=ax8, label='Composite Score')
        
        # 9. Heatmap of parameter interactions
        ax9 = fig.add_subplot(gs[2, 0])
        corr_data = results_df[['beta', 'rel_sus_latentslow', 'tb_mortality', 'composite_score']].corr()
        im = ax9.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
        ax9.set_xticks(range(len(corr_data.columns)))
        ax9.set_yticks(range(len(corr_data.columns)))
        ax9.set_xticklabels(corr_data.columns, rotation=45)
        ax9.set_yticklabels(corr_data.columns)
        ax9.set_title('Parameter Correlations')
        
        # Add correlation values to heatmap
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # 10-12. Box plots for each parameter
        ax10 = fig.add_subplot(gs[2, 1])
        results_df['beta_cat'] = pd.cut(results_df['beta'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        results_df['rel_sus_cat'] = pd.cut(results_df['rel_sus_latentslow'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        results_df['tb_mort_cat'] = pd.cut(results_df['tb_mortality'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # Box plot for beta categories
        box_data = [results_df[results_df['beta_cat'] == cat]['composite_score'].values 
                    for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                    if len(results_df[results_df['beta_cat'] == cat]) > 0]
        box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                      if len(results_df[results_df['beta_cat'] == cat]) > 0]
        
        if box_data:
            bp = ax10.boxplot(box_data, tick_labels=box_labels)
            ax10.set_xlabel('Beta Categories')
            ax10.set_ylabel('Composite Score')
            ax10.set_title('Score Distribution by Beta Categories')
            ax10.tick_params(axis='x', rotation=45)
            ax10.grid(True, alpha=0.3)
        
        # Box plots for relative susceptibility
        ax11 = fig.add_subplot(gs[2, 2])
        box_data = [results_df[results_df['rel_sus_cat'] == cat]['composite_score'].values 
                    for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                    if len(results_df[results_df['rel_sus_cat'] == cat]) > 0]
        box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                      if len(results_df[results_df['rel_sus_cat'] == cat]) > 0]
        
        if box_data:
            bp = ax11.boxplot(box_data, tick_labels=box_labels)
            ax11.set_xlabel('Rel Sus Categories')
            ax11.set_ylabel('Composite Score')
            ax11.set_title('Score Distribution by Rel Sus Categories')
            ax11.tick_params(axis='x', rotation=45)
            ax11.grid(True, alpha=0.3)
        
        # Box plots for TB mortality
        ax12 = fig.add_subplot(gs[2, 3])
        box_data = [results_df[results_df['tb_mort_cat'] == cat]['composite_score'].values 
                    for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                    if len(results_df[results_df['tb_mort_cat'] == cat]) > 0]
        box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                      if len(results_df[results_df['tb_mort_cat'] == cat]) > 0]
        
        if box_data:
            bp = ax12.boxplot(box_data, tick_labels=box_labels)
            ax12.set_xlabel('TB Mortality Categories')
            ax12.set_ylabel('Composite Score')
            ax12.set_title('Score Distribution by TB Mortality Categories')
            ax12.tick_params(axis='x', rotation=45)
            ax12.grid(True, alpha=0.3)
        
        plt.suptitle('TB Model Calibration Parameter Sweep Results', fontsize=20, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_violin_plots(self, results_df, timestamp, save_path=None):
        """
        Create focused violin plots showing score distributions for each parameter.
        
        Args:
            results_df: DataFrame with sweep results
            timestamp: Timestamp for file naming
            save_path: Optional path to save the plot
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Violin plot for Beta values
        beta_groups = results_df.groupby('beta')['composite_score'].apply(list).reset_index()
        violin_data = [group for group in beta_groups['composite_score']]
        violin_labels = [f'{beta:.3f}' for beta in beta_groups['beta']]
        
        parts1 = ax1.violinplot(violin_data, positions=range(len(violin_data)))
        ax1.set_xticks(range(len(violin_labels)))
        ax1.set_xticklabels(violin_labels, rotation=45)
        ax1.set_xlabel('Beta (Transmission Rate)')
        ax1.set_ylabel('Composite Calibration Score')
        ax1.set_title('Score Distribution by Beta Value')
        ax1.grid(True, alpha=0.3)
        
        # Add mean points
        means = [np.mean(data) for data in violin_data]
        ax1.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
        ax1.legend()
        
        # 2. Violin plot for Relative Susceptibility values
        rel_sus_groups = results_df.groupby('rel_sus_latentslow')['composite_score'].apply(list).reset_index()
        violin_data = [group for group in rel_sus_groups['composite_score']]
        violin_labels = [f'{rel_sus:.2f}' for rel_sus in rel_sus_groups['rel_sus_latentslow']]
        
        parts2 = ax2.violinplot(violin_data, positions=range(len(violin_data)))
        ax2.set_xticks(range(len(violin_labels)))
        ax2.set_xticklabels(violin_labels, rotation=45)
        ax2.set_xlabel('Relative Susceptibility (Latent)')
        ax2.set_ylabel('Composite Calibration Score')
        ax2.set_title('Score Distribution by Relative Susceptibility Value')
        ax2.grid(True, alpha=0.3)
        
        # Add mean points
        means = [np.mean(data) for data in violin_data]
        ax2.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
        ax2.legend()
        
        # 3. Violin plot for TB Mortality values
        tb_mort_groups = results_df.groupby('tb_mortality')['composite_score'].apply(list).reset_index()
        violin_data = [group for group in tb_mort_groups['composite_score']]
        violin_labels = [f'{tb_mort:.1e}' for tb_mort in tb_mort_groups['tb_mortality']]
        
        parts3 = ax3.violinplot(violin_data, positions=range(len(violin_data)))
        ax3.set_xticks(range(len(violin_labels)))
        ax3.set_xticklabels(violin_labels, rotation=45)
        ax3.set_xlabel('TB Mortality Rate')
        ax3.set_ylabel('Composite Calibration Score')
        ax3.set_title('Score Distribution by TB Mortality Value')
        ax3.grid(True, alpha=0.3)
        
        # Add mean points
        means = [np.mean(data) for data in violin_data]
        ax3.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
        ax3.legend()
        
        plt.tight_layout()
        plt.suptitle('TB Model Calibration: Violin Plots by Parameter Value', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig