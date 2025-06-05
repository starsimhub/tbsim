import os
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from typing import Dict, List, Tuple
import starsim as ss


def plot_results(flat_results, keywords=None, exclude=('15',), n_cols=5,
                 dark=True, cmap='tab20', heightfold=3, style='default'):
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