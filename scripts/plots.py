import os
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
from typing import Dict, List, Tuple
import starsim as ss



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