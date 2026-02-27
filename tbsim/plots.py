"""Plotting utilities for visualizing TB simulation results."""

# Standard library
import os
import re
import sys

# External packages - data/numerical
import numpy as np

# External packages - plotting
import matplotlib.pyplot as plt

# External packages - project dependencies
import sciris as sc
import starsim as ss

__all__ = ['plot']

def plot(
    results,
    select=None,
    title='',
    savefig=False,
    output_dir=None,
    theme='light',
    n_cols=6,
    row_height=1.5,
    style=None,
):
    """Plot simulation results (MultiSim, Sim, or flat results dict).

    Parameters
    ----------
    results
        MultiSim, single Sim, or dict of flat results per scenario.
    select : None, list, or dict, optional
        Which metrics to plot. Interpreted like pandas column selection
        (e.g. ``DataFrame.filter()``):

        - **None** : all metrics (except names containing 'None').
        - **list** : same as ``dict(items=list)``; use ``'~pattern'`` in the
          list to exclude metric names containing that pattern.
        - **dict** :
          - ``items`` : list of exact metric names (like pandas ``filter(items=...)``).
          - ``like`` : str or list of str; keep names containing any (``filter(like=...)``).
          - ``regex`` : str; keep names matching the regex (``filter(regex=...)``).
          - ``exclude`` : list of patterns (or ``'~pattern'``); drop names containing any.

        Examples::

            plot(msim)                                    # all metrics
            plot(msim, select=['incidence', 'prevalence']) # items=['incidence','prevalence']
            plot(msim, select=['incidence', '~None'])      # items + exclude 'None'
            plot(msim, select=dict(like='prev'))          # names containing 'prev'
            plot(msim, select=dict(like=['prev', 'death']))
            plot(msim, select=dict(regex=r'^tb_.*'))      # names matching regex
            plot(msim, select=dict(items=['a'], exclude=['internal']))
    title : str, optional
        Figure suptitle. Default ''.
    savefig : bool, optional
        If True, save PNG to output_dir. Default False.
    output_dir : str, optional
        Directory for saved figure. Default 'results'.
    theme : {'light', 'dark'}, optional
        Color theme. Default 'light'.
    n_cols : int, optional
        Number of subplot columns. Default 6.
    row_height : float, optional
        Subplot row height (inches). Default 1.5.
    style : dict, optional
        Override plot style. Keys (all optional) include: mpl_style, cmap, alpha (opacity of lines/markers),
        plot_type ('line' or 'scatter'), line_width, line_style, use_markers,
        marker_size, marker_styles, marker_edge_width, grid_color, grid_alpha,
        grid_linewidth, spine_linewidth, title_fontsize, legend_fontsize,
        axis_label_fontsize, tick_fontsize, shared_legend, legend_position.
        Example: ``style=dict(cmap='plasma', use_markers=True)``.

    Notes
    -----
    Missing metrics in a scenario are filled with zeros.
    Non-1D result values are reduced with nanmean.
    """
    opts = {**_DEFAULT_STYLE, **(style or {})}

    try:
        flat_results = _normalize_results(results)
        if not flat_results:
            print("No scenarios to plot.")
            return
        flat_results = _validate_flat_results(flat_results)
    except (TypeError, ValueError) as e:
        print(f"Cannot plot: {e}")
        return

    dark = str(theme).lower() == 'dark'
    fg = '#eaeaea' if dark else '#111'
    ax_bg = '#2b2b2b' if dark else '#f0f0f0'
    fig_bg = '#1e1e1e' if dark else '#f7f7f7'
    grid_color = '#9a9a9a' if dark else '#777'

    try:
        plt.style.use(opts['mpl_style'])
    except Exception:
        print(f"Warning: {opts['mpl_style']} style not found. Using default style.")
        plt.style.use('default')

    all_metrics = {m for flat in flat_results.values() for m in flat}
    metrics = _select_metrics(all_metrics, select, flat_results)
    if not metrics:
        print("No metrics to plot.")
        return

    flat_results, _ = _fill_missing_metrics(flat_results, metrics)

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, row_height * n_rows + 1))
    axs = np.array(axs).flatten()

    all_x_min = all_x_max = None
    for flat in flat_results.values():
        for m in metrics:
            x, _ = _as_1d_xy(flat.get(m))
            tmin, tmax = _safe_min_max(x)
            if tmin is not None:
                all_x_min = tmin if all_x_min is None else min(all_x_min, tmin)
                all_x_max = tmax if all_x_max is None else max(all_x_max, tmax)

    fig.patch.set_facecolor(fig_bg)
    for ax in axs:
        ax.set_facecolor(ax_bg)
        ax.set_axisbelow(True)
        ax.tick_params(colors=fg, which='both', width=opts['spine_linewidth'])
        for spine in ax.spines.values():
            spine.set_visible(False)

    n_scen = len(flat_results)
    try:
        _cm = plt.colormaps.get_cmap(opts['cmap'])
    except Exception:
        _cm = plt.cm.get_cmap(opts['cmap'])

    def palette(j):
        # Just get the next color from the colormap
        return _cm(j)

    mstyles = opts['marker_styles'] or ['o', 's', 'D', '^', 'v', 'P', 'X', '*', 'h', '8', 'p', '<', '>', 'H', 'd']
    use_markers = bool(opts.get('use_markers', False))

    for i, metric in enumerate(metrics):
        ax = axs[i]
        # Get the label from the first available result object
        metric_label = metric
        for flat in flat_results.values():
            result = flat.get(metric)
            if result is not None:
                if hasattr(result, 'full_label'):
                    metric_label = result.full_label
                elif hasattr(result, 'label') and result.label:
                    metric_label = result.label
                break
        
        for j, (scen, flat) in enumerate(flat_results.items()):
            x, y = _as_1d_xy(flat.get(metric))
            if x is None:
                continue
            color = palette(j)
            # ls = '--' if j == 0 else '-'
            ls = '-'
            marker = mstyles[j % len(mstyles)] if use_markers else None
            if opts['plot_type'] == 'scatter':
                ax.scatter(x, y, label=scen, color=color, s=max(1, float(opts['marker_size'])) ** 2,
                           marker=marker, alpha=opts['alpha'], edgecolor=grid_color, linewidths=opts['marker_edge_width'])
            else:
                ax.plot(x, y, lw=opts['line_width'], label=scen, color=color, linestyle=ls,
                        alpha=opts['alpha'], marker=marker, markersize=opts['marker_size'],
                        markeredgewidth=opts['marker_edge_width'], markeredgecolor=grid_color,
                        zorder=10 if j == 0 else 5)
        ax.set_title(metric_label, fontsize=opts['title_fontsize'], fontweight='light', color=fg)
        ax.set_xlabel('Time', fontsize=opts['axis_label_fontsize'], color=fg)
        ax.tick_params(axis='both', labelsize=opts['tick_fontsize'], colors=fg)
        ax.grid(True, color=opts['grid_color'], alpha=opts['grid_alpha'], linestyle='--', linewidth=opts['grid_linewidth'])
        if all_x_min is not None and all_x_max is not None:
            ax.set_xlim(all_x_min, all_x_max)
        if not opts['shared_legend']:
            leg = ax.legend(fontsize=opts['legend_fontsize'], frameon=True, loc='best')
            if leg:
                leg.get_frame().set_alpha(0.8)
                leg.get_frame().set_facecolor(fig_bg)
                for t in leg.get_texts():
                    t.set_color(fg)

    for ax in axs[len(metrics):]:
        fig.delaxes(ax)

    if opts['shared_legend']:
        all_handles, all_labels, seen = [], [], set()
        for ax in axs[:len(metrics)]:
            for h, lbl in zip(*ax.get_legend_handles_labels()):
                if lbl and lbl not in seen:
                    all_handles.append(h)
                    all_labels.append(lbl)
                    seen.add(lbl)
        if all_handles and all_labels:
            leg = fig.legend(all_handles, all_labels, loc=opts['legend_position'], fontsize=opts['legend_fontsize'],
                             frameon=True, fancybox=True)
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_facecolor(fig_bg)
            leg.get_frame().set_edgecolor(grid_color)
            for t in leg.get_texts():
                t.set_color(fg)

    if title:
        fig.suptitle(title, fontsize=12, color=fg)

    plt.tight_layout(pad=2.0)
    if savefig:
        try:
            out = out_to(output_dir)
            fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
            print(f"Saved figure to {out}")
        except OSError as e:
            print(f"Warning: could not save figure: {e}")
    plt.show()

    return


_DEFAULT_STYLE = dict(
    mpl_style   = 'default',
    cmap        = 'tab10',
    alpha       = 0.85,
    plot_type   = 'line',
    line_width  = 1.2,
    marker_size = 0.5,
    use_markers = False,
    marker_styles      = None,
    marker_edge_width  = 0.2,
    grid_color         = 'white',
    grid_alpha         = 0.25,
    grid_linewidth     = 0.2,
    spine_linewidth    = 0.5,
    title_fontsize     = 10,
    legend_fontsize    = 7,
    axis_label_fontsize = 6,
    tick_fontsize      = 6,
    shared_legend      = True,
    legend_position    = 'upper left',
    line_style         = '-',
)

def _normalize_results(results):
    """Convert MultiSim, Sim, or dict into a dict of label -> flat result dict."""
    if hasattr(results, 'sims') and results.sims is not None:
        # MultiSim: unpack sims and flatten each
        return {sim.label: sim.results.flatten() for sim in results.sims}
    if hasattr(results, 'results') and hasattr(results, 'label'):
        # Single Sim
        return {results.label: results.results.flatten()}
    if isinstance(results, dict):
        return results
    raise TypeError(
        f"results must be MultiSim, Sim, or dict of flat results; got {type(results).__name__}"
    )


def _validate_flat_results(flat_results):
    """Ensure each value is a dict of metric name -> result with timevec and values."""
    if not flat_results:
        raise ValueError("flat_results is empty")
    out = {}
    for label, flat in flat_results.items():
        if not isinstance(flat, dict):
            raise TypeError(f"flat_results[{label!r}] must be dict; got {type(flat).__name__}")
        normalized = {}
        for k, v in flat.items():
            sk = str(k)
            if not hasattr(v, 'timevec') or not hasattr(v, 'values'):
                raise TypeError(
                    f"flat_results[{label!r}][{sk}] must have timevec and values; got {type(v).__name__}"
                )
            normalized[sk] = v
        out[str(label)] = normalized
    return out


def _parse_exclude_patterns(items):
    """Return list of patterns from '~Something' or list like ['~None','~temp'] (strip leading ~)."""
    if not items:
        return []
    it = (items,) if isinstance(items, str) else items
    return [p[1:] if isinstance(p, str) and p.startswith('~') else str(p) for p in it if p]


def _select_metrics(all_metrics, select_spec, flat_results):
    """Filter all_metrics by select_spec (None/list/dict with items, like, regex, exclude)."""
    available = {m for m in all_metrics if any(m in flat for flat in flat_results.values())}

    if select_spec is None:
        exclude = _parse_exclude_patterns(['~None'])
        return sorted(m for m in available if not any(pat in m for pat in exclude))

    exclude_patterns = []
    resolved = set()

    if isinstance(select_spec, (list, tuple)):
        include_items = [x for x in select_spec if not (isinstance(x, str) and x.startswith('~'))]
        exclude_patterns = _parse_exclude_patterns([x for x in select_spec if isinstance(x, str) and x.startswith('~')])
        if include_items:
            resolved = {m for m in include_items if m in available}
        else:
            resolved = available
    elif isinstance(select_spec, dict):
        exclude_patterns = _parse_exclude_patterns(select_spec.get('exclude', []))
        if 'items' in select_spec:
            resolved |= {m for m in select_spec['items'] if m in available}
        if 'like' in select_spec:
            pats = select_spec['like'] if isinstance(select_spec['like'], (list, tuple)) else [select_spec['like']]
            for p in pats:
                resolved |= {m for m in available if p in m}
        if 'regex' in select_spec:
            pat = re.compile(select_spec['regex'])
            resolved |= {m for m in available if pat.search(m)}
    else:
        resolved = available

    exclude_fn = lambda m: any(pat in m for pat in exclude_patterns)
    return sorted(m for m in resolved if not exclude_fn(m))


def _fill_missing_metrics(flat_results, metrics):
    """Fill missing metrics in each scenario with zero series using a reference timevec."""
    ref_timevec = {}
    for flat in flat_results.values():
        for m in metrics:
            if m in flat and m not in ref_timevec:
                ref_timevec[m] = flat[m].timevec
    # Fill missing
    for label, flat in flat_results.items():
        for m in metrics:
            if m not in flat and m in ref_timevec:
                tv = ref_timevec[m]
                n = len(tv)
                # Create a Result-like object with zeros
                class _ZeroResult:
                    def __init__(self, timevec, values):
                        self.timevec = timevec
                        self.values = np.asarray(values, dtype=float)
                flat[m] = _ZeroResult(tv, np.zeros(n))
    return flat_results, ref_timevec


def _as_1d_xy(result):
    """Return (x, y) 1D arrays from a result with timevec and values; (None, None) if invalid."""
    if result is None or not hasattr(result, 'timevec') or not hasattr(result, 'values'):
        return None, None

    x = np.asarray(result.timevec)
    if x.size == 0:
        return None, None
    x = np.ravel(x)

    y = np.asarray(result.values)
    if y.size == 0:
        return x, np.asarray([])
    if y.ndim > 1:
        y = np.nanmean(y, axis=tuple(range(1, y.ndim)))
    y = np.ravel(y)

    n = min(len(x), len(y))
    return x[:n], y[:n]


def _safe_min_max(x):
    """Return (min, max) for array x, or (None, None) if empty or invalid."""
    if x is None:
        return None, None
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None, None
        return np.nanmin(a), np.nanmax(a)
    except (TypeError, ValueError):
        return None, None


def out_to(outdir):
    """Return a timestamped output file path inside outdir, creating the directory if needed."""
    try:
        timestamp = sc.now(dateformat='%Y%m%d_%H%M%S')
        if hasattr(sys.modules.get('__main__'), '__file__'):
            script_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        else:
            script_dir = os.getcwd()
        outdir = 'results' if outdir is None else outdir
        outdir = os.path.join(script_dir, outdir)
        os.makedirs(outdir, exist_ok=True)
        out = os.path.join(outdir, f'scenarios_{timestamp}.png')
        return out
    except OSError as e:
        raise OSError(f"Cannot create output path for figure: {e}") from e
    
