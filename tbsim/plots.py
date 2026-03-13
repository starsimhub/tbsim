"""
Plotting utilities for visualizing TB simulation results.

The main entry point is :func:`plot`, which accepts a ``MultiSim``, a single
``Sim``, or a pre-built flat results dict.  Use the ``select`` argument to
filter metrics (by name, substring, or regex) and ``theme``/``style`` to
control appearance.  See :func:`plot` for the full parameter list and examples.
"""

import re
import sys

import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt

__all__ = ['plot']


def plot(results, select=None, title='', savefig=False, filename='tbsim.png',
         output_dir=None, theme='light', n_cols=6, row_height=1.5, style=None, show=True):
    """Plot simulation results (MultiSim, Sim, or flat results dict).

    Args:
        results:      MultiSim, single Sim, or dict of flat results per scenario.
        select:       Metrics to plot. None = all. List of exact names, or a dict
                      with keys ``like`` (substring), ``regex``, ``items``, and/or
                      ``exclude``. Prefix a list entry with ``'~'`` to exclude it.
        title (str):  Figure suptitle.
        savefig (bool): Save the figure to disk.
        filename (str): Output filename. Default ``'tbsim.png'``.
        output_dir (str): Output folder. Default ``'results'``.
        theme (str):  ``'light'`` or ``'dark'``.
        n_cols (int): Number of subplot columns. Default 6.
        row_height (float): Row height in inches. Default 1.5.
        style (str):  Any ``ss.style()``-compatible style name
                      (e.g. ``'starsim'``, ``'seaborn-v0_8-whitegrid'``).
        show (bool):  Call ``plt.show()`` when done.

    Returns:
        matplotlib.figure.Figure

    Examples::

        tbsim.plot(msim)                                           # all metrics
        tbsim.plot(msim, select=['n_infectious', 'prevalence'])    # exact names
        tbsim.plot(msim, select=dict(like='prevalence'))           # substring match
        tbsim.plot(msim, select=dict(regex=r'^n_'))                # regex match
        tbsim.plot(msim, select=['~15+', '~None'])                 # exclude patterns
        tbsim.plot(msim, theme='dark', style='starsim', n_cols=3)  # appearance
        tbsim.plot(msim, savefig=True, filename='abc.png')         # save to disk
        tbsim.get_tb(sim).plot()                                   # via disease object
    """
    try:
        flat_results = _normalize_results(results)
        if not flat_results:
            sc.printred("No scenarios to plot.")
            return
        flat_results = _validate_flat_results(flat_results)
    except (TypeError, ValueError) as e:
        sc.printred(f"Cannot plot: {e}")
        return

    dark = str(theme).lower() == 'dark'
    fg = '#eaeaea' if dark else '#111'
    ax_bg = '#2b2b2b' if dark else '#f0f0f0'
    fig_bg = '#1e1e1e' if dark else '#f7f7f7'

    all_metrics = {m for flat in flat_results.values() for m in flat}
    metrics = _select_metrics(all_metrics, select, flat_results)
    if not metrics:
        sc.printred("No metrics to plot.")
        return

    flat_results, _ = _fill_missing_metrics(flat_results, metrics)

    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(metrics) / n_cols))

    try:
        style_ctx = ss.style(style)
    except (ValueError, OSError):
        sc.printyellow(f"Warning: style '{style}' not found; using default.")
        style_ctx = ss.style(None)

    with style_ctx:
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
            for spine in ax.spines.values():
                spine.set_visible(False)

        try:
            _cm = plt.colormaps.get_cmap('tab10')
        except (AttributeError, KeyError):
            _cm = plt.cm.get_cmap('tab10')

        all_handles, all_labels, seen = [], [], set()

        for i, metric in enumerate(metrics):
            ax = axs[i]
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
                line, = ax.plot(x, y, lw=1.2, label=scen, color=_cm(j), alpha=0.85, zorder=10 if j == 0 else 5)
                if scen not in seen:
                    all_handles.append(line)
                    all_labels.append(scen)
                    seen.add(scen)

            ax.set_title(metric_label, fontsize=10, fontweight='light', color=fg)
            ax.set_xlabel('Time', fontsize=6, color=fg)
            ax.tick_params(axis='both', labelsize=6, colors=fg)
            ax.grid(True, color='white' if dark else 'gray', alpha=0.25, linestyle='--', linewidth=0.2)
            if all_x_min is not None and all_x_max is not None:
                ax.set_xlim(all_x_min, all_x_max)

        for ax in axs[len(metrics):]:
            fig.delaxes(ax)

        if all_handles:
            leg = fig.legend(all_handles, all_labels, loc='upper left', fontsize=7, frameon=True, fancybox=True)
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_facecolor(fig_bg)
            for t in leg.get_texts():
                t.set_color(fg)

        if title:
            fig.suptitle(title, fontsize=12, color=fg)

        plt.tight_layout(pad=2.0)

        if savefig:
            out = _out_path(filename, output_dir)
            sc.savefig(out, fig=fig)

        if show:
            plt.show()

    return fig


def _flatten_by_result_name(flat):
    """Re-key a flat result dict using each result's own name, dropping the module prefix."""
    out = {}
    for prefixed_key, result in flat.items():
        key = result.name.lower() if getattr(result, 'name', None) else prefixed_key
        out[key] = result
    return out


def _normalize_results(results):
    """Convert MultiSim, Sim, or dict into ``{label: flat_result_dict}``.

    Uses ``ss.utils.match_result_keys(key=None)`` to flatten each sim's
    results, then re-keys by result name so metrics align across sims that
    use different module class names (e.g. ``TB_LSHTM`` vs ``TB_LSHTM_Acute``).
    """
    if hasattr(results, 'sims') and results.sims is not None:
        return {sim.label: _flatten_by_result_name(ss.utils.match_result_keys(sim.results, key=None))
                for sim in results.sims}
    if hasattr(results, 'results') and hasattr(results, 'label'):
        return {results.label: _flatten_by_result_name(ss.utils.match_result_keys(results.results, key=None))}
    if isinstance(results, dict):
        return results
    raise TypeError(f"results must be MultiSim, Sim, or dict of flat results; got {type(results).__name__}")


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
    """Return exclude-pattern strings, stripping a leading ``'~'`` if present."""
    if not items:
        return []
    it = (items,) if isinstance(items, str) else items
    return [p[1:] if isinstance(p, str) and p.startswith('~') else str(p) for p in it if p]


def _select_metrics(all_metrics, select_spec, flat_results):
    """Return a sorted list of result names to plot, filtered by *select_spec*.

    Args:
        all_metrics (set): All metric names found across all scenarios.
        select_spec (None/list/dict): Filter specification; see :func:`plot`.
        flat_results (dict): The flat results dict (used to check availability).

    Returns:
        list[str]: Sorted metric names to plot.
    """
    available = {m for m in all_metrics if any(m in flat for flat in flat_results.values())}

    if select_spec is None:
        exclude = _parse_exclude_patterns(['~None'])
        return sorted(m for m in available if not any(pat in m for pat in exclude))

    exclude_patterns = []
    resolved = set()

    if isinstance(select_spec, str):
        resolved = {select_spec} if select_spec in available else set()
    elif isinstance(select_spec, (list, tuple, set)):
        include_items = [x for x in select_spec if not (isinstance(x, str) and x.startswith('~'))]
        exclude_patterns = _parse_exclude_patterns([x for x in select_spec if isinstance(x, str) and x.startswith('~')])
        resolved = {m for m in include_items if m in available} if include_items else available
    elif isinstance(select_spec, dict):
        exclude_patterns = _parse_exclude_patterns(select_spec.get('exclude', []))
        if 'items' in select_spec:
            resolved |= {m for m in sc.tolist(select_spec['items']) if m in available}
        if 'like' in select_spec:
            for p in sc.tolist(select_spec['like']):
                resolved |= {m for m in available if str(p) in m}
        if 'regex' in select_spec:
            pat = re.compile(str(select_spec['regex']))
            resolved |= {m for m in available if pat.search(m)}
        if not any(k in select_spec for k in ('items', 'like', 'regex')):
            resolved = available
    else:
        resolved = available

    return sorted(m for m in resolved if not any(pat in m for pat in exclude_patterns))


def _fill_missing_metrics(flat_results, metrics):
    """Fill missing result names in each scenario with a zero time series."""
    ref_timevec = {}
    for flat in flat_results.values():
        for m in metrics:
            if m in flat and m not in ref_timevec:
                ref_timevec[m] = flat[m].timevec
    for flat in flat_results.values():
        for m in metrics:
            if m not in flat and m in ref_timevec:
                tv = ref_timevec[m]
                flat[m] = _ZeroResult(tv, np.zeros(len(tv)))
    return flat_results, ref_timevec


class _ZeroResult:
    """Minimal result-like object used to pad missing metrics with zeros."""

    def __init__(self, timevec, values):
        self.timevec = timevec
        self.values = np.asarray(values, dtype=float)


def _as_1d_xy(result):
    """Return ``(x, y)`` 1-D arrays from a result object; ``(None, None)`` if invalid."""
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
    """Return ``(min, max)`` for array *x*, or ``(None, None)`` if empty/invalid."""
    if x is None:
        return None, None
    try:
        a = np.asarray(x)
        if a.size == 0:
            return None, None
        return np.nanmin(a), np.nanmax(a)
    except (TypeError, ValueError):
        return None, None


_run_timestamp = None  # shared across all plot() calls in one Python session


def _out_path(filename, output_dir):
    """Return a full output path inside a timestamped run subfolder.
    """
    global _run_timestamp
    if _run_timestamp is None:
        _run_timestamp = sc.now(dateformat='%Y%m%d_%H%M%S')
    try:
        main = sys.modules.get('__main__')
        script_dir = sc.thispath(main.__file__, aspath=False) if hasattr(main, '__file__') else sc.thispath(aspath=False)
        output_dir = 'results' if output_dir is None else output_dir
        return sc.makefilepath(filename, folder=[script_dir, output_dir, _run_timestamp], makedirs=True)
    except OSError as e:
        raise OSError(f"Cannot create output path for figure: {e}") from e
