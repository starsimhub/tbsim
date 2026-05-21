"""
Plotting utilities for visualizing TB simulation results.

The main entry point is :func:`plot`, which accepts a ``MultiSim``, a single
``Sim``, or a pre-built flat results dict.  Use the ``select`` argument to
filter metrics (by name, substring, or regex) and ``style`` to
control appearance.  See :func:`plot` for the full parameter list and examples.
"""

import re
import sys
import numpy as np
import sciris as sc
import starsim as ss
import matplotlib.pyplot as plt
import networkx as nx

__all__ = ['plot', 'plot_household']


def plot(results, select=None, title='', filename=None, n_cols=None, row_height=1.5, style=None, show=True):
    """Plot simulation results (MultiSim, Sim, or flat results dict).

    Args:
        results:      MultiSim, single Sim, or dict of flat results per scenario.
        select:       What to plot. None = all. List of exact names, or a dict
                      with keys ``like`` (substring), ``regex``, ``items``, and/or
                      ``exclude``. Prefix a list entry with ``'~'`` to exclude it.
        title (str):  Figure suptitle.
        filename (str/path): Output filename; if provided, save figure
        n_cols (int): Number of subplot columns
        row_height (float): Row height in inches. Default 1.5.
        style (str):  Any ``ss.style()``-compatible style name (e.g. ``'starsim'``, ``'seaborn-v0_8-whitegrid'``).
        show (bool):  Call ``plt.show()`` when done.

    Returns:
        matplotlib.figure.Figure

    Examples::

        tbsim.plot(msim)                                           # all metrics
        tbsim.plot(msim, select=['n_infectious', 'prevalence'])    # exact names
        tbsim.plot(msim, select=dict(like='prevalence'))           # substring match
        tbsim.plot(msim, select=dict(regex=r'^n_'))                # regex match
        tbsim.plot(msim, select=['~15+', '~None'])                 # exclude patterns
        tbsim.plot(msim, style='starsim', n_cols=3)                # appearance
        tbsim.plot(msim, filename='abc.png')                       # save to disk
        tbsim.get_tb(sim).plot()                                   # via disease object
    """
    flat_results = _normalize_results(results)
    if not flat_results:
        sc.printred("No scenarios to plot.")
        return
    flat_results = _validate_flat_results(flat_results)

    all_metrics = {m for flat in flat_results.values() for m in flat}
    metrics = _select_metrics(all_metrics, select, flat_results)
    if not metrics:
        sc.printred("No metrics to plot.")
        return

    flat_results, _ = _fill_missing_metrics(flat_results, metrics)

    n_rows, n_cols  = sc.getrowscols(len(metrics), ncols=n_cols)

    with ss.style(style):
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

        for ax in axs:
            ax.set_axisbelow(True)
            for spine in ax.spines.values():
                spine.set_visible(False)

        _cm = plt.colormaps.get_cmap('tab10')
        all_handles = []
        all_labels = []
        seen = set()

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

            ax.set_title(metric_label, fontsize=12, fontweight='light')
            ax.set_xlabel('Time', fontsize=7)
            ax.tick_params(axis='both', labelsize=6)
            ax.grid(True, color='gray', alpha=0.25, linestyle='--', linewidth=0.2)
            if all_x_min is not None and all_x_max is not None:
                ax.set_xlim(all_x_min, all_x_max)

        for ax in axs[len(metrics):]:
            fig.delaxes(ax)

        if all_handles:
            leg = fig.legend(all_handles, all_labels, loc='upper left', fontsize=8, frameon=True, fancybox=True)
            leg.get_frame().set_alpha(0.9)

        if title:
            fig.suptitle(title, fontsize=13)

        plt.tight_layout(pad=2.0)

        if filename:
            filepath = sc.makefilepath(filename, makedirs=True)
            sc.savefig(filepath, fig=fig)

        if show:
            plt.show()

    return fig


def _rename_results(flat):
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
    use different module class names (e.g. ``TB`` vs ``TBAcute``).
    """
    if isinstance(results, ss.MultiSim):
        sims = results.sims
        return {sim.label: _rename_results(ss.utils.match_result_keys(sim.results, key=None)) for sim in sims}
    elif isinstance(results, ss.Sim):
        sim = results
        return {sim.label: _rename_results(ss.utils.match_result_keys(sim.results, key=None))} # TODO: can probably just be sim.results.flatten()?
    elif isinstance(results, dict):
        return results
    else:
        errormsg = f"Results must be MultiSim, Sim, or dict of flat results; got {type(results).__name__}"
        raise TypeError(errormsg)


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


def _as_1d_xy(result): # TODO: is this needed?
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


def _households_from_input(households_or_network, return_network=False):
    """Normalize household input to a list of UID lists."""
    # Case 1: explicit list of households
    if isinstance(households_or_network, (list, tuple)):
        households = [list(hh) for hh in households_or_network if len(hh)]
        if households:
            return (households, None) if return_network else households

    # Case 2: Sim passed in
    if isinstance(households_or_network, ss.Sim):
        sim = households_or_network
        hh_net = None
        for net in sim.networks.values():
            if hasattr(net, 'household_ids') or hasattr(net, 'hhs'):
                hh_net = net
                break
        if hh_net is None:
            raise ValueError('No household network found in provided sim')
    else:
        hh_net = households_or_network

    # Case 3: Household network with household_ids
    if hasattr(hh_net, 'household_ids'):
        hh_state = hh_net.household_ids
        hh_arr = np.asarray(hh_state, dtype=float)
        valid = np.isfinite(hh_arr) & (hh_arr >= 0)
        if not np.any(valid):
            return ([], hh_net) if return_network else []
        hh_ids = np.unique(hh_arr[valid]).astype(int)
        households = [list(np.asarray((hh_state == hhid).uids, dtype=int)) for hhid in hh_ids]
        households = [hh for hh in households if len(hh)]
        return (households, hh_net) if return_network else households

    # Case 4: tbsim HouseholdNet with hhs list
    if hasattr(hh_net, 'hhs'):
        households = [list(hh) for hh in hh_net.hhs if len(hh)]
        return (households, hh_net) if return_network else households

    raise TypeError(
        'households_or_network must be one of: list of households, ss.Sim, '
        'or household network with household_ids/hhs'
    )


def plot_household(
    households_or_network,
    title='Household Network Structure',
    figsize=(12, 8),
    max_households=30,
    edge_mode='auto',
    layout='ring',
    layout_seed=123,
    show_labels=False,
    show=True,
    filename=None,
):
    """
    Plot the structure of households and intra-household connections.

    This function creates a network visualization where:
    - Nodes represent individual agents
    - Edges represent household connections (actual network edges by default)
    - Different colors represent different households
    - Node size is proportional to household size

    Args:
        households_or_network: one of:
            - list of households (each household is a list/array of UIDs)
            - an ``ss.Sim`` with a network exposing ``household_ids``
            - a household network exposing ``household_ids`` or ``hhs``
        title (str): Title for the plot
        figsize (tuple): Figure size (width, height)
        max_households (int/None): maximum households to render (largest first).
            Use None to render all (can be very dense for large sims).
        edge_mode (str): one of ``'auto'``, ``'actual'``, ``'complete'``.
            - ``'actual'``: draw true edges from the network object
            - ``'complete'``: reconstruct full within-household cliques
            - ``'auto'`` (default): use ``'actual'`` for sim/network input and
              ``'complete'`` for explicit household-list input
        layout (str): one of ``'ring'``, ``'grid'``, ``'spring'``, ``'kamada'``.
            - ``'ring'``: household-clustered circular layout (legacy style)
            - ``'grid'``: household clusters arranged in rows/columns
            - ``'spring'``: force-directed layout (less donut-like)
            - ``'kamada'``: Kamada-Kawai layout (compact force-directed)
        layout_seed (int): random seed used for ``layout='spring'``.
        show_labels (bool): whether to draw UID labels inside nodes.
            Defaults to False for cleaner household figures.
        show (bool): call ``plt.show()``.
        filename (str/path/None): optional path to save figure.

    Returns:
        networkx.Graph: The NetworkX graph object for further analysis
    """
    households, hh_net = _households_from_input(households_or_network, return_network=True)
    if len(households) == 0:
        raise ValueError('No households available to plot')

    if edge_mode not in ['auto', 'actual', 'complete']:
        raise ValueError(f"edge_mode must be one of ['auto', 'actual', 'complete'], not {edge_mode!r}")
    if layout not in ['ring', 'grid', 'spring', 'kamada']:
        raise ValueError(f"layout must be one of ['ring', 'grid', 'spring', 'kamada'], not {layout!r}")

    has_network_edges = (
        hh_net is not None and
        hasattr(hh_net, 'edges') and
        hasattr(hh_net.edges, 'p1') and
        hasattr(hh_net.edges, 'p2')
    )

    if edge_mode == 'auto':
        use_actual_edges = has_network_edges
    elif edge_mode == 'actual':
        if not has_network_edges:
            raise ValueError("edge_mode='actual' requires a sim/network input with edge arrays")
        use_actual_edges = True
    else:
        use_actual_edges = False

    # For readability, cap rendered households (largest first)
    if max_households is not None and len(households) > max_households:
        households = sorted(households, key=len, reverse=True)[:max_households]

    # Create NetworkX graph
    G = nx.Graph()

    # Add all agents as nodes
    all_agents = [agent for hh in households for agent in hh]
    G.add_nodes_from(all_agents)

    # Add household membership as node attributes
    node_to_hh = {}
    for hh_idx, household in enumerate(households):
        for agent in household:
            G.nodes[agent]['household'] = hh_idx
            G.nodes[agent]['household_size'] = len(household)
            node_to_hh[int(agent)] = hh_idx

    # Add edges within households
    edges_by_hh = {hh_idx: [] for hh_idx in range(len(households))}
    if use_actual_edges:
        p1 = np.asarray(hh_net.edges.p1, dtype=int)
        p2 = np.asarray(hh_net.edges.p2, dtype=int)
        for a, b in zip(p1, p2):
            ha = node_to_hh.get(int(a), None)
            hb = node_to_hh.get(int(b), None)
            if ha is None or hb is None:
                continue
            if ha == hb:
                edges_by_hh[ha].append((int(a), int(b)))
    else:
        for hh_idx, household in enumerate(households):
            if len(household) > 1:
                for i in range(len(household)):
                    for j in range(i + 1, len(household)):
                        edges_by_hh[hh_idx].append((household[i], household[j]))

    for hh_idx, hh_edges in edges_by_hh.items():
        if len(hh_edges):
            G.add_edges_from(hh_edges, household=hh_idx)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colors for households (continuous map gives many distinct colors)
    cmap = plt.colormaps.get_cmap('gist_ncar')
    household_colors = cmap(np.linspace(0.02, 0.98, len(households)))

    # Create layout positions
    if layout == 'ring':
        pos = {}
        if len(households) == 1:
            # Single household - use circular layout
            pos = nx.circular_layout(G)
        else:
            # Multiple households - arrange in circle with internal clustering
            household_angles = np.linspace(0, 2*np.pi, len(households), endpoint=False)

            for hh_idx, household in enumerate(households):
                # Base position for this household
                base_radius = 3.0
                hh_x = base_radius * np.cos(household_angles[hh_idx])
                hh_y = base_radius * np.sin(household_angles[hh_idx])

                if len(household) == 1:
                    pos[household[0]] = (hh_x, hh_y)
                else:
                    # Arrange household members in a small circle
                    agent_angles = np.linspace(0, 2*np.pi, len(household), endpoint=False)
                    inner_radius = 0.3 + 0.1 * len(household)

                    for agent_idx, agent in enumerate(household):
                        pos[agent] = (hh_x + inner_radius * np.cos(agent_angles[agent_idx]),
                                      hh_y + inner_radius * np.sin(agent_angles[agent_idx]))
    elif layout == 'grid':
        pos = {}
        n_hh = len(households)
        n_cols = max(1, int(np.ceil(np.sqrt(n_hh))))
        x_spacing = 3.0
        y_spacing = 3.0

        for hh_idx, household in enumerate(households):
            row = hh_idx // n_cols
            col = hh_idx % n_cols
            hh_x = col * x_spacing
            hh_y = -row * y_spacing

            if len(household) == 1:
                pos[household[0]] = (hh_x, hh_y)
            else:
                agent_angles = np.linspace(0, 2*np.pi, len(household), endpoint=False)
                inner_radius = 0.25 + 0.08 * np.sqrt(len(household))
                for agent_idx, agent in enumerate(household):
                    pos[agent] = (hh_x + inner_radius * np.cos(agent_angles[agent_idx]),
                                  hh_y + inner_radius * np.sin(agent_angles[agent_idx]))

        if len(pos):
            xs = np.array([xy[0] for xy in pos.values()])
            ys = np.array([xy[1] for xy in pos.values()])
            x0 = 0.5 * (xs.min() + xs.max())
            y0 = 0.5 * (ys.min() + ys.max())
            for k in pos.keys():
                x, y = pos[k]
                pos[k] = (x - x0, y - y0)
    elif layout == 'spring':
        pos = nx.spring_layout(G, seed=int(layout_seed))
    else:
        pos = nx.kamada_kawai_layout(G)

    # Draw network by household
    for hh_idx, household in enumerate(households):
        # Node sizes proportional to household size
        node_sizes = [80 + 12 * len(household) for _ in household]

        # Draw nodes for this household
        node_positions = {node: pos[node] for node in household}
        nx.draw_networkx_nodes(
            G,
            node_positions,
            nodelist=household,
            node_color=[household_colors[hh_idx]] * len(household),
            node_size=node_sizes,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.8,
            ax=ax,
        )

        # Draw edges for this household
        household_edges = edges_by_hh[hh_idx]
        if household_edges:
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=household_edges,
                edge_color=household_colors[hh_idx],
                width=1.0,
                alpha=0.6,
                ax=ax,
            )

    # Draw labels (optional; defaults off to avoid clutter)
    if show_labels and len(all_agents) <= 200:
        nx.draw_networkx_labels(G, pos, font_size=6, font_weight='normal', font_color='black', ax=ax)

    # Add axis limits with padding so edge nodes/labels aren't clipped
    if len(pos):
        xs = np.array([xy[0] for xy in pos.values()])
        ys = np.array([xy[1] for xy in pos.values()])
        dx = xs.max() - xs.min()
        dy = ys.max() - ys.min()
        pad_x = max(0.5, 0.08 * dx)
        pad_y = max(0.5, 0.08 * dy)
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    # Create legend
    legend_elements = []
    for hh_idx, household in enumerate(households):
        legend_elements.append(plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=household_colors[hh_idx],
            markersize=8, alpha=0.8, markeredgecolor='black',
            label=f'Household {hh_idx + 1} (n={len(household)})'
        ))

    total_agents = len(all_agents)
    total_edges = G.number_of_edges()
    avg_household_size = np.mean([len(hh) for hh in households])
    edge_mode_label = 'actual' if use_actual_edges else 'complete'
    stats_summary = (
        f'Network Statistics | agents: {total_agents} | households: {len(households)} | '
        f'edges: {total_edges} | edge mode: {edge_mode_label} | avg household size: {avg_household_size:.1f}'
    )

    if len(households) <= 25:
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.text(0.02, 0.015, stats_summary, ha='left', va='bottom', fontsize=8)
    ax.axis('off')
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.95))

    if filename is not None:
        fig.savefig(filename, dpi=200, bbox_inches='tight', pad_inches=0.25)
    if show:
        plt.show()

    # Print network statistics

    print('Network Statistics:')
    print(f'  Total agents: {total_agents}')
    print(f'  Total households: {len(households)}')
    print(f'  Total edges: {total_edges}')
    print(f'  Edge mode: {edge_mode_label}')
    print(f'  Average household size: {avg_household_size:.1f}')
    if len(households) <= 30:
        print(f'  Household sizes: {[len(hh) for hh in households]}')

    return G