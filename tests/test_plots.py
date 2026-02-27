"""Test plotting utilities with real MultiSim and flattened results."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import tbsim

from tbsim.plots import (
    _as_1d_xy,
    _fill_missing_metrics,
    _normalize_results,
    _parse_exclude_patterns,
    _safe_min_max,
    _select_metrics,
    _validate_flat_results,
    plot,
)

n_agents = 300
do_plot = False
sc.options(interactive=False)  # Keep CI and pytest runs headless by default.


def make_three_sims():
    """Create three minimal TB sims for plotting tests."""
    pars = dict(dt=ss.days(14), start=ss.date('2000-01-01'), stop=ss.date('2002-12-31'), verbose=0)
    tb_pars = dict(init_prev=0.2, beta=ss.peryear(1.0))
    net_pars = dict(n_contacts=ss.poisson(lam=5), dur=30)
    sims = []
    for label in ('Scenario A', 'Scenario B', 'Scenario C'):
        sims.append(ss.Sim(
            label=label,
            n_agents=n_agents,
            diseases=tbsim.TB_LSHTM(name='tb', pars=tb_pars),
            networks=ss.RandomNet(pars=net_pars),
            pars=pars,
        ))
    return sims


def make_msim():
    """Run and return a MultiSim for this module."""
    msim = ss.MultiSim(sims=make_three_sims())
    msim.run(parallel=False)
    return msim


def make_flat_results(msim):
    """Return flattened results keyed by sim label."""
    return {sim.label: sim.results.flatten() for sim in msim.sims}


@pytest.fixture(scope='module')
def msim():
    return make_msim()


@pytest.fixture(scope='module')
def one_sim(msim):
    return msim.sims[0]


@pytest.fixture(scope='module')
def flat_results(msim):
    return make_flat_results(msim)


@sc.timer()
def test_normalize_results_from_multisim(msim, flat_results):
    out = _normalize_results(msim)
    assert set(out.keys()) == set(flat_results.keys()), 'Expected identical scenario labels from MultiSim normalization'
    for label in out.keys():
        assert set(out[label].keys()) == set(flat_results[label].keys()), f'Expected same metrics for scenario {label}'
        for key in out[label].keys():
            out_result = out[label][key]
            ref_result = flat_results[label][key]
            np.testing.assert_array_equal(np.ravel(out_result.timevec), np.ravel(ref_result.timevec))
            np.testing.assert_array_equal(np.ravel(out_result.values), np.ravel(ref_result.values))


@sc.timer()
def test_normalize_results_from_one_sim(one_sim):
    out = _normalize_results(one_sim)
    assert len(out) == 1, 'Expected a single scenario after normalizing one Sim'
    assert one_sim.label in out, f'Expected scenario label {one_sim.label} in normalized output'
    assert out[one_sim.label].keys() == one_sim.results.flatten().keys(), 'Expected identical flattened metrics'


@sc.timer()
def test_normalize_results_from_flat_dict(flat_results):
    assert _normalize_results(flat_results) is flat_results, 'Expected dict input to be returned unchanged'


@sc.timer()
def test_validate_flat_results_accepts_real_flat(flat_results):
    out = _validate_flat_results(flat_results)
    assert out.keys() == flat_results.keys(), 'Expected validated output to keep all scenario labels'
    for label in out.keys():
        for key, result in out[label].items():
            assert hasattr(result, 'timevec') and hasattr(result, 'values'), f'Expected {label}:{key} to include timevec and values'


@sc.timer()
def test_select_metrics_with_real_flat(flat_results):
    all_metrics = set()
    for flat in flat_results.values():
        all_metrics |= set(flat.keys())
    out = _select_metrics(all_metrics, None, flat_results)
    assert len(out) >= 1, 'Expected at least one metric to be selected by default'
    assert not any('None' in metric for metric in out), "Expected default metric selection to exclude names containing 'None'"


@sc.timer()
def test_select_metrics_string_and_like(flat_results):
    all_metrics = {metric for flat in flat_results.values() for metric in flat}
    a_metric = next(iter(all_metrics))
    out_str = _select_metrics(all_metrics, a_metric, flat_results)
    assert out_str == [a_metric], 'Expected string select input to choose one exact metric'

    out_like = _select_metrics(all_metrics, dict(like='incidence'), flat_results)
    assert isinstance(out_like, list), 'Expected like-based metric selection to return a list'


@sc.timer()
def test_fill_missing_metrics_with_real_flat(flat_results):
    metrics = list(next(iter(flat_results.values())).keys())[:2]
    copied = {label: dict(flat) for label, flat in flat_results.items()}
    filled, ref = _fill_missing_metrics(copied, metrics)
    for label in filled.keys():
        for metric in metrics:
            assert metric in filled[label], f'Expected metric {metric} in scenario {label} after filling'
            result = filled[label][metric]
            assert len(np.ravel(result.timevec)) == len(np.ravel(result.values)), 'Expected aligned time and value lengths for filled metrics'
            if metric in ref:
                np.testing.assert_array_equal(np.ravel(result.timevec), np.ravel(ref[metric]))
    for label in flat_results.keys():
        for metric in metrics:
            if metric in flat_results[label]:
                np.testing.assert_array_equal(filled[label][metric].values, flat_results[label][metric].values)


@sc.timer()
def test_as_1d_xy_with_real_result(flat_results):
    first_flat = next(iter(flat_results.values()))
    first_key = next(iter(first_flat.keys()))
    result = first_flat[first_key]
    x, y = _as_1d_xy(result)
    assert x is not None and y is not None and len(x) == len(y) and len(x) > 0, 'Expected non-empty aligned 1D x/y arrays'
    n = min(len(np.ravel(result.timevec)), len(np.ravel(result.values)))
    np.testing.assert_array_equal(x, np.ravel(result.timevec)[:n])
    y_expected = np.ravel(result.values) if np.asarray(result.values).ndim <= 1 else np.nanmean(np.asarray(result.values), axis=tuple(range(1, np.asarray(result.values).ndim)))
    np.testing.assert_array_equal(y, np.ravel(y_expected)[:n])


@sc.timer()
def test_safe_min_max_with_real_x(flat_results):
    first_flat = next(iter(flat_results.values()))
    first_key = next(iter(first_flat.keys()))
    result = first_flat[first_key]
    x, _ = _as_1d_xy(result)
    lo, hi = _safe_min_max(x)
    assert lo is not None and hi is not None and lo <= hi, 'Expected finite ordered min/max values'
    np.testing.assert_almost_equal(lo, np.nanmin(x))
    np.testing.assert_almost_equal(hi, np.nanmax(x))


@sc.timer()
def test_parse_exclude_patterns():
    assert _parse_exclude_patterns(['~None']) == ['None'], 'Expected leading "~" to be stripped from patterns'
    assert _parse_exclude_patterns([]) == [], 'Expected empty exclude list to remain empty'


@sc.timer()
def test_sim_results_plausible(flat_results):
    first_flat = next(iter(flat_results.values()))
    for name, result in list(first_flat.items())[:5]:
        x, y = _as_1d_xy(result)
        if x is not None and y is not None and len(y) > 0:
            assert np.all(np.isfinite(y)), f'Expected finite values for metric {name}'
            if 'n_' in name or 'incidence' in name or 'prevalence' in name:
                assert np.all(y >= 0), f'Expected non-negative values for metric {name}'


@sc.timer()
def test_plot_with_flat_results(flat_results):
    fig = plot(flat_results, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None, 'Expected plotting flat results to return a figure'


@sc.timer()
def test_plot_with_one_sim(one_sim):
    fig = plot(one_sim, savefig=False, n_cols=2, show=do_plot)
    assert fig is not None, 'Expected plotting one Sim to return a figure'


@sc.timer()
def test_plot_with_multisim(msim):
    fig = plot(msim, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None, 'Expected plotting MultiSim to return a figure'


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()

    msim = make_msim()
    one_sim = msim.sims[0]
    flat_results = make_flat_results(msim)

    test_normalize_results_from_multisim(msim, flat_results)
    test_normalize_results_from_one_sim(one_sim)
    test_normalize_results_from_flat_dict(flat_results)
    test_validate_flat_results_accepts_real_flat(flat_results)
    test_select_metrics_with_real_flat(flat_results)
    test_select_metrics_string_and_like(flat_results)
    test_fill_missing_metrics_with_real_flat(flat_results)
    test_as_1d_xy_with_real_result(flat_results)
    test_safe_min_max_with_real_x(flat_results)
    test_parse_exclude_patterns()
    test_sim_results_plausible(flat_results)
    test_plot_with_flat_results(flat_results)
    test_plot_with_one_sim(one_sim)
    test_plot_with_multisim(msim)

    T.toc()
