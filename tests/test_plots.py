"""Test tbsim.plots with real MultiSim and flattened results (up to 150 lines)."""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest
import starsim as ss
import tbsim
from tbsim.plots import (_as_1d_xy, _fill_missing_metrics, _normalize_results, _parse_exclude_patterns, _safe_min_max, _select_metrics, _validate_flat_results, plot)

def make_three_sims():
    """Three minimal TB_LSHTM sims with short runs for fast tests."""
    pars = dict(dt=ss.days(14), start=ss.date('2000-01-01'), stop=ss.date('2002-12-31'), verbose=0)
    tb_pars = dict(init_prev=0.2, beta=ss.peryear(1.0))
    net_pars = dict(n_contacts=ss.poisson(lam=5), dur=30)
    sims = []
    for label in ('Scenario A', 'Scenario B', 'Scenario C'):
        sims.append(ss.Sim(
            label=label,
            diseases=tbsim.TB_LSHTM(name='tb', pars=tb_pars),
            networks=ss.RandomNet(pars=net_pars),
            pars=pars,
        ))
    return sims

@pytest.fixture(scope='module')
def msim():
    """Run 3 sims as MultiSim once per test module."""
    sims = make_three_sims()
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=False)
    return msim


@pytest.fixture(scope='module')
def one_sim(msim):
    """Extract a single Sim from the MultiSim."""
    return msim.sims[0]


@pytest.fixture(scope='module')
def flat_results(msim):
    """Flattened results from MultiSim: dict label -> flat result dict."""
    return {sim.label: sim.results.flatten() for sim in msim.sims}


def test_normalize_results_from_multisim(msim, flat_results):
    """_normalize_results(MultiSim) equals manual flatten: same labels, metrics, and data."""
    out = _normalize_results(msim)
    assert set(out.keys()) == set(flat_results.keys())
    for label in out:
        assert set(out[label].keys()) == set(flat_results[label].keys())
        for k in out[label]:
            o, f = out[label][k], flat_results[label][k]
            np.testing.assert_array_equal(np.ravel(o.timevec), np.ravel(f.timevec))
            np.testing.assert_array_equal(np.ravel(o.values), np.ravel(f.values))


def test_normalize_results_from_one_sim(one_sim):
    """_normalize_results(single Sim) returns one scenario."""
    out = _normalize_results(one_sim)
    assert len(out) == 1
    assert one_sim.label in out
    assert out[one_sim.label].keys() == one_sim.results.flatten().keys()


def test_normalize_results_from_flat_dict(flat_results):
    """_normalize_results(dict) returns the same dict."""
    assert _normalize_results(flat_results) is flat_results


def test_validate_flat_results_accepts_real_flat(flat_results):
    """_validate_flat_results accepts flattened sim results."""
    out = _validate_flat_results(flat_results)
    assert out.keys() == flat_results.keys()
    for label in out:
        for k, v in out[label].items():
            assert hasattr(v, 'timevec') and hasattr(v, 'values')


def test_select_metrics_with_real_flat(flat_results):
    """_select_metrics filters real metric names; default excludes 'None'."""
    all_metrics = set()
    for flat in flat_results.values():
        all_metrics |= set(flat.keys())
    out = _select_metrics(all_metrics, None, flat_results)
    assert len(out) >= 1
    assert not any('None' in m for m in out)


def test_select_metrics_like_with_real_flat(flat_results):
    all_metrics = {m for flat in flat_results.values() for m in flat}
    out = _select_metrics(all_metrics, dict(like='incidence'), flat_results)
    assert any('incidence' in m for m in out) or len(out) == 0


def test_fill_missing_metrics_with_real_flat(flat_results):
    """_fill_missing_metrics preserves existing data; filled series match ref timevec length."""
    metrics = list(next(iter(flat_results.values())).keys())[:2]
    filled, ref = _fill_missing_metrics(flat_results, metrics)
    for label in filled:
        for m in metrics:
            assert m in filled[label]
            r = filled[label][m]
            assert len(np.ravel(r.timevec)) == len(np.ravel(r.values))
            if m in ref:
                np.testing.assert_array_equal(np.ravel(r.timevec), np.ravel(ref[m]))
    for label in flat_results:
        for m in metrics:
            if m in flat_results[label]:
                np.testing.assert_array_equal(filled[label][m].values, flat_results[label][m].values)


def test_as_1d_xy_with_real_result(flat_results):
    """_as_1d_xy returns correct 1D arrays: x from timevec, y from values (same length)."""
    first_flat = next(iter(flat_results.values()))
    first_key = next(iter(first_flat.keys()))
    r = first_flat[first_key]
    x, y = _as_1d_xy(r)
    assert x is not None and y is not None and len(x) == len(y) and len(x) > 0
    n = min(len(np.ravel(r.timevec)), len(np.ravel(r.values)))
    np.testing.assert_array_equal(x, np.ravel(r.timevec)[:n])
    y_expected = np.ravel(r.values) if np.asarray(r.values).ndim <= 1 else np.nanmean(np.asarray(r.values), axis=tuple(range(1, np.asarray(r.values).ndim)))
    np.testing.assert_array_equal(y, np.ravel(y_expected)[:n])


def test_safe_min_max_with_real_x(flat_results):
    """_safe_min_max returns correct (nanmin, nanmax) of the array."""
    first_flat = next(iter(flat_results.values()))
    first_key = next(iter(first_flat.keys()))
    r = first_flat[first_key]
    x, _ = _as_1d_xy(r)
    lo, hi = _safe_min_max(x)
    assert lo is not None and hi is not None and lo <= hi
    np.testing.assert_almost_equal(lo, np.nanmin(x))
    np.testing.assert_almost_equal(hi, np.nanmax(x))


def test_parse_exclude_patterns():
    """_parse_exclude_patterns strips leading tilde and handles empty."""
    assert _parse_exclude_patterns(['~None']) == ['None']
    assert _parse_exclude_patterns([]) == []


def test_sim_results_plausible(flat_results):
    """Flattened sim results: key series are finite and (where relevant) non-negative."""
    first_flat = next(iter(flat_results.values()))
    for name, r in list(first_flat.items())[:5]:
        x, y = _as_1d_xy(r)
        if x is not None and y is not None and len(y) > 0:
            assert np.all(np.isfinite(y)), f"{name} should be finite"
            if 'n_' in name or 'incidence' in name or 'prevalence' in name:
                assert np.all(y >= 0), f"{name} should be non-negative"


def test_plot_with_flat_results(flat_results):
    plot(flat_results, savefig=False, n_cols=3)


def test_plot_with_one_sim(one_sim):
    plot(one_sim, savefig=False, n_cols=2)


def test_plot_with_multisim(msim):
    plot(msim, savefig=False, n_cols=3)
