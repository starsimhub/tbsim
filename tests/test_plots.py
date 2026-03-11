"""Tests for tbsim plotting utilities."""

import pathlib
import sys

import matplotlib.pyplot as plt
import pytest
import sciris as sc
import starsim as ss

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tbsim

do_plot = False


def flatten_by_result_name(flat):
    return {
        (result.name.lower() if getattr(result, 'name', None) else key): result
        for key, result in flat.items()
    }


def make_msim(use_alt_model=False):
    pars = dict(dt=ss.days(14), start=ss.date('2000-01-01'), stop=ss.date('2002-12-31'), verbose=0)
    sims = []
    for i, label in enumerate(('Scenario A', 'Scenario B', 'Scenario C')):
        disease_cls = ss.SIS if (use_alt_model and i == 0) else ss.SIR
        sims.append(ss.Sim(
            label=label,
            n_agents=300,
            diseases=disease_cls(),
            networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30)),
            pars=pars,
        ))
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=False)
    return msim


def make_flat_results(msim):
    return {
        sim.label: flatten_by_result_name(ss.utils.match_result_keys(sim.results, key=None))
        for sim in msim.sims
    }


@pytest.fixture(scope='module')
def msim():
    return make_msim()


@pytest.fixture(scope='module')
def mixed_msim():
    return make_msim(use_alt_model=True)


@pytest.fixture(scope='module')
def flat_results(msim):
    return make_flat_results(msim)


@sc.timer()
def test_plot_accepts_all_input_types(msim, flat_results):
    """The public API should work for Sim, MultiSim, and flat dict inputs."""
    fig = tbsim.plot(msim.sims[0], savefig=False, n_cols=2, show=do_plot)
    assert fig is not None
    plt.close(fig)

    fig = tbsim.plot(msim, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None
    plt.close(fig)

    fig = tbsim.plot(flat_results, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None
    plt.close(fig)


@sc.timer()
def test_plot_handles_mixed_tb_modules(mixed_msim):
    """Differently named model results should still align on shared metrics."""
    fig = tbsim.plot(mixed_msim, select=['n_infected', 'prevalence'], savefig=False, n_cols=2, show=do_plot)
    assert fig is not None
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert any('Infect' in title for title in titles)
    assert any('Prevalence' in title for title in titles)
    plt.close(fig)


@sc.timer()
def test_plot_select_and_fill_missing_metric(flat_results):
    """Selected metrics should still plot when one scenario is missing them."""
    patched = {label: flat.copy() for label, flat in flat_results.items()}
    metric = 'prevalence'
    patched['Scenario B'].pop(metric, None)
    fig = tbsim.plot(patched, select=[metric], savefig=False, n_cols=1, show=do_plot)
    assert fig is not None
    assert len(fig.axes[0].lines) == len(patched)
    assert 'Prevalence' in fig.axes[0].get_title()
    plt.close(fig)


@sc.timer()
def test_plot_saves_and_falls_back_from_bad_style(msim, tmp_path):
    """Saving should work even when the requested matplotlib style is invalid."""
    fig = tbsim.plot(
        msim,
        select=['prevalence'],
        savefig=True,
        output_dir=str(tmp_path),
        style=dict(mpl_style='definitely-not-a-style'),
        n_cols=1,
        show=do_plot,
    )
    assert fig is not None
    assert list(tmp_path.glob('scenarios_*.png'))
    plt.close(fig)


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    msim = make_msim()
    mixed_msim = make_msim(use_alt_model=True)
    flat_results = make_flat_results(msim)
    test_plot_accepts_all_input_types(msim, flat_results)
    test_plot_handles_mixed_tb_modules(mixed_msim)
    test_plot_select_and_fill_missing_metric(flat_results)
    T.toc()
