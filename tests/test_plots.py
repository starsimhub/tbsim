"""Tests for tbsim plotting utilities."""

import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import tbsim
from tbsim.plots import _flatten_by_result_name

do_plot = False


def make_msim(use_acute=False):
    pars = dict(dt=ss.days(14), start=ss.date('2000-01-01'), stop=ss.date('2002-12-31'), verbose=0)
    tb_pars = dict(init_prev=0.2, beta=ss.peryear(1.0))
    sims = [
        ss.Sim(label=label, n_agents=300,
               diseases=(tbsim.TB_LSHTM_Acute if (use_acute and i == 0) else tbsim.TB_LSHTM)(name='tb', pars=tb_pars),
               networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30)),
               pars=pars)
        for i, label in enumerate(('Scenario A', 'Scenario B', 'Scenario C'))
    ]
    msim = ss.MultiSim(sims=sims)
    msim.run(parallel=False)
    return msim


def make_flat_results(msim):
    return {
        sim.label: _flatten_by_result_name(ss.utils.match_result_keys(sim.results, key=None))
        for sim in msim.sims
    }


def test_plot_accepts_all_input_types():
    """The public API works for Sim, MultiSim, and flat dict inputs."""
    msim = make_msim()
    flat = make_flat_results(msim)

    fig = tbsim.plot(msim.sims[0], savefig=False, n_cols=2, show=do_plot)
    assert fig is not None
    plt.close(fig)

    fig = tbsim.plot(msim, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None
    plt.close(fig)

    fig = tbsim.plot(flat, savefig=False, n_cols=3, show=do_plot)
    assert fig is not None
    plt.close(fig)


def test_plot_handles_mixed_tb_modules():
    """TB_LSHTM and TB_LSHTM_Acute results align on shared metrics."""
    msim = make_msim(use_acute=True)
    fig = tbsim.plot(msim, select=['n_infectious', 'prevalence_active'], savefig=False, n_cols=2, show=do_plot)
    assert fig is not None
    titles = {ax.get_title() for ax in fig.axes if ax.get_title()}
    assert any('Infectious' in t or 'infectious' in t for t in titles)
    plt.close(fig)


def test_plot_select_and_fill_missing_metric():
    """A metric missing in one scenario is zero-filled and still plotted."""
    msim = make_msim()
    flat = make_flat_results(msim)
    patched = {label: d.copy() for label, d in flat.items()}
    metric = 'n_infectious'
    patched['Scenario B'].pop(metric, None)
    fig = tbsim.plot(patched, select=[metric], savefig=False, n_cols=1, show=do_plot)
    assert fig is not None
    assert len(fig.axes[0].lines) == len(patched)
    plt.close(fig)


def test_plot_saves_and_falls_back_from_bad_style(tmp_path, capsys):
    """savefig works and an invalid style name falls back to the default."""
    msim = make_msim()
    fig = tbsim.plot(
        msim,
        select=['n_infectious'],
        savefig=True,
        filename='test_out.png',
        output_dir=str(tmp_path),
        style='definitely-not-a-style',
        n_cols=1,
        show=do_plot,
    )
    assert fig is not None
    assert list(tmp_path.rglob('test_out.png'))  # file lives in a timestamped subfolder
    assert 'Warning' in capsys.readouterr().out
    plt.close(fig)


if __name__ == '__main__':
    do_plot = True
    sc.options(interactive=do_plot)
    T = sc.timer()
    test_plot_accepts_all_input_types()
    test_plot_handles_mixed_tb_modules()
    test_plot_select_and_fill_missing_metric()
    T.toc()
