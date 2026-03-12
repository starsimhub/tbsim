"""Tests for the DxDelivery diagnostic intervention."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_sim(n_agents=1000, interventions=None):
    """Create and return a tbsim.Sim with HSB and optional interventions."""
    hsb = tbsim.HealthSeekingBehavior()
    all_interventions = [hsb] + (interventions or [])
    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=all_interventions,
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), rand_seed=42),
        tb_pars=dict(init_prev=0.30),
    )
    return sim


def test_dx_delivery_runs():
    """DxDelivery completes a full run with HealthSeekingBehavior."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    sim = make_sim(interventions=[dx])
    sim.run()

    assert sim.results.dxdelivery.n_tested.sum() > 0
    assert sim.results.dxdelivery.n_positive.sum() > 0


def test_dx_delivery_diagnoses_agents():
    """DxDelivery sets diagnosed=True for positive results."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    sim = make_sim(n_agents=1000, interventions=[dx])
    sim.run()

    n_diagnosed = sim.people.diagnosed.count()
    assert n_diagnosed > 0


def test_dx_delivery_custom_result_state():
    """DxDelivery with custom result_state auto-registers and sets that state."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert(), result_state='screen_positive')
    sim = make_sim(interventions=[dx])
    sim.run()

    assert 'screen_positive' in sim.people.states
    n_screen_pos = sim.people.screen_positive.count()
    assert n_screen_pos > 0


def test_dx_delivery_cascade():
    """Two DxDelivery steps can be chained: screen -> confirm."""
    screen = tbsim.DxDelivery(
        name='screen',
        product=tbsim.CAD(),
        coverage=0.9,
        result_state='screen_positive',
    )
    confirm = tbsim.DxDelivery(
        name='confirm',
        product=tbsim.Xpert(),
        coverage=0.8,
        eligibility=lambda sim: sim.people.screen_positive.uids,
        result_state='diagnosed',
    )
    sim = make_sim(n_agents=1000, interventions=[screen, confirm])
    sim.run()

    assert sim.results.screen.n_tested.sum() > 0
    assert sim.results.confirm.n_tested.sum() > 0


def test_dx_delivery_coverage():
    """DxDelivery with coverage < 1.0 tests fewer agents."""
    dx_full = tbsim.DxDelivery(product=tbsim.Xpert(), coverage=1.0)
    sim_full = make_sim(n_agents=1000, interventions=[dx_full])
    sim_full.run()
    n_full = sim_full.results.dxdelivery.n_tested.sum()

    dx_half = tbsim.DxDelivery(product=tbsim.Xpert(), coverage=0.5)
    sim_half = make_sim(n_agents=1000, interventions=[dx_half])
    sim_half.run()
    n_half = sim_half.results.dxdelivery.n_tested.sum()

    assert n_half < n_full


# ---------------------------------------------------------------------------
# BetaByYear tests
# ---------------------------------------------------------------------------

def test_beta_intervention_changes_beta():
    """BetaByYear modifies beta at the specified year with TB_LSHTM."""
    initial_beta = 0.01
    x_beta = 0.5
    intervention_year = 2005
    stop_year = 2010

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start=f'{intervention_year-1}-01-01', stop=f'{stop_year}-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=[beta_intv],
        pars=sim_pars,
    )
    sim.init()

    pars = tbsim.get_tb(sim).pars
    assert np.isclose(pars.beta.value, initial_beta)

    while sim.t.now('year') < intervention_year:
        sim.run_one_step()
    assert np.isclose(pars.beta.value, initial_beta)

    sim.run_one_step()
    expected_beta = initial_beta * x_beta
    assert np.isclose(pars.beta.value, expected_beta)

    sim.run_one_step()
    assert np.isclose(pars.beta.value, expected_beta)


def test_beta_intervention_with_acute():
    """BetaByYear works with TB_LSHTM_Acute."""
    initial_beta = 0.02
    x_beta = 0.6
    intervention_year = 2005

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start='2004-01-01', stop='2007-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM_Acute(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=[beta_intv], pars=sim_pars)
    sim.init()

    while sim.t.now('year') < intervention_year:
        sim.run_one_step()

    sim.run_one_step()
    expected_beta = initial_beta * x_beta
    beta_val = tbsim.get_tb(sim).pars.beta
    actual = beta_val.value if hasattr(beta_val, 'value') else float(beta_val)
    assert np.isclose(actual, expected_beta)


def test_beta_multiple_years():
    """BetaByYear applies different x_beta values at multiple years."""
    initial_beta = 0.1
    years = [2002, 2005]
    x_betas = [0.5, 0.8]

    pop = ss.People(n_agents=50)
    tb = tbsim.TB_LSHTM(name='tb', pars=dict(beta=initial_beta, init_prev=0.25))
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': years, 'x_beta': x_betas})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=[beta_intv],
                 pars=dict(start='2001-01-01', stop='2007-01-01', dt=ss.days(7), rand_seed=42))
    sim.init()

    while sim.t.now('year') < 2002:
        sim.run_one_step()
    sim.run_one_step()
    assert np.isclose(tbsim.get_tb(sim).pars.beta.value, initial_beta * 0.5)

    while sim.t.now('year') < 2005:
        sim.run_one_step()
    sim.run_one_step()
    assert np.isclose(tbsim.get_tb(sim).pars.beta.value, initial_beta * 0.5 * 0.8)
