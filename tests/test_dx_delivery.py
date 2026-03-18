"""Tests for the DxDelivery diagnostic intervention."""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
from tbsim import TBS


def make_sim(n_agents=1000, interventions=None, **kwargs):
    """Create and return a tbsim.Sim with HSB and optional interventions."""
    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=interventions,
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-12-31')),
        tb_pars=dict(init_prev=0.30),
        **kwargs
    )
    return sim


def test_dx_delivery_runs():
    """DxDelivery completes a full run with HealthSeekingBehavior."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    sim = make_sim(interventions=dx)
    sim.run()

    assert sim.results.dxdelivery.n_tested.sum() > 0
    assert sim.results.dxdelivery.n_positive.sum() > 0


def test_dx_delivery_diagnoses_agents():
    """DxDelivery sets diagnosed=True for positive results."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    sim = make_sim(n_agents=1000, interventions=dx)
    sim.run()

    n_diagnosed = sim.people.dxdelivery.diagnosed.count()
    assert n_diagnosed > 0


def test_dx_delivery_custom_result_state():
    """DxDelivery with custom result_state auto-registers and sets that state."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert(), result_state='screen_positive')
    sim = make_sim(interventions=dx)
    sim.run()

    n_screen_pos = sim.people.dxdelivery.screen_positive.count()
    assert n_screen_pos > 0


def test_dx_delivery_cascade():
    """Three DxDelivery steps can be chained: screen -> confirm."""
    hsb = tbsim.HealthSeekingBehavior()
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
        eligibility=lambda sim: sim.people.screen.screen_positive.uids,
        result_state='diagnosed',
    )
    sim = make_sim(n_agents=1000, interventions=[hsb, screen, confirm])
    sim.run()

    assert sim.results.screen.n_tested.sum() > 0
    assert sim.results.confirm.n_tested.sum() > 0


def test_result_expiry_resets_state():
    """With short result_validity, fewer agents are diagnosed at end of sim vs no expiry."""
    dx_expiry = tbsim.DxDelivery(product=tbsim.Xpert(), result_state='diagnosed', result_validity=ss.days(30))
    sim_expiry = make_sim(n_agents=500, interventions=dx_expiry)
    sim_expiry.run()
    n_diagnosed_expiry = sim_expiry.people.dxdelivery.diagnosed.count()

    dx_persist = tbsim.DxDelivery(product=tbsim.Xpert(), result_state='diagnosed', result_validity=None)
    sim_persist = make_sim(n_agents=500, interventions=dx_persist)
    sim_persist.run()
    n_diagnosed_persist = sim_persist.people.dxdelivery.diagnosed.count()

    # With expiry, fewer agents should be diagnosed at end (most expired)
    assert n_diagnosed_expiry < n_diagnosed_persist, \
        f"Expected fewer diagnosed with expiry ({n_diagnosed_expiry}) vs persist ({n_diagnosed_persist})"


def test_result_expiry_none_persists():
    """result_validity=None preserves result state indefinitely (default behavior)."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert(), result_state='diagnosed', result_validity=None)
    sim = make_sim(n_agents=500, interventions=dx)
    sim.run()

    # Some agents should still be diagnosed at the end
    assert sim.people.dxdelivery.diagnosed.count() > 0


def test_result_expiry_re_eligible():
    """Agents with expired results become eligible for re-testing."""
    dx = tbsim.DxDelivery(
        product=tbsim.Xpert(),
        result_state='diagnosed',
        result_validity=ss.days(30),
    )
    sim = make_sim(n_agents=500, interventions=dx)
    sim.run()

    # Over a long sim with short validity, agents should be tested multiple times
    max_times = sim.people.dxdelivery.n_times_tested.values.max()
    assert max_times > 1, "With short validity, agents should be re-tested after expiry"


def test_dx_delivery_coverage():
    """DxDelivery with coverage < 1.0 tests fewer agents."""
    dx_full = tbsim.DxDelivery(product=tbsim.Xpert(), coverage=1.0)
    sim_full = make_sim(n_agents=1000, interventions=dx_full)
    sim_full.run()
    n_full = sim_full.results.dxdelivery.n_tested.sum()

    dx_half = tbsim.DxDelivery(product=tbsim.Xpert(), coverage=0.5)
    sim_half = make_sim(n_agents=1000, interventions=dx_half)
    sim_half.run()
    n_half = sim_half.results.dxdelivery.n_tested.sum()

    assert n_half < n_full


# ---------------------------------------------------------------------------
# BetaByYear tests
# ---------------------------------------------------------------------------

def test_beta_intervention_changes_beta():
    """BetaByYear modifies beta at the specified year with TB."""
    initial_beta = 0.01
    x_beta = 0.5
    intervention_year = 2005
    stop_year = 2010

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start=f'{intervention_year-1}-01-01', stop=f'{stop_year}-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TB(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=beta_intv,
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
    """BetaByYear works with TBAcute."""
    initial_beta = 0.02
    x_beta = 0.6
    intervention_year = 2005

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start='2004-01-01', stop='2007-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TBAcute(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': intervention_year, 'x_beta': x_beta})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=beta_intv, pars=sim_pars)
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
    tb = tbsim.TB(name='tb', pars=dict(beta=initial_beta, init_prev=0.25))
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': years, 'x_beta': x_betas})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=beta_intv,
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
