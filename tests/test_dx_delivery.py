"""Tests for the DxDelivery diagnostic intervention."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_sim(n_agents=1000):
    """Create sim with HSB + Dx delivery for testing."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), rand_seed=42)
    return pop, tb, net, pars


def test_dx_delivery_runs():
    """DxDelivery completes a full run with HealthSeekingBehavior."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # Should have tested some people
    assert sim.results.dxdelivery.n_tested.values.sum() > 0
    assert sim.results.dxdelivery.n_positive.values.sum() > 0


def test_dx_delivery_diagnoses_agents():
    """DxDelivery sets diagnosed=True for positive results."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=1000)
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # Some agents should be diagnosed
    n_diagnosed = np.sum(np.asarray(sim.people.diagnosed))
    assert n_diagnosed > 0


def test_dx_delivery_custom_result_state():
    """DxDelivery with custom result_state auto-registers and sets that state."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = DxDelivery(product=xpert(), result_state='screen_positive')
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()

    # The custom state should exist and have some True values
    assert hasattr(sim.people, 'screen_positive')
    n_screen_pos = np.sum(np.asarray(sim.people.screen_positive))
    assert n_screen_pos > 0


def test_dx_delivery_cascade():
    """Two DxDelivery steps can be chained: screen -> confirm."""
    from tbsim.interventions.dx_products import cad_cxr, xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=1000)
    hsb = tbsim.HealthSeekingBehavior()

    screen = DxDelivery(
        name='screen',
        product=cad_cxr(),
        coverage=0.9,
        result_state='screen_positive',
    )
    confirm = DxDelivery(
        name='confirm',
        product=xpert(),
        coverage=0.8,
        eligibility=lambda sim: sim.people.screen_positive.uids,
        result_state='diagnosed',
    )

    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, screen, confirm], pars=pars)
    sim.run()

    # Both steps should have recorded results
    assert sim.results.screen.n_tested.values.sum() > 0
    assert sim.results.confirm.n_tested.values.sum() > 0


def test_dx_delivery_coverage():
    """DxDelivery with coverage < 1.0 tests fewer agents."""
    from tbsim.interventions.dx_products import xpert
    from tbsim.interventions.diagnostics import DxDelivery

    pop, tb, net, pars = make_sim(n_agents=1000)
    hsb = tbsim.HealthSeekingBehavior()
    dx_full = DxDelivery(product=xpert(), coverage=1.0)
    sim_full = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx_full], pars=pars)
    sim_full.run()
    n_full = sim_full.results.dxdelivery.n_tested.values.sum()

    pop2, tb2, net2, pars2 = make_sim(n_agents=1000)
    hsb2 = tbsim.HealthSeekingBehavior()
    dx_half = DxDelivery(product=xpert(), coverage=0.5)
    sim_half = ss.Sim(people=pop2, diseases=tb2, networks=net2, interventions=[hsb2, dx_half], pars=pars2)
    sim_half.run()
    n_half = sim_half.results.dxdelivery.n_tested.values.sum()

    # Half coverage should test substantially fewer (with noise)
    assert n_half < n_full


# ---------------------------------------------------------------------------
# BetaByYear tests
# ---------------------------------------------------------------------------

def make_sim_acute(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), dt=ss.days(7)):
    """Standard sim components using TB_LSHTM_Acute."""
    pop = ss.People(n_agents=agents)
    tb = tbsim.TB_LSHTM_Acute(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars


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


# ---------------------------------------------------------------------------
# Immigration tests
# ---------------------------------------------------------------------------

import pytest


@pytest.mark.xfail(reason='Immigration class is known non-functional')
def test_immigration_runs():
    """Immigration intervention runs with TB_LSHTM."""
    from tbsim.interventions.immigration import Immigration
    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'))
    immig = Immigration()
    sim = ss.Sim(people=pop, diseases=tb, networks=net, demographics=immig, pars=pars)
    sim.run()


def test_simple_immigration_runs():
    """SimpleImmigration intervention runs with TB_LSHTM."""
    from tbsim.interventions.immigration import SimpleImmigration
    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'))
    immig = SimpleImmigration(immigration_rate=10)
    sim = ss.Sim(people=pop, diseases=tb, networks=net, demographics=immig, pars=pars)
    sim.run()
