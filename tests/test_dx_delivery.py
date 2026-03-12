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
