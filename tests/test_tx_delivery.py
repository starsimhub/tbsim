"""Tests for the TxDelivery treatment intervention."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_sim(n_agents=1000):
    """Create sim components for testing."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), rand_seed=42)
    return pop, tb, net, pars


def test_tx_delivery_runs():
    """TxDelivery completes a full run with HSB + DxDelivery upstream."""
    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(product=tbsim.dots())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    assert sim.results.txdelivery.n_treated.values.sum() > 0
    assert sim.results.txdelivery.n_success.values.sum() > 0


def test_tx_delivery_clears_tb():
    """TxDelivery moves successful agents to CLEARED state."""
    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(product=tbsim.dots())
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    # Some agents should have been successfully treated
    n_success = np.sum(np.asarray(sim.interventions.txdelivery.tb_treatment_success))
    assert n_success > 0


def test_tx_delivery_high_efficacy():
    """High-efficacy product should have more successes than failures."""
    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(product=tbsim.first_line())  # 95% efficacy
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    total_success = sim.results.txdelivery.cum_success.values[-1]
    total_failure = sim.results.txdelivery.cum_failure.values[-1]
    assert total_success > total_failure


def test_tx_delivery_custom_eligibility():
    """TxDelivery with custom eligibility function works."""
    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.DxDelivery(product=tbsim.xpert())
    tx = tbsim.TxDelivery(
        product=tbsim.dots(),
        eligibility=lambda sim: (sim.people.diagnosed & sim.people.alive).uids,
    )
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()

    assert sim.results.txdelivery.n_treated.values.sum() > 0


def test_full_cascade():
    """Full cascade: HSB -> screen -> confirm -> treat."""
    pop, tb, net, pars = make_sim()
    hsb = tbsim.HealthSeekingBehavior()

    screen = tbsim.DxDelivery(
        name='screen',
        product=tbsim.cad_cxr(),
        coverage=0.9,
        result_state='screen_positive',
    )
    confirm = tbsim.DxDelivery(
        name='confirm',
        product=tbsim.xpert(),
        coverage=0.8,
        eligibility=lambda sim: sim.people.screen_positive.uids,
        result_state='diagnosed',
    )
    treat = tbsim.TxDelivery(product=tbsim.dots())

    sim = ss.Sim(people=pop, diseases=tb, networks=net,
                 interventions=[hsb, screen, confirm, treat], pars=pars)
    sim.run()

    # All stages should have processed agents
    assert sim.results.screen.n_tested.values.sum() > 0
    assert sim.results.confirm.n_tested.values.sum() > 0
    assert sim.results.txdelivery.n_treated.values.sum() > 0
