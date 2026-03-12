"""Tests for the TxDelivery treatment intervention."""

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


def test_tx_delivery_runs():
    """TxDelivery completes a full run with HSB + DxDelivery upstream."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    tx = tbsim.TxDelivery(product=tbsim.DOTS())
    sim = make_sim(interventions=[dx, tx])
    sim.run()

    assert sim.results.txdelivery.n_treated.sum() > 0
    assert sim.results.txdelivery.n_success.sum() > 0


def test_tx_delivery_clears_tb():
    """TxDelivery moves successful agents to CLEARED state."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    tx = tbsim.TxDelivery(product=tbsim.DOTS())
    sim = make_sim(interventions=[dx, tx])
    sim.run()

    n_success = sim.interventions.txdelivery.tb_treatment_success.count()
    assert n_success > 0


def test_tx_delivery_high_efficacy():
    """High-efficacy product should have more successes than failures."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    tx = tbsim.TxDelivery(product=tbsim.FirstLine())  # 95% efficacy
    sim = make_sim(interventions=[dx, tx])
    sim.run()

    total_success = sim.results.txdelivery.cum_success[-1]
    total_failure = sim.results.txdelivery.cum_failure[-1]
    assert total_success > total_failure


def test_tx_delivery_custom_eligibility():
    """TxDelivery with custom eligibility function works."""
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    tx = tbsim.TxDelivery(
        product=tbsim.DOTS(),
        eligibility=lambda sim: (sim.people.diagnosed & sim.people.alive).uids,
    )
    sim = make_sim(interventions=[dx, tx])
    sim.run()

    assert sim.results.txdelivery.n_treated.sum() > 0


def test_full_cascade():
    """Full cascade: HSB -> screen -> confirm -> treat."""
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
    treat = tbsim.TxDelivery(product=tbsim.DOTS())

    sim = make_sim(interventions=[screen, confirm, treat])
    sim.run()

    assert sim.results.screen.n_tested.sum() > 0
    assert sim.results.confirm.n_tested.sum() > 0
    assert sim.results.txdelivery.n_treated.sum() > 0
