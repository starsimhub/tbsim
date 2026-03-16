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


def make_tx_sim(n_agents=50, use_acute=False, **tb_pars):
    """Create a minimal sim with TxDelivery for unit-testing step_start_treatment."""
    tb_model = 'lshtm_acute' if use_acute else 'lshtm'
    dx = tbsim.DxDelivery(product=tbsim.Xpert())
    tx = tbsim.TxDelivery(product=tbsim.DOTS())
    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=[dx, tx],
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-12-31')),
        tb_pars=tb_pars or None,
        tb_model=tb_model,
    )
    sim.init()
    return sim, tbsim.get_tb(sim), sim.interventions.txdelivery


# --- step_start_treatment unit tests ---

def test_start_treatment_latent_cleared():
    """step_start_treatment on INFECTION (latent) sets state to CLEARED immediately."""
    sim, tb, tx = make_tx_sim(n_agents=50)
    uids = ss.uids([1, 2, 3])
    tb.state[uids] = TBSL.INFECTION
    tx._elig_uids = uids
    tx.step_start_treatment()
    assert np.all(tb.state[uids] == TBSL.CLEARED)
    assert not tb.infected[uids].any()
    assert tb.susceptible[uids].all()


def test_start_treatment_active_to_treatment():
    """step_start_treatment on NON_INFECTIOUS/ASYMPTOMATIC/SYMPTOMATIC sets state to TREATMENT."""
    sim, tb, tx = make_tx_sim(n_agents=50)
    for state in [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]:
        uids = ss.uids([0])
        tb.state[uids] = state
        tx._elig_uids = uids
        tx.step_start_treatment()
        assert np.all(tb.state[uids] == TBSL.TREATMENT)
        assert tb.on_treatment[uids].all()


def test_start_treatment_empty_uids():
    """step_start_treatment with empty uids does not raise."""
    sim, tb, tx = make_tx_sim(n_agents=10)
    tx._elig_uids = ss.uids()
    tx.step_start_treatment()


def test_start_treatment_acute_latent_cleared():
    """TB_LSHTM_Acute: step_start_treatment on ACUTE or INFECTION sets state to CLEARED."""
    sim, tb, tx = make_tx_sim(n_agents=20, use_acute=True)
    tb.state[ss.uids([0])] = TBSL.ACUTE
    tb.state[ss.uids([1])] = TBSL.INFECTION
    tx._elig_uids = ss.uids([0, 1])
    tx.step_start_treatment()
    assert tb.state[0] == TBSL.CLEARED
    assert tb.state[1] == TBSL.CLEARED


def test_start_treatment_mixed_latent_active_ignores_cleared():
    """step_start_treatment with mix of INFECTION, SYMPTOMATIC, CLEARED: only INFECTION and SYMPTOMATIC are changed."""
    sim, tb, tx = make_tx_sim(n_agents=50)
    tb.state[ss.uids([0])] = TBSL.INFECTION
    tb.state[ss.uids([1])] = TBSL.SYMPTOMATIC
    tb.state[ss.uids([2])] = TBSL.CLEARED
    tx._elig_uids = ss.uids([0, 1, 2])
    tx.step_start_treatment()
    assert tb.state[0] == TBSL.CLEARED    # INFECTION → CLEARED
    assert tb.state[1] == TBSL.TREATMENT  # SYMPTOMATIC → TREATMENT
    assert tb.state[2] == TBSL.CLEARED    # CLEARED stays CLEARED (not affected)


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
        eligibility=lambda sim: (sim.people.dxdelivery.diagnosed & sim.people.alive).uids,
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
        eligibility=lambda sim: sim.people.screen.screen_positive.uids,
        result_state='diagnosed',
    )
    treat = tbsim.TxDelivery(product=tbsim.DOTS())

    sim = make_sim(interventions=[screen, confirm, treat])
    sim.run()

    assert sim.results.screen.n_tested.sum() > 0
    assert sim.results.confirm.n_tested.sum() > 0
    assert sim.results.txdelivery.n_treated.sum() > 0
