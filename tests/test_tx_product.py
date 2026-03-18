"""Tests for the Tx treatment product."""

import pytest
import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS


def make_tx_sim(n_agents=200, tx_product=None):
    """Create a minimal sim with TB, HSB, Dx, and a Tx product for testing.

    Returns the sim after init. Access the Tx product via sim.interventions.
    """
    interventions = [tbsim.HealthSeekingBehavior(), tbsim.DxDelivery(product=tbsim.Xpert())]
    if tx_product is not None:
        interventions.append(tbsim.TxDelivery(product=tx_product))

    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=interventions,
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01')),
        tb_pars=dict(init_prev=0.30),
    )
    sim.init()
    return sim


def test_tx_basic():
    """Tx with efficacy=1.0 returns all agents as success."""
    sim = make_tx_sim(tx_product=tbsim.Tx(efficacy=1.0))
    tx = sim.interventions.txdelivery.product
    for _ in range(10):
        sim.run_one_step()

    uids = sim.people.alive.uids[:20]
    results = tx.administer(sim, uids)
    assert 'success' in results
    assert 'failure' in results
    assert len(results['success']) == len(uids)
    assert len(results['failure']) == 0


def test_tx_zero_efficacy():
    """Tx with efficacy=0.0 returns all agents as failure."""
    sim = make_tx_sim(tx_product=tbsim.Tx(efficacy=0.0))
    tx = sim.interventions.txdelivery.product
    for _ in range(10):
        sim.run_one_step()

    uids = sim.people.alive.uids[:20]
    results = tx.administer(sim, uids)
    assert len(results['success']) == 0
    assert len(results['failure']) == len(uids)


def test_tx_drug_type_overrides_efficacy():
    """Tx with drug_type overrides the efficacy parameter."""
    tx = tbsim.Tx(efficacy=0.5, drug_type='first_line_combo')
    # first_line_combo has 95% cure rate, which should override 0.5
    assert np.isclose(tx.efficacy, 0.95)


def test_all_tx_products_in_sim():
    """All Tx product classes can be instantiated and have expected efficacies."""
    # Verify efficacies
    assert np.isclose(tbsim.DOTS().efficacy, 0.85)
    assert np.isclose(tbsim.DOTSImproved().efficacy, 0.90)
    assert np.isclose(tbsim.FirstLine().efficacy, 0.95)
    assert np.isclose(tbsim.SecondLine().efficacy, 0.75)

    # Verify one can be administered in a sim
    sim = make_tx_sim(tx_product=tbsim.DOTS())
    tx = sim.interventions.txdelivery.product
    for _ in range(10):
        sim.run_one_step()
    uids = sim.people.alive.uids[:10]
    results = tx.administer(sim, uids)
    assert len(results['success']) + len(results['failure']) == len(uids)


def test_tx_reproducible_with_same_seed():
    """Tx.administer() produces identical results for same rand_seed (CRN)."""

    def run_tx(seed, consume_global_state=False):
        sim = tbsim.Sim(
            n_agents=200,
            interventions=[tbsim.HealthSeekingBehavior(), tbsim.DxDelivery(product=tbsim.Xpert()), tbsim.TxDelivery(product=tbsim.DOTS())],
            sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'), rand_seed=seed),
            tb_pars=dict(init_prev=0.30),
        )
        sim.init()
        if consume_global_state:
            _ = np.random.random(100)
        tx = sim.interventions.txdelivery.product
        uids = sim.people.alive.uids[:50]
        return tx.administer(sim, uids)

    r1 = run_tx(42, consume_global_state=False)
    r2 = run_tx(42, consume_global_state=True)

    assert np.array_equal(r1['success'], r2['success']), \
        "Same seed should produce identical Tx outcomes regardless of global numpy state"


if __name__ == '__main__':
    pytest.main(["-x", "-v", __file__])
