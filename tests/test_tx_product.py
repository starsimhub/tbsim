"""Tests for the Tx treatment product."""

import pytest
import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL


def make_tx_sim(n_agents=200):
    """Create a minimal sim with TB for testing Tx products."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    sim = ss.Sim(people=pop, diseases=tb, networks=net, pars=pars)
    sim.init()
    return sim


def test_tx_basic():
    """Tx with efficacy=1.0 returns all agents as success."""
    from tbsim.interventions.tx_products import Tx
    tx = Tx(efficacy=1.0)
    sim = make_tx_sim()
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
    from tbsim.interventions.tx_products import Tx
    tx = Tx(efficacy=0.0)
    sim = make_tx_sim()
    for _ in range(10):
        sim.run_one_step()

    uids = sim.people.alive.uids[:20]
    results = tx.administer(sim, uids)
    assert len(results['success']) == 0
    assert len(results['failure']) == len(uids)


def test_tx_drug_type_overrides_efficacy():
    """Tx with drug_type overrides the efficacy parameter."""
    from tbsim.interventions.tx_products import Tx
    from tbsim.interventions.drug_types import TBDrugType
    tx = Tx(efficacy=0.5, drug_type=TBDrugType.FIRST_LINE_COMBO)
    # FIRST_LINE_COMBO has 95% cure rate, which should override 0.5
    assert np.isclose(tx.efficacy, 0.95)


def test_dots_factory():
    """dots() returns a Tx with DOTS cure probability."""
    from tbsim.interventions.tx_products import dots
    tx = dots()
    assert np.isclose(tx.efficacy, 0.85)


def test_first_line_factory():
    """first_line() returns a Tx with first-line cure probability."""
    from tbsim.interventions.tx_products import first_line
    tx = first_line()
    assert np.isclose(tx.efficacy, 0.95)


def test_dots_runs_in_sim():
    """dots() product can be administered in a running sim."""
    from tbsim.interventions.tx_products import dots
    tx = dots()
    sim = make_tx_sim()
    for _ in range(10):
        sim.run_one_step()
    uids = sim.people.alive.uids[:10]
    results = tx.administer(sim, uids)
    assert len(results['success']) + len(results['failure']) == len(uids)


def test_tx_reproducible_with_same_seed():
    """Tx.administer() produces identical results for same rand_seed (CRN).

    The bug: Tx.administer() uses np.random.random() which draws from numpy's
    global random state rather than a starsim-seeded distribution. This means
    the results are sensitive to anything that perturbs the global state between
    two runs with the same rand_seed, breaking CRN reproducibility.
    """

    def run_tx(seed, consume_global_state=False):
        pop = ss.People(n_agents=200)
        tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
        net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
        pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'), rand_seed=seed)
        sim = ss.Sim(people=pop, diseases=tb, networks=net, pars=pars)
        sim.init()
        if consume_global_state:
            # Simulate other code drawing from numpy global state before administer
            _ = np.random.random(100)
        tx = tbsim.dots()
        uids = sim.people.alive.uids[:50]
        return tx.administer(sim, uids)

    r1 = run_tx(42, consume_global_state=False)
    r2 = run_tx(42, consume_global_state=True)  # same seed, but global state perturbed

    # Same seed → identical success UIDs regardless of external numpy state perturbation.
    # With proper CRN (starsim distribution), this must hold.
    # With np.random.random() (the bug), it fails because global state is consumed.
    assert np.array_equal(r1['success'], r2['success']), \
        "Same seed should produce identical Tx outcomes regardless of global numpy state"
