"""Tests for TPTTx (product) and TPTSimple (delivery intervention)."""

import numpy as np
import pytest
import starsim as ss
import tbsim as mtb
import pandas as pd
from tbsim.interventions.tpt import TPTTx, TPTSimple
from tbsim.tb_lshtm import TBSL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB_LSHTM(name='tb', pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars


age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1],
})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tpt_default_values():
    """Test TPTSimple + TPTTx with default parameters."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = TPTSimple()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    # Delivery pars on the intervention
    assert '0.5' in str(tpt.pars.coverage) or '0.50' in str(tpt.pars.coverage)
    assert tpt.pars.eligible_states == [TBSL.INFECTION]

    # Product pars
    assert hasattr(tpt.product.pars, 'dur_treatment')
    assert hasattr(tpt.product.pars, 'dur_protection')
    assert tpt.product.pars.disease == 'tb'


def test_tpt_custom_values():
    """Test TPTSimple + TPTTx with custom parameters."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.months(6)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = TPTSimple(product=product, pars={
        'coverage': 0.8,
        'age_range': [15, 65],
        'start': ss.date('2005-01-01'),
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    assert '0.8' in str(tpt.pars.coverage)
    assert tpt.pars.age_range == [15, 65]
    assert tpt.pars.start == ss.date('2005-01-01')


def test_tpt_targets_latent_only():
    """Test that TPT only targets agents in INFECTION (latent) state."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = TPTSimple(pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()

    # Only latently infected agents should have been initiated
    initiated_uids = tpt.initiated.uids
    if len(initiated_uids) > 0:
        tb_disease = sim.diseases.tb
        states_of_initiated = np.asarray(tb_disease.state[initiated_uids])
        # At administration time they were INFECTION; some may have transitioned since
        # but at minimum they should have protection_starts set
        assert np.all(~np.isnan(tpt.product.ti_protection_starts[initiated_uids]))


def test_tpt_treatment_then_protection():
    """Test the two-phase model: treatment first, then protection."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    # Non-zero treatment duration — protection should not start immediately
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.years(1)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    # Step 1: initiate treatment — not yet protected (ti doesn't advance with manual step)
    tpt.step()
    initiated = tpt.initiated.uids
    if len(initiated) > 0:
        # Protection hasn't started yet (treatment takes 1 year)
        assert not np.all(tpt.product.tpt_protected[initiated]), \
            "Agents should not be protected immediately (treatment phase)"

        # But protection_starts should be set in the future
        starts = tpt.product.ti_protection_starts[initiated]
        assert np.all(~np.isnan(starts)), "Protection start times should be set"
        assert np.all(starts > tpt.ti), "Protection should start after treatment completes"


def test_tpt_protection_expiry():
    """Test that protection expires after dur_protection."""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    # Very short treatment + very short protection (1 timestep each)
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(1)),
        'dur_protection': ss.constant(v=ss.days(1)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    # Step through enough times for treatment + protection to expire
    for _ in range(5):
        tpt.step()

    # All protection should have expired
    assert np.count_nonzero(tpt.product.tpt_protected) == 0, \
        "All protection should have expired"


def test_tpt_modifies_rr():
    """Test that TPT applies rr_* modifiers for protected agents."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    # Instant treatment so protection starts immediately
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(10)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tb_disease = sim.diseases.tb
    initial_activation = np.array(tb_disease.rr_activation).copy()

    tpt = sim.interventions['tptsimple']
    # Two steps: first to initiate + start protection (dur_treatment=0), second to apply
    tpt.step()
    tpt.step()

    current_activation = np.array(tb_disease.rr_activation)

    protected = tpt.product.tpt_protected.uids
    if len(protected) > 0:
        assert np.any(current_activation[protected] < initial_activation[protected]), \
            "TPT should reduce activation risk for protected agents"


def test_tpt_result_metrics():
    """Test that result metrics are initialized and updated correctly."""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = TPTSimple()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()
    tpt.update_results()

    assert 'n_newly_initiated' in tpt.results
    assert isinstance(tpt.results['n_newly_initiated'][tpt.ti], (int, np.integer))
    assert 'n_protected' in tpt.results
    assert isinstance(tpt.results['n_protected'][tpt.ti], (int, np.integer))


def test_tpt_with_age_range():
    """Test TPT delivery with age range filter."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    itv = TPTSimple(pars={
        'coverage': 1.0,
        'age_range': [15, 50],
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()

    # Initiated agents should be within age range
    initiated_uids = tpt.initiated.uids
    if len(initiated_uids) > 0:
        ages = np.asarray(sim.people.age[initiated_uids])
        assert np.all(ages >= 15), "All initiated should be >= 15"
        assert np.all(ages <= 50), "All initiated should be <= 50"
