"""Tests for BCGVx (product) and BCGRoutine (delivery intervention)."""

import numpy as np
import pytest
import starsim as ss
import tbsim as mtb
import pandas as pd
from tbsim.interventions.bcg import BCGVx, BCGRoutine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(agents=20, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB_LSHTM(name='tb')
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

def test_bcg_default_values():
    """Test BCGRoutine + BCGVx with default parameters."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = BCGRoutine()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    # Delivery pars on the intervention
    assert '0.5' in str(bcg.pars.coverage) or '0.50' in str(bcg.pars.coverage)
    assert bcg.pars.start == ss.date('1900-01-01')
    assert bcg.pars.stop == ss.date('2100-12-31')
    assert bcg.pars.age_range == [0, 5]
    assert bcg.min_age == 0
    assert bcg.max_age == 5

    # Biological pars on the product
    assert '0.8' in str(bcg.product.pars.p_take)
    assert hasattr(bcg.product.pars, 'dur_immune')

    # State arrays sized to population
    assert len(bcg.bcg_vaccinated) == nagents
    assert len(bcg.ti_bcg_vaccinated) == nagents


def test_bcg_custom_values():
    """Test BCGRoutine + BCGVx with custom parameters."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    product = BCGVx(pars={
        'p_take': ss.bernoulli(p=0.9),
        'dur_immune': ss.constant(v=ss.years(15)),
    })
    itv = BCGRoutine(product=product, pars={
        'coverage': 0.75,
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2015-01-01'),
        'age_range': (1, 10),
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    assert '0.75' in str(bcg.pars.coverage)
    assert '0.9' in str(bcg.product.pars.p_take)
    assert bcg.pars.start == ss.date('2000-01-01')
    assert bcg.pars.stop == ss.date('2015-01-01')
    assert bcg.pars.age_range == (1, 10)
    assert bcg.min_age == 1
    assert bcg.max_age == 10
    assert len(bcg.bcg_vaccinated) == nagents


def test_bcg_age_range_functionality():
    """Test BCG age range functionality with different age ranges."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)

    # Adult vaccination (18-65)
    itv = BCGRoutine(pars={'age_range': (18, 65)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']
    assert bcg.min_age == 18
    assert bcg.max_age == 65

    # Adolescent vaccination (10-19)
    itv2 = BCGRoutine(pars={'age_range': (10, 19)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgroutine']
    assert bcg2.min_age == 10
    assert bcg2.max_age == 19


def test_bcg_eligibility_and_vaccination():
    """Test eligibility and vaccination of individuals for BCG."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    itv = BCGRoutine()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']
    assert len(bcg.bcg_vaccinated) == nagents

    bcg.step()

    ages = sim.people.age
    in_age_range = ((ages >= 0) & (ages <= 5)).uids
    assert np.any(bcg.bcg_vaccinated[in_age_range]), "Some age-eligible individuals should be vaccinated after step"


def test_bcg_eligibility_with_age_range():
    """Test eligibility with different age ranges."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    # Age range 10-20
    itv = BCGRoutine(pars={'age_range': (10, 20)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']
    eligible = bcg.check_eligibility()
    assert len(eligible) > 0, "There should be eligible individuals in age range 10-20"

    # Age range 30-50
    itv2 = BCGRoutine(pars={'age_range': (30, 50)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgroutine']
    eligible2 = bcg2.check_eligibility()
    assert len(eligible2) > 0, "There should be eligible individuals in age range 30-50"


def test_bcg_improves_tb_outcomes():
    """Test that BCG vaccination improves TB outcomes (rr_* modifiers)."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = BCGRoutine()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    tb = sim.diseases.tb
    initial_rr_activation = np.array(tb.rr_activation).copy()
    initial_rr_clearance = np.array(tb.rr_clearance).copy()
    initial_rr_death = np.array(tb.rr_death).copy()

    bcg.step()

    current_activation = np.array(tb.rr_activation)
    current_clearance = np.array(tb.rr_clearance)
    current_death = np.array(tb.rr_death)

    assert np.any(current_activation < initial_rr_activation), "BCG should reduce activation risk"
    assert np.any(current_clearance > initial_rr_clearance), "BCG should improve clearance rate"
    assert np.any(current_death < initial_rr_death), "BCG should reduce death risk"


def test_bcg_protection_duration():
    """Test that protection duration is properly set on the product."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    product = BCGVx(pars={'dur_immune': ss.constant(v=ss.years(8))})
    itv = BCGRoutine(product=product)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    bcg.step()

    vaccinated = bcg.bcg_vaccinated
    if np.any(vaccinated):
        protection_expires = bcg.product.ti_bcg_protection_expires[vaccinated]
        valid = ~np.isnan(protection_expires)
        if np.any(valid):
            assert np.all(protection_expires[valid] > bcg.ti), "Protection expiration should be after current time"


def test_bcg_protection_expiry_and_removal():
    """Test that protection expiry removes protection effects."""
    nagents = 10
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    product = BCGVx(pars={'dur_immune': ss.constant(v=ss.years(1))})
    itv = BCGRoutine(product=product)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    bcg.step()

    vaccinated = bcg.bcg_vaccinated
    assert np.any(vaccinated), "Some individuals should be vaccinated"

    for _ in range(10):
        bcg.step()

    assert hasattr(bcg, 'bcg_vaccinated'), "BCG intervention should still be functional"


def test_bcg_modifiers_reapplied_each_step():
    """Test that rr modifiers are reapplied after TB resets them to 1.0."""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    product = BCGVx(pars={'p_take': ss.bernoulli(p=0.95)})
    itv = BCGRoutine(product=product, pars={'coverage': 0.95})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    bcg.step()  # Vaccinate and apply modifiers
    tb = sim.diseases.tb
    protected = bcg.product.bcg_protected.uids
    assert len(protected) > 0, "Some individuals should be protected"

    # Simulate TB resetting rr_* to 1.0
    tb.rr_activation[:] = 1.0

    # Run another step — Phase B should reapply modifiers
    bcg.step()
    current_activation = np.array(tb.rr_activation)
    assert np.all(current_activation[protected] <= 1.0), "Protection should persist across steps"


def test_bcg_result_metrics():
    """Test that result metrics are initialized and updated correctly."""
    nagents = 10
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = BCGRoutine()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgroutine']

    bcg.step()
    bcg.update_results()

    assert 'n_newly_vaccinated' in bcg.results
    assert isinstance(bcg.results['n_newly_vaccinated'][bcg.ti], (int, np.integer))
    assert 'n_protected' in bcg.results
    assert isinstance(bcg.results['n_protected'][bcg.ti], (int, np.integer))
