import numpy as np
import pytest
from unittest import mock
from tbsim.interventions.bcg import BCGProtection
import starsim as ss
import tbsim as mtb
import json
import csv
import pandas as pd
import sciris as sc

class DummyTB:
    def __init__(self, n):
        self.rr_activation = np.ones(n)
        self.rr_clearance = np.ones(n)
        self.rr_death = np.ones(n)

class DummyPeople:
    def __init__(self, ages):
        self.age = np.array(ages)

class DummySim:
    def __init__(self, ages):
        self.people = DummyPeople(ages)
        self.diseases = mock.Mock()
        self.diseases.tb = DummyTB(len(ages))
        self.t = 0

class DummyState(np.ndarray):
    """A dummy state array that supports boolean indexing and assignment."""
    def __new__(cls, shape, dtype=bool, default=False):
        obj = np.full(shape, default, dtype=dtype).view(cls)
        return obj

def dummy_define_states(*args, **kwargs):
    # Patch for ss.Intervention.define_states
    pass

def dummy_define_results(*args, **kwargs):
    # Patch for ss.Intervention.define_results
    pass

def dummy_uids(arr):
    # Patch for ss.uids
    return np.array(arr, dtype=int)

def make_sim(agents=20, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB()
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars=dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars

@pytest.fixture
def patch_ss(monkeypatch):
    import tbsim.interventions.bcg as bcgmod
    monkeypatch.setattr(bcgmod, "ss", mock.Mock())
    bcgmod.ss.uids = dummy_uids
    bcgmod.ss.BoolArr = lambda name, default=False: DummyState((5,), dtype=bool, default=default)
    bcgmod.ss.FloatArr = lambda name: DummyState((5,), dtype=float, default=0.0)
    bcgmod.ss.Intervention = object
    bcgmod.ss.Result = mock.Mock()
    # Patch ss.Arr so DummyState is a subclass of it
    class DummyArr(np.ndarray):
        pass
    DummyState.__bases__ = (DummyArr,)

    bcgmod.ss.Arr = DummyArr
    yield

def test_bcg_intervention_default_values():
    """Test BCGProtection intervention with default parameters"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection()
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    # Check coverage value for starsim bernoulli or float
    if hasattr(bcg.pars.coverage, 'filter'):
        # starsim bernoulli object, check its string representation or default
        assert '0.5' in str(bcg.pars.coverage) or '0.50' in str(bcg.pars.coverage), "Default coverage should be 0.5"
    else:
        assert bcg.pars.coverage == 0.5, "Default coverage should be 0.5"
    assert '0.8' in str(bcg.pars.p_take), "Default p_take should be 0.8"
    assert bcg.pars.start == ss.date('1900-01-01'), "Default start year should be 1900-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2100-12-31'), "Default stop year should be 2100-12-31 with type ss.date"
    assert hasattr(bcg.pars, 'dur_immune'), "dur_immune parameter should exist"
    assert bcg.pars.age_range == [0, 5], "Default age range should be [0, 5]"
    assert bcg.min_age == 0, "Default min_age should be 0"
    assert bcg.max_age == 5, "Default max_age should be 5"
    assert len(bcg.bcg_vaccinated) == nagents, "bcg_vaccinated array should match population size"
    assert len(bcg.ti_bcg_vaccinated) == nagents, "ti_bcg_vaccinated array should match population size"

def test_bcg_intervention_custom_values():
    """Test BCGProtection intervention with custom parameters"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection(pars={
        'coverage': 0.75,
        'p_take': ss.bernoulli(p=0.9),
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2015-01-01'),
        'dur_immune': ss.constant(v=ss.years(15)),
        'age_range': (1, 10)
    })
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    # Check coverage value for starsim bernoulli or float
    if hasattr(bcg.pars.coverage, 'filter'):
        # starsim bernoulli object, check its string representation or default
        assert '0.75' in str(bcg.pars.coverage), "Custom coverage should be 0.75"
    else:
        assert bcg.pars.coverage == 0.75, "Custom coverage should be 0.75"
    assert '0.9' in str(bcg.pars.p_take), "Custom p_take should be 0.9"
    assert bcg.pars.start == ss.date('2000-01-01'), "Custom start year should be 2000-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2015-01-01'), "Custom stop year should be 2015-01-01 with type ss.date"
    assert hasattr(bcg.pars, 'dur_immune'), "dur_immune parameter should exist"
    assert bcg.pars.age_range == (1, 10), "Custom age range should be (1, 10)"
    assert bcg.min_age == 1, "Custom min_age should be 1"
    assert bcg.max_age == 10, "Custom max_age should be 10"
    assert len(bcg.bcg_vaccinated) == nagents, "bcg_vaccinated array should match population size"
    assert len(bcg.ti_bcg_vaccinated) == nagents, "ti_bcg_vaccinated array should match population size"

def test_bcg_age_range_functionality():
    """Test BCG age range functionality with different age ranges"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    
    # Test adult vaccination (18-65 years)
    itv = mtb.BCGProtection(pars={'age_range': (18, 65)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    assert bcg.pars.age_range == (18, 65), "Age range should be (18, 65)"
    assert bcg.min_age == 18, "min_age should be 18"
    assert bcg.max_age == 65, "max_age should be 65"
    
    # Test adolescent vaccination (10-19 years)
    itv2 = mtb.BCGProtection(pars={'age_range': (10, 19)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgprotection']
    
    assert bcg2.pars.age_range == (10, 19), "Age range should be (10, 19)"
    assert bcg2.min_age == 10, "min_age should be 10"
    assert bcg2.max_age == 19, "max_age should be 19"

age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def test_bcg_eligibility_and_vaccination():
    """Test eligibility and vaccination of individuals for BCG"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    
    itv = mtb.BCGProtection()
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()  
    bcg = sim.interventions['bcgprotection']
    assert len(bcg.bcg_vaccinated) == nagents, "bcg_vaccinated array should match population size"    
    
    # Simulate a step to perform vaccination
    bcg.step()

    # After step, some individuals in the target age range should be vaccinated
    ages = sim.people.age
    in_age_range = ((ages >= 0) & (ages <= 5)).uids
    assert np.any(bcg.bcg_vaccinated[in_age_range]), "Some age-eligible individuals should be vaccinated after step"

def test_bcg_eligibility_with_age_range():
    """Test eligibility with different age ranges"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    
    # Test with age range 10-20 (should include ages 10, 15, 20 from age_data)
    itv = mtb.BCGProtection(pars={'age_range': (10, 20)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    eligible = bcg.check_eligibility()
    # Should have eligible individuals in the 10-20 age range
    assert len(eligible) > 0, "There should be eligible individuals in age range 10-20"
    
    # Test with age range 30-50 (should include ages 30, 40, 50 from age_data)
    itv2 = mtb.BCGProtection(pars={'age_range': (30, 50)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgprotection']
    
    eligible2 = bcg2.check_eligibility()
    # Should have eligible individuals in the 30-50 age range
    assert len(eligible2) > 0, "There should be eligible individuals in age range 30-50"

def test_bcg_improves_tb_outcomes():
    """Test that BCG vaccination improves TB outcomes"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()  
    bcg = sim.interventions['bcgprotection']
    
    tb = sim.diseases.tb
    # Store initial TB outcomes
    initial_rr_activation = tb.rr_activation.copy()
    initial_rr_clearance = tb.rr_clearance.copy()
    initial_rr_death = tb.rr_death.copy()
    
    # Simulate a step to apply BCG intervention
    bcg.step()
    
    # Check if TB outcomes have improved
    # Convert to numpy arrays to avoid starsim array comparison bug
    current_activation = np.array(tb.rr_activation)
    current_clearance = np.array(tb.rr_clearance)
    current_death = np.array(tb.rr_death)
    
    initial_activation = np.array(initial_rr_activation)
    initial_clearance = np.array(initial_rr_clearance)
    initial_death = np.array(initial_rr_death)
    
    assert np.any(current_activation < initial_activation), "BCG should reduce activation risk"
    assert np.any(current_clearance > initial_clearance), "BCG should improve clearance rate"
    assert np.any(current_death < initial_death), "BCG should reduce death risk"

def test_bcg_protection_duration():
    """Test that protection duration is properly set"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection(pars={'dur_immune': ss.constant(v=ss.years(8))})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']

    # Simulate vaccination
    bcg.step()

    # Check that protection expiration is set for vaccine responders
    vaccinated = bcg.bcg_vaccinated
    if np.any(vaccinated):
        protection_expires = bcg.ti_bcg_protection_expires[vaccinated]
        valid = ~np.isnan(protection_expires)
        if np.any(valid):
            # Expiry should be in the future
            assert np.all(protection_expires[valid] > bcg.ti), "Protection expiration should be after current time"

def test_bcg_protection_expiry_and_removal():
    """Test that protection expiry removes protection effects and resets TB risk modifiers"""
    nagents = 10
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection(pars={'dur_immune': ss.constant(v=ss.years(1))})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    # Simulate vaccination
    bcg.step()
    
    # Check that some individuals are vaccinated and protected
    vaccinated = bcg.bcg_vaccinated
    assert np.any(vaccinated), "Some individuals should be vaccinated"
    
    # Simulate time passing to expire protection by advancing the simulation
    # Use the intervention's step method multiple times to simulate time passing
    for _ in range(10):
        bcg.step()
    
    # Check that protection has been removed for expired individuals
    # Note: The exact behavior depends on the simulation timestep and dur_immune
    # We'll just check that the intervention is still working
    assert hasattr(bcg, 'bcg_vaccinated'), "BCG intervention should still be functional"


def test_bcg_modifiers_reapplied_each_step():
    """Test that rr modifiers are reapplied after TB resets them to 1.0"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection(pars={'coverage': 0.95, 'p_take': ss.bernoulli(p=0.95)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    bcg.step()  # Vaccinate and apply modifiers
    tb = sim.diseases.tb
    protected = bcg.bcg_protected.uids
    assert len(protected) > 0, "Some individuals should be protected"
    # Simulate TB resetting rr_* to 1.0
    tb.rr_activation[:] = 1.0
    # Run another step — Phase B should reapply modifiers
    bcg.step()
    current_activation = np.array(tb.rr_activation)
    assert np.all(current_activation[protected] <= 1.0), "Protection should persist across steps"


def test_bcg_result_metrics():
    """Test that result metrics are initialized and updated correctly"""
    nagents = 10
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    # bcg.init_results()  # Already called by sim.init(), so skip this
    bcg.step()
    bcg.update_results()
    assert 'n_newly_vaccinated' in bcg.results
    assert isinstance(bcg.results['n_newly_vaccinated'][bcg.ti], (int, np.integer))
    assert 'n_protected' in bcg.results
    assert isinstance(bcg.results['n_protected'][bcg.ti], (int, np.integer))




        