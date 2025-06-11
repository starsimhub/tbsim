import numpy as np
import pytest
from unittest import mock
from tbsim.interventions.bcg import BCGProtection, BCGProb
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

def make_sim(agents=20, start=sc.date('2000-01-01'), stop=sc.date('2020-12-31'), dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta': ss.beta(0.01), 'init_prev': 0.25})
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

def test_bcgprob_sampling():
    prob = BCGProb()
    a = prob.activation(size=100)
    c = prob.clearance(size=100)
    d = prob.death(size=100)
    assert np.all((a >= 0.5) & (a <= 0.65))
    assert np.all((c >= 1.3) & (c <= 1.5))
    assert np.all((d >= 0.05) & (d <= 0.15))

def test_bcgprob_file_json(tmp_path):
    # Test loading from JSON file
    data = {
        "activation": {"min": 0.6, "max": 0.7},
        "clearance": {"min": 1.4, "max": 1.6},
        "death": {"min": 0.1, "max": 0.2}
    }
    f = tmp_path / "bcgprob.json"
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(data, fp)
    prob = BCGProb(from_file=str(f))
    a = prob.activation(size=10)
    assert np.all((a >= 0.6) & (a <= 0.7))

def test_bcgprob_file_csv(tmp_path):
    # Test loading from CSV file
    f = tmp_path / "bcgprob.csv"
    with open(f, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["name", "min", "max"])
        writer.writerow(["activation", 0.6, 0.7])
        writer.writerow(["clearance", 1.4, 1.6])
        writer.writerow(["death", 0.1, 0.2])
    prob = BCGProb(from_file=str(f))
    a = prob.activation(size=10)
    assert np.all((a >= 0.6) & (a <= 0.7))

def test_bcg_intervention_default_values():
    # Test BCGProtection intervention with default parameters
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection()
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    assert bcg.coverage== 0.6, "Default coverage should be 0.6"
    assert bcg.efficacy == 0.8, "Default efficacy should be 0.8"
    assert bcg.start == sc.date('1900-01-01'), "Default start year should be 1900-01-01 with type sc.date"
    assert bcg.stop == sc.date('2100-12-31'), "Default stop year should be 2100-12-31 with type sc.date"
    assert bcg.duration == 10, "Default duration should be 10 years"
    assert bcg.age_limit == 5, "Default age limit should be 5 years"
    assert len(bcg.vaccinated) == nagents, "Vaccinated array should match population size"
    assert len(bcg.ti_bcgvaccinated) == nagents, "ti_bcgvaccinated array should match population size"

    
def test_bcg_intervention_custom_values():
    # Test BCGProtection intervention with custom parameters
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection(pars={
        'coverage': 0.75,
        'efficacy': 0.9,
        'start': sc.date('2000-01-01'),
        'stop' : sc.date('2015-01-01'),
        'duration': 15,
        'age_limit': 10
    })
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    assert bcg.coverage == 0.75, "Custom coverage should be 0.75"
    assert bcg.efficacy == 0.9, "Custom efficacy should be 0.9"
    assert bcg.start == sc.date('2000-01-01'), "Custom start year should be 2000-01-01 with type sc.date"
    assert bcg.stop == sc.date('2015-01-01'), "Custom stop year should be 2015-01-01 with type sc.date"
    assert bcg.duration == 15, "Custom duration should be 15 years"
    assert bcg.age_limit == 10, "Custom age limit should be 10 years"
    assert len(bcg.vaccinated) == nagents, "Vaccinated array should match population size"
    assert len(bcg.ti_bcgvaccinated) == nagents, "ti_bcgvaccinated array should match population size"

age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def test_bcg_eligibility_and_vaccination():
    # this tests checks the eligibility and vaccination of individuals for BCG
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    # pars['start'] = sc.date('2000-01-01')
    # pars['stop'] = sc.date('2015-01-01')
    inv_pars={
        # 'start': sc.date('2000-01-01'),         
        # 'stop' : sc.date('2014-01-01'),
    }
    itv = mtb.BCGProtection(inv_pars)
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()  
    bcg = sim.interventions['bcgprotection']
    assert len(bcg.vaccinated) == nagents, "Vaccinated array should match population size"    
    
    # Considering default values, only ages <= 5 should be eligible
    eligible = bcg.check_eligibility()
    assert len(eligible) > 0, "There should be eligible individuals for vaccination"
    assert np.all(bcg.vaccinated[eligible] == False), "Eligible individuals should not be vaccinated yet"
    # Simulate a step to perform vaccination
    bcg.step()
    # After step, eligible individuals should be vaccinated
    assert np.any(bcg.vaccinated[eligible]), "Some eligible individuals should be vaccinated after step"
    
def test_bcg_improves_tb_outcomes():
    # this tests checks the improvement in TB outcomes due to BCG vaccination
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
    assert np.any(tb.rr_activation < initial_rr_activation), "BCG should reduce activation risk"
    assert np.any(tb.rr_clearance > initial_rr_clearance), "BCG should improve clearance rate"
    assert np.any(tb.rr_death < initial_rr_death), "BCG should reduce death risk"
    
        