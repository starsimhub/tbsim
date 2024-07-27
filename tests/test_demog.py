import pytest
from unittest.mock import MagicMock
import numpy as np
import starsim as ss
import tbsim as mtb

from tbsim.demographics import NewBornsSocialIntroduction

# Define another function to create a tuberculosis simulation with different parameters
def make_tb_simplified(agents=1000, start=2000, end=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(dict(beta = 0.001, init_prev = 0.25))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt = dt, start = start, end = end), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

def test_generate_and_associate_offspring_no_newborns(setup_simulation):
    pregnancy, sim = setup_simulation
    pregnancy.make_embryos = MagicMock(return_value=[])
    conceive_uids = [1, 2, 3]
    result = pregnancy.generate_and_associate_offspring(conceive_uids, include_nutritional_states=False, network='HarlemNet')
    assert result == []
    print('test_generate_and_associate_offspring_no_newborns')

def test_generate_and_associate_offspring_with_newborns(setup_simulation):
    pregnancy, sim = setup_simulation
    pregnancy.make_embryos = MagicMock(return_value=[4, 5, 6])
    conceive_uids = [1, 2, 3]
    sim.people.hhid = {1: 'A', 2: 'B', 3: 'C'}
    sim.people.arm = {1: 'X', 2: 'Y', 3: 'Z'}
    sim.networks['HarlemNet'].find_contacts = MagicMock(return_value=[7, 8, 9])

    result = pregnancy.generate_and_associate_offspring(conceive_uids, include_nutritional_states=False, network='HarlemNet')

    assert result == [4, 5, 6]
    assert sim.people.hhid[4] == 'A'
    assert sim.people.hhid[5] == 'B'
    assert sim.people.hhid[6] == 'C'
    assert sim.people.arm[4] == 'X'
    assert sim.people.arm[5] == 'Y'
    assert sim.people.arm[6] == 'Z'
    print('test_generate_and_associate_offspring_with_newborns')

def test_assign_malnutritional_states_to_newborns(setup_simulation):
    sim = make_tb_simplified(agents=1500, start=2000, end=2020, dt=7/365)
    pregnancy = NewBornsSocialIntroduction(sim)
    newborn_uids = [4, 5, 6]
    conceive_uids = [1, 2, 3]
    sim.diseases['malnutrition'].micro_state = {1: 'micro1', 2: 'micro2', 3: 'micro3'}
    sim.diseases['malnutrition'].macro_state = {1: 'macro1', 2: 'macro2', 3: 'macro3'}

    pregnancy.assign_malnutritional_states_to_newborns(newborn_uids, conceive_uids, sim)

    assert sim.diseases['malnutrition'].micro_state[4] == 'micro1'
    assert sim.diseases['malnutrition'].micro_state[5] == 'micro2'
    assert sim.diseases['malnutrition'].micro_state[6] == 'micro3'
    assert sim.diseases['malnutrition'].macro_state[4] == 'macro1'
    assert sim.diseases['malnutrition'].macro_state[5] == 'macro2'
    assert sim.diseases['malnutrition'].macro_state[6] == 'macro3'
    print('test_assign_malnutritional_states_to_newborns')
    
if __name__ == '__main__':
    pytest.main()