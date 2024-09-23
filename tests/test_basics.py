import pytest
import numpy as np
import starsim as ss
import tbsim as mtb

def make_tb_simplified(agents=20, start=2000, end=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta': 0.01, 'init_prev': 0.25})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt=dt, start=start, end=end), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

def test_initial_states():
    tb = mtb.TB()
    print(tb.states)
    assert isinstance(tb.susceptible, ss.BoolArr)
    assert isinstance(tb.infected, ss.BoolArr)
    assert isinstance(tb.rel_sus, ss.FloatArr)
    assert isinstance(tb.rel_trans, ss.FloatArr)
    assert isinstance(tb.ti_infected, ss.FloatArr)
    assert isinstance(tb.state, ss.FloatArr)
    assert isinstance(tb.active_tb_state, ss.FloatArr)
    assert isinstance(tb.rr_activation, ss.FloatArr)
    assert isinstance(tb.rr_clearance, ss.FloatArr)
    assert isinstance(tb.rr_death, ss.FloatArr)
    assert isinstance(tb.on_treatment, ss.BoolArr)
    assert isinstance(tb.ti_active, ss.FloatArr)
    assert isinstance(tb.ti_active, ss.FloatArr)
    assert isinstance(tb.ti_active, ss.FloatArr)
    
def test_tb_initialization():
    tb = mtb.TB()
    assert tb.pars['init_prev'] is not None
    assert isinstance(tb.pars['rate_LS_to_presym'], float)
    assert isinstance(tb.pars['rate_LF_to_presym'], float)
    assert isinstance(tb.pars['rate_presym_to_active'], float)
    assert isinstance(tb.pars['rate_active_to_clear'], float)
    assert isinstance(tb.pars['rate_exptb_to_dead'], float)
    assert isinstance(tb.pars['rate_smpos_to_dead'], float)
    assert isinstance(tb.pars['rate_smneg_to_dead'], float)
    assert isinstance(tb.pars['rel_trans_presymp'], float)
    assert isinstance(tb.pars['rel_trans_smpos'], float)
    assert isinstance(tb.pars['rel_trans_smneg'], float)
    assert isinstance(tb.pars['rel_trans_exptb'], float)
    assert isinstance(tb.pars['rel_trans_treatment'], float)
    
def test_default_parameters():
    tb = mtb.TB()
    print(tb)
    assert tb.pars['init_prev'] is not None
    assert isinstance(tb.pars['rate_LS_to_presym'], float)
    assert isinstance(tb.pars['rate_LF_to_presym'], float)
    # assert isinstance(tb.pars['rate_active_to_cure'], float)
    assert isinstance(tb.pars['rate_exptb_to_dead'], float)
    assert isinstance(tb.pars['rate_smpos_to_dead'], float)
    assert isinstance(tb.pars['rate_smneg_to_dead'], float)
    assert isinstance(tb.pars['rel_trans_smpos'], float)

def test_tb_infectious():
    tb = mtb.TB()
    tb.state[:] = mtb.TBS.ACTIVE_PRESYMP
    assert tb.infectious.all()
    tb.state[:] = mtb.TBS.ACTIVE_SMPOS
    assert tb.infectious.all()
    tb.state[:] = mtb.TBS.ACTIVE_SMNEG
    assert tb.infectious.all()
    tb.state[:] = mtb.TBS.ACTIVE_EXPTB
    assert tb.infectious.all()
    tb.state[:] = mtb.TBS.NONE
    assert not tb.infectious.any()

def test_set_prognoses():
    sim = make_tb_simplified()
    sim.run()
    tb = sim.diseases['tb']
    before = tb.state.copy()
    uids = ss.uids([1, 2, 3, 7, 9])
    tb.set_prognoses(uids)
    after = tb.state
    assert not np.array_equal(before, after)
    print("Before: ", before)
    print("After: ", after)

def test_update_pre():
    sim = make_tb_simplified(agents=300)
    sim.initialize()
    tb = sim.diseases['tb']
    assert len(tb.state[tb.state == mtb.TBS.NONE]) > 0
    sim.run()
    assert len(tb.state[tb.state == mtb.TBS.LATENT_SLOW]) > 0
    assert len(tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG]) > 0

    print("none", tb.state[tb.state == mtb.TBS.NONE])
    print("Slow:", tb.state[tb.state == mtb.TBS.LATENT_SLOW])
    print("Fast:", tb.state[tb.state == mtb.TBS.LATENT_FAST])
    print("Active Presymp:", tb.state[tb.state == mtb.TBS.ACTIVE_PRESYMP])
    print("Active ExpTB:", tb.state[tb.state == mtb.TBS.ACTIVE_EXPTB])
    print("Active Smear Negative: ", tb.state[tb.state == mtb.TBS.ACTIVE_SMNEG])
    print("Active Smear Positive: ", tb.state[tb.state == mtb.TBS.ACTIVE_SMPOS])

def test_update_death_with_uids():
    sim = make_tb_simplified(agents=300)
    sim.initialize()
    tb = sim.diseases['tb']
    # sim.run()
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = True
    tb.rel_trans[uids] = 1.0
    
    tb.update_death(uids)
    
    assert not tb.susceptible[uids].any()
    assert not tb.infected[uids].any()
    assert (tb.rel_trans[uids] == 0).all()
    
def test_update_death_no_uids():
    sim = make_tb_simplified(agents=300)
    sim.initialize()
    tb = sim.diseases['tb']
        
    initial_susceptible = tb.susceptible.copy()
    initial_infected = tb.infected.copy()
    initial_rel_trans = tb.rel_trans.copy()
    
    tb.update_death([])
    
    assert np.array_equal(tb.susceptible, initial_susceptible)
    assert np.array_equal(tb.infected, initial_infected)
    assert np.array_equal(tb.rel_trans, initial_rel_trans)

@pytest.fixture
def tb():
    sim = make_tb_simplified(agents=300)
    sim.initialize()
    tb = sim.diseases['tb']
    return tb

# When no individuals have active TB, no treatment should be started
def test_start_treatment_no_active_tb(tb):
    uids = ss.uids([1, 2, 3])
    tb.state[uids] = mtb.TBS.LATENT_SLOW  # No active TB
    num_treated = tb.start_treatment(uids)
    assert num_treated == 0
    assert not tb.on_treatment[uids].any()
    assert (tb.rr_death[uids] == 1).all()  # Default value

#When all individuals have active TB, treatment should be started for all.
def test_start_treatment_with_active_tb(tb):
    uids = ss.uids([1, 2, 3])
    tb.state[uids] = mtb.TBS.ACTIVE_SMPOS  # Active TB
    num_treated = tb.start_treatment(uids)
    assert num_treated == len(uids)
    assert tb.on_treatment[uids].all()
    assert (tb.rr_death[uids] == 0).all()
    
# When there is a mix of individuals with and without active TB, only those with active TB should start treatment.
def test_start_treatment_mixed_states(tb):
    uids_active = ss.uids([1, 2])
    uids_non_active = ss.uids([3, 4])
    tb.state[uids_active] = mtb.TBS.ACTIVE_SMPOS  # Active TB
    tb.state[uids_non_active] = mtb.TBS.LATENT_SLOW  # No active TB
    all_uids = uids_active + uids_non_active
    num_treated = tb.start_treatment(all_uids)
    assert num_treated == len(uids_active)
    assert tb.on_treatment[uids_active].all()
    assert not tb.on_treatment[uids_non_active].any()
    assert (tb.rr_death[uids_active] == 0).all()
    assert (tb.rr_death[uids_non_active] == 1).all()  # Default value

#When individuals have active extrapulmonary TB, treatment should be started for al
def test_start_treatment_exptb(tb):
    uids = ss.uids([1, 2, 3])
    tb.state[uids] = mtb.TBS.ACTIVE_EXPTB  # Active extrapulmonary TB
    num_treated = tb.start_treatment(uids)
    assert num_treated == len(uids)
    assert tb.on_treatment[uids].all()
    assert (tb.rr_death[uids] == 0).all()

def test_set_prognoses_susceptible_to_infected(tb):
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    
    tb.set_prognoses(uids)
    
    assert not tb.susceptible[uids].any()
    assert tb.infected[uids].all()
    
def test_set_prognoses_reltrans_het(tb):
    uids = ss.uids([1, 2, 3])
    tb.pars.reltrans_het.rvs = lambda uids: np.array([0.5, 0.7, 0.9])
    
    tb.set_prognoses(uids)
    
    assert np.array_equal(tb.reltrans_het[uids], np.array([0.5, 0.7, 0.9]))       

# Determining the active TB state.
def test_set_prognoses_active_tb_state(tb):
    uids = ss.uids([1, 2, 3])
    tb.pars.active_state.rvs = lambda uids: np.array([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    tb.set_prognoses(uids)
    
    assert np.array_equal(tb.active_tb_state[uids], np.array([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]))

# Updating the result count of new infections.
def test_set_prognoses_new_infections_count(tb):
    uids = ss.uids([1, 2, 3])
    initial_count = tb.results['new_infections'][tb.sim.ti]
    
    tb.set_prognoses(uids)
    
    assert tb.results['new_infections'][tb.sim.ti] == initial_count + len(uids)

if __name__ == '__main__':
    pytest.main()
