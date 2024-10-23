import pytest
import numpy as np
import starsim as ss
import tbsim as mtb

def make_tb_simplified(agents=20, start=2000, stop=2020, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta': ss.beta(0.01), 'init_prev': 0.25})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt=dt, start=start, stop=stop), demographics=dems)
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
    assert isinstance(tb.pars['rate_LS_to_presym'], ss.rate)
    assert isinstance(tb.pars['rate_LF_to_presym'], ss.rate)
    assert isinstance(tb.pars['rate_presym_to_active'], ss.rate)
    assert isinstance(tb.pars['rate_active_to_clear'], ss.rate)
    assert isinstance(tb.pars['rate_exptb_to_dead'], ss.rate)
    assert isinstance(tb.pars['rate_smpos_to_dead'], ss.rate)
    assert isinstance(tb.pars['rate_smneg_to_dead'], ss.rate)
    assert isinstance(tb.pars['rel_trans_presymp'], float)
    assert isinstance(tb.pars['rel_trans_smpos'], float)
    assert isinstance(tb.pars['rel_trans_smneg'], float)
    assert isinstance(tb.pars['rel_trans_exptb'], float)
    assert isinstance(tb.pars['rel_trans_treatment'], float)
    
def test_default_parameters():
    tb = mtb.TB()
    print(tb)
    assert tb.pars['init_prev'] is not None
    assert isinstance(tb.pars['rate_LS_to_presym'], ss.rate)
    assert isinstance(tb.pars['rate_LF_to_presym'], ss.rate)
    # assert isinstance(tb.pars['rate_active_to_cure'], ss.rate)
    assert isinstance(tb.pars['rate_exptb_to_dead'], ss.rate)
    assert isinstance(tb.pars['rate_smpos_to_dead'], ss.rate)
    assert isinstance(tb.pars['rate_smneg_to_dead'], ss.rate)
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
    sim.init()
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
    sim.init()
    tb = sim.diseases['tb']
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = True
    tb.rel_trans[uids] = 1.0
    
    tb.step_die(uids)
    
    assert not tb.susceptible[uids].any()
    assert not tb.infected[uids].any()
    assert (tb.rel_trans[uids] == 0).all()
    
def test_step_die_no_uids():
    sim = make_tb_simplified(agents=300)
    sim.init()
    tb = sim.diseases['tb']
        
    initial_susceptible = tb.susceptible.copy()
    initial_infected = tb.infected.copy()
    initial_rel_trans = tb.rel_trans.copy()
    
    tb.step_die([])
    
    assert np.array_equal(tb.susceptible, initial_susceptible)
    assert np.array_equal(tb.infected, initial_infected)
    assert np.array_equal(tb.rel_trans, initial_rel_trans)

@pytest.fixture
def tb():
    sim = make_tb_simplified(agents=300)
    sim.init()
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
    all_uids = ss.uids.concat(uids_active, uids_non_active) 
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
    tb.pars.reltrans_het.rvs = lambda uids: np.array([0.5, 0.7, 0.9], dtype=np.float32)
    
    tb.set_prognoses(uids)
    
    # Ensure both arrays are of the same data type when comparing
    expected = np.array([0.5, 0.7, 0.9], dtype=np.float32)
    actual = tb.reltrans_het[uids]
    assert np.array_equal(actual, expected), f"Expected {expected}, but got {actual}"      

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

def test_p_latent_to_presym():
    # Setup: Create a simulated TB instance and prepare relevant data
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']
    tb.state = np.full(tb.state.shape, mtb.TBS.LATENT_SLOW)  # Assuming all agents are in the latent slow TB state

    # Initialize unique IDs for testing; choose IDs such that some are in latent fast and some in latent slow
    latent_slow_uids = ss.uids([1, 2, 3])  # Latent slow TB
    latent_fast_uids = ss.uids([4, 5, 6])  # Latent fast TB
    tb.state[latent_fast_uids] = mtb.TBS.LATENT_FAST

    # Expected: Verify that probabilities are correctly calculated based on TB states
    probabilities_slow = mtb.TB.p_latent_to_presym(tb, sim, latent_slow_uids) 
    probabilities_fast = mtb.TB.p_latent_to_presym(tb, sim, latent_fast_uids) 

    # The rate should be different for slow and fast latent TB states
    assert np.all(probabilities_slow < probabilities_fast), "Fast progression should have higher transition probabilities"

    # Each returned probability must be between 0 and 1
    assert np.all(0 <= probabilities_slow) and np.all(probabilities_slow <= 1)
    assert np.all(0 <= probabilities_fast) and np.all(probabilities_fast <= 1)

    # Additional checks for expected behavior or known values could be added here, such as:
    # Assert that no probability is zero unless explicitly set so by model parameters
    assert not np.any(probabilities_slow == 0)
    assert not np.any(probabilities_fast == 0)

def test_p_presym_to_active():
    # Setup: Create a simulated TB instance and prepare relevant data
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']
    tb.state = np.full(tb.state.shape, mtb.TBS.ACTIVE_PRESYMP)  # Assuming all agents are in the pre-symptomatic active TB state

    # Initialize unique IDs for testing; all are in pre-symptomatic state
    presym_uids = ss.uids([1, 2, 3, 4, 5])

    # Expected: Verify that probabilities are correctly calculated based on TB state
    probabilities = mtb.TB.p_presym_to_active(tb, sim, presym_uids)

    # Assert that all probabilities are the same if rate is constant across all individuals
    expected_rate = np.full(len(presym_uids), fill_value=tb.pars.rate_presym_to_active)
    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the rate"

    # Each returned probability must be between 0 and 1
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1), "Probabilities should be valid (between 0 and 1)"

    # Ensure that all states were correctly assumed to be pre-symptomatic in the test
    assert (tb.state[presym_uids] == mtb.TBS.ACTIVE_PRESYMP).all(), "All tested UIDs should be in the pre-symptomatic state"


def test_p_active_to_clear():
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']

    # Set some individuals to active TB states
    active_uids = ss.uids([1, 2, 3, 4, 5])
    tb.state[active_uids] = np.random.choice([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB], size=len(active_uids))

    # Assume individuals 1, 2 are on treatment
    tb.on_treatment[ss.uids([1, 2])] = True

    # Calculate probabilities
    probabilities = mtb.TB.p_active_to_clear(tb, sim, active_uids)

    # Check that probabilities are within the expected range (0 to 1)
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1)

    # Check that the rates for those on treatment are correctly applied
    expected_rate = np.full(len(active_uids), fill_value=tb.pars.rate_active_to_clear)
    expected_rate[:2] = tb.pars.rate_treatment_to_clear  # Adjust for treatment
    expected_rate *= tb.rr_clearance[active_uids]  # Adjust for relative clearance

    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the adjusted rates"

    # Additional check for correct handling of treatment effects
    treatment_effect = tb.pars.rate_treatment_to_clear > tb.pars.rate_active_to_clear
    assert treatment_effect, "Treatment should generally provide a higher clearance rate"

def test_p_active_to_death( ):
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']

    # Set some individuals to different active TB states
    active_uids = ss.uids([1, 2, 3, 4, 5])
    tb.state[active_uids] = np.random.choice([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB], size=len(active_uids))

    # Calculate probabilities
    probabilities = mtb.TB.p_active_to_death(tb, sim, active_uids)

    # Check that probabilities are within the expected range (0 to 1)
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1)

    # Expected rates based on TB state and individual death rate adjustments
    expected_rate = np.full(len(active_uids), fill_value=tb.pars.rate_exptb_to_dead)
    expected_rate[tb.state[active_uids] == mtb.TBS.ACTIVE_SMPOS] = tb.pars.rate_smpos_to_dead
    expected_rate[tb.state[active_uids] == mtb.TBS.ACTIVE_SMNEG] = tb.pars.rate_smneg_to_dead
    expected_rate *= tb.rr_death[active_uids]

    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the adjusted rates"

    # Additional check to ensure rates are applied correctly
    assert (tb.state[active_uids] == mtb.TBS.ACTIVE_SMPOS).any() or (tb.state[active_uids] == mtb.TBS.ACTIVE_SMNEG).any() or (tb.state[active_uids] == mtb.TBS.ACTIVE_EXPTB).any(), "Ensure at least some active TB states are set for testing"


def test_latent_to_active_presymptomatic_transition():
    sim = make_tb_simplified(agents=500)
    sim.init()
    tb = sim.diseases['tb']
    # Setup individuals in latent states
    latent_uids = ss.uids(np.arange(50)) 
    tb.state[latent_uids] = np.random.choice([mtb.TBS.LATENT_SLOW, mtb.TBS.LATENT_FAST], size=len(latent_uids))

    # Manually execute the transition step
    tb.step()

    # Check if any latent have transitioned to pre-symptomatic
    transitioned = tb.state[latent_uids] == mtb.TBS.ACTIVE_PRESYMP
    print(transitioned)
    assert transitioned.any(), "At least one latent TB should transition to pre-symptomatic."

def test_presymptomatic_to_active_transition():
    # Since we work with percentages, we make sure we have enough
    # individuals in the pre-symptomatic state to have a good chance of transitioning
    sim = make_tb_simplified(agents=500)
    sim.init()
    tb = sim.diseases['tb']
    # Setup some individuals to pre-symptomatic
    presym_uids = ss.uids(np.arange(250)) 
    tb.state[presym_uids] = mtb.TBS.ACTIVE_PRESYMP

    # Manually execute the transition step
    tb.step()

    # Check if any pre-symptomatic have transitioned to active
    transitioned = (
        (tb.state[presym_uids] == mtb.TBS.ACTIVE_SMPOS)
        | (tb.state[presym_uids] == mtb.TBS.ACTIVE_SMNEG)
        | (tb.state[presym_uids] == mtb.TBS.ACTIVE_EXPTB)
    )
    # transitioned = transitioned[transitioned] # Filter out only those that transitioned - uncomment if you want to see the True values
    assert transitioned.any(), "At least one pre-symptomatic should transition to an active state."

def test_active_to_cleared_transition():
    # Increasing number of agents even higher as the clearance rate is very low
    sim = make_tb_simplified(agents=5000)
    sim.init()
    tb = sim.diseases['tb']
    # Setup individuals in active TB states
    active_uids = ss.uids(np.arange(4000))  # Also test with a large number of agents
    
    tb.state[active_uids] = np.random.choice(
        [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB],
        size=len(active_uids),
    )

    # Manually execute the transition step
    tb.step()

    # Check if any active TB patients have cleared the infection
    cleared = tb.state[active_uids] == mtb.TBS.NONE
    assert cleared.any(), "At least one active TB patient should have cleared the infection."
    
def test_active_to_death_transition():
    # Setup individuals in active TB states
    sim = make_tb_simplified(agents=5000)
    sim.init()
    tb = sim.diseases['tb']
    active_uids = ss.uids(np.arange(4000))  # Also test with a large number of agents
    tb.state[active_uids] = np.random.choice(
        [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB],
        size=len(active_uids),
    )

    # Manually execute the transition step
    tb.step()

    # Check if any active TB patients have died
    died = tb.state[active_uids] == mtb.TBS.DEAD
    # total = len(died[died])
    assert died.any(), "At least one active TB patient should have transitioned to death."
  
if __name__ == '__main__':
    pytest.main()
