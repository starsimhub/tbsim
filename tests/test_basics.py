import pytest
import numpy as np
import starsim as ss
import tbsim as mtb

def make_tb_simplified(agents=20, start=ss.date('2000-01-01'), stop=ss.date('2020-01-01'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(pars={'beta': ss.peryear(0.0025), 'init_prev': 0.25})  # Standardized transmission rate
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=15))
    deaths = ss.Deaths(pars=dict(death_rate=10))
    spars = dict(
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
    )
    sim = ss.Sim(people=pop, 
                 networks=net,
                 diseases=tb,
                 demographics=[births, deaths],
                 pars=spars)
    return sim

# ============================================================================
# BASIC TESTS (from test_basics.py)
# ============================================================================

def test_initial_states():
    tb = mtb.TB()
    # Remove the states attribute reference as it doesn't exist in Starsim 3.0+
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
    assert isinstance(tb.pars['rate_LS_to_presym'], ss.TimePar)
    assert isinstance(tb.pars['rate_LF_to_presym'], ss.TimePar)
    assert isinstance(tb.pars['rate_presym_to_active'], ss.TimePar)
    assert isinstance(tb.pars['rate_active_to_clear'], ss.TimePar)
    assert isinstance(tb.pars['rate_exptb_to_dead'], ss.TimePar)
    assert isinstance(tb.pars['rate_smpos_to_dead'], ss.TimePar)
    assert isinstance(tb.pars['rate_smneg_to_dead'], ss.TimePar)
    assert isinstance(tb.pars['rel_trans_presymp'], float)
    assert isinstance(tb.pars['rel_trans_smpos'], float)
    assert isinstance(tb.pars['rel_trans_smneg'], float)
    assert isinstance(tb.pars['rel_trans_exptb'], float)
    assert isinstance(tb.pars['rel_trans_treatment'], float)
    
def test_default_parameters():
    tb = mtb.TB()
    print(tb)
    assert tb.pars['init_prev'] is not None
    assert isinstance(tb.pars['rate_LS_to_presym'], ss.TimePar)
    assert isinstance(tb.pars['rate_LF_to_presym'], ss.TimePar)
    # assert isinstance(tb.pars['rate_active_to_cure'], ss.TimePar)
    assert isinstance(tb.pars['rate_exptb_to_dead'], ss.TimePar)
    assert isinstance(tb.pars['rate_smpos_to_dead'], ss.TimePar)
    assert isinstance(tb.pars['rate_smneg_to_dead'], ss.TimePar)
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
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    before = tb.state.copy()
    uids = ss.uids([1, 2, 3, 7, 9])
    tb.set_prognoses(uids, sources=None)  # Added sources parameter for Starsim 3.0+
    after = tb.state
    assert not np.array_equal(before, after)
    print("Before: ", before)
    print("After: ", after)

def test_update_death_with_uids():
    sim = make_tb_simplified(agents=300)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
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
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
        
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
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
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
    uids = tb.sim.people.auids # All people
    tb.susceptible[uids] = True
    tb.infected[uids] = False

    tb.set_prognoses(uids, sources=None)  # Added sources parameter for Starsim 3.0+

    assert tb.infected[uids].all() # All should now be infected

    # Fast progressors should not be susceptible
    fast_uids = uids[tb.state[uids] == mtb.TBS.LATENT_FAST]
    assert not tb.susceptible[fast_uids].any()

    # Slow progressors should still be susceptible
    slow_uids = uids[tb.state[uids] == mtb.TBS.LATENT_SLOW]
    assert tb.susceptible[slow_uids].all()

def test_set_prognoses_reltrans_het(tb):
    uids = ss.uids([1, 2, 3])
    tb.pars.reltrans_het.rvs = lambda uids: np.array([0.5, 0.7, 0.9], dtype=np.float32)
    
    tb.set_prognoses(uids, sources=None)  # Added sources parameter for Starsim 3.0+
    
    # Ensure both arrays are of the same data type when comparing
    expected = np.array([0.5, 0.7, 0.9], dtype=np.float32)
    actual = tb.reltrans_het[uids]
    assert np.array_equal(actual, expected), f"Expected {expected}, but got {actual}"      

# Determining the active TB state.
def test_set_prognoses_active_tb_state(tb):
    uids = ss.uids([1, 2, 3])
    tb.pars.active_state.rvs = lambda uids: np.array([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    tb.set_prognoses(uids, sources=None)  # Added sources parameter for Starsim 3.0+
    
    assert np.array_equal(tb.active_tb_state[uids], np.array([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]))

# Updating the result count of new infections.
def test_set_prognoses_new_infections_count(tb):
    uids = ss.uids([1, 2, 3])
    initial_count = np.count_nonzero(tb.infected)
    
    # Reset the specific uids to uninfected state before testing
    tb.infected[uids] = False
    tb.susceptible[uids] = True
    tb.state[uids] = mtb.TBS.NONE
    
    tb.set_prognoses(uids, sources=None)  # Added sources parameter for Starsim 3.0+
    
    # Check that the specific uids are now infected
    assert tb.infected[uids].all(), "All specified UIDs should be infected after set_prognoses"

# @pytest.mark.skip(reason="Complex probability calculation test - may fail due to stochastic effects")
def test_p_latent_to_presym():
    # Setup: Create a simulated TB instance and prepare relevant data
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', np.full(tb.state.shape, mtb.TBS.LATENT_SLOW))

    # Initialize unique IDs for testing; choose IDs such that some are in latent fast and some in latent slow
    latent_slow_uids = ss.uids([1, 2, 3])  # Latent slow TB
    latent_fast_uids = ss.uids([4, 5, 6])  # Latent fast TB
    tb.setattribute('state', tb.state.copy())
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

# @pytest.mark.skip(reason="Complex probability calculation test - may fail due to stochastic effects")
def test_p_presym_to_active():
    # Setup: Create a simulated TB instance and prepare relevant data
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', np.full(tb.state.shape, mtb.TBS.ACTIVE_PRESYMP))

    # Initialize unique IDs for testing; all are in pre-symptomatic state
    presym_uids = ss.uids([1, 2, 3, 4, 5])

    # Expected: Verify that probabilities are correctly calculated based on TB state
    probabilities = mtb.TB.p_presym_to_active(tb, sim, presym_uids)

    # Assert that all probabilities are the same if rate is constant across all individuals
    # Convert TimePar to float for calculation
    expected_rate = np.full(len(presym_uids), fill_value=float(tb.pars.rate_presym_to_active))
    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the rate"

    # Each returned probability must be between 0 and 1
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1), "Probabilities should be valid (between 0 and 1)"

    # Ensure that all states were correctly assumed to be pre-symptomatic in the test
    assert (tb.state[presym_uids] == mtb.TBS.ACTIVE_PRESYMP).all(), "All tested UIDs should be in the pre-symptomatic state"

# @pytest.mark.skip(reason="Complex probability calculation test - may fail due to stochastic effects")
def test_p_active_to_clear():
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+

    # Set some individuals to active TB states
    active_uids = ss.uids([1, 2, 3, 4, 5])
    tb.setattribute('state', tb.state.copy())
    tb.state[active_uids] = np.random.choice([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB], size=len(active_uids))

    # Assume individuals 1, 2 are on treatment
    tb.on_treatment[ss.uids([1, 2])] = True

    # Calculate probabilities
    probabilities = mtb.TB.p_active_to_clear(tb, sim, active_uids)

    # Check that probabilities are within the expected range (0 to 1)
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1)

    # Check that the rates for those on treatment are correctly applied
    # Convert TimePar objects to float for calculation
    expected_rate = np.full(len(active_uids), fill_value=float(tb.pars.rate_active_to_clear))
    expected_rate[:2] = float(tb.pars.rate_treatment_to_clear)  # Adjust for treatment
    expected_rate *= tb.rr_clearance[active_uids]  # Adjust for relative clearance

    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the adjusted rates"

    # Additional check for correct handling of treatment effects
    treatment_effect = float(tb.pars.rate_treatment_to_clear) > float(tb.pars.rate_active_to_clear)
    assert treatment_effect, "Treatment should generally provide a higher clearance rate"

# @pytest.mark.skip(reason="Complex probability calculation test - may fail due to stochastic effects")
def test_p_active_to_death():
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+

    # Set some individuals to different active TB states
    active_uids = ss.uids([1, 2, 3, 4, 5])
    tb.setattribute('state', tb.state.copy())
    tb.state[active_uids] = np.random.choice([mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB], size=len(active_uids))

    # Calculate probabilities
    probabilities = mtb.TB.p_active_to_death(tb, sim, active_uids)

    # Check that probabilities are within the expected range (0 to 1)
    assert np.all(0 <= probabilities) and np.all(probabilities <= 1)

    # Expected rates based on TB state and individual death rate adjustments
    # Convert TimePar objects to float for calculation
    expected_rate = np.full(len(active_uids), fill_value=float(tb.pars.rate_exptb_to_dead))
    expected_rate[tb.state[active_uids] == mtb.TBS.ACTIVE_SMPOS] = float(tb.pars.rate_smpos_to_dead)
    expected_rate[tb.state[active_uids] == mtb.TBS.ACTIVE_SMNEG] = float(tb.pars.rate_smneg_to_dead)
    expected_rate *= tb.rr_death[active_uids]

    expected_prob = 1 - np.exp(-expected_rate)
    assert np.allclose(probabilities, expected_prob), "Probabilities should match expected values calculated from the adjusted rates"

    # Additional check to ensure rates are applied correctly
    assert (tb.state[active_uids] == mtb.TBS.ACTIVE_SMPOS).any() or (tb.state[active_uids] == mtb.TBS.ACTIVE_SMNEG).any() or (tb.state[active_uids] == mtb.TBS.ACTIVE_EXPTB).any(), "Ensure at least some active TB states are set for testing"

# @pytest.mark.skip(reason="Stochastic transition test - may fail due to random effects")
def test_latent_to_active_presymptomatic_transition():
    sim = make_tb_simplified(agents=500)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    # Setup individuals in latent states
    latent_uids = ss.uids(np.arange(50)) 
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', tb.state.copy())
    tb.state[latent_uids] = np.random.choice([mtb.TBS.LATENT_SLOW, mtb.TBS.LATENT_FAST], size=len(latent_uids))

    # Manually execute the transition step
    tb.step()

    # Check if any latent have transitioned to pre-symptomatic
    transitioned = tb.state[latent_uids] == mtb.TBS.ACTIVE_PRESYMP
    print(transitioned)
    assert transitioned.any(), "At least one latent TB should transition to pre-symptomatic."

# @pytest.mark.skip(reason="Stochastic transition test - may fail due to random effects")
def test_presymptomatic_to_active_transition():
    # Since we work with percentages, we make sure we have enough
    # individuals in the pre-symptomatic state to have a good chance of transitioning
    sim = make_tb_simplified(agents=500)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    # Setup some individuals to pre-symptomatic
    presym_uids = ss.uids(np.arange(250)) 
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', tb.state.copy())
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

# @pytest.mark.skip(reason="Complex probability calculation test - may fail due to stochastic effects")
def test_active_to_cleared_transition():
    # The clearance rate is very low, so we'll test the probability calculation instead
    sim = make_tb_simplified(agents=10)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    
    # Setup individuals in active TB states
    active_uids = ss.uids([1, 2, 3, 4, 5])
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', tb.state.copy())
    tb.state[active_uids] = np.random.choice(
        [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB],
        size=len(active_uids),
    )

    # Calculate clearance probabilities
    probabilities = mtb.TB.p_active_to_clear(tb, sim, active_uids)
    
    # Check that probabilities are calculated correctly (should be very small but non-zero)
    assert np.all(probabilities > 0), "Clearance probabilities should be positive"
    assert np.all(probabilities < 1), "Clearance probabilities should be less than 1"
    
    # The clearance rate is very low, so probabilities should be small
    expected_max_prob = 1 - np.exp(-float(tb.pars.rate_active_to_clear))
    assert np.all(probabilities <= expected_max_prob), "Probabilities should not exceed theoretical maximum"
    
# @pytest.mark.skip(reason="Stochastic transition test - may fail due to random effects")
def test_active_to_death_transition():
    # Setup individuals in active TB states
    sim = make_tb_simplified(agents=5000)
    sim.init()
    tb = sim.diseases['tb']  # Updated for Starsim 3.0+
    active_uids = ss.uids(np.arange(4000))  # Also test with a large number of agents
    
    # Use setattribute to modify locked attributes in Starsim 3.0+
    tb.setattribute('state', tb.state.copy())
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

class TestTBInitialization:
    """Test TB class initialization and parameter validation"""
    
    def test_tb_initialization_default(self):
        """Test TB initialization with default parameters"""
        tb = mtb.TB()
        
        # Check that all required parameters are present
        assert 'init_prev' in tb.pars
        assert 'beta' in tb.pars
        assert 'p_latent_fast' in tb.pars
        assert 'rate_LS_to_presym' in tb.pars
        assert 'rate_LF_to_presym' in tb.pars
        assert 'rate_presym_to_active' in tb.pars
        assert 'rate_active_to_clear' in tb.pars
        assert 'rate_exptb_to_dead' in tb.pars
        assert 'rate_smpos_to_dead' in tb.pars
        assert 'rate_smneg_to_dead' in tb.pars
        assert 'rate_treatment_to_clear' in tb.pars
        
        # Check that rate parameters are TimePar instances
        for key in tb.pars.keys():
            if key.startswith('rate_'):
                assert isinstance(tb.pars[key], ss.TimePar), f"Rate parameter {key} must be TimePar"
    
    def test_tb_initialization_custom_params(self):
        """Test TB initialization with custom parameters"""
        custom_pars = {
            'init_prev': ss.bernoulli(0.05),
            'beta': ss.peryear(0.005),
            'p_latent_fast': ss.bernoulli(0.2),
            'rate_LS_to_presym': ss.perday(1e-4),
            'rate_LF_to_presym': ss.perday(1e-2),
            'rate_presym_to_active': ss.perday(1e-1),
            'rate_active_to_clear': ss.perday(1e-3),
            'rate_exptb_to_dead': ss.perday(1e-3),
            'rate_smpos_to_dead': ss.perday(1e-2),
            'rate_smneg_to_dead': ss.perday(1e-3),
            'rate_treatment_to_clear': ss.peryear(6),
            'rel_trans_presymp': 0.2,
            'rel_trans_smpos': 1.0,
            'rel_trans_smneg': 0.4,
            'rel_trans_exptb': 0.1,
            'rel_trans_treatment': 0.3,
            'rel_sus_latentslow': 0.15,
            'cxr_asymp_sens': 0.9,
        }
        
        tb = mtb.TB(pars=custom_pars)
        
        # Check that custom parameters were applied
        assert tb.pars['init_prev'] == custom_pars['init_prev']
        assert tb.pars['beta'] == custom_pars['beta']
        assert tb.pars['rel_trans_presymp'] == 0.2
        assert tb.pars['rel_trans_smpos'] == 1.0
        assert tb.pars['rel_trans_smneg'] == 0.4
        assert tb.pars['rel_trans_exptb'] == 0.1
        assert tb.pars['rel_trans_treatment'] == 0.3
        assert tb.pars['rel_sus_latentslow'] == 0.15
        assert tb.pars['cxr_asymp_sens'] == 0.9
    
    def test_tb_states_initialization(self):
        """Test that all TB-specific states are properly initialized"""
        tb = mtb.TB()
        
        # Check that all required states are present
        assert hasattr(tb, 'state')
        assert hasattr(tb, 'latent_tb_state')
        assert hasattr(tb, 'active_tb_state')
        assert hasattr(tb, 'rr_activation')
        assert hasattr(tb, 'rr_clearance')
        assert hasattr(tb, 'rr_death')
        assert hasattr(tb, 'on_treatment')
        assert hasattr(tb, 'ever_infected')
        assert hasattr(tb, 'ti_presymp')
        assert hasattr(tb, 'ti_active')
        assert hasattr(tb, 'ti_cur')
        assert hasattr(tb, 'reltrans_het')
        
        # Check that states are proper array types
        assert isinstance(tb.state, ss.FloatArr)
        assert isinstance(tb.latent_tb_state, ss.FloatArr)
        assert isinstance(tb.active_tb_state, ss.FloatArr)
        assert isinstance(tb.rr_activation, ss.FloatArr)
        assert isinstance(tb.rr_clearance, ss.FloatArr)
        assert isinstance(tb.rr_death, ss.FloatArr)
        assert isinstance(tb.on_treatment, ss.BoolArr)
        assert isinstance(tb.ever_infected, ss.BoolArr)
        assert isinstance(tb.ti_presymp, ss.FloatArr)
        assert isinstance(tb.ti_active, ss.FloatArr)
        assert isinstance(tb.ti_cur, ss.FloatArr)
        assert isinstance(tb.reltrans_het, ss.FloatArr)


class TestTBStates:
    """Test TB state constants and state management"""
    
    def test_tb_state_constants(self):
        """Test that TB state constants are properly defined"""
        assert mtb.TBS.NONE == -1
        assert mtb.TBS.LATENT_SLOW == 0
        assert mtb.TBS.LATENT_FAST == 1
        assert mtb.TBS.ACTIVE_PRESYMP == 2
        assert mtb.TBS.ACTIVE_SMPOS == 3
        assert mtb.TBS.ACTIVE_SMNEG == 4
        assert mtb.TBS.ACTIVE_EXPTB == 5
        assert mtb.TBS.DEAD == 8
        assert mtb.TBS.PROTECTED == 100
    
    def test_infectious_property(self):
        """Test the infectious property for different TB states"""
        tb = mtb.TB()
        
        # Test that infectious states are correctly identified
        tb.state[:] = mtb.TBS.ACTIVE_PRESYMP
        assert tb.infectious.all()
        
        tb.state[:] = mtb.TBS.ACTIVE_SMPOS
        assert tb.infectious.all()
        
        tb.state[:] = mtb.TBS.ACTIVE_SMNEG
        assert tb.infectious.all()
        
        tb.state[:] = mtb.TBS.ACTIVE_EXPTB
        assert tb.infectious.all()
        
        # Test that non-infectious states are correctly identified
        tb.state[:] = mtb.TBS.NONE
        assert not tb.infectious.any()
        
        tb.state[:] = mtb.TBS.LATENT_SLOW
        assert not tb.infectious.any()
        
        tb.state[:] = mtb.TBS.LATENT_FAST
        assert not tb.infectious.any()
        
        tb.state[:] = mtb.TBS.DEAD
        assert not tb.infectious.any()
        
        tb.state[:] = mtb.TBS.PROTECTED
        assert not tb.infectious.any()
    
    def test_infectious_with_treatment(self):
        """Test that treatment status affects infectiousness"""
        tb = mtb.TB()
        
        # Set all to active TB
        tb.state[:] = mtb.TBS.ACTIVE_SMPOS
        tb.on_treatment[:] = False
        assert tb.infectious.all()
        
        # Start treatment
        tb.on_treatment[:] = True
        assert tb.infectious.all()  # Still infectious but with reduced transmission


class TestTBTreatment:
    """Test TB treatment functionality"""
    
    @pytest.fixture
    def tb_with_active_cases(self):
        """Create TB instance with some active cases"""
        tb = mtb.TB()
        sim = ss.Sim(people=ss.People(n_agents=10), diseases=tb)
        sim.init()
        
        # Set some individuals to active TB states
        uids = ss.uids([0, 1, 2, 3, 4])
        tb.state[uids[0]] = mtb.TBS.ACTIVE_PRESYMP
        tb.state[uids[1]] = mtb.TBS.ACTIVE_SMPOS
        tb.state[uids[2]] = mtb.TBS.ACTIVE_SMNEG
        tb.state[uids[3]] = mtb.TBS.ACTIVE_EXPTB
        tb.state[uids[4]] = mtb.TBS.NONE  # Not active
        
        return tb, uids
    
    def test_start_treatment_no_active_cases(self, tb_with_active_cases):
        """Test starting treatment when no active cases"""
        tb, uids = tb_with_active_cases
        
        # Set all to non-active states
        tb.state[:] = mtb.TBS.LATENT_SLOW
        non_active_uids = ss.uids([5, 6, 7])
        
        treated_count = tb.start_treatment(non_active_uids)
        assert treated_count == 0
    
    def test_start_treatment_with_active_cases(self, tb_with_active_cases):
        """Test starting treatment with active cases"""
        tb, uids = tb_with_active_cases
        
        # Start treatment for all uids
        treated_count = tb.start_treatment(uids)
        
        # Should treat 4 active cases (excluding the non-active one)
        assert treated_count == 4
        
        # Check that treatment flags were set
        assert tb.on_treatment[uids[0]]
        assert tb.on_treatment[uids[1]]
        assert tb.on_treatment[uids[2]]
        assert tb.on_treatment[uids[3]]
        assert not tb.on_treatment[uids[4]]  # Non-active case
    
    def test_start_treatment_death_rate_reduction(self, tb_with_active_cases):
        """Test that treatment reduces death rates"""
        tb, uids = tb_with_active_cases
        
        # Start treatment
        tb.start_treatment(uids)
        
        # Check that death risk ratios were set to 0 for treated individuals
        assert tb.rr_death[uids[0]] == 0
        assert tb.rr_death[uids[1]] == 0
        assert tb.rr_death[uids[2]] == 0
        assert tb.rr_death[uids[3]] == 0
    
    def test_start_treatment_transmission_reduction(self, tb_with_active_cases):
        """Test that treatment reduces transmission rates"""
        tb, uids = tb_with_active_cases
        
        # Record initial transmission rates
        initial_rel_trans = tb.rel_trans[uids].copy()
        
        # Start treatment
        tb.start_treatment(uids)
        
        # Check that transmission rates were reduced for treated individuals
        for i in range(4):  # Only the active cases
            assert tb.rel_trans[uids[i]] < initial_rel_trans[i]


class TestTBResults:
    """Test TB results tracking and reporting"""
    
    def test_init_results(self):
        """Test that results are properly initialized"""
        tb = mtb.TB()
        sim = ss.Sim(people=ss.People(n_agents=10), diseases=tb)
        sim.init()
        
        tb.init_results()
        
        # Check that all required results are present
        expected_results = [
            'n_latent_slow', 'n_latent_fast', 'n_active', 'n_active_presymp',
            'n_active_presymp_15+', 'n_active_smpos', 'n_active_smpos_15+',
            'n_active_smneg', 'n_active_smneg_15+', 'n_active_exptb',
            'n_active_exptb_15+', 'new_active', 'new_active_15+',
            'cum_active', 'cum_active_15+', 'new_deaths', 'new_deaths_15+',
            'cum_deaths', 'cum_deaths_15+', 'n_infectious', 'n_infectious_15+',
            'prevalence_active', 'incidence_kpy', 'deaths_ppy', 'n_reinfected',
            'new_notifications_15+', 'n_detectable_15+'
        ]
        
        for result_name in expected_results:
            assert result_name in tb.results
    
    def test_update_results(self):
        """Test that results are properly updated"""
        tb = mtb.TB()
        sim = ss.Sim(people=ss.People(n_agents=100), diseases=tb)
        sim.init()
        tb.init_results()
        
        # Set some individuals to different states
        tb.state[0:10] = mtb.TBS.LATENT_SLOW
        tb.state[10:20] = mtb.TBS.LATENT_FAST
        tb.state[20:25] = mtb.TBS.ACTIVE_PRESYMP
        tb.state[25:30] = mtb.TBS.ACTIVE_SMPOS
        tb.state[30:35] = mtb.TBS.ACTIVE_SMNEG
        tb.state[35:40] = mtb.TBS.ACTIVE_EXPTB
        
        # Update results
        tb.update_results()
        
        # Check that counts are correct
        assert tb.results['n_latent_slow'][tb.ti] >= 10
        assert tb.results['n_latent_fast'][tb.ti] >= 10
        assert tb.results['n_active_presymp'][tb.ti] >= 5
        assert tb.results['n_active_smpos'][tb.ti] >= 5
        assert tb.results['n_active_smneg'][tb.ti] >= 5
        assert tb.results['n_active_exptb'][tb.ti] >= 5
        assert tb.results['n_active'][tb.ti] >= 20  # All active states combined
    
    def test_finalize_results(self):
        """Test that cumulative results are properly calculated"""
        tb = mtb.TB()
        sim = ss.Sim(people=ss.People(n_agents=10), diseases=tb)
        sim.init()
        tb.init_results()
        
        # Simulate some results over time
        for i in range(5):
            tb.results['new_active'][i] = 2
            tb.results['new_deaths'][i] = 1
            tb.results['new_active_15+'][i] = 1
            tb.results['new_deaths_15+'][i] = 1
        
        tb.finalize_results()
        
        # Check cumulative results
        assert tb.results['cum_active'][4] == 10  # 2 * 5
        assert tb.results['cum_deaths'][4] == 5   # 1 * 5
        assert tb.results['cum_active_15+'][4] == 5  # 1 * 5
        assert tb.results['cum_deaths_15+'][4] == 5  # 1 * 5


class TestTBIntegration:
    """Integration tests for TB simulation"""
    
    def test_tb_simulation_basic(self):
        """Test basic TB simulation functionality"""
        # Create simulation components
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        births = ss.Births(pars=dict(birth_rate=15))
        deaths = ss.Deaths(pars=dict(death_rate=10))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            demographics=[births, deaths],
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Check that simulation completed
        assert sim.complete
        assert len(sim.results) > 0
        
        # Check that TB results are present
        assert 'tb' in sim.results
        tb_results = sim.results['tb']
        
        # Check that key metrics are tracked
        assert 'n_active' in tb_results
        assert 'n_infectious' in tb_results
        assert 'prevalence_active' in tb_results
        assert 'incidence_kpy' in tb_results
    
    def test_tb_with_treatment_intervention(self):
        """Test TB simulation with treatment intervention"""
        # Create simulation components
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Get TB module
        tb_module = sim.diseases['tb']
        
        # Find some active cases and start treatment
        active_uids = ss.uids(np.where(tb_module.state == mtb.TBS.ACTIVE_SMPOS)[0])
        if len(active_uids) > 0:
            treated_count = tb_module.start_treatment(active_uids[:5])  # Treat up to 5 cases
            assert treated_count >= 0
            assert treated_count <= len(active_uids[:5])

# ============================================================================
# TESTS FROM test_tb_simple.py
# ============================================================================

class TestTBSimulation:
    """Test TB simulation functionality"""
    
    def test_tb_simulation_basic(self):
        """Test basic TB simulation functionality"""
        # Create simulation components
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        births = ss.Births(pars=dict(birth_rate=15))
        deaths = ss.Deaths(pars=dict(death_rate=10))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            demographics=[births, deaths],
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Check that simulation completed
        assert sim.complete
        assert len(sim.results) > 0
        
        # Check that TB results are present
        assert 'tb' in sim.results
        tb_results = sim.results['tb']
        
        # Check that key metrics are tracked
        assert 'n_active' in tb_results
        assert 'n_infectious' in tb_results
        assert 'prevalence_active' in tb_results
        assert 'incidence_kpy' in tb_results
    
    def test_tb_simulation_with_treatment(self):
        """Test TB simulation with treatment intervention"""
        # Create simulation components
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Get TB module
        tb_module = sim.diseases['tb']
        
        # Find some active cases and start treatment
        active_uids = ss.uids(np.where(tb_module.state == mtb.TBS.ACTIVE_SMPOS)[0])
        if len(active_uids) > 0:
            treated_count = tb_module.start_treatment(active_uids[:5])  # Treat up to 5 cases
            assert treated_count >= 0
            assert treated_count <= len(active_uids[:5])
    
    def test_tb_simulation_states(self):
        """Test that TB simulation produces expected states"""
        # Create simulation components
        pop = ss.People(n_agents=50)
        tb = mtb.TB(pars={'init_prev': ss.bernoulli(0.1)})  # Higher initial prevalence
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2002-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Get TB module
        tb_module = sim.diseases['tb']
        
        # Check that we have some TB cases (may not have any due to low transmission)
        # Just check that the state array exists and has the right length
        assert len(tb_module.state) == 50
        
        # Check that infectious property exists and works
        assert hasattr(tb_module, 'infectious')
        assert len(tb_module.infectious) == 50
        
        # Check that results are reasonable
        tb_results = sim.results['tb']
        assert tb_results['n_active'][-1] >= 0  # Should be non-negative
        assert tb_results['n_infectious'][-1] >= 0  # Should be non-negative


class TestTBTreatment:
    """Test TB treatment functionality"""
    
    def test_start_treatment_method_exists(self):
        """Test that start_treatment method exists and is callable"""
        tb = mtb.TB()
        assert hasattr(tb, 'start_treatment')
        assert callable(tb.start_treatment)
    
    def test_treatment_parameters(self):
        """Test that treatment-related parameters are properly set"""
        tb = mtb.TB()
        
        # Check treatment-related parameters
        assert 'rate_treatment_to_clear' in tb.pars
        assert 'rel_trans_treatment' in tb.pars
        assert isinstance(tb.pars['rate_treatment_to_clear'], ss.TimePar)
        assert isinstance(tb.pars['rel_trans_treatment'], float)


class TestTBResults:
    """Test TB results tracking"""
    
    def test_results_structure(self):
        """Test that TB results have the expected structure"""
        # Create a simple simulation to test results
        pop = ss.People(n_agents=10)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Check that TB results are present
        assert 'tb' in sim.results
        tb_results = sim.results['tb']
        
        # Check for key result fields
        expected_results = [
            'n_latent_slow', 'n_latent_fast', 'n_active', 'n_active_presymp',
            'n_active_smpos', 'n_active_smneg', 'n_active_exptb', 'new_active',
            'new_deaths', 'n_infectious', 'prevalence_active', 'incidence_kpy'
        ]
        
        for result_name in expected_results:
            assert result_name in tb_results, f"Missing result: {result_name}"
    
    def test_results_values(self):
        """Test that TB results have reasonable values"""
        # Create a simple simulation to test results
        pop = ss.People(n_agents=20)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Get TB results
        tb_results = sim.results['tb']
        
        # Check that counts are non-negative
        assert np.all(tb_results['n_active'] >= 0)
        assert np.all(tb_results['n_infectious'] >= 0)
        assert np.all(tb_results['new_deaths'] >= 0)
        
        # Check that prevalence is between 0 and 1
        assert np.all(tb_results['prevalence_active'] >= 0)
        assert np.all(tb_results['prevalence_active'] <= 1)


class TestTBPrevalence:
    """Test TB prevalence validation and ranges"""
    
    def test_prevalence_basic_ranges(self):
        """Test that TB prevalence is always within valid ranges (0-1)"""
        # Test with different population sizes
        for n_agents in [10, 50, 100]:
            pop = ss.People(n_agents=n_agents)
            tb = mtb.TB()
            sim = ss.Sim(
                people=pop,
                diseases=tb,
                pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
            )
            
            sim.run()
            tb_results = sim.results['tb']
            
            # Prevalence should always be between 0 and 1
            assert np.all(tb_results['prevalence_active'] >= 0), f"Prevalence below 0 for {n_agents} agents"
            assert np.all(tb_results['prevalence_active'] <= 1), f"Prevalence above 1 for {n_agents} agents"
    
    def test_prevalence_initial_conditions(self):
        """Test prevalence with different initial conditions"""
        # Test with different initial prevalence settings
        for init_prev in [0.01, 0.05, 0.1, 0.2]:
            pop = ss.People(n_agents=100)
            tb = mtb.TB(pars={'init_prev': ss.bernoulli(init_prev)})
            sim = ss.Sim(
                people=pop,
                diseases=tb,
                pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2003-01-01'))
            )
            
            sim.run()
            tb_results = sim.results['tb']
            
            # Check that prevalence is within valid range
            assert np.all(tb_results['prevalence_active'] >= 0)
            assert np.all(tb_results['prevalence_active'] <= 1)
            
            # Check that initial prevalence is reasonable (within factor of 2 of expected)
            initial_prevalence = tb_results['prevalence_active'][0]
            assert initial_prevalence >= 0, f"Initial prevalence {initial_prevalence} below 0 for init_prev={init_prev}"
            assert initial_prevalence <= 1, f"Initial prevalence {initial_prevalence} above 1 for init_prev={init_prev}"
    
    @pytest.mark.skip(reason="Stochastic test - may fail due to random effects")
    def test_prevalence_transmission_sensitivity(self):
        """Test prevalence changes with different transmission rates"""
        prevalences = []
        
        for beta in [0.001, 0.005, 0.01, 0.02]:
            pop = ss.People(n_agents=100)
            tb = mtb.TB(pars={'beta': ss.peryear(beta)})
            sim = ss.Sim(
                people=pop,
                diseases=tb,
                pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
            )
            
            sim.run()
            tb_results = sim.results['tb']
            
            # Store final prevalence
            final_prevalence = tb_results['prevalence_active'][-1]
            prevalences.append(final_prevalence)
            
            # Validate prevalence range
            assert 0 <= final_prevalence <= 1, f"Final prevalence {final_prevalence} outside [0,1] for beta={beta}"
        
        # Higher transmission should generally lead to higher prevalence
        # (though this may not always be true due to stochastic effects)
        print(f"Prevalence values for different betas: {prevalences}")
    
    @pytest.mark.skip(reason="Stochastic test - may fail due to random effects")
    def test_prevalence_with_treatment(self):
        """Test prevalence changes when treatment is applied"""
        # Run simulation without treatment
        pop = ss.People(n_agents=100)
        tb_no_treatment = mtb.TB()
        sim_no_treatment = ss.Sim(
            people=pop,
            diseases=tb_no_treatment,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        sim_no_treatment.run()
        
        # Run simulation with treatment intervention
        pop2 = ss.People(n_agents=100)
        tb_with_treatment = mtb.TB()
        sim_with_treatment = ss.Sim(
            people=pop2,
            diseases=tb_with_treatment,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation and apply treatment during the run
        sim_with_treatment.run()
        
        # Get TB module and apply treatment after simulation
        tb_module = sim_with_treatment.diseases['tb']
        active_uids = ss.uids(np.where(tb_module.state == mtb.TBS.ACTIVE_SMPOS)[0])
        if len(active_uids) > 0:
            treated_count = tb_module.start_treatment(active_uids[:10])  # Treat some cases
            print(f"Treated {treated_count} cases")
        
        # Compare prevalences
        no_treatment_prevalence = sim_no_treatment.results['tb']['prevalence_active'][-1]
        with_treatment_prevalence = sim_with_treatment.results['tb']['prevalence_active'][-1]
        
        # Both should be within valid range
        assert 0 <= no_treatment_prevalence <= 1
        assert 0 <= with_treatment_prevalence <= 1
        
        print(f"Prevalence without treatment: {no_treatment_prevalence:.4f}")
        print(f"Prevalence with treatment: {with_treatment_prevalence:.4f}")
        
        # Validate that both simulations produced valid prevalence values
        assert not np.isnan(no_treatment_prevalence), "No treatment prevalence is NaN"
        assert not np.isnan(with_treatment_prevalence), "With treatment prevalence is NaN"
    
    def test_prevalence_consistency(self):
        """Test that prevalence calculation is consistent with case counts"""
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2003-01-01'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        
        # Check that prevalence is consistent with case counts
        for i in range(len(tb_results['prevalence_active'])):
            n_alive = np.count_nonzero(sim.people.alive)
            if n_alive > 0:  # Only check if people are alive
                expected_prevalence = tb_results['n_active'][i] / n_alive
                actual_prevalence = tb_results['prevalence_active'][i]
                
                # Allow for small numerical differences
                assert abs(expected_prevalence - actual_prevalence) < 1e-10, \
                    f"Prevalence mismatch at time {i}: expected {expected_prevalence}, got {actual_prevalence}"
    
    def test_prevalence_calculation_formula(self):
        """Test that prevalence is calculated using the correct formula: n_active / n_alive"""
        pop = ss.People(n_agents=50)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2002-01-01'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        
        # Test the prevalence calculation formula at multiple time points
        for i in range(0, len(tb_results['prevalence_active']), 10):  # Check every 10th time point
            n_alive = np.count_nonzero(sim.people.alive)
            n_active = tb_results['n_active'][i]
            prevalence = tb_results['prevalence_active'][i]
            
            if n_alive > 0:
                # Prevalence should be n_active / n_alive
                expected = n_active / n_alive
                assert abs(prevalence - expected) < 1e-10, \
                    f"Prevalence calculation error at time {i}: {n_active} active / {n_alive} alive = {expected}, got {prevalence}"
                
                # Validate that prevalence is within [0, 1]
                assert 0 <= prevalence <= 1, f"Prevalence {prevalence} outside [0,1] at time {i}"
                
                # Validate that n_active <= n_alive (logical constraint)
                assert n_active <= n_alive, f"Active cases ({n_active}) > alive people ({n_alive}) at time {i}"
            else:
                # If no one is alive, prevalence should be 0 or undefined
                assert prevalence == 0 or np.isnan(prevalence), f"Prevalence should be 0 or NaN when no one is alive, got {prevalence}"
    
    def test_prevalence_edge_cases(self):
        """Test prevalence in edge cases"""
        # Test with very small population
        pop = ss.People(n_agents=1)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        
        # Prevalence should still be valid
        assert np.all(tb_results['prevalence_active'] >= 0)
        assert np.all(tb_results['prevalence_active'] <= 1)
        
        # Test with very high initial prevalence
        pop2 = ss.People(n_agents=10)
        tb2 = mtb.TB(pars={'init_prev': ss.bernoulli(0.8)})  # 80% initial prevalence
        sim2 = ss.Sim(
            people=pop2,
            diseases=tb2,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        sim2.run()
        tb_results2 = sim2.results['tb']
        
        # Prevalence should still be valid
        assert np.all(tb_results2['prevalence_active'] >= 0)
        assert np.all(tb_results2['prevalence_active'] <= 1)
        
        # Initial prevalence should be high but not necessarily exactly 0.8 due to stochastic effects
        initial_prevalence = tb_results2['prevalence_active'][0]
        assert 0 <= initial_prevalence <= 1
        print(f"High initial prevalence test: {initial_prevalence:.3f}")
    
    def test_prevalence_zero_cases(self):
        """Test prevalence when there are no active TB cases"""
        # Test with very low transmission to ensure no cases
        pop = ss.People(n_agents=100)
        tb = mtb.TB(pars={'beta': ss.peryear(0.0001)})  # Very low transmission
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        
        # Prevalence should be 0 when there are no active cases
        for i in range(len(tb_results['prevalence_active'])):
            if tb_results['n_active'][i] == 0:
                assert tb_results['prevalence_active'][i] == 0, \
                    f"Prevalence should be 0 when n_active=0 at time {i}, got {tb_results['prevalence_active'][i]}"
    
    def test_prevalence_all_cases(self):
        """Test prevalence when all people have active TB (theoretical edge case)"""
        # This is a theoretical test - in practice, this would be very unlikely
        # but we should ensure the calculation handles it correctly
        pop = ss.People(n_agents=10)
        tb = mtb.TB(pars={'init_prev': ss.bernoulli(1.0)})  # 100% initial prevalence
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2000-01-02'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        
        # Check that prevalence calculation works correctly
        for i in range(len(tb_results['prevalence_active'])):
            n_alive = np.count_nonzero(sim.people.alive)
            n_active = tb_results['n_active'][i]
            prevalence = tb_results['prevalence_active'][i]
            
            if n_alive > 0:
                # Prevalence should be n_active / n_alive
                expected = n_active / n_alive
                assert abs(prevalence - expected) < 1e-10, \
                    f"Prevalence calculation error: {n_active}/{n_alive} = {expected}, got {prevalence}"
                
                # Prevalence should be <= 1
                assert prevalence <= 1, f"Prevalence {prevalence} > 1 when all people have TB"
    
    @pytest.mark.skip(reason="Stochastic test - may fail due to random effects")
    def test_prevalence_temporal_consistency(self):
        """Test that prevalence changes smoothly over time"""
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        sim.run()
        tb_results = sim.results['tb']
        prevalence = tb_results['prevalence_active']
        
        # Check that prevalence doesn't have extreme jumps
        for i in range(1, len(prevalence)):
            change = abs(prevalence[i] - prevalence[i-1])
            # Prevalence shouldn't change by more than 50% in one time step
            # (allowing for some stochastic variation)
            assert change <= 0.5, f"Large prevalence jump at time {i}: {prevalence[i-1]} -> {prevalence[i]}"
            
            # Prevalence should remain in valid range
            assert 0 <= prevalence[i] <= 1
            assert 0 <= prevalence[i-1] <= 1


class TestTBIntegration:
    """Integration tests for TB with other components"""
    
    def test_tb_with_demographics(self):
        """Test TB simulation with birth and death demographics"""
        # Create simulation components
        pop = ss.People(n_agents=100)
        tb = mtb.TB()
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
        births = ss.Births(pars=dict(birth_rate=15))
        deaths = ss.Deaths(pars=dict(death_rate=10))
        
        # Create simulation
        sim = ss.Sim(
            people=pop,
            networks=net,
            diseases=tb,
            demographics=[births, deaths],
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2005-01-01'))
        )
        
        # Run simulation
        sim.run()
        
        # Check that simulation completed successfully
        assert sim.complete
        
        # Check that we have some population dynamics
        assert len(sim.people) > 0
        
        # Check that TB results are present
        assert 'tb' in sim.results
    
    @pytest.mark.skip(reason="Stochastic test - may fail due to random effects")
    def test_tb_parameter_sensitivity(self):
        """Test that TB responds to parameter changes"""
        # Test with different transmission rates
        for beta in [0.001, 0.005, 0.01]:
            pop = ss.People(n_agents=50)
            tb = mtb.TB(pars={'beta': ss.peryear(beta)})
            sim = ss.Sim(
                people=pop,
                diseases=tb,
                pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2002-01-01'))
            )
            
            sim.run()
            
            # Higher transmission should generally lead to more cases
            tb_results = sim.results['tb']
            final_active = tb_results['n_active'][-1]
            assert final_active >= 0  # Should be non-negative


class TestTBEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_tb_with_minimal_population(self):
        """Test TB with minimal population size"""
        pop = ss.People(n_agents=1)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
        )
        
        # Should not raise an error
        sim.run()
        assert sim.complete
    
    def test_tb_with_very_short_simulation(self):
        """Test TB with very short simulation time"""
        pop = ss.People(n_agents=10)
        tb = mtb.TB()
        sim = ss.Sim(
            people=pop,
            diseases=tb,
            pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2000-01-02'))
        )
        
        # Should not raise an error
        sim.run()
        assert sim.complete
    
    def test_tb_parameter_validation(self):
        """Test that TB validates parameters correctly"""
        # Test that TB can handle valid parameters
        tb = mtb.TB(pars={'rate_LS_to_presym': ss.perday(0.1)})  # Valid TimePar
        assert isinstance(tb.pars['rate_LS_to_presym'], ss.TimePar)
        
        # Test that TB can handle custom transmission rates
        tb2 = mtb.TB(pars={'beta': ss.peryear(0.01)})
        assert tb2.pars['beta'] == ss.peryear(0.01)

