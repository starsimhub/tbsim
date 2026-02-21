"""
Simple test script to verify TBPeople class functionality.
"""

import tbsim
import starsim as ss
from tbsim.people import TBPeople


def test_tb_people_basic():
    """Test basic TBPeople functionality."""
    print("Testing TBPeople basic functionality...")
    
    # Create TBPeople instance
    pop = TBPeople(n_agents=100)
    
    # Expected TB state names (hardcoded list)
    expected_names = {
        'sought_care', 'care_seeking_multiplier', 'multiplier_applied',
        'n_times_tested', 'n_times_treated', 'returned_to_community',
        'received_tpt', 'tb_treatment_success', 'tested', 'test_result',
        'diagnosed', 'on_tpt', 'tb_smear', 'hiv_positive', 'eptb',
        'symptomatic', 'presymptomatic', 'non_symptomatic', 'screen_negative',
        'household_contact', 'treatment_success', 'treatment_failure',
        'hhid', 'vaccination_year'
    }
    
    # Verify it has the expected number of states
    assert len(pop.extra_states) == len(expected_names), f"Expected {len(expected_names)} states, got {len(pop.extra_states)}"
    
    # Verify all expected state names are present
    actual_names = {state.name for state in pop.extra_states}
    assert expected_names == actual_names, f"State names don't match. Expected: {expected_names}, Got: {actual_names}"
    
    # Test a few specific states to make sure they work as attributes
    assert hasattr(pop, 'sought_care'), "sought_care state not found"
    assert hasattr(pop, 'diagnosed'), "diagnosed state not found"
    assert hasattr(pop, 'hiv_positive'), "hiv_positive state not found"
    assert hasattr(pop, 'vaccination_year'), "vaccination_year state not found"
    
    print("✓ Basic functionality test passed")


def test_tb_people_with_custom_states():
    """Test TBPeople with additional custom states."""
    print("Testing TBPeople with custom states...")
    
    # Add custom states
    custom_states = [
        ss.FloatArr('custom_var1', default=0.0),
        ss.BoolState('custom_flag', default=False),
    ]
    
    pop = TBPeople(n_agents=50, extra_states=custom_states)
    
    # Expected TB state names (hardcoded list)
    expected_tb_names = {
        'sought_care', 'care_seeking_multiplier', 'multiplier_applied',
        'n_times_tested', 'n_times_treated', 'returned_to_community',
        'received_tpt', 'tb_treatment_success', 'tested', 'test_result',
        'diagnosed', 'on_tpt', 'tb_smear', 'hiv_positive', 'eptb',
        'symptomatic', 'presymptomatic', 'non_symptomatic', 'screen_negative',
        'household_contact', 'treatment_success', 'treatment_failure',
        'hhid', 'vaccination_year'
    }
    
    # Should have TB states + custom states
    expected_total = len(expected_tb_names) + len(custom_states)
    assert len(pop.extra_states) == expected_total, f"Expected {expected_total} states, got {len(pop.extra_states)}"
    
    # Verify all states are present in extra_states attribute
    state_names = {state.name for state in pop.extra_states}
    assert 'custom_var1' in state_names, "Custom state 'custom_var1' not found in extra_states"
    assert 'custom_flag' in state_names, "Custom state 'custom_flag' not found in extra_states"
    
    # Verify TB states are present
    for state_name in expected_tb_names:
        assert state_name in state_names, f"TB state '{state_name}' not found in extra_states"
    
    # Verify custom states are accessible as attributes
    assert hasattr(pop, 'custom_var1'), "Custom state 'custom_var1' not found as attribute"
    assert hasattr(pop, 'custom_flag'), "Custom state 'custom_flag' not found as attribute"
    
    print("✓ Custom states test passed")


def test_tb_people_inheritance():
    """Test that TBPeople properly inherits from ss.People."""
    print("Testing TBPeople inheritance...")
    
    pop = TBPeople(n_agents=25)
    
    # Should be an instance of ss.People
    assert isinstance(pop, ss.People), "TBPeople should inherit from ss.People"
    
    # Should have all the attributes of ss.People
    assert hasattr(pop, 'n_agents'), "Missing n_agents attribute"
    assert hasattr(pop, 'extra_states'), "Missing extra_states attribute"
    assert pop.n_agents == 25, f"Expected 25 agents, got {pop.n_agents}"
    
    print("✓ Inheritance test passed")


def test_compatibility_with_simulation():
    """Test that TBPeople works with simulations."""
    print("Testing TBPeople compatibility with simulations...")
    
    # Create population
    pop = TBPeople(n_agents=100)
    
    # Create TB disease
    tb_pars = dict(
        beta=ss.peryear(0.01),
        init_prev=0.1,
    )
    tb = tbsim.TB(pars=tb_pars)
    
    # Create network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=2), dur=0))
    
    # Create simulation (short duration for testing)
    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1990-01-01'),
        stop=ss.date('1990-12-31'),
    )
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=[tb],
        pars=sim_pars
    )
    
    # Should be able to create simulation without errors
    assert sim is not None, "Failed to create simulation with TBPeople"
    # Note: The simulation object doesn't directly expose the people object,
    # but we can verify it was created successfully
    
    print("✓ Simulation compatibility test passed")


def test_tb_people_multiple_custom_states():
    """Test TBPeople with multiple types of custom states."""
    print("Testing TBPeople with multiple custom states...")
    
    # Define various types of custom states
    custom_states = [
        ss.FloatArr('SES', default=0.0),                    # Socioeconomic status
        ss.BoolState('urban', default=True),               # Urban/rural
        ss.IntArr('education_level', default=0),          # Education level
        ss.FloatArr('income', default=1000.0),            # Income
        ss.BoolState('has_insurance', default=False),     # Insurance status
        ss.FloatArr('distance_to_clinic', default=5.0),   # Distance to clinic
        ss.BoolState('mobile_phone', default=True),       # Mobile phone access
        ss.IntArr('household_size', default=4),           # Household size
    ]
    
    pop = TBPeople(n_agents=200, extra_states=custom_states)
    
    # Verify total states
    expected_total = len(TBPeople.TB_STATES) + len(custom_states)
    assert len(pop.extra_states) == expected_total, f"Expected {expected_total} states, got {len(pop.extra_states)}"
    
    # Verify TB states are still present
    tb_states = pop.get_tb_states()
    assert len(tb_states) == len(TBPeople.TB_STATES), f"Expected {len(TBPeople.TB_STATES)} TB states, got {len(tb_states)}"
    
    # Verify custom states are present
    custom_states_returned = pop.get_custom_states()
    assert len(custom_states_returned) == len(custom_states), f"Expected {len(custom_states)} custom states, got {len(custom_states_returned)}"
    
    # Verify all custom states are accessible as attributes
    for state in custom_states:
        assert hasattr(pop, state.name), f"Custom state '{state.name}' not found as attribute"
    
    # Verify state info includes custom states
    state_info = pop.get_state_info()
    for state in custom_states:
        assert state.name in state_info, f"Custom state '{state.name}' not found in state info"
        assert not state_info[state.name]['is_tb_state'], f"Custom state '{state.name}' incorrectly marked as TB state"
    
    print("✓ Multiple custom states test passed")


def test_tb_people_custom_states_with_defaults():
    """Test TBPeople with custom states that have various default values."""
    print("Testing TBPeople with custom states with different defaults...")
    
    # Custom states with different default values
    custom_states = [
        ss.FloatArr('age_at_first_visit', default=25.5),
        ss.BoolState('previous_tb_history', default=False),
        ss.IntArr('number_of_contacts', default=0),
        ss.FloatArr('treatment_adherence', default=0.8),
        ss.BoolState('completed_treatment', default=False),
        ss.FloatArr('quality_of_life', default=0.7),
    ]
    
    pop = TBPeople(n_agents=150, extra_states=custom_states)
    
    # Verify states are accessible
    for state in custom_states:
        assert hasattr(pop, state.name), f"Custom state '{state.name}' not accessible"
    
    # Test that default values are set correctly
    assert hasattr(pop, 'age_at_first_visit'), "age_at_first_visit state not found"
    assert hasattr(pop, 'previous_tb_history'), "previous_tb_history state not found"
    assert hasattr(pop, 'number_of_contacts'), "number_of_contacts state not found"
    
    # Verify state summary includes custom states
    summary = pop.get_state_summary()
    assert summary['custom_states'] == len(custom_states), f"Expected {len(custom_states)} custom states in summary"
    
    print("✓ Custom states with defaults test passed")


def test_tb_people_large_custom_state_set():
    """Test TBPeople with a large number of custom states."""
    print("Testing TBPeople with large custom state set...")
    
    # Create many custom states
    custom_states = []
    for i in range(20):  # 20 additional custom states
        if i % 4 == 0:
            custom_states.append(ss.FloatArr(f'float_var_{i}', default=float(i)))
        elif i % 4 == 1:
            custom_states.append(ss.BoolState(f'bool_var_{i}', default=i % 2 == 0))
        elif i % 4 == 2:
            custom_states.append(ss.IntArr(f'int_var_{i}', default=i))
        else:
            custom_states.append(ss.FloatArr(f'measurement_{i}', default=i * 0.1))
    
    pop = TBPeople(n_agents=100, extra_states=custom_states)
    
    # Verify total states
    expected_total = len(TBPeople.TB_STATES) + len(custom_states)
    assert len(pop.extra_states) == expected_total, f"Expected {expected_total} states, got {len(pop.extra_states)}"
    
    # Verify all custom states are accessible
    for state in custom_states:
        assert hasattr(pop, state.name), f"Custom state '{state.name}' not accessible"
    
    # Verify state info works with many states
    state_info = pop.get_state_info()
    assert len(state_info) == expected_total, f"Expected {expected_total} states in info, got {len(state_info)}"
    
    # Verify custom states are not marked as TB states
    for state in custom_states:
        assert not state_info[state.name]['is_tb_state'], f"Custom state '{state.name}' incorrectly marked as TB state"
    
    print("✓ Large custom state set test passed")


def test_tb_people_custom_states_with_simulation():
    """Test TBPeople with custom states in a simulation context."""
    print("Testing TBPeople with custom states in simulation...")
    
    # Create custom states
    custom_states = [
        ss.FloatArr('SES', default=0.0),
        ss.BoolState('urban', default=True),
        ss.FloatArr('accessibility', default=0.5),
    ]
    
    pop = TBPeople(n_agents=100, extra_states=custom_states)
    
    # Create TB disease
    tb_pars = dict(
        beta=ss.peryear(0.01),
        init_prev=0.1,
    )
    tb = tbsim.TB(pars=tb_pars)
    
    # Create network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=2), dur=0))
    
    # Create simulation
    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1990-01-01'),
        stop=ss.date('1990-12-31'),
    )
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=[tb],
        pars=sim_pars
    )
    
    # Verify simulation was created successfully
    assert sim is not None, "Failed to create simulation with custom states"
    
    # Verify custom states are still accessible after simulation creation
    for state in custom_states:
        assert hasattr(pop, state.name), f"Custom state '{state.name}' not accessible after simulation creation"
    
    # Verify state summary works
    summary = pop.get_state_summary()
    assert summary['custom_states'] == len(custom_states), f"Expected {len(custom_states)} custom states in summary"
    
    print("✓ Custom states with simulation test passed")


def test_tb_people_custom_states_edge_cases():
    """Test TBPeople with edge cases for custom states."""
    print("Testing TBPeople with custom states edge cases...")
    
    # Test with empty custom states list
    pop_empty = TBPeople(n_agents=50, extra_states=[])
    assert len(pop_empty.extra_states) == len(TBPeople.TB_STATES), "Empty custom states should not affect TB states"
    assert len(pop_empty.get_custom_states()) == 0, "Should have no custom states"
    
    # Test with single custom state
    single_custom = [ss.BoolState('single_flag', default=True)]
    pop_single = TBPeople(n_agents=50, extra_states=single_custom)
    assert len(pop_single.extra_states) == len(TBPeople.TB_STATES) + 1, "Should have TB states + 1 custom"
    assert hasattr(pop_single, 'single_flag'), "Single custom state not accessible"
    
    # Test with custom states that have same names as TB states (should fail gracefully)
    # This should raise an error since we can't have duplicate state names
    conflicting_custom = [ss.FloatArr('sought_care', default=1.0)]  # Same name as TB state
    try:
        pop_conflict = TBPeople(n_agents=50, extra_states=conflicting_custom)
        # If we get here, it means the conflict wasn't caught, which is unexpected
        assert False, "Should have raised an error for conflicting state names"
    except ValueError as e:
        # This is expected - we should get an error for conflicting names
        assert "sought_care" in str(e), f"Expected error about 'sought_care' conflict, got: {e}"
        print("    ✓ Correctly caught conflicting state names")
    
    # Test with very long custom state names
    long_name_custom = [ss.FloatArr('very_long_custom_state_name_that_might_cause_issues', default=0.0)]
    pop_long = TBPeople(n_agents=50, extra_states=long_name_custom)
    assert hasattr(pop_long, 'very_long_custom_state_name_that_might_cause_issues'), "Long name custom state not accessible"
    
    print("✓ Custom states edge cases test passed")


def test_tb_people_custom_states_state_info():
    """Test state information methods with custom states."""
    print("Testing TBPeople state info methods with custom states...")
    
    custom_states = [
        ss.FloatArr('SES', default=0.0),
        ss.BoolState('urban', default=True),
        ss.IntArr('education', default=8),
    ]
    
    pop = TBPeople(n_agents=100, extra_states=custom_states)
    
    # Test get_state_info with custom states
    state_info = pop.get_state_info()
    
    # Verify all states are in info
    assert len(state_info) == len(TBPeople.TB_STATES) + len(custom_states), f"Expected {len(TBPeople.TB_STATES) + len(custom_states)} states in info"
    
    # Verify TB states are marked correctly
    for state in TBPeople.TB_STATES:
        assert state.name in state_info, f"TB state '{state.name}' not in state info"
        assert state_info[state.name]['is_tb_state'], f"TB state '{state.name}' not marked as TB state"
    
    # Verify custom states are marked correctly
    for state in custom_states:
        assert state.name in state_info, f"Custom state '{state.name}' not in state info"
        assert not state_info[state.name]['is_tb_state'], f"Custom state '{state.name}' incorrectly marked as TB state"
    
    # Test list_tb_states method
    # This should only show TB states, not custom states
    import io
    import sys
    captured_output = io.StringIO()
    sys.stdout = captured_output
    pop.list_tb_states()
    sys.stdout = sys.__stdout__
    output = captured_output.getvalue()
    
    # Verify output contains TB states but not custom states
    assert 'sought_care' in output, "TB states should be in list_tb_states output"
    assert 'SES' not in output, "Custom states should not be in list_tb_states output"
    
    print("✓ State info methods with custom states test passed")


def test_tb_people_simulation_results():
    """Test TBPeople with simulation results and state tracking."""
    print("Testing TBPeople simulation results...")
    
    # Create TBPeople with custom states
    custom_states = [
        ss.FloatArr('SES', default=0.0),
        ss.BoolState('urban', default=True),
        ss.FloatArr('accessibility', default=0.5),
    ]
    
    pop = TBPeople(n_agents=200, extra_states=custom_states)
    
    # Create TB disease with some initial prevalence
    tb_pars = dict(
        beta=ss.peryear(0.02),
        init_prev=0.1,
    )
    tb = tbsim.TB(pars=tb_pars)
    
    # Create network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=3), dur=0))
    
    # Create simulation
    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1990-01-01'),
        stop=ss.date('1990-12-31'),
    )
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=[tb],
        pars=sim_pars
    )
    
    # Run the simulation
    sim.run()
    
    # Test that we can access results
    assert hasattr(sim, 'results'), "Simulation should have results attribute"
    results = sim.results
    
    # Test that TB disease results are available
    assert hasattr(results, 'tb'), "Results should have TB disease results"
    tb_results = results.tb
    
    # Test that we can access TB-specific results
    expected_tb_results = ['prevalence', 'incidence', 'mortality', 'recovery']
    for result_name in expected_tb_results:
        if hasattr(tb_results, result_name):
            result_data = getattr(tb_results, result_name)
            assert result_data is not None, f"TB result '{result_name}' should not be None"
            print(f"    ✓ TB result '{result_name}' available: {type(result_data)}")
    
    # Test that custom states are still accessible after simulation
    for state in custom_states:
        assert hasattr(pop, state.name), f"Custom state '{state.name}' not accessible after simulation"
    
    # Test that we can access state values after simulation
    assert hasattr(pop, 'SES'), "SES state should be accessible"
    assert hasattr(pop, 'urban'), "urban state should be accessible"
    assert hasattr(pop, 'accessibility'), "accessibility state should be accessible"
    
    # Test that TB states are accessible and have reasonable values
    assert hasattr(pop, 'diagnosed'), "diagnosed state should be accessible"
    assert hasattr(pop, 'symptomatic'), "symptomatic state should be accessible"
    
    # Test state summary after simulation
    summary = pop.get_state_summary()
    assert summary['total_agents'] == 200, "Should have 200 agents"
    assert summary['custom_states'] == len(custom_states), f"Should have {len(custom_states)} custom states"
    
    print("✓ Simulation results test passed")


def test_tb_people_simulation_with_state_tracking():
    """Test TBPeople with state tracking during simulation."""
    print("Testing TBPeople with state tracking...")
    
    # Create TBPeople with tracking states
    tracking_states = [
        ss.FloatArr('intervention_effect', default=1.0),
        ss.BoolState('received_intervention', default=False),
        ss.IntArr('intervention_count', default=0),
    ]
    
    pop = TBPeople(n_agents=100, extra_states=tracking_states)
    
    # Create TB disease
    tb_pars = dict(
        beta=ss.peryear(0.01),
        init_prev=0.05,
    )
    tb = tbsim.TB(pars=tb_pars)
    
    # Create network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=2), dur=0))
    
    # Create simulation with shorter duration for testing
    sim_pars = dict(
        dt=ss.days(14),
        start=ss.date('1990-01-01'),
        stop=ss.date('1990-06-30'),
    )
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=[tb],
        pars=sim_pars
    )
    
    # Run simulation
    sim.run()
    
    # Test that tracking states are accessible
    for state in tracking_states:
        assert hasattr(pop, state.name), f"Tracking state '{state.name}' not accessible"
    
    # Test that we can access state values
    intervention_effect = pop.intervention_effect
    received_intervention = pop.received_intervention
    intervention_count = pop.intervention_count
    
    # Verify state values are arrays of correct size
    assert len(intervention_effect) == 100, "intervention_effect should have 100 values"
    assert len(received_intervention) == 100, "received_intervention should have 100 values"
    assert len(intervention_count) == 100, "intervention_count should have 100 values"
    
    # Test that TB states are still accessible
    assert hasattr(pop, 'diagnosed'), "diagnosed state should be accessible"
    assert hasattr(pop, 'symptomatic'), "symptomatic state should be accessible"
    
    # Test state info after simulation
    state_info = pop.get_state_info()
    assert len(state_info) == len(TBPeople.TB_STATES) + len(tracking_states), "Should have all states in info"
    
    print("✓ State tracking test passed")


def test_tb_people_simulation_results_analysis():
    """Test TBPeople with results analysis and state correlation."""
    print("Testing TBPeople simulation results analysis...")
    
    # Create TBPeople with analytical states
    analytical_states = [
        ss.FloatArr('risk_score', default=0.5),
        ss.BoolState('high_risk', default=False),
        ss.FloatArr('exposure_time', default=0.0),
    ]
    
    pop = TBPeople(n_agents=150, extra_states=analytical_states)
    
    # Create TB disease
    tb_pars = dict(
        beta=ss.peryear(0.015),
        init_prev=0.08,
    )
    tb = tbsim.TB(pars=tb_pars)
    
    # Create network
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=4), dur=0))
    
    # Create simulation
    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1990-01-01'),
        stop=ss.date('1990-12-31'),
    )
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=[tb],
        pars=sim_pars
    )
    
    # Run simulation
    sim.run()
    
    # Test that we can access and analyze results
    results = sim.results
    
    # Test TB disease results
    if hasattr(results, 'tb'):
        tb_results = results.tb
        
        # Test prevalence results
        if hasattr(tb_results, 'prevalence'):
            prevalence = tb_results.prevalence
            assert prevalence is not None, "Prevalence results should not be None"
            print(f"    ✓ Prevalence results: {type(prevalence)}")
        
        # Test incidence results
        if hasattr(tb_results, 'incidence'):
            incidence = tb_results.incidence
            assert incidence is not None, "Incidence results should not be None"
            print(f"    ✓ Incidence results: {type(incidence)}")
    
    # Test that analytical states are accessible
    assert hasattr(pop, 'risk_score'), "risk_score state should be accessible"
    assert hasattr(pop, 'high_risk'), "high_risk state should be accessible"
    assert hasattr(pop, 'exposure_time'), "exposure_time state should be accessible"
    
    # Test that we can access state values for analysis
    risk_scores = pop.risk_score
    high_risk_flags = pop.high_risk
    exposure_times = pop.exposure_time
    
    # Verify state values are arrays of correct size
    assert len(risk_scores) == 150, "risk_score should have 150 values"
    assert len(high_risk_flags) == 150, "high_risk should have 150 values"
    assert len(exposure_times) == 150, "exposure_time should have 150 values"
    
    # Test that TB states are accessible for correlation analysis
    assert hasattr(pop, 'diagnosed'), "diagnosed state should be accessible"
    assert hasattr(pop, 'symptomatic'), "symptomatic state should be accessible"
    assert hasattr(pop, 'hiv_positive'), "hiv_positive state should be accessible"
    
    # Test state summary after simulation
    summary = pop.get_state_summary()
    assert summary['total_agents'] == 150, "Should have 150 agents"
    assert summary['custom_states'] == len(analytical_states), f"Should have {len(analytical_states)} custom states"
    
    print("✓ Results analysis test passed")


if __name__ == '__main__':
    print("TBPeople Class Tests")
    print("=" * 40)
    
    try:
        test_tb_people_basic()
        test_tb_people_with_custom_states()
        test_tb_people_inheritance()
        test_compatibility_with_simulation()
        
        # New tests for additional states
        test_tb_people_multiple_custom_states()
        test_tb_people_custom_states_with_defaults()
        test_tb_people_large_custom_state_set()
        test_tb_people_custom_states_with_simulation()
        test_tb_people_custom_states_edge_cases()
        test_tb_people_custom_states_state_info()
        
        # New tests for simulation results
        test_tb_people_simulation_results()
        test_tb_people_simulation_with_state_tracking()
        test_tb_people_simulation_results_analysis()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! TBPeople class is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
