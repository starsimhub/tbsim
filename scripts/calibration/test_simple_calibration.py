"""
Simple test script for TB calibration functionality

This script tests basic calibration functions to ensure they work correctly.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_data_creation():
    """Test that South Africa data can be created"""
    try:
        from tb_calibration_south_africa import create_south_africa_data
        sa_data = create_south_africa_data()
        
        # Check that data has expected structure
        assert 'case_notifications' in sa_data
        assert 'age_prevalence' in sa_data
        assert 'targets' in sa_data
        
        # Check case notifications data
        cn_df = sa_data['case_notifications']
        assert len(cn_df) == 5  # 5 years of data
        assert 'rate_per_100k' in cn_df.columns
        
        # Check age prevalence data
        ap_df = sa_data['age_prevalence']
        assert len(ap_df) == 6  # 6 age groups
        assert 'prevalence_per_100k' in ap_df.columns
        
        print("✓ Data creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data creation test failed: {e}")
        return False

def test_calibration_score():
    """Test that calibration score calculation works"""
    try:
        from tb_calibration_sweep import calculate_calibration_score
        
        # Create mock simulation object with required attributes
        class MockSim:
            def __init__(self):
                self.results = {
                    'timevec': [pd.Timestamp(f'{year}-01-01') for year in range(2000, 2021)],
                    'tb': {
                        'prevalence_active': np.random.uniform(0.001, 0.01, 21)
                    },
                    'tbdiagnostic': {
                        'n_test_positive': pd.Series(np.random.randint(10, 100, 21))
                    },
                    'n_alive': np.random.randint(1000, 2000, 21)
                }
                # Mock people object for age stratification
                class MockPeople:
                    def __init__(self):
                        self.alive = np.ones(1000, dtype=bool)
                        self.age = np.random.randint(15, 80, 1000)
                
                self.people = MockPeople()
                
                # Mock diseases object
                class MockDiseases:
                    def __init__(self):
                        class MockTB:
                            def __init__(self):
                                self.state = np.random.choice([0, 1, 2, 3], 1000)
                        self.tb = MockTB()
                
                self.diseases = MockDiseases()
        
        # Create mock South Africa data
        sa_data = {
            'case_notifications': pd.DataFrame({
                'rate_per_100k': [650, 950, 980, 834, 554]
            }),
            'age_prevalence': pd.DataFrame({
                'prevalence_per_100k': [850, 1200, 1400, 1600, 1800, 2200]
            }),
            'targets': {
                'overall_prevalence_2018': 0.852
            }
        }
        
        mock_sim = MockSim()
        score = calculate_calibration_score(mock_sim, sa_data)
        
        # Check that score has expected structure
        assert 'composite_score' in score
        assert 'notification_mape' in score
        assert 'age_prev_mape' in score
        
        print("✓ Calibration score test passed")
        return True
        
    except Exception as e:
        print(f"✗ Calibration score test failed: {e}")
        return False

def test_simple_simulation():
    """Test that a simple TB simulation can run without HIV"""
    try:
        import starsim as ss
        import tbsim as mtb
        
        # Create a simple simulation with just TB, no HIV
        sim_pars = dict(
            unit='day',
            dt=30,
            start=ss.date('2000-01-01'),
            stop=ss.date('2010-01-01'),
            rand_seed=42,
            verbose=0,
        )
        
        # Simple demographics
        demog = [
            ss.Births(birth_rate=25, unit='day', dt=30),  # Higher birth rate
            ss.Deaths(death_rate=5, unit='day', dt=30, rate_units=1),  # Lower death rate
        ]
        
        # Create population
        people = ss.People(n_agents=500, extra_states=mtb.get_extrastates())
        
        # Simple TB parameters
        tb_pars = dict(
            beta=ss.rate_prob(0.01, unit='day'),  # Lower transmission rate
            init_prev=ss.bernoulli(p=0.05),  # Lower initial prevalence
            rel_sus_latentslow=0.1,
            rate_LS_to_presym=ss.perday(1e-5),  # Much slower progression
            rate_LF_to_presym=ss.perday(1e-3),
            rate_active_to_clear=ss.perday(1e-4),
            rate_smpos_to_dead=ss.perday(1e-5),  # Much lower mortality
            rate_exptb_to_dead=ss.perday(1e-6),
            rate_smneg_to_dead=ss.perday(1e-6),
        )
        
        tb = mtb.TB(pars=tb_pars)
        
        # Simple network
        net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=3), dur=0))
        
        # Run simulation
        sim = ss.Sim(
            people=people,
            diseases=[tb],
            networks=net,
            demographics=demog,
            pars=sim_pars,
        )
        sim.run()
        
        # Check that simulation completed
        assert hasattr(sim, 'results')
        assert 'timevec' in sim.results
        assert 'tb' in sim.results
        
        print("✓ Simple simulation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Simple simulation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Running TB calibration tests...\n")
    
    tests = [
        test_data_creation,
        test_calibration_score,
        test_simple_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 