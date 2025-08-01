"""
Simple test script for TB optimization with conservative parameters
"""

import numpy as np
import pandas as pd
import datetime
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb_calibration_south_africa import (
    create_south_africa_data,
    run_calibration_simulation,
    calculate_calibration_score
)


def test_single_simulation():
    """
    Test a single simulation to make sure everything works
    """
    
    print("=== Testing Single Simulation ===")
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    print("✓ Created South Africa data")
    
    # Test with conservative parameters
    print("Testing with conservative parameters: β=0.015, rel_sus=0.10, mort=2e-4")
    
    try:
        # Run simulation
        sim = run_calibration_simulation(
            beta=0.015,
            rel_sus_latentslow=0.10,
            tb_mortality=2e-4,
            n_agents=300,  # Very small for testing
            years=100      # Short for testing
        )
        print("✓ Simulation completed successfully")
        
        # Calculate score
        score_metrics = calculate_calibration_score(sim, sa_data)
        print("✓ Calibration score calculated")
        
        print(f"\nResults:")
        print(f"  Composite Score: {score_metrics['composite_score']:.2f}")
        print(f"  Overall Prevalence: {score_metrics['model_overall_prev']:.3f}%")
        print(f"  Notification MAPE: {score_metrics['notification_mape']:.1f}%")
        print(f"  Age Prevalence MAPE: {score_metrics['age_prev_mape']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        print(f"Full error: {traceback.format_exc()}")
        return False


def test_small_optimization():
    """
    Test a very small optimization with just a few parameter combinations
    """
    
    print("\n=== Testing Small Optimization ===")
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    
    # Very small parameter ranges
    beta_range = np.array([0.015, 0.020])
    rel_sus_range = np.array([0.10, 0.15])
    tb_mortality_range = np.array([2e-4, 3e-4])
    
    print(f"Testing {len(beta_range) * len(rel_sus_range) * len(tb_mortality_range)} combinations")
    
    results = []
    
    for beta in beta_range:
        for rel_sus in rel_sus_range:
            for tb_mortality in tb_mortality_range:
                
                print(f"\nTesting: β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}")
                
                try:
                    # Run simulation
                    sim = run_calibration_simulation(
                        beta=beta,
                        rel_sus_latentslow=rel_sus,
                        tb_mortality=tb_mortality,
                        n_agents=300,
                        years=100
                    )
                    
                    # Calculate score
                    score_metrics = calculate_calibration_score(sim, sa_data)
                    
                    # Store results
                    result = {
                        'beta': beta,
                        'rel_sus_latentslow': rel_sus,
                        'tb_mortality': tb_mortality,
                        'composite_score': score_metrics['composite_score'],
                        'model_overall_prev': score_metrics['model_overall_prev']
                    }
                    results.append(result)
                    
                    print(f"  ✓ Success! Score: {score_metrics['composite_score']:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    continue
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('composite_score')
        
        print(f"\n=== OPTIMIZATION RESULTS ===")
        print(f"Successful simulations: {len(results)}")
        print(f"\nBest result:")
        best = results_df.iloc[0]
        print(f"  β={best['beta']:.3f}, rel_sus={best['rel_sus_latentslow']:.2f}, "
              f"mort={best['tb_mortality']:.1e}")
        print(f"  Score: {best['composite_score']:.2f}")
        print(f"  Overall Prevalence: {best['model_overall_prev']:.3f}%")
        
        return True
    else:
        print("❌ No successful simulations!")
        return False


def main():
    """
    Main test function
    """
    
    print("TB Model Optimization Test")
    print("Testing with conservative parameters to identify issues\n")
    
    # Test single simulation first
    single_success = test_single_simulation()
    
    if single_success:
        print("\n✓ Single simulation test passed!")
        
        # Test small optimization
        opt_success = test_small_optimization()
        
        if opt_success:
            print("\n✓ Small optimization test passed!")
            print("The optimization framework is working correctly.")
        else:
            print("\n❌ Small optimization test failed!")
            print("There may be issues with the optimization loop.")
    else:
        print("\n❌ Single simulation test failed!")
        print("There are fundamental issues with the simulation setup.")


if __name__ == '__main__':
    main() 