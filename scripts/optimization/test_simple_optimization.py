"""
Simple TB Optimization Test Script

This script provides basic testing functionality for the TB model parameter optimization
framework. It tests individual simulations and small optimization runs with conservative
parameters to identify and debug issues before running large-scale optimizations.

Purpose:
--------
- Verify that the TB calibration simulation runs correctly
- Test single simulation execution
- Test small-scale optimization with minimal parameter combinations
- Identify issues early before running expensive large-scale optimizations
- Provide quick feedback on optimization framework functionality

Components:
-----------
- Single simulation test with conservative parameters
- Small optimization with 2x2x2 = 8 parameter combinations
- Conservative population size (300 agents) for fast execution
- Short simulation period (100 years) for testing

Usage:
------
    python scripts/optimization/test_simple_optimization.py

Output:
-------
- Test results for single simulation
- Test results for small optimization
- Success/failure indicators
- Parameter combinations and scores for successful runs
- Diagnostic information if tests fail

Notes:
------
Run this script before running full optimizations to ensure:
- All dependencies are correctly installed
- Calibration functions work properly
- South Africa data is accessible
- Parameter ranges are appropriate
- No fundamental simulation issues exist
"""

import numpy as np
import pandas as pd
import datetime
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.burn_in.run_tb_burn_in_South_Africa import (
    create_south_africa_data,
    run_calibration_simulation,
    calculate_calibration_score
)


def test_single_simulation():
    """
    Test a single TB calibration simulation with conservative parameters.
    
    This function runs one complete simulation to verify that the calibration
    framework is functioning correctly. It uses conservative parameters that
    should reliably complete without errors.
    
    Test Parameters:
    ----------------
    - beta = 0.015 (conservative transmission rate)
    - rel_sus_latentslow = 0.10 (low susceptibility)
    - tb_mortality = 2e-4 (moderate mortality)
    - n_agents = 300 (small population for speed)
    - years = 100 (short simulation for testing)
    
    Returns
    -------
    bool
        True if the test passed successfully, False if it failed
        
    Output:
    -------
    Prints:
    - Test progress and status
    - Calibration score if successful
    - Overall prevalence achieved
    - Notification and age prevalence MAPEs
    - Full error traceback if test fails
    
    Notes
    -----
    If this test fails, there are likely fundamental issues with:
    - Module imports or dependencies
    - Calibration simulation setup
    - South Africa data access
    - Parameter value ranges
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
    Test a small-scale optimization with minimal parameter combinations.
    
    This function tests the complete optimization loop with a minimal number
    of parameter combinations (2×2×2 = 8 combinations). It verifies that:
    - Multiple simulations can run sequentially
    - Results are properly collected and sorted
    - Best parameters are correctly identified
    - No errors occur during the optimization loop
    
    Parameter Ranges:
    -----------------
    - beta: [0.015, 0.020]
    - rel_sus_latentslow: [0.10, 0.15]
    - tb_mortality: [2e-4, 3e-4]
    
    Returns
    -------
    bool
        True if at least one simulation succeeded and optimization completed,
        False if all simulations failed
        
    Output:
    -------
    Prints:
    - Progress for each parameter combination
    - Success/failure status for each run
    - Summary of successful simulations
    - Best parameter combination and score
    - Overall prevalence for best parameters
    
    Notes
    -----
    This is a lightweight test using:
    - 300 agents (very small for speed)
    - 100 years simulation (shorter for testing)
    - Only 8 parameter combinations (2^3)
    
    If this test passes, the full optimization framework should work correctly.
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
    Main test function to run all TB optimization tests sequentially.
    
    Orchestrates the complete testing workflow:
    1. Runs single simulation test to verify basic functionality
    2. If single test passes, runs small optimization test
    3. Reports overall testing status and identifies issues
    
    The tests are run in sequence with early termination if fundamental
    issues are detected.
    
    Test Sequence:
    --------------
    1. Single simulation test
       - Tests basic calibration framework
       - Uses one parameter set
       - Fast execution (~10-30 seconds)
    
    2. Small optimization test (only if single test passes)
       - Tests optimization loop
       - Uses 8 parameter combinations
       - Moderate execution (~2-5 minutes)
    
    Exit Status Messages:
    ---------------------
    - "Single simulation test passed" - Basic framework works
    - "Small optimization test passed" - Optimization framework works
    - "Single simulation test failed" - Fundamental setup issues
    - "Small optimization test failed" - Issues with optimization loop
    
    Notes
    -----
    Run this before attempting large-scale optimizations to catch issues early.
    The tests use conservative parameters to maximize likelihood of success.
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