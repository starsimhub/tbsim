"""
Example: Using the Centralized CalibrationPlotter Class

This script demonstrates how to use the centralized CalibrationPlotter class
from tbsim.utils for creating various types of calibration plots.
"""

import numpy as np
import pandas as pd
import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the centralized plotting class and utilities
from tbsim.utils import CalibrationPlotter, create_south_africa_data, calculate_calibration_score
from tb_calibration_south_africa import run_calibration_simulation


def example_basic_usage():
    """
    Example 1: Basic usage of CalibrationPlotter
    """
    print("=== Example 1: Basic CalibrationPlotter Usage ===")
    
    # Create a plotter instance
    plotter = CalibrationPlotter(style='default')
    
    # Run a simple simulation
    print("Running simulation...")
    sim = run_calibration_simulation(
        beta=0.020,
        rel_sus_latentslow=0.15,
        tb_mortality=3e-4,
        n_agents=500,
        years=100
    )
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Create calibration comparison plot
    print("Creating calibration comparison plot...")
    fig = plotter.plot_calibration_comparison(
        sim, 
        sa_data, 
        timestamp,
        save_path=f"example_calibration_comparison_{timestamp}.pdf"
    )
    
    print(f"✓ Calibration comparison plot saved")
    return sim, sa_data, timestamp


def example_parameter_sweep_plots():
    """
    Example 2: Creating parameter sweep plots with synthetic data
    """
    print("\n=== Example 2: Parameter Sweep Plots ===")
    
    # Create synthetic sweep results for demonstration
    np.random.seed(42)
    n_simulations = 30
    
    # Generate synthetic parameter combinations
    betas = np.random.uniform(0.01, 0.03, n_simulations)
    rel_sus = np.random.uniform(0.05, 0.25, n_simulations)
    tb_mort = np.random.uniform(1e-4, 5e-4, n_simulations)
    
    # Generate synthetic scores (lower is better)
    base_score = 50
    score_noise = np.random.normal(0, 10, n_simulations)
    composite_scores = base_score + score_noise + (betas - 0.02)**2 * 1000 + (rel_sus - 0.15)**2 * 200
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'beta': betas,
        'rel_sus_latentslow': rel_sus,
        'tb_mortality': tb_mort,
        'composite_score': composite_scores,
        'simulation_number': range(1, n_simulations + 1)
    })
    
    # Create plotter
    plotter = CalibrationPlotter()
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Create comprehensive sweep results plot
    print("Creating comprehensive sweep results plot...")
    sweep_fig = plotter.plot_sweep_results(
        results_df, 
        timestamp,
        save_path=f"example_sweep_results_{timestamp}.pdf"
    )
    
    # Create focused violin plots
    print("Creating focused violin plots...")
    violin_fig = plotter.plot_violin_plots(
        results_df, 
        timestamp,
        save_path=f"example_violin_plots_{timestamp}.pdf"
    )
    
    print(f"✓ Sweep plots saved")
    return results_df, timestamp


def example_custom_plotter():
    """
    Example 3: Customizing the CalibrationPlotter
    """
    print("\n=== Example 3: Customized CalibrationPlotter ===")
    
    # Create a plotter with custom settings
    custom_plotter = CalibrationPlotter(
        style='seaborn-v0_8',  # Different style
        figsize=(16, 10)       # Custom figure size
    )
    
    # Run simulation
    sim = run_calibration_simulation(
        beta=0.015,
        rel_sus_latentslow=0.20,
        tb_mortality=3e-4,
        n_agents=300,
        years=80
    )
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Create plot with custom settings
    print("Creating customized calibration plot...")
    fig = custom_plotter.plot_calibration_comparison(
        sim, 
        sa_data, 
        timestamp,
        save_path=f"example_custom_plot_{timestamp}.pdf"
    )
    
    print(f"✓ Customized plot saved")
    return sim, timestamp


def example_batch_processing():
    """
    Example 4: Batch processing multiple simulations
    """
    print("\n=== Example 4: Batch Processing ===")
    
    # Define parameter sets to test
    parameter_sets = [
        {'beta': 0.010, 'rel_sus_latentslow': 0.10, 'tb_mortality': 2e-4},
        {'beta': 0.015, 'rel_sus_latentslow': 0.15, 'tb_mortality': 3e-4},
        {'beta': 0.020, 'rel_sus_latentslow': 0.20, 'tb_mortality': 4e-4},
    ]
    
    # Create plotter
    plotter = CalibrationPlotter()
    sa_data = create_south_africa_data()
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Process each parameter set
    for i, params in enumerate(parameter_sets):
        print(f"Processing parameter set {i+1}/{len(parameter_sets)}: {params}")
        
        # Run simulation
        sim = run_calibration_simulation(
            beta=params['beta'],
            rel_sus_latentslow=params['rel_sus_latentslow'],
            tb_mortality=params['tb_mortality'],
            n_agents=200,
            years=60
        )
        
        # Calculate score
        score = calculate_calibration_score(sim, sa_data)
        
        # Create plot
        fig = plotter.plot_calibration_comparison(
            sim, 
            sa_data, 
            f"{timestamp}_set_{i+1}",
            save_path=f"example_batch_set_{i+1}_{timestamp}.pdf"
        )
        
        print(f"  Score: {score['composite_score']:.2f}")
    
    print(f"✓ Batch processing completed")


def main():
    """
    Run all examples
    """
    print("TB Model CalibrationPlotter Examples")
    print("=" * 50)
    
    # Example 1: Basic usage
    sim1, sa_data, timestamp1 = example_basic_usage()
    
    # Example 2: Parameter sweep plots
    results_df, timestamp2 = example_parameter_sweep_plots()
    
    # Example 3: Custom plotter
    sim3, timestamp3 = example_custom_plotter()
    
    # Example 4: Batch processing
    example_batch_processing()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the generated PDF files to see the plots.")
    print("\nGenerated files:")
    print(f"- example_calibration_comparison_{timestamp1}.pdf")
    print(f"- example_sweep_results_{timestamp2}.pdf")
    print(f"- example_violin_plots_{timestamp2}.pdf")
    print(f"- example_custom_plot_{timestamp3}.pdf")
    print(f"- example_batch_set_1_{timestamp1}.pdf")
    print(f"- example_batch_set_2_{timestamp1}.pdf")
    print(f"- example_batch_set_3_{timestamp1}.pdf")


if __name__ == '__main__':
    main() 