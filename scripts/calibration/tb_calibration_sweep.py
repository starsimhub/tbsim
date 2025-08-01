"""
TB Model Calibration Parameter Sweep for South Africa

This script performs a systematic parameter sweep to find optimal calibration parameters
that best match South Africa TB data including case notifications and age-stratified prevalence.
"""

import numpy as np
import pandas as pd
import datetime
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tb_calibration_south_africa import (
    create_south_africa_data, 
    run_calibration_simulation,
    compute_case_notifications,
    compute_age_stratified_prevalence,
    create_calibration_report
)


def calculate_calibration_score(sim, sa_data):
    """
    Calculate a composite calibration score based on multiple metrics
    
    Args:
        sim: Simulation object
        sa_data: South Africa data dictionary
    
    Returns:
        dict: Calibration metrics and composite score
    """
    
    # Compute model outputs
    notifications = compute_case_notifications(sim)
    age_prevalence = compute_age_stratified_prevalence(sim)
    
    # Case notification fit
    years = list(notifications.keys())
    model_rates = np.array([notifications[year]['rate_per_100k'] for year in years])
    data_rates = sa_data['case_notifications']['rate_per_100k'].values
    
    notification_rmse = np.sqrt(np.mean((model_rates - data_rates)**2))
    notification_mape = np.mean(np.abs((model_rates - data_rates) / data_rates)) * 100
    
    # Age prevalence fit
    age_groups = list(age_prevalence.keys())
    model_age_prev = np.array([age_prevalence[group]['prevalence_per_100k'] for group in age_groups])
    data_age_prev = sa_data['age_prevalence']['prevalence_per_100k'].values
    
    age_prev_rmse = np.sqrt(np.mean((model_age_prev - data_age_prev)**2))
    age_prev_mape = np.mean(np.abs((model_age_prev - data_age_prev) / data_age_prev)) * 100
    
    # Overall prevalence fit
    time_years = np.array([d.year for d in sim.results['timevec']])
    active_prev = sim.results['tb']['prevalence_active']
    target_idx = np.argmin(np.abs(time_years - 2018))
    model_overall_prev = active_prev[target_idx] * 100
    target_overall_prev = sa_data['targets']['overall_prevalence_2018']
    
    overall_prev_error = abs(model_overall_prev - target_overall_prev)
    
    # Composite score (lower is better)
    # Weight different components based on importance
    composite_score = (
        0.4 * notification_mape +  # Case notifications (40% weight)
        0.4 * age_prev_mape +      # Age prevalence (40% weight)
        0.2 * (overall_prev_error * 100)  # Overall prevalence (20% weight)
    )
    
    return {
        'notification_rmse': notification_rmse,
        'notification_mape': notification_mape,
        'age_prev_rmse': age_prev_rmse,
        'age_prev_mape': age_prev_mape,
        'overall_prev_error': overall_prev_error,
        'model_overall_prev': model_overall_prev,
        'target_overall_prev': target_overall_prev,
        'composite_score': composite_score
    }


def run_calibration_sweep(beta_range, rel_sus_range, tb_mortality_range, 
                         n_agents=500, years=150, max_simulations=50):
    """
    Run a systematic parameter sweep for calibration
    
    Args:
        beta_range: Array of beta values to test
        rel_sus_range: Array of relative susceptibility values to test
        tb_mortality_range: Array of TB mortality values to test
        n_agents: Number of agents per simulation
        years: Simulation duration
        max_simulations: Maximum number of simulations to run
    
    Returns:
        dict: Sweep results with best parameters
    """
    
    print("Starting TB Model Calibration Parameter Sweep...")
    print(f"Parameter ranges:")
    print(f"  Beta: {beta_range}")
    print(f"  Rel Sus: {rel_sus_range}")
    print(f"  TB Mortality: {tb_mortality_range}")
    print(f"  Max simulations: {max_simulations}")
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    
    # Initialize results storage
    results = []
    best_score = float('inf')
    best_params = None
    best_sim = None
    
    # Calculate total combinations
    total_combinations = len(beta_range) * len(rel_sus_range) * len(tb_mortality_range)
    simulations_run = 0
    
    print(f"\nTotal parameter combinations: {total_combinations}")
    print("Running simulations...")
    
    start_time = time.time()
    
    for beta in beta_range:
        for rel_sus in rel_sus_range:
            for tb_mortality in tb_mortality_range:
                
                if simulations_run >= max_simulations:
                    print(f"\nReached maximum simulations ({max_simulations})")
                    break
                
                simulations_run += 1
                print(f"Simulation {simulations_run}/{min(total_combinations, max_simulations)}: "
                      f"β={beta:.4f}, rel_sus={rel_sus:.3f}, mort={tb_mortality:.1e}")
                
                try:
                    # Run simulation
                    sim = run_calibration_simulation(
                        beta=beta,
                        rel_sus_latentslow=rel_sus,
                        tb_mortality=tb_mortality,
                        n_agents=n_agents,
                        years=years
                    )
                    
                    # Calculate calibration score
                    score_metrics = calculate_calibration_score(sim, sa_data)
                    
                    # Store results
                    result = {
                        'beta': beta,
                        'rel_sus_latentslow': rel_sus,
                        'tb_mortality': tb_mortality,
                        'simulation_number': simulations_run,
                        'score_metrics': score_metrics,
                        'composite_score': score_metrics['composite_score']
                    }
                    results.append(result)
                    
                    # Update best if better
                    if score_metrics['composite_score'] < best_score:
                        best_score = score_metrics['composite_score']
                        best_params = (beta, rel_sus, tb_mortality)
                        best_sim = sim
                        print(f"  ✓ New best score: {best_score:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Simulation failed: {e}")
                    continue
            
            if simulations_run >= max_simulations:
                break
        
        if simulations_run >= max_simulations:
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Create results summary
    results_df = pd.DataFrame(results)
    
    # Sort by composite score
    results_df = results_df.sort_values('composite_score')
    
    # Create sweep summary
    sweep_summary = {
        'timestamp': datetime.datetime.now().strftime("%Y_%m_%d_%H%M"),
        'parameter_ranges': {
            'beta_range': beta_range.tolist(),
            'rel_sus_range': rel_sus_range.tolist(),
            'tb_mortality_range': tb_mortality_range.tolist()
        },
        'simulation_settings': {
            'n_agents': n_agents,
            'years': years,
            'max_simulations': max_simulations,
            'simulations_run': simulations_run,
            'elapsed_time_minutes': elapsed_time / 60
        },
        'best_parameters': {
            'beta': best_params[0] if best_params else None,
            'rel_sus_latentslow': best_params[1] if best_params else None,
            'tb_mortality': best_params[2] if best_params else None,
            'composite_score': best_score
        },
        'top_10_results': results_df.head(10).to_dict('records')
    }
    
    return sweep_summary, results_df, best_sim, sa_data


def plot_sweep_results(results_df, timestamp):
    """
    Create plots showing sweep results
    
    Args:
        results_df: DataFrame with sweep results
        timestamp: Timestamp for file naming
    """
    
    import matplotlib.pyplot as plt
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Score distribution
    ax1.hist(results_df['composite_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Composite Calibration Score')
    ax1.set_ylabel('Number of Simulations')
    ax1.set_title('Distribution of Calibration Scores')
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta vs Score
    ax2.scatter(results_df['beta'], results_df['composite_score'], alpha=0.6, s=30)
    ax2.set_xlabel('Beta (Transmission Rate)')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Beta vs Calibration Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rel Sus vs Score
    ax3.scatter(results_df['rel_sus_latentslow'], results_df['composite_score'], alpha=0.6, s=30)
    ax3.set_xlabel('Relative Susceptibility (Latent)')
    ax3.set_ylabel('Composite Score')
    ax3.set_title('Relative Susceptibility vs Calibration Score')
    ax3.grid(True, alpha=0.3)
    
    # 4. TB Mortality vs Score
    ax4.scatter(results_df['tb_mortality'], results_df['composite_score'], alpha=0.6, s=30)
    ax4.set_xlabel('TB Mortality Rate')
    ax4.set_ylabel('Composite Score')
    ax4.set_title('TB Mortality vs Calibration Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('TB Model Calibration Parameter Sweep Results', fontsize=16, y=1.02)
    
    filename = f"calibration_sweep_results_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """
    Main function to run the calibration sweep
    """
    
    print("=== TB Model Calibration Parameter Sweep ===")
    
    # Define parameter ranges for sweep
    # Start with broader ranges and can refine later
    beta_range = np.array([0.010, 0.015, 0.020, 0.025, 0.030])
    rel_sus_range = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
    tb_mortality_range = np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
    
    # Run sweep
    sweep_summary, results_df, best_sim, sa_data = run_calibration_sweep(
        beta_range=beta_range,
        rel_sus_range=rel_sus_range,
        tb_mortality_range=tb_mortality_range,
        n_agents=500,  # Smaller population for faster sweep
        years=150,     # Shorter simulation for faster sweep
        max_simulations=50  # Limit total simulations
    )
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Save results
    results_df.to_csv(f"calibration_sweep_results_{timestamp}.csv", index=False)
    
    # Save sweep summary
    with open(f"calibration_sweep_summary_{timestamp}.json", 'w') as f:
        json.dump(sweep_summary, f, indent=2, default=str)
    
    # Create plots
    plot_sweep_results(results_df, timestamp)
    
    # Create detailed report for best simulation
    if best_sim is not None:
        print(f"\nCreating detailed report for best parameters...")
        report = create_calibration_report(best_sim, sa_data, f"{timestamp}_best")
        
        # Print best results
        print(f"\n=== BEST CALIBRATION PARAMETERS ===")
        print(f"Beta: {sweep_summary['best_parameters']['beta']:.4f}")
        print(f"Relative Susceptibility: {sweep_summary['best_parameters']['rel_sus_latentslow']:.3f}")
        print(f"TB Mortality: {sweep_summary['best_parameters']['tb_mortality']:.1e}")
        print(f"Composite Score: {sweep_summary['best_parameters']['composite_score']:.2f}")
        print(f"================================")
    
    # Print top 5 results
    print(f"\n=== TOP 5 CALIBRATION RESULTS ===")
    for i, result in enumerate(sweep_summary['top_10_results'][:5]):
        print(f"{i+1}. β={result['beta']:.4f}, rel_sus={result['rel_sus_latentslow']:.3f}, "
              f"mort={result['tb_mortality']:.1e}, score={result['composite_score']:.2f}")
    
    print(f"\nCalibration sweep completed!")
    print(f"Files created with timestamp: {timestamp}")
    
    return sweep_summary, results_df, best_sim


if __name__ == '__main__':
    sweep_summary, results_df, best_sim = main() 