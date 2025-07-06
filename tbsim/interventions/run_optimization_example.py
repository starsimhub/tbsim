"""
Example: TB Model Parameter Optimization for South Africa Calibration

This script demonstrates how to run a focused parameter optimization
to find better calibration parameters that match South Africa data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
from tb_calibration_south_africa import (
    create_south_africa_data,
    run_calibration_simulation,
    compute_case_notifications,
    compute_age_stratified_prevalence,
    calculate_calibration_score
)


def run_focused_optimization():
    """
    Run a focused parameter optimization with higher transmission rates
    """
    
    print("=== TB Model Parameter Optimization for South Africa ===")
    print("Running focused optimization with higher transmission rates...")
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    
    # Define parameter ranges for optimization
    # Start with more conservative ranges to ensure simulations complete
    beta_range = np.array([0.015, 0.020, 0.025, 0.030])
    rel_sus_range = np.array([0.10, 0.15, 0.20, 0.25])
    tb_mortality_range = np.array([2e-4, 3e-4, 4e-4])
    
    print(f"Parameter ranges:")
    print(f"  Beta: {beta_range}")
    print(f"  Rel Sus: {rel_sus_range}")
    print(f"  TB Mortality: {tb_mortality_range}")
    
    # Store results
    results = []
    best_score = float('inf')
    best_params = None
    best_sim = None
    
    total_combinations = len(beta_range) * len(rel_sus_range) * len(tb_mortality_range)
    print(f"\nTotal combinations to test: {total_combinations}")
    
    start_time = time.time()
    
    for i, beta in enumerate(beta_range):
        for j, rel_sus in enumerate(rel_sus_range):
            for k, tb_mortality in enumerate(tb_mortality_range):
                
                combination_num = i * len(rel_sus_range) * len(tb_mortality_range) + \
                                j * len(tb_mortality_range) + k + 1
                
                print(f"\nTest {combination_num}/{total_combinations}: "
                      f"β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}")
                
                try:
                    print(f"    Running simulation...")
                    # Run simulation with smaller population for speed
                    sim = run_calibration_simulation(
                        beta=beta,
                        rel_sus_latentslow=rel_sus,
                        tb_mortality=tb_mortality,
                        n_agents=400,  # Smaller for faster optimization
                        years=120      # Shorter for faster optimization
                    )
                    
                    print(f"    Calculating calibration score...")
                    # Calculate calibration score
                    score_metrics = calculate_calibration_score(sim, sa_data)
                    
                    # Store results
                    result = {
                        'beta': beta,
                        'rel_sus_latentslow': rel_sus,
                        'tb_mortality': tb_mortality,
                        'combination': combination_num,
                        'composite_score': score_metrics['composite_score'],
                        'notification_mape': score_metrics['notification_mape'],
                        'age_prev_mape': score_metrics['age_prev_mape'],
                        'overall_prev_error': score_metrics['overall_prev_error'],
                        'model_overall_prev': score_metrics['model_overall_prev']
                    }
                    results.append(result)
                    
                    # Update best if better
                    if score_metrics['composite_score'] < best_score:
                        best_score = score_metrics['composite_score']
                        best_params = (beta, rel_sus, tb_mortality)
                        best_sim = sim
                        print(f"  ✓ NEW BEST! Score: {best_score:.2f}")
                        print(f"     Overall prevalence: {score_metrics['model_overall_prev']:.3f}%")
                        print(f"     Notification MAPE: {score_metrics['notification_mape']:.1f}%")
                        print(f"     Age prevalence MAPE: {score_metrics['age_prev_mape']:.1f}%")
                    else:
                        print(f"  Score: {score_metrics['composite_score']:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Failed: {e}")
                    import traceback
                    print(f"    Full error: {traceback.format_exc()}")
                    continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Create results DataFrame
    if len(results) == 0:
        print("⚠️  No successful simulations completed!")
        print("This could be due to:")
        print("  - Parameter values causing simulation failures")
        print("  - Memory or computational issues")
        print("  - Import or dependency problems")
        return None, None, None
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('composite_score')
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Save results
    results_df.to_csv(f"optimization_results_{timestamp}.csv", index=False)
    
    # Print summary
    print(f"\n=== OPTIMIZATION COMPLETED ===")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Tests completed: {len(results)}/{total_combinations}")
    
    if best_params:
        print(f"\nBEST PARAMETERS FOUND:")
        print(f"  Beta: {best_params[0]:.3f}")
        print(f"  Relative Susceptibility: {best_params[1]:.2f}")
        print(f"  TB Mortality: {best_params[2]:.1e}")
        print(f"  Composite Score: {best_score:.2f}")
        
        # Show detailed results for best parameters
        best_result = results_df.iloc[0]
        print(f"\nBEST RESULTS:")
        print(f"  Overall Prevalence: {best_result['model_overall_prev']:.3f}%")
        print(f"  Notification MAPE: {best_result['notification_mape']:.1f}%")
        print(f"  Age Prevalence MAPE: {best_result['age_prev_mape']:.1f}%")
        print(f"  Overall Prevalence Error: {best_result['overall_prev_error']:.3f} percentage points")
    
    # Show top 5 results
    print(f"\nTOP 5 RESULTS:")
    for i, result in results_df.head(5).iterrows():
        print(f"{i+1}. β={result['beta']:.3f}, rel_sus={result['rel_sus_latentslow']:.2f}, "
              f"mort={result['tb_mortality']:.1e}, score={result['composite_score']:.2f}")
    
    # Create optimization plots
    create_optimization_plots(results_df, timestamp)
    
    return results_df, best_sim, best_params


def create_optimization_plots(results_df, timestamp):
    """
    Create plots showing optimization results
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Score distribution
    ax1.hist(results_df['composite_score'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Composite Calibration Score')
    ax1.set_ylabel('Number of Parameter Combinations')
    ax1.set_title('Distribution of Calibration Scores')
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta vs Score
    scatter1 = ax2.scatter(results_df['beta'], results_df['composite_score'], 
                          c=results_df['rel_sus_latentslow'], cmap='viridis', alpha=0.7, s=50)
    ax2.set_xlabel('Beta (Transmission Rate)')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Beta vs Calibration Score (colored by rel_sus)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax2, label='Relative Susceptibility')
    
    # 3. Rel Sus vs Score
    scatter2 = ax3.scatter(results_df['rel_sus_latentslow'], results_df['composite_score'], 
                          c=results_df['beta'], cmap='plasma', alpha=0.7, s=50)
    ax3.set_xlabel('Relative Susceptibility (Latent)')
    ax3.set_ylabel('Composite Score')
    ax3.set_title('Relative Susceptibility vs Calibration Score (colored by beta)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax3, label='Beta')
    
    # 4. Overall prevalence vs Score
    ax4.scatter(results_df['model_overall_prev'], results_df['composite_score'], alpha=0.7, s=50)
    ax4.axvline(0.852, color='red', linestyle='--', label='Target: 0.852%')
    ax4.set_xlabel('Model Overall Prevalence (%)')
    ax4.set_ylabel('Composite Score')
    ax4.set_title('Overall Prevalence vs Calibration Score')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('TB Model Parameter Optimization Results', fontsize=16, y=1.02)
    
    filename = f"optimization_plots_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def compare_with_initial_results():
    """
    Compare optimization results with initial demo results
    """
    
    print("\n=== COMPARISON WITH INITIAL RESULTS ===")
    
    # Initial demo results (from the first run)
    initial_results = {
        'beta': 0.020,
        'rel_sus_latentslow': 0.15,
        'tb_mortality': 3e-4,
        'composite_score': 78.1,  # Average MAPE
        'notification_mape': 86.4,
        'age_prev_mape': 69.7,
        'overall_prev_error': 0.356,
        'model_overall_prev': 0.496
    }
    
    print("Initial Demo Results:")
    print(f"  Parameters: β={initial_results['beta']:.3f}, rel_sus={initial_results['rel_sus_latentslow']:.2f}, mort={initial_results['tb_mortality']:.1e}")
    print(f"  Composite Score: {initial_results['composite_score']:.1f}")
    print(f"  Overall Prevalence: {initial_results['model_overall_prev']:.3f}%")
    print(f"  Notification MAPE: {initial_results['notification_mape']:.1f}%")
    print(f"  Age Prevalence MAPE: {initial_results['age_prev_mape']:.1f}%")
    
    print("\nOptimization Results (if available):")
    print("Run the optimization to see improvements!")
    
    return initial_results


def main():
    """
    Main function to run the optimization example
    """
    
    print("TB Model Parameter Optimization Example")
    print("This example shows how to find better calibration parameters")
    print("that match South Africa TB data more closely.\n")
    
    # Run the optimization
    results_df, best_sim, best_params = run_focused_optimization()
    
    # Check if optimization was successful
    if results_df is None:
        print("\n❌ Optimization failed - no successful simulations completed.")
        print("Please check the error messages above and try again.")
        return
    
    # Compare with initial results
    initial_results = compare_with_initial_results()
    
    # If we have best simulation, show detailed comparison
    if best_sim is not None:
        print(f"\n=== DETAILED COMPARISON WITH BEST PARAMETERS ===")
        
        # Get South Africa data
        sa_data = create_south_africa_data()
        
        # Compute outputs for best simulation
        notifications = compute_case_notifications(best_sim)
        age_prevalence = compute_age_stratified_prevalence(best_sim)
        
        print(f"\nBest Parameters: β={best_params[0]:.3f}, rel_sus={best_params[1]:.2f}, mort={best_params[2]:.1e}")
        
        print(f"\nCase Notifications (per 100,000):")
        for year, data in notifications.items():
            data_rate = sa_data['case_notifications'][sa_data['case_notifications']['year'] == year]['rate_per_100k'].iloc[0]
            pct_diff = ((data['rate_per_100k'] - data_rate) / data_rate) * 100
            print(f"  {year}: Model={data['rate_per_100k']:.0f}, Data={data_rate:.0f}, Diff={pct_diff:.1f}%")
        
        print(f"\nAge Prevalence (per 100,000):")
        for age_group, data in age_prevalence.items():
            data_rate = sa_data['age_prevalence'][sa_data['age_prevalence']['age_group'] == age_group]['prevalence_per_100k'].iloc[0]
            pct_diff = ((data['prevalence_per_100k'] - data_rate) / data_rate) * 100
            print(f"  {age_group}: Model={data['prevalence_per_100k']:.0f}, Data={data_rate:.0f}, Diff={pct_diff:.1f}%")
    
    print(f"\nOptimization example completed!")
    print(f"Check the generated files for detailed results.")


if __name__ == '__main__':
    main() 