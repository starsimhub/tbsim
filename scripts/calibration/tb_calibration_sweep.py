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


def plot_violin_plots(results_df, timestamp):
    """
    Create focused violin plots showing score distributions for each parameter
    
    Args:
        results_df: DataFrame with sweep results
        timestamp: Timestamp for file naming
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Violin plot for Beta values
    # Group by actual beta values
    beta_groups = results_df.groupby('beta')['composite_score'].apply(list).reset_index()
    violin_data = [group for group in beta_groups['composite_score']]
    violin_labels = [f'{beta:.3f}' for beta in beta_groups['beta']]
    
    parts1 = ax1.violinplot(violin_data, positions=range(len(violin_data)))
    ax1.set_xticks(range(len(violin_labels)))
    ax1.set_xticklabels(violin_labels, rotation=45)
    ax1.set_xlabel('Beta (Transmission Rate)')
    ax1.set_ylabel('Composite Calibration Score')
    ax1.set_title('Score Distribution by Beta Value')
    ax1.grid(True, alpha=0.3)
    
    # Add mean points
    means = [np.mean(data) for data in violin_data]
    ax1.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
    ax1.legend()
    
    # 2. Violin plot for Relative Susceptibility values
    rel_sus_groups = results_df.groupby('rel_sus_latentslow')['composite_score'].apply(list).reset_index()
    violin_data = [group for group in rel_sus_groups['composite_score']]
    violin_labels = [f'{rel_sus:.2f}' for rel_sus in rel_sus_groups['rel_sus_latentslow']]
    
    parts2 = ax2.violinplot(violin_data, positions=range(len(violin_data)))
    ax2.set_xticks(range(len(violin_labels)))
    ax2.set_xticklabels(violin_labels, rotation=45)
    ax2.set_xlabel('Relative Susceptibility (Latent)')
    ax2.set_ylabel('Composite Calibration Score')
    ax2.set_title('Score Distribution by Relative Susceptibility Value')
    ax2.grid(True, alpha=0.3)
    
    # Add mean points
    means = [np.mean(data) for data in violin_data]
    ax2.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
    ax2.legend()
    
    # 3. Violin plot for TB Mortality values
    tb_mort_groups = results_df.groupby('tb_mortality')['composite_score'].apply(list).reset_index()
    violin_data = [group for group in tb_mort_groups['composite_score']]
    violin_labels = [f'{tb_mort:.1e}' for tb_mort in tb_mort_groups['tb_mortality']]
    
    parts3 = ax3.violinplot(violin_data, positions=range(len(violin_data)))
    ax3.set_xticks(range(len(violin_labels)))
    ax3.set_xticklabels(violin_labels, rotation=45)
    ax3.set_xlabel('TB Mortality Rate')
    ax3.set_ylabel('Composite Calibration Score')
    ax3.set_title('Score Distribution by TB Mortality Value')
    ax3.grid(True, alpha=0.3)
    
    # Add mean points
    means = [np.mean(data) for data in violin_data]
    ax3.plot(range(len(means)), means, 'ro-', markersize=8, label='Mean Score')
    ax3.legend()
    
    plt.tight_layout()
    plt.suptitle('TB Model Calibration: Violin Plots by Parameter Value', fontsize=16, y=1.02)
    
    filename = f"calibration_violin_plots_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_sweep_results(results_df, timestamp):
    """
    Create plots showing sweep results including violin plots
    
    Args:
        results_df: DataFrame with sweep results
        timestamp: Timestamp for file naming
    """
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create a larger figure with more subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Score distribution histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results_df['composite_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Composite Calibration Score')
    ax1.set_ylabel('Number of Simulations')
    ax1.set_title('Distribution of Calibration Scores')
    ax1.grid(True, alpha=0.3)
    
    # 2. Beta vs Score scatter
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(results_df['beta'], results_df['composite_score'], alpha=0.6, s=30)
    ax2.set_xlabel('Beta (Transmission Rate)')
    ax2.set_ylabel('Composite Score')
    ax2.set_title('Beta vs Calibration Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rel Sus vs Score scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(results_df['rel_sus_latentslow'], results_df['composite_score'], alpha=0.6, s=30)
    ax3.set_xlabel('Relative Susceptibility (Latent)')
    ax3.set_ylabel('Composite Score')
    ax3.set_title('Relative Susceptibility vs Calibration Score')
    ax3.grid(True, alpha=0.3)
    
    # 4. TB Mortality vs Score scatter
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.scatter(results_df['tb_mortality'], results_df['composite_score'], alpha=0.6, s=30)
    ax4.set_xlabel('TB Mortality Rate')
    ax4.set_ylabel('Composite Score')
    ax4.set_title('TB Mortality vs Calibration Score')
    ax4.grid(True, alpha=0.3)
    
    # 5. Violin plot for Beta
    ax5 = fig.add_subplot(gs[1, 0])
    # Create categorical bins for beta values
    beta_bins = pd.cut(results_df['beta'], bins=5, labels=False)
    violin_data = []
    violin_labels = []
    for i in range(5):
        mask = beta_bins == i
        if mask.any():
            violin_data.append(results_df.loc[mask, 'composite_score'].values)
            beta_range = results_df.loc[mask, 'beta']
            violin_labels.append(f'{beta_range.min():.3f}-{beta_range.max():.3f}')
    
    if violin_data:
        parts = ax5.violinplot(violin_data, positions=range(len(violin_data)))
        ax5.set_xticks(range(len(violin_labels)))
        ax5.set_xticklabels(violin_labels, rotation=45)
        ax5.set_xlabel('Beta Range')
        ax5.set_ylabel('Composite Score')
        ax5.set_title('Score Distribution by Beta Range')
        ax5.grid(True, alpha=0.3)
    
    # 6. Violin plot for Relative Susceptibility
    ax6 = fig.add_subplot(gs[1, 1])
    rel_sus_bins = pd.cut(results_df['rel_sus_latentslow'], bins=5, labels=False)
    violin_data = []
    violin_labels = []
    for i in range(5):
        mask = rel_sus_bins == i
        if mask.any():
            violin_data.append(results_df.loc[mask, 'composite_score'].values)
            rel_sus_range = results_df.loc[mask, 'rel_sus_latentslow']
            violin_labels.append(f'{rel_sus_range.min():.2f}-{rel_sus_range.max():.2f}')
    
    if violin_data:
        parts = ax6.violinplot(violin_data, positions=range(len(violin_data)))
        ax6.set_xticks(range(len(violin_labels)))
        ax6.set_xticklabels(violin_labels, rotation=45)
        ax6.set_xlabel('Rel Sus Range')
        ax6.set_ylabel('Composite Score')
        ax6.set_title('Score Distribution by Rel Sus Range')
        ax6.grid(True, alpha=0.3)
    
    # 7. Violin plot for TB Mortality
    ax7 = fig.add_subplot(gs[1, 2])
    tb_mort_bins = pd.cut(results_df['tb_mortality'], bins=5, labels=False)
    violin_data = []
    violin_labels = []
    for i in range(5):
        mask = tb_mort_bins == i
        if mask.any():
            violin_data.append(results_df.loc[mask, 'composite_score'].values)
            tb_mort_range = results_df.loc[mask, 'tb_mortality']
            violin_labels.append(f'{tb_mort_range.min():.1e}-{tb_mort_range.max():.1e}')
    
    if violin_data:
        parts = ax7.violinplot(violin_data, positions=range(len(violin_data)))
        ax7.set_xticks(range(len(violin_labels)))
        ax7.set_xticklabels(violin_labels, rotation=45)
        ax7.set_xlabel('TB Mortality Range')
        ax7.set_ylabel('Composite Score')
        ax7.set_title('Score Distribution by TB Mortality Range')
        ax7.grid(True, alpha=0.3)
    
    # 8. 3D scatter plot (Beta vs Rel Sus vs Score)
    ax8 = fig.add_subplot(gs[1, 3], projection='3d')
    scatter = ax8.scatter(results_df['beta'], results_df['rel_sus_latentslow'], 
                         results_df['composite_score'], 
                         c=results_df['composite_score'], cmap='viridis', alpha=0.6)
    ax8.set_xlabel('Beta')
    ax8.set_ylabel('Rel Sus')
    ax8.set_zlabel('Composite Score')
    ax8.set_title('3D: Beta vs Rel Sus vs Score')
    plt.colorbar(scatter, ax=ax8, label='Composite Score')
    
    # 9. Heatmap of parameter interactions
    ax9 = fig.add_subplot(gs[2, 0])
    # Create correlation matrix for parameters and score
    corr_data = results_df[['beta', 'rel_sus_latentslow', 'tb_mortality', 'composite_score']].corr()
    im = ax9.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1)
    ax9.set_xticks(range(len(corr_data.columns)))
    ax9.set_yticks(range(len(corr_data.columns)))
    ax9.set_xticklabels(corr_data.columns, rotation=45)
    ax9.set_yticklabels(corr_data.columns)
    ax9.set_title('Parameter Correlations')
    
    # Add correlation values to heatmap
    for i in range(len(corr_data.columns)):
        for j in range(len(corr_data.columns)):
            text = ax9.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    # 10. Box plots for each parameter
    ax10 = fig.add_subplot(gs[2, 1])
    # Create discrete categories for each parameter
    results_df['beta_cat'] = pd.cut(results_df['beta'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    results_df['rel_sus_cat'] = pd.cut(results_df['rel_sus_latentslow'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    results_df['tb_mort_cat'] = pd.cut(results_df['tb_mortality'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Box plot for beta categories
    box_data = [results_df[results_df['beta_cat'] == cat]['composite_score'].values 
                for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                if len(results_df[results_df['beta_cat'] == cat]) > 0]
    box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                  if len(results_df[results_df['beta_cat'] == cat]) > 0]
    
    if box_data:
        bp = ax10.boxplot(box_data, tick_labels=box_labels)
        ax10.set_xlabel('Beta Categories')
        ax10.set_ylabel('Composite Score')
        ax10.set_title('Score Distribution by Beta Categories')
        ax10.tick_params(axis='x', rotation=45)
        ax10.grid(True, alpha=0.3)
    
    # 11. Box plots for relative susceptibility
    ax11 = fig.add_subplot(gs[2, 2])
    box_data = [results_df[results_df['rel_sus_cat'] == cat]['composite_score'].values 
                for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                if len(results_df[results_df['rel_sus_cat'] == cat]) > 0]
    box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                  if len(results_df[results_df['rel_sus_cat'] == cat]) > 0]
    
    if box_data:
        bp = ax11.boxplot(box_data, tick_labels=box_labels)
        ax11.set_xlabel('Rel Sus Categories')
        ax11.set_ylabel('Composite Score')
        ax11.set_title('Score Distribution by Rel Sus Categories')
        ax11.tick_params(axis='x', rotation=45)
        ax11.grid(True, alpha=0.3)
    
    # 12. Box plots for TB mortality
    ax12 = fig.add_subplot(gs[2, 3])
    box_data = [results_df[results_df['tb_mort_cat'] == cat]['composite_score'].values 
                for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                if len(results_df[results_df['tb_mort_cat'] == cat]) > 0]
    box_labels = [cat for cat in ['Very Low', 'Low', 'Medium', 'High', 'Very High'] 
                  if len(results_df[results_df['tb_mort_cat'] == cat]) > 0]
    
    if box_data:
        bp = ax12.boxplot(box_data, tick_labels=box_labels)
        ax12.set_xlabel('TB Mortality Categories')
        ax12.set_ylabel('Composite Score')
        ax12.set_title('Score Distribution by TB Mortality Categories')
        ax12.tick_params(axis='x', rotation=45)
        ax12.grid(True, alpha=0.3)
    
    plt.suptitle('TB Model Calibration Parameter Sweep Results', fontsize=20, y=0.98)
    
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
    plot_violin_plots(results_df, timestamp)
    
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