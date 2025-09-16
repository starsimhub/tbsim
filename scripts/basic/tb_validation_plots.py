#!/usr/bin/env python3
"""
TB Simulation Validation Plots

This module provides comprehensive validation and analysis plotting functions
for TB simulation results. It includes dwell time analysis, prevalence monitoring,
and model accuracy visualization.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
from tbsim.analyzers import DwtAnalyzer
from tbsim.tb import TBS

def create_validation_plots(sim):
    """Create comprehensive validation and accuracy plots for the TB model"""
    
    # Get the dwell time analyzer
    dwt_analyzer = None
    for analyzer in sim.analyzers:
        if isinstance(analyzer, DwtAnalyzer):
            dwt_analyzer = analyzer
            break
    
    if dwt_analyzer is None:
        print("Warning: DwtAnalyzer not found. Skipping validation plots.")
        return
    
    print("Creating comprehensive model validation and accuracy plots...")
    
    # 1. Dwell Time Validation Plot
    print("1. Creating dwell time validation plot...")
    dwt_analyzer.plot_dwell_time_validation()
    
    # 2. Sankey Diagram for State Transitions
    print("2. Creating Sankey diagram for state transitions...")
    dwt_analyzer.sankey_agents()
    
    # 3. Enhanced State Transition Network Graph
    print("3. Creating enhanced state transition network graph...")
    dwt_analyzer.graph_state_transitions_enhanced(
        subtitle="TB Model State Transitions - Model Accuracy Analysis",
        colormap='viridis',
        figsize=(16, 8),
        node_size_scale=1200,
        edge_width_scale=10
    )
    
    # 4. Histogram with KDE for Dwell Time Distribution
    print("4. Creating dwell time distribution histogram with KDE...")
    dwt_analyzer.histogram_with_kde()
    
    # 5. Kaplan-Meier Survival Curves
    print("5. Creating Kaplan-Meier survival curves...")
    dwt_analyzer.kaplan_meier()
    
    # 6. Reinfection Analysis
    print("6. Creating reinfection analysis plot...")
    dwt_analyzer.reinfections()
    
    # 7. Interactive Bar Chart for Dwell Times
    print("7. Creating interactive bar chart for dwell times...")
    dwt_analyzer.interactive_bars()
    
    # 8. Network Graph with Curved Edges
    print("8. Creating network graph with curved edges...")
    dwt_analyzer.graph_state_transitions_curved(
        subtitle="TB Model Accuracy - State Transition Flows",
        colormap='viridis'
    )

def monitor_prevalence_stability(sim, target_prevalence=0.01, tolerance=0.002):
    """
    Monitor and report on prevalence stability throughout the simulation
    
    Args:
        sim: Simulation object
        target_prevalence: Target prevalence level
        tolerance: Acceptable deviation from target (±tolerance)
    """
    results = sim.results
    tb_results = results['tb']
    
    # Calculate prevalence over time
    total_pop = results['n_alive']
    active_tb = tb_results['n_active']
    prevalence = active_tb / total_pop
    
    # Calculate stability metrics
    mean_prevalence = np.mean(prevalence)
    std_prevalence = np.std(prevalence)
    cv_prevalence = (std_prevalence / mean_prevalence) * 100 if mean_prevalence > 0 else 0
    
    # Check if within target range
    min_target = target_prevalence - tolerance
    max_target = target_prevalence + tolerance
    within_target = np.sum((prevalence >= min_target) & (prevalence <= max_target))
    target_percentage = (within_target / len(prevalence)) * 100
    
    print(f"\n{'='*60}")
    print("PREVALENCE STABILITY ANALYSIS")
    print(f"{'='*60}")
    print(f"Target Prevalence: {target_prevalence:.1%} (±{tolerance:.1%})")
    print(f"Mean Prevalence: {mean_prevalence:.3%}")
    print(f"Std Deviation: {std_prevalence:.3%}")
    print(f"Coefficient of Variation: {cv_prevalence:.1f}%")
    print(f"Time within target: {target_percentage:.1f}%")
    
    # Stability assessment
    if cv_prevalence < 5:
        print("✓ EXCELLENT: Very stable prevalence (CV < 5%)")
    elif cv_prevalence < 10:
        print("✓ GOOD: Stable prevalence (CV < 10%)")
    elif cv_prevalence < 20:
        print("⚠ MODERATE: Somewhat variable prevalence (CV < 20%)")
    else:
        print("⚠ POOR: Highly variable prevalence (CV ≥ 20%)")
    
    if target_percentage > 80:
        print("✓ EXCELLENT: Prevalence within target range >80% of time")
    elif target_percentage > 60:
        print("✓ GOOD: Prevalence within target range >60% of time")
    else:
        print("⚠ NEEDS IMPROVEMENT: Prevalence often outside target range")
    
    print(f"{'='*60}")
    
    return {
        'mean_prevalence': mean_prevalence,
        'std_prevalence': std_prevalence,
        'cv_prevalence': cv_prevalence,
        'target_percentage': target_percentage,
        'prevalence_series': prevalence
    }

def calculate_model_accuracy_metrics(sim):
    """Calculate key accuracy metrics for the TB model"""
    
    print("\n" + "="*60)
    print("MODEL ACCURACY METRICS")
    print("="*60)
    
    # Get basic simulation results
    results = sim.results
    
    # Calculate key epidemiological metrics
    final_time_idx = -1
    
    # Population dynamics
    total_pop = results['n_alive'][final_time_idx]
    total_deaths = results['cum_deaths'][final_time_idx]
    
    # TB-specific metrics
    tb_results = results['tb']
    active_tb = tb_results['n_active'][final_time_idx]
    latent_tb = tb_results['n_latent_slow'][final_time_idx] + tb_results['n_latent_fast'][final_time_idx]
    recovered = tb_results['n_ever_infected'][final_time_idx] - tb_results['n_active'][final_time_idx] - latent_tb
    
    # Calculate rates
    active_tb_rate = (active_tb / total_pop) * 100 if total_pop > 0 else 0
    latent_tb_rate = (latent_tb / total_pop) * 100 if total_pop > 0 else 0
    recovered_rate = (recovered / total_pop) * 100 if total_pop > 0 else 0
    mortality_rate = (total_deaths / (total_pop + total_deaths)) * 100 if (total_pop + total_deaths) > 0 else 0
    
    # Time series stability (coefficient of variation)
    active_tb_cv = np.std(tb_results['n_active']) / np.mean(tb_results['n_active']) * 100
    latent_tb_total = tb_results['n_latent_slow'] + tb_results['n_latent_fast']
    latent_tb_cv = np.std(latent_tb_total) / np.mean(latent_tb_total) * 100
    
    print(f"Final Population: {total_pop:,.0f}")
    print(f"Total Deaths: {total_deaths:,.0f}")
    print(f"Mortality Rate: {mortality_rate:.2f}%")
    print()
    print("TB Prevalence Rates (Final):")
    print(f"  Active TB: {active_tb_rate:.2f}%")
    print(f"  Latent TB: {latent_tb_rate:.2f}%")
    print(f"  Recovered: {recovered_rate:.2f}%")
    print()
    print("Model Stability (Coefficient of Variation):")
    print(f"  Active TB CV: {active_tb_cv:.2f}%")
    print(f"  Latent TB CV: {latent_tb_cv:.2f}%")
    
    # Model validation indicators
    print()
    print("Model Validation Indicators:")
    
    # Check if model reached equilibrium
    if active_tb_cv < 10:
        print("  ✓ Model reached stable equilibrium (low CV)")
    else:
        print("  ⚠ Model shows high variability (high CV)")
    
    # Check realistic prevalence ranges
    if 0.1 <= active_tb_rate <= 5.0:
        print("  ✓ Active TB prevalence within realistic range (0.1-5%)")
    else:
        print("  ⚠ Active TB prevalence outside typical range")
    
    if 10 <= latent_tb_rate <= 50:
        print("  ✓ Latent TB prevalence within realistic range (10-50%)")
    else:
        print("  ⚠ Latent TB prevalence outside typical range")
    
    print("="*60)

def create_accuracy_dashboard(sim):
    """Create a comprehensive accuracy dashboard with multiple subplots"""
    
    print("Creating comprehensive accuracy dashboard...")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle('TB Model Accuracy Dashboard', fontsize=14, fontweight='bold')
    
    results = sim.results
    timevec = results['timevec']
    
    # 1. Population Dynamics
    ax1 = axes[0, 0]
    ax1.plot(timevec, results['n_alive'], color='#440154', label='Alive', linewidth=2)  # Dark purple
    ax1.plot(timevec, results['cum_deaths'], color='#fde725', label='Dead', linewidth=2)  # Yellow
    ax1.set_title('Population Dynamics', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. TB Prevalence Over Time
    ax2 = axes[0, 1]
    tb_results = results['tb']
    ax2.plot(timevec, tb_results['n_active'], color='#fde725', label='Active TB', linewidth=2)  # Yellow
    latent_total = tb_results['n_latent_slow'] + tb_results['n_latent_fast']
    ax2.plot(timevec, latent_total, color='#35b779', label='Latent TB', linewidth=2)  # Green
    recovered_total = tb_results['n_ever_infected'] - tb_results['n_active'] - latent_total
    ax2.plot(timevec, recovered_total, color='#31688e', label='Recovered', linewidth=2)  # Blue
    ax2.set_title('TB Prevalence Over Time', fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. TB Incidence and Mortality
    ax3 = axes[0, 2]
    ax3.plot(timevec, tb_results['new_active'], color='#31688e', label='New Active Cases', linewidth=2)  # Blue
    ax3.plot(timevec, tb_results['new_deaths'], color='#35b779', label='New Deaths', linewidth=2)  # Green
    ax3.set_title('TB Incidence and Mortality', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Prevalence Rates (Percentage)
    ax4 = axes[1, 0]
    total_pop = results['n_alive']
    active_rate = (tb_results['n_active'] / total_pop) * 100
    latent_rate = (latent_total / total_pop) * 100
    recovered_rate = (recovered_total / total_pop) * 100
    
    ax4.plot(timevec, active_rate, color='#fde725', label='Active TB %', linewidth=2)  # Yellow
    ax4.plot(timevec, latent_rate, color='#35b779', label='Latent TB %', linewidth=2)  # Green
    ax4.plot(timevec, recovered_rate, color='#31688e', label='Recovered %', linewidth=2)  # Blue
    ax4.set_title('TB Prevalence Rates (%)', fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Percentage')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Model Stability Metrics
    ax5 = axes[1, 1]
    # Calculate rolling coefficient of variation
    window = min(50, len(active_rate) // 4)
    if window > 1:
        active_cv = np.array([np.std(active_rate[max(0, i-window):i+1]) / np.mean(active_rate[max(0, i-window):i+1]) * 100 
                             if np.mean(active_rate[max(0, i-window):i+1]) > 0 else 0
                             for i in range(len(active_rate))])
        latent_cv = np.array([np.std(latent_rate[max(0, i-window):i+1]) / np.mean(latent_rate[max(0, i-window):i+1]) * 100 
                             if np.mean(latent_rate[max(0, i-window):i+1]) > 0 else 0
                             for i in range(len(latent_rate))])
        
        ax5.plot(timevec, active_cv, color='#fde725', label='Active TB CV', linewidth=2)  # Yellow
        ax5.plot(timevec, latent_cv, color='#35b779', label='Latent TB CV', linewidth=2)  # Green
        ax5.axhline(y=10, color='#440154', linestyle='--', alpha=0.7, label='Stability Threshold')  # Dark purple
        ax5.set_title('Model Stability (Coefficient of Variation)', fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('CV (%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # 6. Final State Summary
    ax6 = axes[1, 2]
    final_idx = -1
    categories = ['Active TB', 'Latent TB', 'Recovered', 'Susceptible']
    values = [
        tb_results['n_active'][final_idx],
        latent_total[final_idx],
        recovered_total[final_idx],
        total_pop[final_idx] - (tb_results['n_active'][final_idx] + 
                               latent_total[final_idx] + 
                               recovered_total[final_idx])
    ]
    colors = ['#fde725', '#35b779', '#31688e', '#440154']  # Viridis colors: yellow, green, blue, dark purple
    
    bars = ax6.bar(categories, values, color=colors, alpha=0.7)
    ax6.set_title('Final State Distribution', fontweight='bold')
    ax6.set_ylabel('Count')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def create_prevalence_analysis_plot(sim, target_prevalence=0.01):
    """Create detailed prevalence analysis plot"""
    
    results = sim.results
    tb_results = results['tb']
    
    # Calculate prevalence over time
    total_pop = results['n_alive']
    active_tb = tb_results['n_active']
    prevalence = active_tb / total_pop
    timevec = results['timevec']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle('TB Prevalence Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Prevalence over time with target range
    ax1.plot(timevec, prevalence * 100, color='#31688e', linewidth=2, label='Active TB Prevalence')  # Blue
    ax1.axhline(y=target_prevalence * 100, color='#fde725', linestyle='--', 
                label=f'Target ({target_prevalence:.1%})', linewidth=2)  # Yellow
    ax1.axhline(y=(target_prevalence + 0.002) * 100, color='#35b779', linestyle=':', alpha=0.7)  # Green
    ax1.axhline(y=(target_prevalence - 0.002) * 100, color='#35b779', linestyle=':', alpha=0.7)  # Green
    ax1.fill_between(timevec, (target_prevalence - 0.002) * 100, (target_prevalence + 0.002) * 100, 
                     alpha=0.1, color='#35b779', label='Target Range')  # Green
    ax1.set_title('Active TB Prevalence Over Time', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prevalence (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prevalence distribution histogram
    ax2.hist(prevalence * 100, bins=30, alpha=0.7, color='#31688e', edgecolor='#440154')  # Blue with dark purple edge
    ax2.axvline(x=target_prevalence * 100, color='#fde725', linestyle='--', linewidth=2, label='Target')  # Yellow
    ax2.set_title('Prevalence Distribution', fontweight='bold')
    ax2.set_xlabel('Prevalence (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def run_comprehensive_validation(sim, target_prevalence=0.01):
    """
    Run comprehensive validation analysis on a TB simulation
    
    Args:
        sim: Simulation object
        target_prevalence: Target prevalence for analysis
    """
    print("Running comprehensive TB simulation validation...")
    print("="*60)
    
    # 1. Prevalence stability analysis
    prevalence_stats = monitor_prevalence_stability(sim, target_prevalence)
    
    # 2. Model accuracy metrics
    calculate_model_accuracy_metrics(sim)
    
    # 3. Create accuracy dashboard
    create_accuracy_dashboard(sim)
    
    # 4. Create prevalence analysis plot
    create_prevalence_analysis_plot(sim, target_prevalence)
    
    # 5. Create validation plots (if DwtAnalyzer is available)
    create_validation_plots(sim)
    
    return prevalence_stats

if __name__ == '__main__':
    # Example usage
    print("TB Validation Plots Module")
    print("This module provides validation and plotting functions for TB simulations.")
    print("Import and use the functions in your simulation scripts.")
    print("\nExample usage:")
    print("from tb_validation_plots import run_comprehensive_validation")
    print("stats = run_comprehensive_validation(sim, target_prevalence=0.01)")
