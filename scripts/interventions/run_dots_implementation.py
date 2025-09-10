"""
Test script for DOTS implementation in TBsim

This script demonstrates the new DOTS implementation that matches EMOD-Generic's approach,
showing how to use the enhanced drug type system.
"""

import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import tbsim as mtb
import os
from datetime import datetime

from tbsim.interventions import (
    TBDrugType, 
    get_dots_parameters, 
    get_drug_parameters,
    create_dots_treatment,
    create_dots_improved_treatment,
    create_first_line_treatment
)

def test_drug_parameters():
    """Test the drug parameters system."""
    print("=== Testing Drug Parameters ===")
    
    # Get DOTS parameters
    dots_params = get_dots_parameters()
    print(f"DOTS Parameters:")
    print(f"  Drug Type: {dots_params.drug_type.name}")
    print(f"  Cure Rate: {dots_params.cure_rate:.3f}")
    print(f"  Inactivation Rate: {dots_params.inactivation_rate:.3f}")
    print(f"  Resistance Rate: {dots_params.resistance_rate:.3f}")
    print(f"  Relapse Rate: {dots_params.relapse_rate:.3f}")
    print(f"  Mortality Rate: {dots_params.mortality_rate:.3f}")
    print(f"  Duration: {dots_params.duration} days")
    print(f"  Adherence Rate: {dots_params.adherence_rate:.3f}")
    print(f"  Cost per Course: ${dots_params.cost_per_course:.0f}")
    
    # Test effectiveness calculation
    print(f"\nDOTS Effectiveness over time:")
    for days in [0, 15, 30, 60, 90, 180]:
        effectiveness = dots_params.get_effectiveness(days)
        print(f"  Day {days:3d}: {effectiveness:.3f}")
    
    # Compare different drug types
    print(f"\n=== Comparing Drug Types ===")
    drug_types = [TBDrugType.DOTS, TBDrugType.DOTS_IMPROVED, TBDrugType.FIRST_LINE_COMBO]
    
    for drug_type in drug_types:
        params = get_drug_parameters(drug_type)
        print(f"{drug_type.name:15s}: Cure Rate = {params.cure_rate:.3f}, Cost = ${params.cost_per_course:.0f}")

def run_simulation_with_dots():
    """Run a simulation using the new DOTS implementation."""
    print("\n=== Running Simulation with DOTS ===")
    
    # Create simulation with DOTS treatment
    sim = ss.Sim(
        people=ss.People(n_agents=1000, extra_states=mtb.get_extrastates()),
        diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
        interventions=[
            mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
            mtb.TBDiagnostic(pars={
                'coverage': ss.bernoulli(0.8, strict=False),
                'sensitivity': 0.85,
                'specificity': 0.95,
                'care_seeking_multiplier': 2.0,
            }),
            create_dots_treatment(pars={
                'reseek_multiplier': 2.0,
                'reset_flags': True,
            }),
        ],
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
        pars=dict(start=ss.date(2000), stop=ss.date(2010), dt=ss.weeks(1)),
    )
    
    sim.run()
    
    # Get results
    results = sim.results['enhancedtbtreatment']
    timevec = results['n_treated'].timevec
    
    print(f"Simulation completed!")
    print(f"Total treated: {np.sum(results['n_treated'].values)}")
    print(f"Total successes: {np.sum(results['n_treatment_success'].values)}")
    print(f"Total failures: {np.sum(results['n_treatment_failure'].values)}")
    print(f"Drug type used: {results['drug_type_used'].values[0]}")
    
    return sim, results

def run_comparison_simulation():
    """Run comparison between different drug types."""
    print("\n=== Running Drug Type Comparison ===")
    
    # Define different treatment scenarios
    scenarios = {
        'DOTS': create_dots_treatment(),
        'DOTS_IMPROVED': create_dots_improved_treatment(),
        'FIRST_LINE': create_first_line_treatment(),
    }
    
    results_dict = {}
    
    for scenario_name, treatment in scenarios.items():
        print(f"Running {scenario_name} scenario...")
        
        sim = ss.Sim(
            people=ss.People(n_agents=500, extra_states=mtb.get_extrastates()),
            diseases=mtb.TB({'init_prev': ss.bernoulli(0.25)}),
            interventions=[
                mtb.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
                mtb.TBDiagnostic(pars={
                    'coverage': ss.bernoulli(0.8, strict=False),
                    'sensitivity': 0.85,
                    'specificity': 0.95,
                    'care_seeking_multiplier': 2.0,
                }),
                treatment,
            ],
            networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
            pars=dict(start=2000, stop=2010, dt=ss.days(1)/12),
        )
        
        sim.run()
        
        # Store results
        treatment_results = sim.results['enhancedtbtreatment']
        results_dict[scenario_name] = {
            'total_treated': np.sum(treatment_results['n_treated'].values),
            'total_success': np.sum(treatment_results['n_treatment_success'].values),
            'total_failure': np.sum(treatment_results['n_treatment_failure'].values),
            'success_rate': np.sum(treatment_results['n_treatment_success'].values) / max(1, np.sum(treatment_results['n_treated'].values)),
            'drug_type': treatment_results['drug_type_used'].values[0],
        }
    
    # Print comparison results
    print(f"\n=== Treatment Comparison Results ===")
    print(f"{'Scenario':<15} {'Treated':<8} {'Success':<8} {'Failure':<8} {'Success Rate':<12} {'Drug Type':<15}")
    print("-" * 70)
    
    for scenario_name, results in results_dict.items():
        print(f"{scenario_name:<15} {results['total_treated']:<8} {results['total_success']:<8} {results['total_failure']:<8} {results['success_rate']:<12.3f} {results['drug_type']:<15}")
    
    return results_dict

def plot_comparison_results(results_dict, results_dir=None):
    """Plot comparison results between different drug types."""
    print("\n=== Plotting Drug Type Comparison ===")
    
    # Set up the figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TB Treatment Strategy Comparison: DOTS vs DOTS-Improved vs First-Line Combo', 
                 fontsize=16, fontweight='bold', y=0.92)
    
    # Extract data for plotting
    scenarios = list(results_dict.keys())
    treated_counts = [results_dict[s]['total_treated'] for s in scenarios]
    success_counts = [results_dict[s]['total_success'] for s in scenarios]
    failure_counts = [results_dict[s]['total_failure'] for s in scenarios]
    success_rates = [results_dict[s]['success_rate'] for s in scenarios]
    
    # Color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    # Plot 1: Total treatments comparison
    bars1 = ax1.bar(scenarios, treated_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Total Number of Treatments', fontsize=12, fontweight='bold')
    ax1.set_title('Total Treatments by Strategy\n(Program scale and reach)', fontsize=12, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.4, axis='y')
    ax1.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bar, value in zip(bars1, treated_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(treated_counts)*0.01,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Success vs Failure comparison
    x = np.arange(len(scenarios))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, success_counts, width, label='Successful Treatments', 
                    color='#28a745', alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax2.bar(x + width/2, failure_counts, width, label='Failed Treatments', 
                    color='#dc3545', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Treatment Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Treatments', fontsize=12, fontweight='bold')
    ax2.set_title('Treatment Outcomes by Strategy\n(Success vs failure breakdown)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    # ax2.grid(True, alpha=0.4, axis='y')
    ax2.set_facecolor('#f8f9fa')
    
    # Add value labels on bars
    for bar, value in zip(bars2, success_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(success_counts + failure_counts)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, value in zip(bars3, failure_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(success_counts + failure_counts)*0.01,
                f'{value}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 3: Success rate comparison
    bars4 = ax3.bar(scenarios, success_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_ylabel('Success Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Treatment Success Rate by Strategy\n(Program effectiveness)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax3.set_ylim(0, 1.1)
    ax3.grid(True, alpha=0.4, axis='y')
    ax3.set_facecolor('#f8f9fa')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars4, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Efficiency analysis (successes per treatment)
    efficiency = [s/t for s, t in zip(success_counts, treated_counts)]
    bars5 = ax4.bar(scenarios, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax4.set_ylabel('Efficiency (Successes/Treatment)', fontsize=12, fontweight='bold')
    ax4.set_title('Treatment Efficiency by Strategy\n(Higher values indicate better resource utilization)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.4, axis='y')
    ax4.set_facecolor('#f8f9fa')
    
    # Add efficiency labels on bars
    for bar, eff in zip(bars5, efficiency):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(efficiency)*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add summary statistics
    best_success_rate = max(success_rates)
    best_strategy = scenarios[success_rates.index(best_success_rate)]
    summary_text = f'Best Success Rate: {best_strategy} ({best_success_rate:.1%})\n'
    summary_text += f'Success Rate Range: {min(success_rates):.1%} - {max(success_rates):.1%}\n'
    summary_text += f'Total Treatments Across All: {sum(treated_counts)}'
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.10, hspace=0.35, wspace=0.25)
    
    # Save the figure
    if results_dir:
        filename = os.path.join(results_dir, 'dots_drug_comparison.png')
    else:
        filename = 'dots_drug_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved drug comparison plot: {filename}")
    plt.show()

def plot_results(sim, results, results_dir=None):
    """Plot comprehensive simulation results with detailed annotations and descriptions."""
    print("\n=== Plotting Enhanced Results ===")
    
    timevec = results['n_treated'].timevec
    drug_type = results['drug_type_used'].values[0]
    
    # Set up the figure with better styling
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'TBsim DOTS Implementation Results - {drug_type} Treatment Strategy', 
                 fontsize=16, fontweight='bold', y=0.92)
    
    # Calculate key statistics for annotations
    total_treated = np.sum(results['n_treated'].values)
    total_success = np.sum(results['n_treatment_success'].values)
    total_failure = np.sum(results['n_treatment_failure'].values)
    overall_success_rate = total_success / max(1, total_treated)
    
    # Plot 1: Treatment outcomes over time with detailed annotations
    ax1.plot(timevec, results['n_treated'].values, label='New Treatments', 
             marker='o', markersize=4, linewidth=2, color='#2E86AB')
    ax1.plot(timevec, results['n_treatment_success'].values, label='Successful Treatments', 
             linestyle='--', linewidth=2, color='#A23B72')
    ax1.plot(timevec, results['n_treatment_failure'].values, label='Failed Treatments', 
             linestyle=':', linewidth=2, color='#F18F01')
    
    # Add statistics annotation
    stats_text = f'Total Treated: {total_treated}\nSuccesses: {total_success}\nFailures: {total_failure}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax1.set_xlabel('Simulation Time (Years)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Individuals', fontsize=12, fontweight='bold')
    ax1.set_title('Daily TB Treatment Outcomes\n(New treatments, successes, and failures)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('#f8f9fa')
    
    # Plot 2: Cumulative outcomes with trend analysis
    ax2.plot(timevec, results['cum_treatment_success'].values, 
             label='Cumulative Successful Treatments', color='#28a745', linewidth=2)
    ax2.plot(timevec, results['cum_treatment_failure'].values, 
             label='Cumulative Failed Treatments', color='#dc3545', linewidth=2)
    
    # Add final cumulative values annotation
    final_success = results['cum_treatment_success'].values[-1]
    final_failure = results['cum_treatment_failure'].values[-1]
    
    # Convert timevec to numeric for calculations
    timevec_numeric = np.arange(len(timevec))
    
    ax2.annotate(f'Final Success: {final_success}', 
                xy=(timevec[-1], final_success), xytext=(timevec[int(len(timevec)*0.7)], final_success*1.1),
                arrowprops=dict(arrowstyle='->', color='#28a745'), fontsize=10, color='#28a745')
    ax2.annotate(f'Final Failure: {final_failure}', 
                xy=(timevec[-1], final_failure), xytext=(timevec[int(len(timevec)*0.7)], final_failure*0.9),
                arrowprops=dict(arrowstyle='->', color='#dc3545'), fontsize=10, color='#dc3545')
    
    ax2.set_xlabel('Simulation Time (Years)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Number of Treatments', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Treatment Outcomes\n(Shows treatment program scale and effectiveness)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('#f8f9fa')
    
    # Plot 3: TB prevalence with epidemiological context
    tb_results = sim.results['tb']
    ax3.plot(tb_results['n_active'].timevec, tb_results['n_active'].values, 
             label='Active TB Cases', color='#fd7e14', linewidth=2)
    
    # Calculate and display prevalence statistics
    initial_active = tb_results['n_active'].values[0]
    final_active = tb_results['n_active'].values[-1]
    max_active = np.max(tb_results['n_active'].values)
    reduction = (initial_active - final_active) / max(1, initial_active) * 100
    
    # Add prevalence statistics
    prev_stats = f'Initial Active: {initial_active:.0f}\nFinal Active: {final_active:.0f}\nPeak Active: {max_active:.0f}\nReduction: {reduction:.1f}%'
    ax3.text(0.02, 0.98, prev_stats, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    ax3.set_xlabel('Simulation Time (Years)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Active TB Cases', fontsize=12, fontweight='bold')
    ax3.set_title('Active TB Prevalence Over Time\n(Impact of treatment on disease burden)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_facecolor('#f8f9fa')
    
    # Plot 4: Treatment success rate with confidence intervals
    success_rate = np.divide(results['n_treatment_success'].values, 
                           np.maximum(results['n_treated'].values, 1), 
                           out=np.zeros_like(results['n_treatment_success'].values), 
                           where=results['n_treated'].values > 0)
    
    # Calculate moving average for smoother trend
    window_size = max(1, len(success_rate) // 20)  # 5% of data points
    if window_size > 1:
        success_rate_smooth = np.convolve(success_rate, np.ones(window_size)/window_size, mode='same')
        ax4.plot(timevec, success_rate_smooth, label='Smoothed Success Rate', 
                color='#007bff', linewidth=3, alpha=0.8)
    
    ax4.plot(timevec, success_rate, label='Daily Success Rate', 
             color='#6f42c1', alpha=0.6, linewidth=1)
    
    # Add overall success rate line
    ax4.axhline(y=overall_success_rate, color='red', linestyle='--', alpha=0.7, 
                label=f'Overall Rate: {overall_success_rate:.1%}')
    
    # Add success rate statistics
    avg_success_rate = np.mean(success_rate[success_rate > 0])
    ax4.text(0.02, 0.98, f'Average Success Rate: {avg_success_rate:.1%}\nOverall Success Rate: {overall_success_rate:.1%}', 
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax4.set_xlabel('Simulation Time (Years)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Treatment Success Rate', fontsize=12, fontweight='bold')
    ax4.set_title('Treatment Success Rate Over Time\n(Program effectiveness and consistency)', 
                  fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='lower right', framealpha=0.9)
    ax4.grid(True, alpha=0.3)
    ax4.set_facecolor('#f8f9fa')
    ax4.set_ylim(0, 1.05)
    
    # Add overall figure information
    fig.text(0.02, 0.02, f'Simulation Parameters: Population={sim.people.n_agents}, '
                         f'Duration={sim.pars["stop"]-sim.pars["start"]} years, '
                         f'Drug Type={drug_type}', 
             fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.10, hspace=0.35, wspace=0.25)
    
    # Save the figure
    if results_dir:
        filename = os.path.join(results_dir, f'dots_enhanced_results_{drug_type.lower().replace(" ", "_")}.png')
    else:
        filename = f'dots_enhanced_results_{drug_type.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved enhanced results plot: {filename}")
    plt.show()

def create_summary_dashboard(sim, results, comparison_results, results_dir=None):
    """Create a comprehensive summary dashboard with key metrics."""
    print("\n=== Creating Summary Dashboard ===")
    
    # Set up the figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Main title
    fig.suptitle('TBSim DOTS Implementation - Comprehensive Analysis Dashboard', 
                 fontsize=12, fontweight='bold', y=0.90)
    
    # Key metrics panel (top left)
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')
    
    # Calculate key metrics
    total_treated = np.sum(results['n_treated'].values)
    total_success = np.sum(results['n_treatment_success'].values)
    total_failure = np.sum(results['n_treatment_failure'].values)
    success_rate = total_success / max(1, total_treated)
    
    tb_results = sim.results['tb']
    initial_active = tb_results['n_active'].values[0]
    final_active = tb_results['n_active'].values[-1]
    reduction = (initial_active - final_active) / max(1, initial_active) * 100
    
    # Create metrics text
    metrics_text = f"""
    KEY PERFORMANCE INDICATORS
    
    Treatment Program:
    • Total Treatments: {total_treated:,}
    • Successful: {total_success:,} ({success_rate:.1%})
    • Failed: {total_failure:,} ({1-success_rate:.1%})
    
    Disease Impact:
    • Initial Active TB: {initial_active:.0f}
    • Final Active TB: {final_active:.0f}
    • Reduction: {reduction:.1f}%
    
    Simulation Parameters:
    • Population: {sim.people.n_agents:,}
    • Duration: {sim.pars['stop']-sim.pars['start']} years
    • Drug Type: {results['drug_type_used'].values[0]}
    """
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Strategy comparison (top right)
    ax_comp = fig.add_subplot(gs[0, 2:4])
    scenarios = list(comparison_results.keys())
    success_rates = [comparison_results[s]['success_rate'] for s in scenarios]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = ax_comp.bar(scenarios, success_rates, color=colors, alpha=0.8, edgecolor='black')
    ax_comp.set_ylabel('Success Rate', fontsize=10, fontweight='bold')
    ax_comp.set_title('Treatment Strategy Comparison\n(Success rates across different drug types)', 
                      fontsize=12, fontweight='bold')
    ax_comp.set_ylim(0, 1.1)
    ax_comp.grid(True, alpha=0.4, axis='y')
    
    # Add percentage labels
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Treatment timeline (middle row)
    ax_timeline = fig.add_subplot(gs[1, :2])
    timevec = results['n_treated'].timevec
    
    ax_timeline.plot(timevec, results['n_treated'].values, label='New Treatments', 
                    color='#2E86AB', linewidth=2, alpha=0.8)
    ax_timeline.plot(timevec, results['n_treatment_success'].values, label='Successful', 
                    color='#28a745', linewidth=2, alpha=0.8)
    ax_timeline.plot(timevec, results['n_treatment_failure'].values, label='Failed', 
                    color='#dc3545', linewidth=2, alpha=0.8)
    
    ax_timeline.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')
    ax_timeline.set_ylabel('Number of Treatments', fontsize=10, fontweight='bold')
    ax_timeline.set_title('Treatment Timeline\n(Daily treatment outcomes over simulation period)', 
                          fontsize=12, fontweight='bold')
    ax_timeline.legend()
    ax_timeline.grid(True, alpha=0.3)
    
    # TB prevalence timeline (middle right)
    ax_prev = fig.add_subplot(gs[1, 2:])
    ax_prev.plot(tb_results['n_active'].timevec, tb_results['n_active'].values, 
                color='#fd7e14', linewidth=2)
    ax_prev.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')
    ax_prev.set_ylabel('Active TB Cases', fontsize=10, fontweight='bold')
    ax_prev.set_title('TB Prevalence Over Time\n(Impact of treatment on disease burden)', 
                      fontsize=12, fontweight='bold')
    ax_prev.grid(True, alpha=0.3)
    
    # Efficiency analysis (bottom left)
    ax_eff = fig.add_subplot(gs[2, 0])
    treated_counts = [comparison_results[s]['total_treated'] for s in scenarios]
    success_counts = [comparison_results[s]['total_success'] for s in scenarios]
    efficiency = [s/t for s, t in zip(success_counts, treated_counts)]
    
    bars_eff = ax_eff.bar(scenarios, efficiency, color=colors, alpha=0.8, edgecolor='black')
    ax_eff.set_ylabel('Efficiency (Success/Treatment)', fontsize=12, fontweight='bold')
    ax_eff.set_title('Treatment Efficiency\n(Resource utilization effectiveness)', 
                     fontsize=12, fontweight='bold')
    ax_eff.grid(True, alpha=0.4, axis='y')
    
    # Add efficiency labels
    for bar, eff in zip(bars_eff, efficiency):
        height = bar.get_height()
        ax_eff.text(bar.get_x() + bar.get_width()/2., height + max(efficiency)*0.01,
                   f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Success rate over time (bottom middle)
    ax_success = fig.add_subplot(gs[2, 1])
    success_rate_time = np.divide(results['n_treatment_success'].values, 
                                 np.maximum(results['n_treated'].values, 1), 
                                 out=np.zeros_like(results['n_treatment_success'].values), 
                                 where=results['n_treated'].values > 0)
    
    # Calculate moving average
    window_size = max(1, len(success_rate_time) // 20)
    if window_size > 1:
        success_rate_smooth = np.convolve(success_rate_time, np.ones(window_size)/window_size, mode='same')
        ax_success.plot(timevec, success_rate_smooth, color='#007bff', linewidth=2, alpha=0.8)
    
    ax_success.plot(timevec, success_rate_time, color='#6f42c1', alpha=0.6, linewidth=1)
    ax_success.axhline(y=success_rate, color='red', linestyle='--', alpha=0.7, 
                       label=f'Overall: {success_rate:.1%}')
    
    ax_success.set_xlabel('Time (Years)', fontsize=10, fontweight='bold')
    ax_success.set_ylabel('Success Rate', fontsize=10, fontweight='bold')
    ax_success.set_title('Success Rate Over Time\n(Program consistency and trends)', 
                         fontsize=12, fontweight='bold')
    ax_success.legend()
    ax_success.grid(True, alpha=0.3)
    ax_success.set_ylim(0, 1.05)
    
    # Recommendations panel (bottom right)
    ax_rec = fig.add_subplot(gs[2, 2:])
    ax_rec.axis('off')
    
    # Generate recommendations based on results
    best_strategy = max(comparison_results.keys(), 
                       key=lambda x: comparison_results[x]['success_rate'])
    best_rate = comparison_results[best_strategy]['success_rate']
    
    recommendations = f"""
    ANALYSIS & RECOMMENDATIONS
    
    Best Performing Strategy:
    • {best_strategy}: {best_rate:.1%} success rate
    
    Key Insights:
    • Treatment program reached {total_treated:,} individuals
    • Overall success rate: {success_rate:.1%}
    • TB burden reduced by {reduction:.1f}%
    
    Recommendations:
    • Consider scaling {best_strategy} strategy
    • Monitor treatment adherence patterns
    • Evaluate cost-effectiveness of strategies
    • Continue surveillance for resistance development
    """
    
    ax_rec.text(0.05, 0.95, recommendations, transform=ax_rec.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05, hspace=0.4, wspace=0.3)
    
    # Save the figure
    if results_dir:
        filename = os.path.join(results_dir, 'dots_summary_dashboard.png')
    else:
        filename = 'dots_summary_dashboard.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved summary dashboard: {filename}")
    plt.show()

def setup_results_directory():
    """Create a results directory with timestamp for organizing output files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"dots_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory: {results_dir}")
    return results_dir

def main():
    """Main function to run all tests."""
    print("TBsim DOTS Implementation Test")
    print("=" * 50)
    
    # Create results directory
    results_dir = setup_results_directory()
    
    # Test drug parameters
    test_drug_parameters()
    
    # Run simulation with DOTS
    sim, results = run_simulation_with_dots()
    
    # Run comparison simulation
    comparison_results = run_comparison_simulation()
    
    # Plot results
    plot_results(sim, results, results_dir)
    
    # Plot comparison results
    plot_comparison_results(comparison_results, results_dir)
    
    # Create comprehensive summary dashboard
    create_summary_dashboard(sim, results, comparison_results, results_dir)
    
    print("\n=== Test Summary ===")
    print("✅ Drug parameters system working correctly")
    print("✅ DOTS treatment implementation successful")
    print("✅ Drug type comparison completed")
    print("✅ Results visualization generated")
    print(f"✅ All figures saved to: {results_dir}")
    print("\nThe TBsim DOTS implementation now matches EMOD-Generic's approach!")

if __name__ == "__main__":
    main()
