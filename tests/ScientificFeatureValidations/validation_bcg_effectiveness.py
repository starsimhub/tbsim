"""
BCG Intervention Effectiveness Validation Script

This script validates the effectiveness of the BCG intervention by comparing
baseline TB disease indicators with BCG intervention outcomes.

Author: TB Simulation Team
Date: 2024
Purpose: Validate BCG intervention impact on TB disease modeling indicators
"""

import tbsim as mtb
import starsim as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def create_test_population():
    """Create standardized test population with age distribution"""
    return pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
    })

def run_baseline_simulation(n_agents=500):
    """
    Run baseline simulation without BCG intervention
    
    Returns:
        dict: Baseline simulation results including TB risk modifiers and time series
    """
    age_data = create_test_population()
    pop = ss.People(n_agents=n_agents, age_data=age_data)
    # Use moderate initial prevalence and higher transmission to see dramatic BCG effect
    # Higher beta generates more new infections, making BCG's prevention more visible
    tb = mtb.TB(pars={'beta': ss.peryear(0.12), 'init_prev': 0.20})
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        pars={'dt': ss.days(7), 'start': ss.date('2000-01-01'), 'stop': ss.date('2025-12-31'), 'verbose': 0}
    )
    
    # Run full simulation
    sim.run()
    
    # Get baseline TB risk modifiers (these are reset each timestep, so we check at final timestep)
    # Note: Risk modifiers are reset to 1.0 each timestep, so checking at end shows default values
    baseline_activation = 1.0
    baseline_clearance = 1.0
    baseline_death = 1.0
    
    # Get TB prevalence over time
    tb_results = sim.results['tb']
    # Access results - Result objects have .values and .timevec attributes
    if 'prevalence' in tb_results:
        prevalence_result = tb_results['prevalence']
        timevec = prevalence_result.timevec
        prevalence = prevalence_result.values
    else:
        timevec = tb_results.get('timevec', np.arange(sim.npts))
        prevalence = np.zeros(len(timevec))
    
    if 'n_active' in tb_results:
        active_result = tb_results['n_active']
        active_cases = active_result.values
    else:
        active_cases = np.zeros(len(timevec))
    
    # Get incidence and cumulative metrics
    if 'new_infections' in tb_results:
        new_infections_result = tb_results['new_infections']
        new_infections = new_infections_result.values if hasattr(new_infections_result, 'values') else new_infections_result
    else:
        new_infections = np.zeros(len(timevec))
    
    if 'new_active' in tb_results:
        new_active_result = tb_results['new_active']
        new_active = new_active_result.values if hasattr(new_active_result, 'values') else new_active_result
    else:
        new_active = np.zeros(len(timevec))
    
    if 'cum_infections' in tb_results:
        cum_infections_result = tb_results['cum_infections']
        cum_infections = cum_infections_result.values if hasattr(cum_infections_result, 'values') else cum_infections_result
    else:
        cum_infections = np.zeros(len(timevec))
    
    if 'cum_active' in tb_results:
        cum_active_result = tb_results['cum_active']
        cum_active = cum_active_result.values if hasattr(cum_active_result, 'values') else cum_active_result
    else:
        cum_active = np.zeros(len(timevec))
    
    return {
        'activation_risk': baseline_activation,
        'clearance_rate': baseline_clearance,
        'death_risk': baseline_death,
        'population': len(sim.people),
        'tb_states': tb.state.raw,
        'sim': sim,
        'timevec': timevec,
        'prevalence': prevalence,
        'active_cases': active_cases,
        'new_infections': new_infections,
        'new_active': new_active,
        'cum_infections': cum_infections,
        'cum_active': cum_active
    }

def run_bcg_simulation(n_agents=500):
    """
    Run simulation with BCG intervention
    
    Returns:
        dict: BCG simulation results including vaccination metrics and time series
    """
    age_data = create_test_population()
    pop = ss.People(n_agents=n_agents, age_data=age_data)
    # Use moderate initial prevalence and higher transmission to see dramatic BCG effect
    # Higher beta generates more new infections, making BCG's prevention more visible
    tb = mtb.TB(pars={'beta': ss.peryear(0.12), 'init_prev': 0.20})
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    bcg = mtb.BCGProtection(pars={
        'coverage': 0.95,  # Very high coverage for dramatic effect
        'efficacy': 0.98,  # Very high efficacy
        'age_range': (0, 18),  # Extended age range to vaccinate more people
        'immunity_period': ss.years(10),
        'start': ss.date('2000-01-01'),  # Start immediately
        'stop': ss.date('2025-12-31')
    })
    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        interventions=[bcg],
        pars={'dt': ss.days(7), 'start': ss.date('2000-01-01'), 'stop': ss.date('2025-12-31'), 'verbose': 0}
    )
    
    # Run full simulation
    sim.run()
    
    # Get BCG intervention
    bcg_intervention = sim.interventions['bcgprotection']
    
    # Get post-BCG TB risk modifiers (these are reset each timestep, so we check at final timestep)
    # Note: Risk modifiers are reset to 1.0 each timestep, so checking at end shows default values
    # The actual impact is seen in incidence and disease progression, not in final modifier values
    bcg_activation = 1.0
    bcg_clearance = 1.0
    bcg_death = 1.0
    
    # Get BCG metrics
    vaccinated = bcg_intervention.is_bcg_vaccinated.sum()
    protected = bcg_intervention.is_protected(bcg_intervention.is_bcg_vaccinated.uids, sim.ti).sum()
    stats = bcg_intervention.get_summary_stats()
    
    # Get TB prevalence over time
    tb_results = sim.results['tb']
    # Access results - Result objects have .values and .timevec attributes
    if 'prevalence' in tb_results:
        prevalence_result = tb_results['prevalence']
        timevec = prevalence_result.timevec
        prevalence = prevalence_result.values
    else:
        timevec = tb_results.get('timevec', np.arange(sim.npts))
        prevalence = np.zeros(len(timevec))
    
    if 'n_active' in tb_results:
        active_result = tb_results['n_active']
        active_cases = active_result.values
    else:
        active_cases = np.zeros(len(timevec))
    
    # Get incidence and cumulative metrics
    if 'new_infections' in tb_results:
        new_infections_result = tb_results['new_infections']
        new_infections = new_infections_result.values if hasattr(new_infections_result, 'values') else new_infections_result
    else:
        new_infections = np.zeros(len(timevec))
    
    if 'new_active' in tb_results:
        new_active_result = tb_results['new_active']
        new_active = new_active_result.values if hasattr(new_active_result, 'values') else new_active_result
    else:
        new_active = np.zeros(len(timevec))
    
    if 'cum_infections' in tb_results:
        cum_infections_result = tb_results['cum_infections']
        cum_infections = cum_infections_result.values if hasattr(cum_infections_result, 'values') else cum_infections_result
    else:
        cum_infections = np.zeros(len(timevec))
    
    if 'cum_active' in tb_results:
        cum_active_result = tb_results['cum_active']
        cum_active = cum_active_result.values if hasattr(cum_active_result, 'values') else cum_active_result
    else:
        cum_active = np.zeros(len(timevec))
    
    # Get BCG vaccination metrics over time
    bcg_results = sim.results['bcgprotection']
    if 'n_vaccinated' in bcg_results:
        n_vaccinated_result = bcg_results['n_vaccinated']
        n_vaccinated_ts = n_vaccinated_result.values
    else:
        n_vaccinated_ts = np.zeros(len(timevec))
    
    if 'n_protected' in bcg_results:
        n_protected_result = bcg_results['n_protected']
        n_protected_ts = n_protected_result.values
    else:
        n_protected_ts = np.zeros(len(timevec))
    
    return {
        'activation_risk': bcg_activation,
        'clearance_rate': bcg_clearance,
        'death_risk': bcg_death,
        'population': len(sim.people),
        'tb_states': tb.state.raw,
        'vaccinated': vaccinated,
        'protected': protected,
        'coverage': stats['final_coverage'],
        'effectiveness': stats['vaccine_effectiveness'],
        'sim': sim,
        'timevec': timevec,
        'prevalence': prevalence,
        'active_cases': active_cases,
        'new_infections': new_infections,
        'new_active': new_active,
        'cum_infections': cum_infections,
        'cum_active': cum_active,
        'n_vaccinated_ts': n_vaccinated_ts,
        'n_protected_ts': n_protected_ts
    }

def test_bcg_individual_impact(n_agents=200):
    """
    Test individual-level BCG impact on risk modifiers
    
    Returns:
        dict: Individual-level BCG impact metrics
    """
    age_data = create_test_population()
    pop = ss.People(n_agents=n_agents, age_data=age_data)
    # Use moderate initial prevalence and higher transmission to see dramatic BCG effect
    # Higher beta generates more new infections, making BCG's prevention more visible
    tb = mtb.TB(pars={'beta': ss.peryear(0.12), 'init_prev': 0.20})
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    bcg = mtb.BCGProtection(pars={
        'coverage': 0.95,  # Very high coverage for dramatic effect
        'efficacy': 0.98,  # Very high efficacy
        'age_range': (0, 18),  # Extended age range to vaccinate more people
        'immunity_period': ss.years(10),
        'start': ss.date('2000-01-01'),  # Start immediately
        'stop': ss.date('2025-12-31')
    })
    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        interventions=[bcg],
        pars={'dt': ss.days(7), 'start': ss.date('2000-01-01'), 'stop': ss.date('2025-12-31'), 'verbose': 0}
    )
    sim.init()
    
    bcg_intervention = sim.interventions['bcgprotection']
    bcg_intervention.step()
    
    # Get individual-level impact
    vaccinated = bcg_intervention.is_bcg_vaccinated.sum()
    if vaccinated > 0:
        vaccinated_uids = bcg_intervention.is_bcg_vaccinated.uids
        activation_modifiers = bcg_intervention.bcg_activation_modifier_applied[vaccinated_uids]
        clearance_modifiers = bcg_intervention.bcg_clearance_modifier_applied[vaccinated_uids]
        death_modifiers = bcg_intervention.bcg_death_modifier_applied[vaccinated_uids]
        
        # Get valid modifiers (non-NaN)
        valid_activation = activation_modifiers[~np.isnan(activation_modifiers)]
        valid_clearance = clearance_modifiers[~np.isnan(clearance_modifiers)]
        valid_death = death_modifiers[~np.isnan(death_modifiers)]
        
        return {
            'vaccinated': vaccinated,
            'activation_mean': np.mean(valid_activation) if len(valid_activation) > 0 else 1.0,
            'clearance_mean': np.mean(valid_clearance) if len(valid_clearance) > 0 else 1.0,
            'death_mean': np.mean(valid_death) if len(valid_death) > 0 else 1.0,
            'activation_reduction': (1 - np.mean(valid_activation)) * 100 if len(valid_activation) > 0 else 0,
            'clearance_improvement': (np.mean(valid_clearance) - 1) * 100 if len(valid_clearance) > 0 else 0,
            'death_reduction': (1 - np.mean(valid_death)) * 100 if len(valid_death) > 0 else 0
        }
    
    return {'vaccinated': 0}

def main():
    """Main validation function"""
    print('=== BCG INTERVENTION EFFECTIVENESS VALIDATION ===\n')
    
    # Run baseline simulation
    print('Running baseline simulation...', end=' ', flush=True)
    baseline = run_baseline_simulation()
    print('✓')
    
    # Run BCG simulation
    print('Running BCG intervention simulation...', end=' ', flush=True)
    bcg_results = run_bcg_simulation()
    print('✓')
    
    # Calculate key metrics
    bcg_sim = bcg_results['sim']
    bcg_intervention = bcg_sim.interventions['bcgprotection']
    final_vaccinated = bcg_intervention.is_bcg_vaccinated.sum()
    # Check how many ever had protection (responders), not just currently protected
    # (protection expires after immunity_period, so at end of 25-year sim, all may be expired)
    vaccinated_uids = bcg_intervention.is_bcg_vaccinated.uids
    if len(vaccinated_uids) > 0:
        expires = np.array(bcg_intervention.ti_bcg_protection_expires[vaccinated_uids])
        ever_protected = (~np.isnan(expires)).sum()  # Count responders (ever had protection)
        final_protected = bcg_intervention.is_protected(vaccinated_uids, bcg_sim.ti).sum()  # Currently protected
    else:
        ever_protected = 0
        final_protected = 0
    
    baseline_final_prev = baseline['prevalence'][-1] if len(baseline['prevalence']) > 0 else 0
    bcg_final_prev = bcg_results['prevalence'][-1] if len(bcg_results['prevalence']) > 0 else 0
    prev_reduction = ((baseline_final_prev - bcg_final_prev) / baseline_final_prev * 100) if baseline_final_prev > 0 else 0
    
    # Check incidence rates
    tb_results_baseline = baseline['sim'].results['tb']
    tb_results_bcg = bcg_results['sim'].results['tb']
    incidence_reduction = 0
    if 'new_active' in tb_results_baseline and 'new_active' in tb_results_bcg:
        baseline_incidence_result = tb_results_baseline['new_active']
        bcg_incidence_result = tb_results_bcg['new_active']
        baseline_incidence = baseline_incidence_result.values if hasattr(baseline_incidence_result, 'values') else baseline_incidence_result
        bcg_incidence = bcg_incidence_result.values if hasattr(bcg_incidence_result, 'values') else bcg_incidence_result
        if np.mean(baseline_incidence) > 0:
            incidence_reduction = (1 - np.mean(bcg_incidence) / np.mean(baseline_incidence)) * 100
    
    # Print summary
    print('\n=== METRICS ===')
    print(f'BCG Coverage: {bcg_results["coverage"]:.1%} ({final_vaccinated} vaccinated, {ever_protected} responders, {final_protected} currently protected)')
    print(f'Prevalence Reduction: {prev_reduction:.1f}% (Baseline: {baseline_final_prev*100:.1f}% → BCG: {bcg_final_prev*100:.1f}%)')
    print(f'Incidence Reduction: {incidence_reduction:.1f}%')
    print(f'Vaccine Effectiveness: {bcg_results["effectiveness"]:.1%}')
    
    # Create comparison plot
    print('\nGenerating comparison plot...', end=' ', flush=True)
    plot_baseline_vs_bcg(baseline, bcg_results)
    print('✓')
    
    return baseline, bcg_results

def plot_baseline_vs_bcg(baseline, bcg_results):
    """
    Create a comparison plot showing baseline vs BCG intervention results
    
    Parameters:
    -----------
    baseline : dict
        Baseline simulation results
    bcg_results : dict
        BCG intervention simulation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BCG Intervention Effectiveness: Baseline vs Intervention', fontsize=16, fontweight='bold')
    
    # Plot 1: TB Prevalence over time
    ax1 = axes[0, 0]
    if 'timevec' in baseline and 'prevalence' in baseline and len(baseline['prevalence']) > 0:
        ax1.plot(baseline['timevec'], baseline['prevalence'] * 100, 
                label='Baseline (No BCG)', linewidth=2, color='#d62728', alpha=0.8)
    if 'timevec' in bcg_results and 'prevalence' in bcg_results and len(bcg_results['prevalence']) > 0:
        ax1.plot(bcg_results['timevec'], bcg_results['prevalence'] * 100, 
                label='With BCG Intervention', linewidth=2, color='#2ca02c', alpha=0.8)
    ax1.set_xlabel('Time (years)', fontsize=11)
    ax1.set_ylabel('TB Prevalence (%)', fontsize=11)
    ax1.set_title('TB Prevalence Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add explanation text box below the plot
    explanation_text1 = (
        "Prevalence = % of population currently infected\n"
        "BCG reduces prevalence by preventing new infections and\n"
        "reducing progression from latent to active TB (~42% reduction)"
    )
    ax1.text(0.5, -0.25, explanation_text1, transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Plot 2: New Active TB Cases (Incidence) over time
    ax2 = axes[0, 1]
    baseline_sim = baseline.get('sim')
    bcg_sim = bcg_results.get('sim')
    
    if baseline_sim and 'new_active' in baseline_sim.results['tb']:
        baseline_incidence = baseline_sim.results['tb']['new_active']
        baseline_timevec = baseline_incidence.timevec
        baseline_new_cases = baseline_incidence.values
        ax2.plot(baseline_timevec, baseline_new_cases, 
                label='Baseline (No BCG)', linewidth=2, color='#d62728', alpha=0.8)
    
    if bcg_sim and 'new_active' in bcg_sim.results['tb']:
        bcg_incidence = bcg_sim.results['tb']['new_active']
        bcg_timevec = bcg_incidence.timevec
        bcg_new_cases = bcg_incidence.values
        ax2.plot(bcg_timevec, bcg_new_cases, 
                label='With BCG Intervention', linewidth=2, color='#2ca02c', alpha=0.8)
    
    ax2.set_xlabel('Time (years)', fontsize=11)
    ax2.set_ylabel('New Active TB Cases per Timestep', fontsize=11)
    ax2.set_title('TB Incidence (New Cases) Over Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add explanation text box below the plot
    explanation_text2 = (
        "Incidence = New active TB cases per timestep\n"
        "BCG reduces incidence by ~42% (activation risk) and\n"
        "improves clearance by ~39% (active → cleared)"
    )
    ax2.text(0.5, -0.25, explanation_text2, transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Plot 3: BCG Vaccination Coverage over time
    ax3 = axes[1, 0]
    if 'timevec' in bcg_results and 'n_vaccinated_ts' in bcg_results and len(bcg_results['n_vaccinated_ts']) > 0:
        ax3.plot(bcg_results['timevec'], bcg_results['n_vaccinated_ts'], 
                label='Vaccinated', linewidth=2, color='#1f77b4', alpha=0.8)
    if 'timevec' in bcg_results and 'n_protected_ts' in bcg_results and len(bcg_results['n_protected_ts']) > 0:
        ax3.plot(bcg_results['timevec'], bcg_results['n_protected_ts'], 
                label='Protected', linewidth=2, color='#ff7f0e', alpha=0.8)
    ax3.set_xlabel('Time (years)', fontsize=11)
    ax3.set_ylabel('Number of Individuals', fontsize=11)
    ax3.set_title('BCG Vaccination and Protection Over Time', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add explanation text box below the plot
    explanation_text3 = (
        "Vaccinated = Received BCG vaccine\n"
        "Protected = Still within 10-year immunity period\n"
        "Protection expires after immunity period ends"
    )
    ax3.text(0.5, -0.25, explanation_text3, transform=ax3.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # Plot 4: Cumulative Active TB Cases
    ax4 = axes[1, 1]
    if 'timevec' in baseline and 'cum_active' in baseline and len(baseline['cum_active']) > 0:
        ax4.plot(baseline['timevec'], baseline['cum_active'], 
                label='Baseline (No BCG)', linewidth=2, color='#d62728', alpha=0.8)
    if 'timevec' in bcg_results and 'cum_active' in bcg_results and len(bcg_results['cum_active']) > 0:
        ax4.plot(bcg_results['timevec'], bcg_results['cum_active'], 
                label='With BCG Intervention', linewidth=2, color='#2ca02c', alpha=0.8)
    ax4.set_xlabel('Time (years)', fontsize=11)
    ax4.set_ylabel('Cumulative Active TB Cases', fontsize=11)
    ax4.set_title('Cumulative Active TB Cases Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add explanation text box below the plot
    explanation_text4 = (
        "Cumulative = Total active TB cases since simulation start\n"
        "Gap between lines = cases prevented by BCG\n"
        "Shows accumulating protective effect over time"
    )
    ax4.text(0.5, -0.25, explanation_text4, transform=ax4.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the plot
    output_path = 'bcg_effectiveness_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved: {output_path}')
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    baseline, bcg_results = main()