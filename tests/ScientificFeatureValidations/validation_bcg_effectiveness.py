"""
BCG Intervention Effectiveness Validation Script

Compares baseline TB simulation with BCG intervention to validate effectiveness.
"""

import tbsim as mtb
import starsim as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Simulation parameters
SIM_PARAMS = {
    'n_agents': 5000,
    'dt': ss.days(7),
    'start': ss.date('2000-01-01'),
    'stop': ss.date('2025-12-31'),
}

TB_PARAMS = {
    'beta': ss.peryear(0.01),
    'init_prev': 0.25,
}

BCG_PARAMS = {
    'coverage': 0.95,
    'efficacy': 0.98,
    'age_range': (0, 18),
    'immunity_period': ss.years(10),
    'start': ss.date('2000-01-01'),
    'stop': ss.date('2025-12-31'),
    'vaccination_timing': ss.uniform(0, 5),  # Rollout over 5 years
    'death_modifier': ss.uniform(0.5, 0.7),
}

def create_population(n_agents=5000):
    """Create test population with age distribution"""
    age_data = pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
    })
    return ss.People(n_agents=n_agents, age_data=age_data)

def extract_results(sim):
    """Extract TB results from simulation"""
    tb_results = sim.results['tb']
    
    # Get timevec from sim.results or TB results
    if 'timevec' in sim.results:
        timevec = sim.results['timevec']
    elif 'timevec' in tb_results:
        timevec = tb_results['timevec']
    else:
        # Fallback: create timevec from simulation time
        timevec = np.array([sim.t.yearvec[i] for i in range(len(sim.t))])
    
    def get_result(key, default=0):
        if key in tb_results:
            result = tb_results[key]
            return result.values if hasattr(result, 'values') else result
        return np.full(len(timevec), default)
    
    return {
        'sim': sim,
        'timevec': timevec,
        'prevalence': get_result('prevalence'),
        'active_cases': get_result('n_active'),
        'new_active': get_result('new_active'),
        'cum_active': get_result('cum_active'),
    }

def run_baseline_simulation():
    """Run baseline simulation without BCG"""
    pop = create_population(SIM_PARAMS['n_agents'])
    tb = mtb.TB(pars=TB_PARAMS)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        pars={**SIM_PARAMS, 'verbose': 0}
    )
    sim.run()
    
    return extract_results(sim)

def run_bcg_simulation():
    """Run simulation with BCG intervention"""
    pop = create_population(SIM_PARAMS['n_agents'])
    tb = mtb.TB(pars=TB_PARAMS)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    bcg = mtb.BCGProtection(pars=BCG_PARAMS)    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        interventions=[bcg],
        pars={**SIM_PARAMS, 'verbose': 0}
    )
    sim.run()
    
    # Extract TB results
    results = extract_results(sim)
    
    # Extract BCG metrics
    bcg_intv = sim.interventions['bcgprotection']
    bcg_results = sim.results['bcgprotection']
    
    vaccinated = bcg_intv.is_bcg_vaccinated.sum()
    vaccinated_uids = bcg_intv.is_bcg_vaccinated.uids
    protected = (bcg_intv.is_bcg_vaccinated[vaccinated_uids] & 
                (sim.ti <= bcg_intv.ti_bcg_protection_expires[vaccinated_uids]) & 
                ~np.isnan(bcg_intv.ti_bcg_protection_expires[vaccinated_uids])).sum() if len(vaccinated_uids) > 0 else 0
    
    # Convert to numpy array for boolean operations
    protection_expires = np.array(bcg_intv.ti_bcg_protection_expires)
    total_responders = np.sum(~np.isnan(protection_expires))
    
    results.update({
        'vaccinated': vaccinated,
        'protected': protected,
        'coverage': vaccinated / len(sim.people) if len(sim.people) > 0 else 0.0,
        'effectiveness': total_responders / vaccinated if vaccinated > 0 else 0.0,
        'n_vaccinated_ts': bcg_results['n_vaccinated'].values if 'n_vaccinated' in bcg_results else np.zeros(len(results['timevec'])),
        'n_protected_ts': bcg_results['n_protected'].values if 'n_protected' in bcg_results else np.zeros(len(results['timevec'])),
    })
    
    return results


def calculate_metrics(baseline, bcg):
    """Calculate comparison metrics between baseline and BCG scenarios"""
    # Prevalence reduction
    prev_baseline = baseline['prevalence'][-1] if len(baseline['prevalence']) > 0 else 0
    prev_bcg = bcg['prevalence'][-1] if len(bcg['prevalence']) > 0 else 0
    prev_reduction = ((prev_baseline - prev_bcg) / prev_baseline * 100) if prev_baseline > 0 else 0
    
    # Incidence reduction
    baseline_inc = baseline['sim'].results['tb']['new_active'].values
    bcg_inc = bcg['sim'].results['tb']['new_active'].values
    incidence_reduction = (1 - np.mean(bcg_inc) / np.mean(baseline_inc)) * 100 if np.mean(baseline_inc) > 0 else 0
    
    # BCG metrics
    bcg_intv = bcg['sim'].interventions['bcgprotection']
    vaccinated_uids = bcg_intv.is_bcg_vaccinated.uids
    ever_protected = np.sum(~np.isnan(bcg_intv.ti_bcg_protection_expires[vaccinated_uids])) if len(vaccinated_uids) > 0 else 0
    
    return {
        'prev_reduction': prev_reduction,
        'incidence_reduction': incidence_reduction,
        'coverage': bcg['coverage'],
        'effectiveness': bcg['effectiveness'],
        'vaccinated': bcg['vaccinated'],
        'ever_protected': ever_protected,
        'currently_protected': bcg['protected'],
    }

def main():
    """Main validation function"""
    print('=== BCG INTERVENTION EFFECTIVENESS VALIDATION ===\n')
    
    print('Running baseline simulation...', end=' ', flush=True)
    baseline = run_baseline_simulation()
    print('✓')
    
    print('Running BCG intervention simulation...', end=' ', flush=True)
    bcg = run_bcg_simulation()
    print('✓')
    
    # Calculate and print metrics
    metrics = calculate_metrics(baseline, bcg)
    print('\n=== RESULTS ===')
    print(f'BCG Coverage: {metrics["coverage"]:.1%} ({metrics["vaccinated"]} vaccinated, {metrics["ever_protected"]} responders)')
    print(f'Prevalence Reduction: {metrics["prev_reduction"]:.1f}%')
    print(f'Incidence Reduction: {metrics["incidence_reduction"]:.1f}%')
    print(f'Vaccine Effectiveness: {metrics["effectiveness"]:.1%}')
    
    # Generate plots
    print('\nGenerating plots...', end=' ', flush=True)
    plot_comparison(baseline, bcg)
    print('✓')
    
    return baseline, bcg

def plot_comparison(baseline, bcg):
    """Create comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('BCG Intervention Effectiveness: Baseline vs Intervention', fontsize=16, fontweight='bold')
    
    time = baseline['timevec']
    
    # Plot 1: Active TB Cases
    ax = axes[0, 0]
    ax.plot(time, baseline['active_cases'], label='Baseline (No BCG)', linewidth=2, color='#d62728', alpha=0.8)
    ax.plot(time, bcg['active_cases'], label='With BCG Intervention', linewidth=2, color='#2ca02c', alpha=0.8)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Number of Active TB Cases')
    ax.set_title('Active TB Cases Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Incidence Rate per 100,000 Population (Bar Chart)
    ax = axes[0, 1]
    baseline_inc = baseline['sim'].results['tb']['new_active'].values
    bcg_inc = bcg['sim'].results['tb']['new_active'].values
    
    # Calculate incidence rates per 100,000 population
    baseline_pop = baseline['sim'].results['n_alive'].values
    bcg_pop = bcg['sim'].results['n_alive'].values
    
    # Annualize incidence (multiply by 52 weeks / dt in weeks)
    dt_weeks = baseline['sim'].dt.days / 7 if hasattr(baseline['sim'].dt, 'days') else 1
    weeks_per_year = 52
    annualization_factor = weeks_per_year / dt_weeks
    
    baseline_inc_rate = (baseline_inc / baseline_pop) * 100000 * annualization_factor
    bcg_inc_rate = (bcg_inc / bcg_pop) * 100000 * annualization_factor
    
    # Create bar chart comparing average incidence rates
    # Group by year for cleaner visualization
    # Handle both datetime objects and numeric time values
    try:
        years = np.array([t.year if hasattr(t, 'year') else int(t) for t in time])
    except (AttributeError, TypeError, ValueError):
        # Fallback: assume time is numeric (year values)
        years = np.array([int(t) for t in time])
    unique_years = np.unique(years)
    
    baseline_avg_by_year = [np.mean(baseline_inc_rate[years == y]) for y in unique_years]
    bcg_avg_by_year = [np.mean(bcg_inc_rate[years == y]) for y in unique_years]
    
    x_pos = np.arange(len(unique_years))
    width = 0.35
    
    ax.bar(x_pos - width/2, baseline_avg_by_year, width, label='Baseline (No BCG)', 
           color='#d62728', alpha=0.8)
    ax.bar(x_pos + width/2, bcg_avg_by_year, width, label='With BCG Intervention', 
           color='#2ca02c', alpha=0.8)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Incidence Rate (per 100,000 per year)')
    ax.set_title('TB Incidence Rate: Baseline vs BCG Intervention')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(y)) for y in unique_years], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: BCG Vaccination/Protection
    ax = axes[1, 0]
    ax.plot(time, bcg['n_vaccinated_ts'], label='Vaccinated', linewidth=2, color='#1f77b4', alpha=0.8)
    ax.plot(time, bcg['n_protected_ts'], label='Protected', linewidth=2, color='#ff7f0e', alpha=0.8)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('BCG Vaccination and Protection Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative Active Cases
    ax = axes[1, 1]
    ax.plot(time, baseline['cum_active'], label='Baseline (No BCG)', linewidth=2, color='#d62728', alpha=0.8)
    ax.plot(time, bcg['cum_active'], label='With BCG Intervention', linewidth=2, color='#2ca02c', alpha=0.8)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Cumulative Active TB Cases')
    ax.set_title('Cumulative Active TB Cases Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bcg_effectiveness_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    baseline, bcg_results = main()
    results = {}
    results['baseline'] = baseline['sim'].results.flatten()
    results['bcg'] = bcg_results['sim'].results.flatten()
    mtb.plot_combined(results, title='BCG Intervention Effectiveness: Baseline vs Intervention', outdir='results/interventions')
    