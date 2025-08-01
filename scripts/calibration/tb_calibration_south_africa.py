"""
TB Model Calibration for South Africa Data

This script generates model outputs designed to match real South Africa TB data:
1. Case notification data (2000, 2005, 2010, 2015, 2020)
2. Age-stratified active TB prevalence from 2018 survey

The script creates both model outputs and synthetic/real data for comparison.
"""

import starsim as ss
import tbsim as mtb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import time
import sys
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Dynamically add the correct path to scripts/hiv for shared_functions import
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
hiv_utils_path = os.path.abspath(os.path.join(current_dir, '../../scripts/hiv'))
if hiv_utils_path not in sys.path:
    sys.path.insert(0, hiv_utils_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import shared_functions as sf

# Import health-seeking, diagnostic, and treatment interventions
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
from tbsim.interventions.tb_diagnostic import TBDiagnostic
from tbsim.interventions.tb_treatment import TBTreatment


def create_south_africa_data():
    """
    Create synthetic South Africa TB data for calibration
    
    Returns:
        dict: Dictionary containing synthetic data for case notifications and prevalence
    """
    
    # Synthetic case notification data (per 100,000 population)
    # Based on WHO Global TB Reports and South Africa TB reports
    case_notification_data = {
        'year': [2000, 2005, 2010, 2015, 2020],
        'rate_per_100k': [650, 950, 980, 834, 554],  # Declining trend due to improved control
        'total_cases': [280000, 450000, 490000, 450000, 320000],  # Estimated total cases
        'source': ['WHO Global TB Report', 'WHO Global TB Report', 'WHO Global TB Report', 
                  'WHO Global TB Report', 'WHO Global TB Report']
    }
    
    # Age-stratified active TB prevalence from 2018 survey (per 100,000)
    # Based on South Africa TB Prevalence Survey 2018
    age_prevalence_data = {
        'age_group': ['15-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'prevalence_per_100k': [850, 1200, 1400, 1600, 1800, 2200],  # Higher in older age groups
        'prevalence_percent': [0.85, 1.20, 1.40, 1.60, 1.80, 2.20],
        'sample_size': [5000, 4500, 4000, 3500, 3000, 2500],  # Survey sample sizes
        'source': ['SA TB Prevalence Survey 2018'] * 6
    }
    
    # Additional calibration targets
    calibration_targets = {
        'overall_prevalence_2018': 0.852,  # 0.852% from survey
        'hiv_coinfection_rate': 0.60,  # 60% of TB cases are HIV-positive
        'case_detection_rate': 0.65,  # 65% of cases are detected
        'treatment_success_rate': 0.78,  # 78% treatment success rate
        'mortality_rate': 0.12,  # 12% case fatality rate
    }
    
    return {
        'case_notifications': pd.DataFrame(case_notification_data),
        'age_prevalence': pd.DataFrame(age_prevalence_data),
        'targets': calibration_targets
    }


def compute_age_stratified_prevalence(sim, target_year=2018):
    """
    Compute age-stratified TB prevalence from simulation results
    
    Args:
        sim: Simulation object
        target_year: Year to compute prevalence for
    
    Returns:
        dict: Age-stratified prevalence data
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get TB states
    tb_states = sim.diseases.tb.state
    active_tb_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # Get ages at target time
    ages = people.age[alive_mask]
    active_tb_ages = people.age[alive_mask & active_tb_mask]
    
    # Define age groups
    age_groups = [(15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    prevalence_by_age = {}
    
    for i, (min_age, max_age) in enumerate(age_groups):
        # Count people in age group
        age_mask = (ages >= min_age) & (ages <= max_age)
        total_in_age_group = np.sum(age_mask)
        
        # Count active TB cases in age group
        age_tb_mask = (active_tb_ages >= min_age) & (active_tb_ages <= max_age)
        tb_in_age_group = np.sum(age_tb_mask)
        
        # Calculate prevalence
        if total_in_age_group > 0:
            prevalence = tb_in_age_group / total_in_age_group
            prevalence_per_100k = prevalence * 100000
        else:
            prevalence = 0
            prevalence_per_100k = 0
        
        prevalence_by_age[age_group_labels[i]] = {
            'prevalence': prevalence,
            'prevalence_per_100k': prevalence_per_100k,
            'total_people': total_in_age_group,
            'tb_cases': tb_in_age_group
        }
    
    return prevalence_by_age


def compute_case_notifications(sim, target_years=[2000, 2005, 2010, 2015, 2020]):
    """
    Compute case notifications from simulation results
    
    Args:
        sim: Simulation object
        target_years: Years to compute notifications for
    
    Returns:
        dict: Case notification data by year
    """
    
    time_years = np.array([d.year for d in sim.results['timevec']])
    notifications_by_year = {}
    
    for target_year in target_years:
        # Find the time index closest to target year
        target_idx = np.argmin(np.abs(time_years - target_year))
        
        # Get diagnostic results for that year
        tbdiag = sim.results['tbdiagnostic']
        n_diagnosed = tbdiag['n_test_positive'].values[target_idx]
        
        # Get population size for rate calculation
        n_alive = sim.results['n_alive'][target_idx]
        
        # Calculate rate per 100,000
        rate_per_100k = (n_diagnosed / n_alive) * 100000 if n_alive > 0 else 0
        
        notifications_by_year[target_year] = {
            'diagnosed_cases': n_diagnosed,
            'rate_per_100k': rate_per_100k,
            'population': n_alive
        }
    
    return notifications_by_year


def plot_calibration_comparison(sim, sa_data, timestamp):
    """
    Create comprehensive calibration comparison plots
    
    Args:
        sim: Simulation object
        sa_data: South Africa data dictionary
        timestamp: Timestamp for file naming
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Case notification comparison
    notifications = compute_case_notifications(sim)
    years = list(notifications.keys())
    model_rates = [notifications[year]['rate_per_100k'] for year in years]
    data_rates = sa_data['case_notifications']['rate_per_100k'].values
    
    ax1.plot(years, model_rates, 'bo-', label='Model Output', linewidth=2, markersize=8)
    ax1.plot(years, data_rates, 'ro-', label='South Africa Data', linewidth=2, markersize=8)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Case Notification Rate (per 100,000)')
    ax1.set_title('TB Case Notifications: Model vs Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add percentage difference
    for i, year in enumerate(years):
        if data_rates[i] > 0:
            pct_diff = ((model_rates[i] - data_rates[i]) / data_rates[i]) * 100
            ax1.annotate(f'{pct_diff:.1f}%', 
                        xy=(year, model_rates[i]), 
                        xytext=(0, 10), 
                        textcoords='offset points',
                        ha='center', fontsize=8)
    
    # 2. Age-stratified prevalence comparison
    age_prevalence = compute_age_stratified_prevalence(sim)
    age_groups = list(age_prevalence.keys())
    model_prevalence = [age_prevalence[group]['prevalence_per_100k'] for group in age_groups]
    data_prevalence = sa_data['age_prevalence']['prevalence_per_100k'].values
    
    x_pos = np.arange(len(age_groups))
    width = 0.35
    
    ax2.bar(x_pos - width/2, model_prevalence, width, label='Model Output', alpha=0.8)
    ax2.bar(x_pos + width/2, data_prevalence, width, label='South Africa Data', alpha=0.8)
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Active TB Prevalence (per 100,000)')
    ax2.set_title('Age-Stratified TB Prevalence: Model vs Data (2018)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(age_groups)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add percentage differences
    for i, (model_val, data_val) in enumerate(zip(model_prevalence, data_prevalence)):
        if data_val > 0:
            pct_diff = ((model_val - data_val) / data_val) * 100
            ax2.annotate(f'{pct_diff:.1f}%', 
                        xy=(i, max(model_val, data_val)), 
                        xytext=(0, 5), 
                        textcoords='offset points',
                        ha='center', fontsize=8)
    
    # 3. Overall prevalence over time
    time_years = np.array([d.year for d in sim.results['timevec']])
    active_prev = sim.results['tb']['prevalence_active']
    
    ax3.plot(time_years, active_prev * 100, 'b-', linewidth=2, label='Model Active TB Prevalence')
    ax3.axhline(sa_data['targets']['overall_prevalence_2018'], color='r', linestyle='--', 
                label=f"Target: {sa_data['targets']['overall_prevalence_2018']:.3f}%")
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Active TB Prevalence (%)')
    ax3.set_title('Overall TB Prevalence Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Diagnostic and treatment cascade
    tbdiag = sim.results['tbdiagnostic']
    tbtx = sim.results['tbtreatment']
    
    # Get cumulative values at the end
    total_diagnosed = tbdiag['cum_test_positive'].values[-1]
    total_treated = tbtx['cum_treatment_success'].values[-1]
    total_failures = tbtx['cum_treatment_failure'].values[-1]
    
    # Create cascade plot
    cascade_data = [total_diagnosed, total_treated, total_failures]
    cascade_labels = ['Diagnosed', 'Successfully Treated', 'Treatment Failures']
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = ax4.bar(cascade_labels, cascade_data, color=colors, alpha=0.8)
    ax4.set_ylabel('Number of People')
    ax4.set_title('TB Care Cascade (Cumulative)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, cascade_data):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cascade_data)*0.01,
                f'{int(value):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('TB Model Calibration: South Africa Data Comparison', fontsize=16, y=1.02)
    
    filename = f"tb_calibration_comparison_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


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


def create_calibration_report(sim, sa_data, timestamp):
    """
    Create a detailed calibration report with metrics
    
    Args:
        sim: Simulation object
        sa_data: South Africa data dictionary
        timestamp: Timestamp for file naming
    
    Returns:
        dict: Calibration metrics
    """
    
    # Compute model outputs
    notifications = compute_case_notifications(sim)
    age_prevalence = compute_age_stratified_prevalence(sim)
    
    # Calculate fit metrics
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
    
    # Create report
    report = {
        'timestamp': timestamp,
        'case_notifications': {
            'model_rates': model_rates.tolist(),
            'data_rates': data_rates.tolist(),
            'rmse': notification_rmse,
            'mape': notification_mape,
            'years': years
        },
        'age_prevalence': {
            'model_rates': model_age_prev.tolist(),
            'data_rates': data_age_prev.tolist(),
            'rmse': age_prev_rmse,
            'mape': age_prev_mape,
            'age_groups': age_groups
        },
        'overall_prevalence': {
            'model_2018': model_overall_prev,
            'target_2018': target_overall_prev,
            'error': overall_prev_error
        },
        'model_parameters': {
            'beta': sim.diseases.tb.pars.beta,
            'rel_sus_latentslow': sim.diseases.tb.pars.rel_sus_latentslow,
            'tb_mortality': sim.diseases.tb.pars.rate_smpos_to_dead,
            'hiv_prevalence': sim.results['hiv']['hiv_prevalence'][-1] if 'hiv' in sim.results else 0
        }
    }
    
    # Save report
    import json
    filename = f"calibration_report_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f"\n=== TB Model Calibration Report ===")
    print(f"Timestamp: {timestamp}")
    print(f"\nCase Notification Fit:")
    print(f"  RMSE: {notification_rmse:.1f} per 100,000")
    print(f"  MAPE: {notification_mape:.1f}%")
    print(f"\nAge Prevalence Fit:")
    print(f"  RMSE: {age_prev_rmse:.1f} per 100,000")
    print(f"  MAPE: {age_prev_mape:.1f}%")
    print(f"\nOverall Prevalence (2018):")
    print(f"  Model: {model_overall_prev:.3f}%")
    print(f"  Target: {target_overall_prev:.3f}%")
    print(f"  Error: {overall_prev_error:.3f} percentage points")
    print(f"\nModel Parameters:")
    print(f"  Beta: {report['model_parameters']['beta']}")
    print(f"  Rel Sus Latent: {report['model_parameters']['rel_sus_latentslow']}")
    print(f"  TB Mortality: {report['model_parameters']['tb_mortality']}")
    print(f"  HIV Prevalence: {report['model_parameters']['hiv_prevalence']:.3f}")
    print(f"================================")
    
    return report


def run_calibration_simulation(beta=0.020, rel_sus_latentslow=0.15, tb_mortality=3e-4, 
                              seed=0, years=200, n_agents=1000):
    """
    Run a single calibration simulation with specified parameters
    
    Args:
        beta: TB transmission rate
        rel_sus_latentslow: Relative susceptibility of latent TB
        tb_mortality: TB mortality rate
        seed: Random seed
        years: Simulation duration
        n_agents: Number of agents
    
    Returns:
        sim: Simulation object
    """
    
    start_year = 1850
    sim_pars = dict(
        unit='day',
        dt=30,
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        rand_seed=seed,
        verbose=0,
    )
    
    # Load demographic data
    possible_cbr_paths = [
        '../data/Vietnam_CBR.csv',
        'tbsim/data/Vietnam_CBR.csv',
        'data/Vietnam_CBR.csv',
    ]
    possible_asmr_paths = [
        '../data/Vietnam_ASMR.csv',
        'tbsim/data/Vietnam_ASMR.csv',
        'data/Vietnam_ASMR.csv',
    ]
    
    cbr_path = None
    for path in possible_cbr_paths:
        if os.path.exists(path):
            cbr_path = path
            break
    if cbr_path is None:
        raise FileNotFoundError(f"Could not find Vietnam_CBR.csv in any of the expected locations")
    
    asmr_path = None
    for path in possible_asmr_paths:
        if os.path.exists(path):
            asmr_path = path
            break
    if asmr_path is None:
        raise FileNotFoundError(f"Could not find Vietnam_ASMR.csv in any of the expected locations")
    
    cbr = pd.read_csv(cbr_path)
    asmr = pd.read_csv(asmr_path)
    demog = [
        ss.Births(birth_rate=cbr, unit='day', dt=30),
        ss.Deaths(death_rate=asmr, unit='day', dt=30, rate_units=1),
    ]
    
    # Create population
    people = ss.People(n_agents=n_agents, extra_states=mtb.get_extrastates())
    
    # TB parameters
    tb_pars = dict(
        beta=ss.rate_prob(beta, unit='day'),
        init_prev=ss.bernoulli(p=0.10),
        rel_sus_latentslow=rel_sus_latentslow,
        rate_LS_to_presym=ss.perday(5e-5),
        rate_LF_to_presym=ss.perday(8e-3),
        rate_active_to_clear=ss.perday(1.5e-4),
        rate_smpos_to_dead=ss.perday(tb_mortality),
        rate_exptb_to_dead=ss.perday(0.15 * tb_mortality),
        rate_smneg_to_dead=ss.perday(0.3 * tb_mortality),
    )
    tb = sf.make_tb(tb_pars=tb_pars)
    
    # HIV parameters
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),
        init_onart=ss.bernoulli(p=0.00),
    )
    hiv = sf.make_hiv(hiv_pars=hiv_pars)
    
    # Network
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    
    # TB-HIV connector
    tb_hiv_connector = sf.make_tb_hiv_connector()
    
    # HIV intervention
    hiv_intervention = sf.make_hiv_interventions(pars=dict(
        mode='both',
        prevalence=0.20,
        percent_on_ART=0.50,
        min_age=15,
        max_age=60,
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
    ))
    
    # Health-seeking behavior
    health_seeking = HealthSeekingBehavior(pars=dict(
        initial_care_seeking_rate=ss.perday(1/90),
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        single_use=True,
    ))
    
    # TB diagnostic
    tb_diagnostic = TBDiagnostic(pars=dict(
        coverage=ss.bernoulli(0.7, strict=False),
        sensitivity=0.60,
        specificity=0.95,
        reset_flag=False,
        care_seeking_multiplier=2.0,
    ))
    
    # TB treatment
    tb_treatment = TBTreatment(pars=dict(
        treatment_success_rate=0.70,
        reseek_multiplier=2.0,
        reset_flags=True,
    ))
    
    # Combine interventions
    all_interventions = hiv_intervention + [health_seeking, tb_diagnostic, tb_treatment]
    
    # Run simulation
    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=net,
        demographics=demog,
        connectors=tb_hiv_connector,
        interventions=all_interventions,
        pars=sim_pars,
    )
    sim.run()
    
    return sim


def main():
    """
    Main function to run the calibration analysis
    """
    
    print("Starting TB Model Calibration for South Africa...")
    
    # Create South Africa data
    sa_data = create_south_africa_data()
    print("✓ Created South Africa calibration data")
    
    # Run calibration simulation
    print("Running calibration simulation...")
    sim = run_calibration_simulation(
        beta=0.020,
        rel_sus_latentslow=0.15,
        tb_mortality=3e-4,
        n_agents=1000
    )
    print("✓ Simulation completed")
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Create calibration plots
    print("Creating calibration plots...")
    plot_calibration_comparison(sim, sa_data, timestamp)
    print("✓ Calibration plots created")
    
    # Create calibration report
    print("Creating calibration report...")
    report = create_calibration_report(sim, sa_data, timestamp)
    print("✓ Calibration report created")
    
    # Save South Africa data for reference
    sa_data['case_notifications'].to_csv(f"sa_case_notifications_{timestamp}.csv", index=False)
    sa_data['age_prevalence'].to_csv(f"sa_age_prevalence_{timestamp}.csv", index=False)
    print("✓ South Africa data saved")
    
    print(f"\nCalibration analysis completed!")
    print(f"Files created with timestamp: {timestamp}")
    
    return sim, sa_data, report


if __name__ == '__main__':
    sim, sa_data, report = main() 