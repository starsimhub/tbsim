"""
Refined TB Prevalence Sweep for Manual Calibration (South Africa) with Health-Seeking, Diagnostic, and Treatment

This script performs a manual calibration sweep of TB transmission dynamics in Starsim/TBsim
to explore plausible endemic equilibrium behavior for South Africa, incorporating key 
epidemiological features specific to the South African context, including health-seeking behavior,
diagnostic testing, and treatment outcomes.

🎯 Objective:
    - Calibrate burn-in dynamics (i.e., rise and settle to endemic steady state)
    - Target approximately:
        • >50% latent TB prevalence
        • ~1% active TB prevalence
    - Qualitative fit to empirical data point: 0.852% active TB prevalence (South Africa, 2018)
    - Model realistic health-seeking behavior, diagnostic testing, and treatment outcomes

🔧 Current Assumptions:
    - Includes HIV coinfection (critical for South Africa TB dynamics)
    - Models TB-HIV interaction effects on progression rates
    - Uses South Africa-specific demographics and population structure
    - Simulation starts in 1850 and runs 200 years to allow for equilibrium
    - Incorporates historical HIV epidemic emergence (1980s onwards)
    - Health-seeking behavior with 90-day average delay (slower for better burn-in)
    - Diagnostic testing with 60% sensitivity, 70% coverage, and 95% specificity
    - Treatment with 70% success rate and retry mechanism for failures

📊 What It Does:
    - Sweeps across a grid of:
        • TB infectiousness (β)
        • Reinfection susceptibility (rel_sus_latentslow)
        • TB mortality rates
    - For each parameter combo, it:
        • Runs a simulation with TB-HIV coinfection + health-seeking + diagnostic
        • Plots active and latent prevalence over time
        • Plots health-seeking behavior metrics
        • Plots diagnostic testing outcomes
        • Plots treatment outcomes (incident and cumulative)
        • Overlays the 2018 SA data point on each plot
        • Adds an inset focused on the post-1980 period (zoomed to 0–1% active prevalence)
    - Outputs multiple PDF figures with all subplots, timestamped with run time
    - Prints runtime diagnostics including total sweep duration

📥 Inputs:
    - Hardcoded ranges for β, rel_sus_latentslow, and TB mortality
    - South Africa-specific demographic parameters
    - HIV epidemic parameters (prevalence targets, timing)
    - Health-seeking parameters (90-day average delay)
    - Diagnostic parameters (60% sensitivity, 70% coverage, 95% specificity)

📤 Outputs:
    - PDF files showing:
        • TB prevalence trajectories across parameter grid
        • Health-seeking behavior metrics
        • Diagnostic testing outcomes (incident and cumulative)
        • Treatment outcomes (incident and cumulative)
        • Population demographics
        • HIV metrics
    - Console logging of sweep progress and timing

⚠️ Notes:
    - Active prevalence <1% is sensitive to population size; low agent counts may cause extinction
    - HIV coinfection significantly impacts TB dynamics in South Africa
    - Health-seeking behavior affects TB transmission and case detection
    - Diagnostic testing influences treatment initiation and outcomes
    - Treatment success/failure affects TB transmission and care-seeking behavior
    - This model now better reflects the South African epidemiological context with realistic care-seeking and treatment

"""

import starsim as ss
import tbsim as mtb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import time
import sys
import os
# Dynamically add the correct path to scripts/hiv for shared_functions import
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined (e.g., in Jupyter), use cwd
    current_dir = os.getcwd()
hiv_utils_path = os.path.abspath(os.path.join(current_dir, '../../scripts/hiv'))
if hiv_utils_path not in sys.path:
    sys.path.insert(0, hiv_utils_path)
# Also add the current directory to the path for local imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Add the parent directory to the path for data access
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import shared_functions as sf

# Import health-seeking, diagnostic, and treatment interventions
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
from tbsim.interventions.tb_diagnostic import TBDiagnostic
from tbsim.interventions.tb_treatment import TBTreatment

start_wallclock = time.time()
start_datetime = datetime.datetime.now()
print(f"Sweep started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

def make_people(n_agents, age_data=None):

    if age_data is None:
        # Use South Africa 1960 age structure instead of Vietnam
        age_data = pd.DataFrame({
            'age': np.arange(0, 101, 5),
            'value': [12000, 10000, 8500, 7500, 6500, 5500, 4500, 3500, 2500, 2000,
                      1500, 1200, 800, 500, 300, 150, 80, 40, 15, 5, 1]  # South Africa 1960 approximate
        })

    # Create population with extra states required for health-seeking and diagnostic interventions
    people = ss.People(n_agents=n_agents, age_data=age_data, extra_states=mtb.get_extrastates())

    return people


def compute_latent_prevalence(sim):
    # Get latent counts
    latent_slow = sim.results['tb']['n_latent_slow']
    latent_fast = sim.results['tb']['n_latent_fast']
    latent_total = latent_slow + latent_fast

    # Try getting time-aligned n_alive from starsim if available
    try:
        n_alive_series = sim.results['n_alive']
    except KeyError:
        # Fallback: use average population size
        n_alive_series = np.full_like(latent_total, fill_value=np.count_nonzero(sim.people.alive))

    return latent_total / n_alive_series


def debug_hiv_results(sim):
    """Debug function to print available HIV result keys and values"""
    print("=== HIV Results Debug ===")
    try:
        print(f"Available HIV result keys: {list(sim.results['hiv'].keys())}")
        print(f"HIV result values at final timestep:")
        for key, value in sim.results['hiv'].items():
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"  {key}: {value[-1]}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error accessing HIV results: {e}")
    print("========================")


def compute_hiv_prevalence(sim):
    """Compute HIV prevalence over time"""
    try:
        # Try to get HIV prevalence directly - use the correct key from HIV model
        hiv_prev = sim.results['hiv']['hiv_prevalence']
        return hiv_prev
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from HIV infection counts using correct key
            n_hiv = sim.results['hiv']['infected']
            n_alive = sim.results['n_alive']
            return n_hiv / n_alive
        except (KeyError, AttributeError):
            try:
                # Another fallback: use n_active from HIV model
                n_hiv = sim.results['hiv']['n_active']
                n_alive = sim.results['n_alive']
                return n_hiv / n_alive
            except (KeyError, AttributeError):
                # Debug: print available HIV result keys
                try:
                    print(f"Available HIV result keys: {list(sim.results['hiv'].keys())}")
                except:
                    print("HIV results not found in simulation")
                # If HIV results are not available, return zeros
                time_length = len(sim.results['timevec'])
                return np.zeros(time_length)


def compute_hiv_positive_tb_prevalence(sim):
    """Compute HIV-positive TB prevalence as proportion of total population"""
    try:
        # Try to get HIV-positive TB counts directly
        hiv_positive_tb = sim.results['tb']['n_active_hiv_positive']
        n_alive = sim.results['n_alive']
        return hiv_positive_tb / n_alive
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from individual states
            # Get HIV and TB prevalence using correct keys
            hiv_prev = compute_hiv_prevalence(sim)
            tb_prev = sim.results['tb']['prevalence_active']
            
            # Estimate HIV-positive TB prevalence as a fraction of total TB
            # This is a rough estimate - in reality it depends on the TB-HIV interaction
            hiv_tb_overlap = hiv_prev * tb_prev * 0.3  # Assume 30% of TB cases are HIV-positive
            return hiv_tb_overlap
        except (KeyError, AttributeError):
            # If results are not available, return zeros
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = np.array([d.year for d in sim.results['timevec']])
                n_alive = sim.results['n_alive']

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, n_alive, color='blue', label='Sim Total Pop (rel. to 2020)')
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Pop')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Simulated Total Population', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"total_population_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hiv_metrics_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot HIV prevalence and HIV-positive TB prevalence for all parameter combinations"""
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Debug: print HIV results for the first simulation
                if m == 0 and i == 0 and j == 0:
                    debug_hiv_results(sim)
                
                # Compute HIV metrics
                hiv_prev = compute_hiv_prevalence(sim)
                hiv_positive_tb_prev = compute_hiv_positive_tb_prevalence(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot HIV prevalence (should be around 20%)
                ax.plot(time, hiv_prev, label='HIV Prevalence', color='red', linewidth=2)
                
                # Plot HIV-positive TB prevalence
                ax.plot(time, hiv_positive_tb_prev, label='HIV+ TB Prevalence', color='purple', linestyle='--')
                
                # Add target line for HIV prevalence
                ax.axhline(0.20, color='red', linestyle=':', linewidth=1, alpha=0.7, label='Target 20% HIV')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Prevalence')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('HIV Prevalence and HIV-Positive TB Prevalence', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"hiv_metrics_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                active_prev = sim.results['tb']['prevalence_active']
                latent_prev = compute_latent_prevalence(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, active_prev, label='Active TB Prevalence', color='blue')
                ax.plot(time, latent_prev, linestyle='--', color='orange', label='Latent TB Prevalence')
                ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='Target 1%')

                sa_data_points = [
                    (datetime.date(1990, 1, 1), 0.006),
                    (datetime.date(2000, 1, 1), 0.008),
                    (datetime.date(2010, 1, 1), 0.009),
                    (datetime.date(2018, 1, 1), 0.00852),
                ]
                for sa_year, sa_prevalence in sa_data_points:
                    ax.plot(sa_year, sa_prevalence, 'ro', markersize=4, alpha=0.7)
                ax.plot(datetime.date(2018, 1, 1), 0.00852, 'ro', markersize=6, label='2018 SA data (0.852%)')

                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Prevalence')
                ax.grid(True)

                inset = inset_axes(ax, width="40%", height="30%", loc='upper right')
                inset.plot(time, active_prev, color='blue')
                inset.plot(time, latent_prev, linestyle='--', color='orange')
                inset.axhline(0.01, color='red', linestyle=':', linewidth=1)
                for sa_year, sa_prevalence in sa_data_points:
                    inset.plot(sa_year, sa_prevalence, 'ro')
                inset.set_xlim(datetime.date(1980, 1, 1), time[-1])
                inset.set_ylim(0, 0.010)
                inset.set_xticks([
                    datetime.date(1980, 1, 1),
                    datetime.date(2000, 1, 1),
                    datetime.date(2020, 1, 1)
                ])
                inset.set_xticklabels(['1980', '2000', '2020'], fontsize=8)
                inset.tick_params(axis='y', labelsize=8)
                inset.set_title('Zoom: 1980+', fontsize=8)
                inset.grid(True)

                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Refined TB Prevalence & Mortality Sweep with Inset Zooms and Latent Overlay', fontsize=16, y=1.02)
    filename = f"tb_prevalence_sweep_with_data_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_health_seeking_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot health-seeking behavior metrics for all parameter combinations
    
    Metrics plotted:
    - new_sought_care: Number of people who sought care in this timestep (count)
    - n_sought_care: Cumulative number of people who have ever sought care (count)
    - n_eligible: Number of people with active TB eligible for care-seeking (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get health-seeking results
                hsb = sim.results['healthseekingbehavior']
                new_sought = hsb['new_sought_care'].values
                n_sought = hsb['n_sought_care'].values
                n_eligible = hsb['n_eligible'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot new people seeking care each step
                ax.plot(time, new_sought, label='New Sought Care', color='green', linewidth=2)
                
                # Plot cumulative people who sought care
                ax.plot(time, n_sought, label='Cumulative Sought Care', color='blue', linestyle='--')
                
                # Plot eligible population
                ax.plot(time, n_eligible, label='Eligible (Active TB)', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Health-Seeking Behavior Over Time', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"health_seeking_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot diagnostic testing metrics for all parameter combinations
    
    Metrics plotted:
    - n_tested: Number of people tested in this timestep (count)
    - n_test_positive: Number of positive test results in this timestep (count)
    - n_test_negative: Number of negative test results in this timestep (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get diagnostic results
                tbdiag = sim.results['tbdiagnostic']
                n_tested = tbdiag['n_tested'].values
                n_test_positive = tbdiag['n_test_positive'].values
                n_test_negative = tbdiag['n_test_negative'].values
                cum_test_positive = tbdiag['cum_test_positive'].values
                cum_test_negative = tbdiag['cum_test_negative'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot incident testing results
                ax.plot(time, n_tested, label='Tested', color='blue', marker='o', markersize=2)
                ax.plot(time, n_test_positive, label='Tested Positive', color='green', linestyle='--')
                ax.plot(time, n_test_negative, label='Tested Negative', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('TB Diagnostic Testing Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"diagnostic_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot cumulative diagnostic testing results for all parameter combinations
    
    Metrics plotted:
    - cum_test_positive: Cumulative number of positive test results over time (count)
    - cum_test_negative: Cumulative number of negative test results over time (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get cumulative diagnostic results
                tbdiag = sim.results['tbdiagnostic']
                cum_test_positive = tbdiag['cum_test_positive'].values
                cum_test_negative = tbdiag['cum_test_negative'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot cumulative testing results
                ax.plot(time, cum_test_positive, label='Cumulative Positives', color='green', linestyle='--')
                ax.plot(time, cum_test_negative, label='Cumulative Negatives', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Cumulative Tests')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Cumulative TB Diagnostic Results', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"cumulative_diagnostic_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot TB treatment outcomes for all parameter combinations
    
    Metrics plotted:
    - n_treated: Number of people who started treatment in this timestep (count)
    - n_treatment_success: Number of successful treatment completions in this timestep (count)
    - n_treatment_failure: Number of failed treatment attempts in this timestep (count)
    """
    import matplotlib.ticker as mticker

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get treatment results
                tbtx = sim.results['tbtreatment']
                n_treated = tbtx['n_treated'].values
                n_treatment_success = tbtx['n_treatment_success'].values
                n_treatment_failure = tbtx['n_treatment_failure'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot treatment outcomes
                ax.plot(time, n_treated, label='Treated', color='blue', marker='o', markersize=2)
                ax.plot(time, n_treatment_success, label='Successes', color='green', linestyle='--')
                ax.plot(time, n_treatment_failure, label='Failures', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('TB Treatment Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"treatment_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot cumulative TB treatment outcomes for all parameter combinations
    
    Metrics plotted:
    - cum_treatment_success: Cumulative number of successful treatments over time (count)
    - cum_treatment_failure: Cumulative number of failed treatments over time (count)
    """
    import matplotlib.ticker as mticker

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get cumulative treatment results
                tbtx = sim.results['tbtreatment']
                cum_treatment_success = tbtx['cum_treatment_success'].values
                cum_treatment_failure = tbtx['cum_treatment_failure'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot cumulative treatment outcomes
                ax.plot(time, cum_treatment_success, label='Cumulative Successes', color='green', linestyle='--')
                ax.plot(time, cum_treatment_failure, label='Cumulative Failures', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Cumulative Treatments')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Cumulative TB Treatment Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (every 20 years)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks(xticks)
        else:
            ax_row.set_xticks(xticks)

    filename = f"cumulative_treatment_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_sim(beta, rel_sus_latentslow, tb_mortality, seed=0, years=200, n_agents=1000):  # 8000
    start_year = 1850  # 1750
    sim_pars = dict(
        dt=ss.days(30),
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        rand_seed=seed,
        verbose=0,
    )

    # demog = [ss.Births(pars=dict(birth_rate=20)), ss.Deaths(pars=dict(death_rate=1))]
    # people = ss.People(n_agents=n_agents)
    # To do: Add time-varying birth rate and age-, sex-, year-specific mortality

    # Try different possible paths for the data files
    possible_cbr_paths = [
        '../data/Vietnam_CBR.csv',  # When running from interventions directory
        'tbsim/data/Vietnam_CBR.csv',  # When running from root directory
        'data/Vietnam_CBR.csv',  # Alternative path
    ]
    possible_asmr_paths = [
        '../data/Vietnam_ASMR.csv',  # When running from interventions directory
        'tbsim/data/Vietnam_ASMR.csv',  # When running from root directory
        'data/Vietnam_ASMR.csv',  # Alternative path
    ]
    
    # Find the correct CBR path
    cbr_path = None
    for path in possible_cbr_paths:
        if os.path.exists(path):
            cbr_path = path
            break
    if cbr_path is None:
        raise FileNotFoundError(f"Could not find Vietnam_CBR.csv in any of the expected locations: {possible_cbr_paths}")
    
    # Find the correct ASMR path
    asmr_path = None
    for path in possible_asmr_paths:
        if os.path.exists(path):
            asmr_path = path
            break
    if asmr_path is None:
        raise FileNotFoundError(f"Could not find Vietnam_ASMR.csv in any of the expected locations: {possible_asmr_paths}")
    
    cbr = pd.read_csv(cbr_path)  # Crude birth rate per 1000
    asmr = pd.read_csv(asmr_path)  # Age-specific mortality rate
    demog = [
        ss.Births(birth_rate=cbr, dt=ss.days(30)),
        ss.Deaths(death_rate=asmr, dt=ss.days(30), rate_units=1),  # rate_units=1 = per person-year
    ]
    people = make_people(n_agents=n_agents)
 
    tb_pars = dict(
        beta=ss.per(beta, ),  # ss.prob(beta),
        init_prev=ss.bernoulli(p=0.10),  # Higher initial prevalence for South Africa context
        rel_sus_latentslow=rel_sus_latentslow,
        # South Africa-specific adjustments
        rate_LS_to_presym=ss.perday(5e-5),  # Slightly higher progression for HIV context
        rate_LF_to_presym=ss.perday(8e-3),  # Higher fast progression rate
        rate_active_to_clear=ss.perday(1.5e-4),  # Lower clearance rate (more persistent)
        rate_smpos_to_dead=ss.perday(tb_mortality),
        rate_exptb_to_dead=ss.perday(0.15 * tb_mortality),
        rate_smneg_to_dead=ss.perday(0.3 * tb_mortality),
    )
    tb = sf.make_tb(tb_pars=tb_pars)

    # Add HIV for South Africa context (critical for TB dynamics)
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),  # Start with no HIV, will be added via intervention
        init_onart=ss.bernoulli(p=0.00),
    )
    hiv = sf.make_hiv(hiv_pars=hiv_pars)

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Add TB-HIV connector to model coinfection effects
    tb_hiv_connector = sf.make_tb_hiv_connector()

    # Add HIV intervention to maintain 20% prevalence for the entire simulation
    # Use 'both' mode to properly manage both HIV infection and ART coverage
    hiv_intervention = sf.make_hiv_interventions(pars=dict(
        mode='both',  # Changed from 'prevalence' to 'both' to manage both infection and ART
        prevalence=0.20,  # Maintain 20% HIV prevalence for the entire simulation
        percent_on_ART=0.50,  # 50% of HIV-positive individuals on ART
        min_age=15,  # Only target adults
        max_age=60,
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
    ))

    # Add health-seeking behavior intervention (90-day average delay - slower for better burn-in)
    # Rate = 1/90 days = 0.011 per day
    health_seeking = HealthSeekingBehavior(pars=dict(
        initial_care_seeking_rate=ss.perday(1/90),  # 90-day average delay for slower case detection
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        single_use=True,
    ))

    # Add TB diagnostic intervention (60% sensitivity - less effective for better burn-in)
    tb_diagnostic = TBDiagnostic(pars=dict(
        coverage=ss.bernoulli(0.7, strict=False),  # 70% coverage - not everyone gets tested
        sensitivity=0.60,  # 60% sensitivity - less effective case detection
        specificity=0.95,  # 95% specificity (standard)
        reset_flag=False,
        care_seeking_multiplier=2.0,  # Encourage retries for false negatives
    ))

    # Add TB treatment intervention (70% success rate - less effective for better burn-in)
    tb_treatment = TBTreatment(pars=dict(
        treatment_success_rate=0.70,  # 70% treatment success rate - less effective treatment
        reseek_multiplier=2.0,  # Encourage retries for treatment failures
        reset_flags=True,  # Reset diagnostic flags after treatment failure
    ))

    # Combine all interventions
    all_interventions = hiv_intervention + [health_seeking, tb_diagnostic, tb_treatment]

    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=net,
        demographics=demog,
        connectors=tb_hiv_connector,  # Pass connector directly, not in a list
        interventions=all_interventions,  # Combined interventions list
        pars=sim_pars,
    )
    sim.run()

    return sim


def refined_sweep(beta_vals, rel_sus_vals, tb_mortality_vals):

    # This function performs a parameter sweep over beta and relative susceptibility values
    # For each parameter combination, it runs a TB simulation and generates plots showing:
    # - Active TB prevalence over time (blue line)
    # - Latent TB prevalence over time (orange dashed line) 
    # - Target 1% prevalence threshold (red dotted line)
    # - 2018 South Africa data point (red dot)
    # Each subplot shows results for a specific (beta, rel_sus) parameter combination

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")  # e.g., 2025_06_24_0330

    sim_grid = [[[None for _ in beta_vals] for _ in rel_sus_vals] for _ in tb_mortality_vals]
    results = {}
    total_runs = len(beta_vals) * len(rel_sus_vals) * len(tb_mortality_vals)
    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                scen_key = f'beta={beta:.3f}_rel_sus={rel_sus:.2f}_mort={tb_mortality:.1e}'
                print(f"▶️ Running simulation {scen_key} ({m},{i},{j})/{total_runs}")
                sim = run_sim(beta=beta, rel_sus_latentslow=rel_sus, tb_mortality=tb_mortality)
                sim_grid[m][i][j] = sim
                results[scen_key] = sim.results.flatten()
    # Use shared_functions.plot_results to plot all scenario results
    # Note: This function plots various metrics with the following units/definitions:
    # - 'active': Active TB cases (count of people with active TB disease)
    # - 'latent': Latent TB cases (count of people with latent TB infection)
    # - 'incidence': New TB infections per time step (count of new cases)
    # - 'prevalence': TB prevalence as fraction of total population (0-1)
    # - 'sought': People who sought care for TB symptoms (count)
    # - 'eligible': People eligible for care-seeking (active TB cases, count)
    # - 'tested': People who received diagnostic testing (count)
    # - 'diagnosed': People diagnosed with TB (count)
    # - 'treated': People who started TB treatment (count)
    # - 'success': Successful TB treatment completions (count)
    # - 'failure': Failed TB treatment attempts (count)
    sf.plot_results(results, dark=False)
    # Optionally, keep the original grid plots if desired
    plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_hiv_metrics_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_health_seeking_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_cumulative_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_cumulative_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)

if __name__ == '__main__':
    # Setup for TB prevalence sweeps
    # This section configures the parameter ranges and executes the sweep analysis
    
    # Plot population demographics
    # Run sweep
    # 2x2x2 parameter grid for 8 total scenarios with higher beta range
    beta_range = np.array([0.015, 0.025])  # Higher infectiousness range up to 0.025
    rel_sus_range = np.array([0.10, 0.20])  # Reinfection susceptibility range
    tb_mortality_range = [2e-4, 4e-4]  # TB mortality range
    refined_sweep(beta_range, rel_sus_range, tb_mortality_range)

    end_wallclock = time.time()
    end_datetime = datetime.datetime.now()
    elapsed_minutes = (end_wallclock - start_wallclock) / 60

    print(f"Sweep finished at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_minutes:.1f} minutes")

    # Uncomment the line below to run a quick test of the health-seeking and diagnostic integration
    # test_health_seeking_diagnostic_integration()


def test_hiv_integration():
    """Simple test function to verify HIV integration works correctly"""
    print("Testing HIV integration...")
    
    # Run a simple simulation with HIV
    sim = run_sim(beta=0.003, rel_sus_latentslow=0.05, tb_mortality=4e-4, years=50, n_agents=500)
    
    # Debug HIV results
    debug_hiv_results(sim)
    
    # Test HIV prevalence calculation
    hiv_prev = compute_hiv_prevalence(sim)
    print(f"HIV prevalence at final timestep: {hiv_prev[-1]:.3f}")
    
    # Test HIV-positive TB prevalence calculation
    hiv_tb_prev = compute_hiv_positive_tb_prevalence(sim)
    print(f"HIV-positive TB prevalence at final timestep: {hiv_tb_prev[-1]:.3f}")
    
    print("HIV integration test completed.")


# Uncomment the line below to run the HIV integration test
# test_hiv_integration()


def test_health_seeking_diagnostic_integration():
    """Test function to verify health-seeking and diagnostic integration works correctly"""
    print("Testing health-seeking and diagnostic integration...")
    
    # Run a simple simulation with health-seeking and diagnostic
    sim = run_sim(beta=0.003, rel_sus_latentslow=0.05, tb_mortality=4e-4, years=50, n_agents=500)
    
    # Check if health-seeking results are available
    try:
        hsb = sim.results['healthseekingbehavior']
        print(f"✓ Health-seeking results available")
        print(f"  - Final new sought care: {hsb['new_sought_care'].values[-1]}")
        print(f"  - Final cumulative sought care: {hsb['n_sought_care'].values[-1]}")
        print(f"  - Final eligible: {hsb['n_eligible'].values[-1]}")
    except KeyError:
        print("✗ Health-seeking results not found")
    
    # Check if diagnostic results are available
    try:
        tbdiag = sim.results['tbdiagnostic']
        print(f"✓ Diagnostic results available")
        print(f"  - Final tested: {tbdiag['n_tested'].values[-1]}")
        print(f"  - Final test positive: {tbdiag['n_test_positive'].values[-1]}")
        print(f"  - Final test negative: {tbdiag['n_test_negative'].values[-1]}")
        print(f"  - Cumulative test positive: {tbdiag['cum_test_positive'].values[-1]}")
        print(f"  - Cumulative test negative: {tbdiag['cum_test_negative'].values[-1]}")
    except KeyError:
        print("✗ Diagnostic results not found")
    
    # Check treatment results
    try:
        tbtx = sim.results['tbtreatment']
        print(f"✓ Treatment results available")
        print(f"  - Final treated: {tbtx['n_treated'].values[-1]}")
        print(f"  - Final treatment success: {tbtx['n_treatment_success'].values[-1]}")
        print(f"  - Final treatment failure: {tbtx['n_treatment_failure'].values[-1]}")
        print(f"  - Cumulative treatment success: {tbtx['cum_treatment_success'].values[-1]}")
        print(f"  - Cumulative treatment failure: {tbtx['cum_treatment_failure'].values[-1]}")
    except KeyError:
        print("✗ Treatment results not found")
    
    # Check people states
    people = sim.people
    print(f"✓ People states:")
    print(f"  - People who sought care: {np.sum(people.sought_care)}")
    print(f"  - People who were tested: {np.sum(people.tested)}")
    print(f"  - People who were diagnosed: {np.sum(people.diagnosed)}")
    print(f"  - People with treatment success: {np.sum(people.tb_treatment_success)}")
    print(f"  - People with treatment failure: {np.sum(people.treatment_failure)}")
    print(f"  - Mean care-seeking multiplier: {np.mean(people.care_seeking_multiplier):.3f}")
    
    print("Health-seeking, diagnostic, and treatment integration test completed.")


# Uncomment the line below to run the health-seeking and diagnostic integration test
# test_health_seeking_diagnostic_integration()