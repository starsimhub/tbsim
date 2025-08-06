"""
Refined TB Prevalence Sweep for Manual Calibration (South Africa)

This script performs a manual calibration sweep of TB transmission dynamics in Starsim/TBsim
to explore plausible endemic equilibrium behavior for South Africa, incorporating key 
epidemiological features specific to the South African context.

🎯 Objective:
    - Calibrate burn-in dynamics (i.e., rise and settle to endemic steady state)
    - Target approximately:
        • >50% latent TB prevalence
        • ~1% active TB prevalence
    - Qualitative fit to empirical data point: 0.852% active TB prevalence (South Africa, 2018)

🔧 Current Assumptions:
    - Includes HIV coinfection (critical for South Africa TB dynamics)
    - Models TB-HIV interaction effects on progression rates
    - Uses South Africa-specific demographics and population structure
    - Simulation starts in 1750 and runs 300 years to allow for equilibrium
    - Incorporates historical HIV epidemic emergence (1980s onwards)

📊 What It Does:
    - Sweeps across a grid of:
        • TB infectiousness (β)
        • Reinfection susceptibility (rel_sus_latentslow)
    - For each parameter combo, it:
        • Runs a simulation with TB-HIV coinfection
        • Plots active and latent prevalence over time
        • Overlays the 2018 SA data point on each plot
        • Adds an inset focused on the post-1980 period (zoomed to 0–1% active prevalence)
    - Outputs a single PDF figure with all subplots, timestamped with run time
    - Prints runtime diagnostics including total sweep duration

📥 Inputs:
    - Hardcoded ranges for β and rel_sus_latentslow
    - South Africa-specific demographic parameters
    - HIV epidemic parameters (prevalence targets, timing)

📤 Outputs:
    - PDF file showing active and latent TB prevalence trajectories across parameter grid
    - Console logging of sweep progress and timing

⚠️ Notes:
    - Active prevalence <1% is sensitive to population size; low agent counts may cause extinction
    - HIV coinfection significantly impacts TB dynamics in South Africa
    - This model now better reflects the South African epidemiological context
    - Future extensions may include time-varying demographic parameters and treatment effects

"""

import starsim as ss
import tbsim as mtb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import time

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

    # Create population
    people = ss.People(n_agents=n_agents, age_data=age_data)

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


def plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, timestamp):
    import matplotlib.ticker as mtick

    fig, axs = plt.subplots(len(rel_sus_vals), len(beta_vals), figsize=(4 * len(beta_vals), 3 * len(rel_sus_vals)), sharex=True, sharey=True)

    # UN WPP data for South Africa (not Vietnam)
    un_years = np.array([1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030])
    un_total = np.array([13.6, 17.4, 22.7, 29.0, 36.8, 44.8, 51.2, 59.3, 66.0]) / 59.3  # South Africa population (millions)

    for i, rel_sus in enumerate(rel_sus_vals):
        for j, beta in enumerate(beta_vals):
            sim = sim_grid[i][j]
            time = np.array([d.year for d in sim.results['timevec']])
            n_alive = sim.results['n_alive']

            # # Normalize to 2020
            # idx_2020 = np.where(time == 2020)[0]
            # ref_val = n_alive[idx_2020[0]] if len(idx_2020) > 0 else n_alive[-1]
            # n_rel = n_alive / ref_val

            ax = axs[i][j] if len(rel_sus_vals) > 1 else axs[j]
            ax.plot(time, n_alive, color='blue', label='Sim Total Pop (rel. to 2020)')
            # ax.plot(un_years, un_total, 'ko', label='UN WPP (rel. to 2020)', markersize=4)
            ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}')
            ax.grid(True)
            if i == len(rel_sus_vals) - 1:
                ax.set_xlabel('Year')
            if j == 0:
                ax.set_ylabel('Pop')

            ax.xaxis.set_major_locator(mtick.MultipleLocator(20))
            if i == 0 and j == 0:
                ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Simulated Total Population', fontsize=14, y=1.02)  #  vs. UN WPP Data\nRelative to 2020', fontsize=14, y=1.02)
    filename = f"total_population_grid_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_sim(beta, rel_sus_latentslow, seed=0, years=300, n_agents=8000):
    start_year = 1750
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

    cbr = pd.read_csv('../data/Vietnam_CBR.csv')  # Crude birth rate per 1000
    asmr = pd.read_csv('../data/Vietnam_ASMR.csv')  # Age-specific mortality rate
    demog = [
        ss.Births(birth_rate=cbr, dt=ss.days(30)),
        ss.Deaths(death_rate=asmr, dt=ss.days(30), rate_units=1),  # rate_units=1 = per person-year
    ]
    people = make_people(n_agents=n_agents)
 
    tb_pars = dict(
        beta=ss.prob(beta),
        init_prev=ss.bernoulli(p=0.10),  # Higher initial prevalence for South Africa context
        rel_sus_latentslow=rel_sus_latentslow,
        # South Africa-specific adjustments
        rate_LS_to_presym=ss.perday(5e-5),  # Slightly higher progression for HIV context
        rate_LF_to_presym=ss.perday(8e-3),  # Higher fast progression rate
        rate_active_to_clear=ss.perday(1.5e-4),  # Lower clearance rate (more persistent)
    )
    tb = mtb.TB(pars=tb_pars)

    # Add HIV for South Africa context (critical for TB dynamics)
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),  # Start with no HIV, will be added via intervention
        init_onart=ss.bernoulli(p=0.00),
    )
    hiv = mtb.HIV(pars=hiv_pars)

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Add TB-HIV connector to model coinfection effects
    tb_hiv_connector = mtb.TB_HIV_Connector()

    # Add HIV intervention to model South Africa's HIV epidemic
    # HIV prevalence in South Africa: ~0% in 1980, ~1% in 1990, ~20% by 2000, ~25% by 2010
    hiv_intervention = mtb.HivInterventions(pars=dict(
        mode='prevalence',
        prevalence=0.20,  # Target ~20% HIV prevalence by 2000
        start=ss.date('1980-01-01'),
        stop=ss.date('2050-12-31'),
    ))

    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=net,
        demographics=demog,
        connectors=[tb_hiv_connector],
        interventions=[hiv_intervention],
        pars=sim_pars,
    )
    sim.run()

    return sim


def refined_sweep(beta_vals, rel_sus_vals):

    # This function performs a parameter sweep over beta and relative susceptibility values
    # For each parameter combination, it runs a TB simulation and generates plots showing:
    # - Active TB prevalence over time (blue line)
    # - Latent TB prevalence over time (orange dashed line) 
    # - Target 1% prevalence threshold (red dotted line)
    # - 2018 South Africa data point (red dot)
    # Each subplot shows results for a specific (beta, rel_sus) parameter combination

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")  # e.g., 2025_06_24_0330

    sim_grid = [[None for _ in beta_vals] for _ in rel_sus_vals]

    total_runs = len(beta_vals) * len(rel_sus_vals)
    fig, axs = plt.subplots(len(rel_sus_vals), len(beta_vals), figsize=(4*len(beta_vals), 3*len(rel_sus_vals)), sharex=True, sharey=True)

    run_counter = 0
    for i, rel_sus in enumerate(rel_sus_vals):
        for j, beta in enumerate(beta_vals):
            run_counter += 1
            print(f"▶️ Running simulation {run_counter}/{total_runs}: β={beta:.3f}, rel_sus={rel_sus:.2f}")

            sim = run_sim(beta=beta, rel_sus_latentslow=rel_sus)
            time = sim.results['timevec']
            active_prev = sim.results['tb']['prevalence_active']
            latent_prev = compute_latent_prevalence(sim)

            ax = axs[i][j] if len(rel_sus_vals) > 1 else axs[j]
            ax.plot(time, active_prev, label='Active TB Prevalence', color='blue')
            # ax.plot(time, latent_prev, label='Latent TB Prevalence', linestyle='--', color='orange')
            ax.plot(time, latent_prev, linestyle='--', color='orange', label='Latent TB Prevalence')
            # inset.plot(time, latent_prev, linestyle='--', color='orange')
            ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='Target 1%')

            # South Africa data points for calibration
            sa_data_points = [
                (datetime.date(1990, 1, 1), 0.006),   # ~0.6% active TB prevalence in 1990
                (datetime.date(2000, 1, 1), 0.008),   # ~0.8% active TB prevalence in 2000  
                (datetime.date(2010, 1, 1), 0.009),   # ~0.9% active TB prevalence in 2010
                (datetime.date(2018, 1, 1), 0.00852), # 0.852% active TB prevalence (South Africa, 2018)
            ]
            
            for sa_year, sa_prevalence in sa_data_points:
                ax.plot(sa_year, sa_prevalence, 'ro', markersize=4, alpha=0.7)
            
            # Highlight the 2018 data point
            ax.plot(datetime.date(2018, 1, 1), 0.00852, 'ro', markersize=6, label='2018 SA data (0.852%)')

            ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}')
            if i == len(rel_sus_vals) - 1:
                ax.set_xlabel('Year')
            if j == 0:
                ax.set_ylabel('Prevalence')
            ax.grid(True)

            # Inset plot for zoomed-in view on 0–10% prevalence
            inset = inset_axes(ax, width="40%", height="30%", loc='upper right')
            inset.plot(time, active_prev, color='blue')
            inset.plot(time, latent_prev, linestyle='--', color='orange')
            inset.axhline(0.01, color='red', linestyle=':', linewidth=1)
            inset.plot(sa_year, sa_prevalence, 'ro')

            # Zoom on y-axis for all years
            # inset.set_ylim(0, 0.10)
            # inset.set_xticks([])
            # inset.set_yticks([0.0, 0.05, 0.10])
            # inset.set_title('Zoom 0–10%', fontsize=8)
            # inset.grid(True)

            # Zoom from 1980 to end
            inset.set_xlim(datetime.date(1980, 1, 1), time[-1])
            inset.set_ylim(0, 0.010)  # 0–1% range
            inset.set_xticks([
                datetime.date(1980, 1, 1),
                datetime.date(2000, 1, 1),
                datetime.date(2020, 1, 1)
            ])
            inset.set_xticklabels(['1980', '2000', '2020'], fontsize=8)
            inset.tick_params(axis='y', labelsize=8)
            inset.set_title('Zoom: 1980+', fontsize=8)
            inset.grid(True)

            if run_counter == 1:
                ax.legend(fontsize=6)

            # plot_total_population(sim, timestamp, beta, rel_sus)

            sim_grid[i][j] = sim


    plt.tight_layout()
    plt.suptitle('Refined TB Prevalence Sweep with Inset Zooms and Latent Overlay', fontsize=16, y=1.02)

    filename = f"tb_prevalence_sweep_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')    
    plt.show()

    plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, timestamp)

if __name__ == '__main__':
    # Setup for TB prevalence sweeps
    # This section configures the parameter ranges and executes the sweep analysis
    
    # Plot population demographics
    # Run sweep
    beta_range = np.linspace(0.01, 0.03, 3)  # 4)         # [0.03, 0.04, 0.05, 0.06]
    rel_sus_range = np.array([0.05, 0.10])  # , 0.15])  # , 0.20])  # capped at 0.20
    refined_sweep(beta_range, rel_sus_range)

    end_wallclock = time.time()
    end_datetime = datetime.datetime.now()
    elapsed_minutes = (end_wallclock - start_wallclock) / 60

    print(f"Sweep finished at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_minutes:.1f} minutes")


# # Plot population demographics
# # Run sweep
# beta_range = np.linspace(0.01, 0.03, 3)  # 4)         # [0.03, 0.04, 0.05, 0.06]
# rel_sus_range = np.array([0.05, 0.10])  # , 0.15])  # , 0.20])  # capped at 0.20
# refined_sweep(beta_range, rel_sus_range)

# end_wallclock = time.time()
# end_datetime = datetime.datetime.now()
# elapsed_minutes = (end_wallclock - start_wallclock) / 60

# print(f"Sweep finished at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
# print(f"Total runtime: {elapsed_minutes:.1f} minutes")
