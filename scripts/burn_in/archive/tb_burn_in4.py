"""
Refined TB Prevalence Sweep for Manual Calibration (South Africa)

This script performs a manual calibration sweep of TB transmission dynamics in Starsim/TBsim
to explore plausible endemic equilibrium behavior for South Africa in the absence of 
interventions.

🎯 Objective:
    - Calibrate burn-in dynamics (i.e., rise and settle to endemic steady state)
    - Target approximately:
        • >50% latent TB prevalence
        • ~1% active TB prevalence
    - Qualitative fit to empirical data point: 0.852% active TB prevalence (South Africa, 2018)

🔧 Current Assumptions:
    - No care-seeking, diagnosis, or treatment modeled
    - HIV is not included
    - Simulation starts in 1750 and runs 300 years to allow for equilibrium
    - Model output is not yet calibrated to time-series case data—only steady-state behavior
    - Population demographics include constant birth/death rates

📊 What It Does:
    - Sweeps across a grid of:
        • TB infectiousness (β)
        • Reinfection susceptibility (rel_sus_latentslow)
    - For each parameter combo, it:
        • Runs a simulation
        • Plots active and latent prevalence over time
        • Overlays the 2018 SA data point on each plot
        • Adds an inset focused on the post-1980 period (zoomed to 0–1% active prevalence)
    - Outputs a single PDF figure with all subplots, timestamped with run time
    - Prints runtime diagnostics including total sweep duration

📥 Inputs:
    - Hardcoded ranges for β and rel_sus_latentslow
    - Adjustable population size (default: 8000 agents)

📤 Outputs:
    - PDF file showing active and latent TB prevalence trajectories across parameter grid
    - Console logging of sweep progress and timing

⚠️ Notes:
    - Active prevalence <1% is sensitive to population size; low agent counts may cause extinction
    - This is a burn-in setup only; no programmatic interventions or burden-reduction dynamics yet
    - Future extensions may include sweeps over latent progression rates and fast/slow proportions

"""

import starsim as ss
import tbsim as mtb
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import time

start_wallclock = time.time()
start_datetime = datetime.datetime.now()
print(f"Sweep started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

def run_sim(beta, rel_sus_latentslow, seed=0, years=300, n_agents=8000):
    start_year = 1750
    sim_pars = dict(
        dt=ss.days(30),
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        rand_seed=seed,
        verbose=0,
    )

    people = ss.People(n_agents=n_agents)

    tb_pars = dict(
        beta=ss.probpermonth(beta),
        init_prev=ss.bernoulli(p=0.01),
        rel_sus_latentslow=rel_sus_latentslow,
    )
    tb = mtb.TB(pars=tb_pars)

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    demog = [ss.Births(pars=dict(birth_rate=10)), ss.Deaths(pars=dict(death_rate=10))]

    sim = ss.Sim(
        people=people,
        diseases=tb,
        networks=net,
        demographics=demog,
        pars=sim_pars,
    )
    sim.run()
    return sim

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

def refined_sweep(beta_vals, rel_sus_vals):
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

            # South Africa data point: 2018, 0.852%
            sa_year = datetime.date(2018, 1, 1)
            sa_prevalence = 0.00852  # 0.852% prevalence
            ax.plot(sa_year, sa_prevalence, 'ro', label='2018 SA data (0.852%)')

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

    plt.tight_layout()
    plt.suptitle('Refined TB Prevalence Sweep with Inset Zooms and Latent Overlay', fontsize=16, y=1.02)

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")  # e.g., 2025_06_24_0330
    filename = f"tb_prevalence_sweep_{timestamp}.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')    
    plt.show()

# Run sweep
beta_range = np.linspace(0.03, 0.06, 4)         # [0.03, 0.04, 0.05, 0.06]
rel_sus_range = np.array([0.05, 0.10, 0.15, 0.20])  # capped at 0.20
refined_sweep(beta_range, rel_sus_range)

end_wallclock = time.time()
end_datetime = datetime.datetime.now()
elapsed_minutes = (end_wallclock - start_wallclock) / 60

print(f"Sweep finished at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {elapsed_minutes:.1f} minutes")