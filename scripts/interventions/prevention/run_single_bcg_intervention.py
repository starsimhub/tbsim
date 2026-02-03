#!/usr/bin/env python
"""
Pure BCG intervention comparison: Baseline vs BCG vaccination.

Compares a TB simulation without interventions (Baseline) against one with
BCG vaccination. Runs both in parallel using ss.parallel().
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

DEFAULT_SPARS = dict(
    dt=ss.days(7),
    start=ss.date('1975-01-01'),
    stop=ss.date('2030-12-31'),
    rand_seed=123,
    verbose=0,
)
n_agents = 2000

DEFAULT_TBPARS = dict(
    beta=ss.peryear(0.025),
    init_prev=ss.bernoulli(p=0.10),
    dt=ss.days(7),
    start=ss.date('1975-02-01'),
    stop=ss.date('2030-12-31'),
)

age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1],
})


def build_sim(label, bcg_intervention=None, spars=None):
    """
    Build a TB simulation, optionally with BCG intervention.

    Args:
        label: Name for the simulation (e.g. 'Baseline', 'BCG').
        bcg_intervention: Dict of BCG parameters, or None for no BCG.
        spars: Optional sim-level parameters to override defaults.

    Returns:
        ss.Sim: Configured simulation (not yet run).
    """
    spars = {**DEFAULT_SPARS, **(spars or {})}
    tbpars = {**DEFAULT_TBPARS}

    interventions = []
    if bcg_intervention is not None:
        interventions.append(mtb.BCGProtection(pars=bcg_intervention))

    pop = ss.People(
        n_agents=n_agents,
        age_data=age_data,
        extra_states=mtb.get_extrastates(),
    )
    tb = mtb.TB(pars=tbpars)
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    ]

    sim = ss.Sim(
        people=pop,
        networks=networks,
        interventions=interventions,
        diseases=[tb],
        pars=spars,
    )
    sim.label = label
    return sim


def run_bcg_comparison(parallel=True, plot=True, outdir=None):
    """
    Run Baseline vs BCG comparison in parallel and optionally plot.

    Args:
        parallel: If True, run sims in parallel; otherwise sequentially.
        plot: If True, plot comparison and optionally save.
        outdir: Directory to save plots (default: results/bcg_comparison).

    Returns:
        dict: Map of scenario name -> flattened results.
    """
    sim_baseline = build_sim(
        label='Baseline',
        bcg_intervention=None,
        spars={'rand_seed': 123},
    )
    sim_bcg = build_sim(
        label='BCG',
        bcg_intervention=dict(
            # Temporal window
            start=ss.date('1980-01-01'),
            stop=ss.date('2020-12-31'),
            # Eligibility
            age_range=[0, 15],
            coverage=ss.bernoulli(p=0.8),
            efficacy=ss.bernoulli(p=0.8),
            # Delivery: when individuals are vaccinated (years from intervention start)
            # ss.constant(0) = immediate; ss.uniform(0, 5) = rollout over 5 years
            delivery=ss.uniform(0, 5),
            # Protection duration and waning
            immunity_period=ss.years(10),
            waning=ss.expon(scale=1.0),
            # TB risk modifiers (applied to rr_activation, rr_clearance, rr_death)
            activation_modifier=ss.uniform(0.5, 0.65),
            clearance_modifier=ss.uniform(1.3, 1.5),
            death_modifier=ss.uniform(0.05, 0.15),
        ),
        spars={'rand_seed': 456},
    )

    sims = [sim_baseline, sim_bcg]
    print(f"\nRunning {len(sims)} simulations {'in parallel' if parallel else 'sequentially'}...")
    # reseed=True gives each sim a distinct seed (123+0, 456+1) so results differ
    msim = ss.parallel(*sims, parallel=parallel, verbose=0, reseed=True)

    results = {}
    for sim in msim.sims:
        if hasattr(sim, 'results') and sim.results is not None:
            results[sim.label] = sim.results.flatten()
            # Sanity: BCG scenario should have vaccinations
            flat = results[sim.label]
            n_vac = flat.get('bcgprotection_cumulative_vaccinated')
            n_vac = int(n_vac.values[-1]) if n_vac is not None else 0
            extra = f" | BCG vaccinations: {n_vac}" if sim.label == 'BCG' else ""
            print(f"  [OK] {sim.label}{extra}")
        else:
            print(f"  [SKIP] {sim.label} (no results)")

    if plot and results:
        import tbsim.utils.plots as pl
        outdir = outdir or os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'bcg_comparison')
        os.makedirs(outdir, exist_ok=True)
        pl.plot_combined(results, heightfold=2, cmap='viridis', outdir=outdir, dark=False, marker_size=1)
        plt.show()

    return results


if __name__ == '__main__':
    run_bcg_comparison(parallel=True, plot=True)
