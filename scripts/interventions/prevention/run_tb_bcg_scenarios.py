#!/usr/bin/env python
"""
BCG intervention scenarios: compare several BCG vaccination strategies.

Runs Baseline and multiple BCG scenarios in parallel using ss.parallel().
Uses RandomNet only (no household networks).
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

N_AGENTS = 2000


def build_sim(label, bcg_intervention=None, spars=None):
    """
    Build a TB simulation, optionally with BCG intervention.

    Args:
        label: Name for the simulation.
        bcg_intervention: Dict or list of BCG parameters, or None for no BCG.
        spars: Optional sim-level parameters.

    Returns:
        ss.Sim: Configured simulation (not yet run).
    """
    spars = {**DEFAULT_SPARS, **(spars or {})}
    tbpars = {**DEFAULT_TBPARS}

    interventions = []
    if bcg_intervention is not None:
        if isinstance(bcg_intervention, dict):
            interventions.append(mtb.BCGProtection(pars=bcg_intervention))
        else:
            for i, params in enumerate(bcg_intervention):
                p = dict(params)
                p['name'] = f'BCG_{i}'
                interventions.append(mtb.BCGProtection(pars=p))

    pop = ss.People(
        n_agents=N_AGENTS,
        age_data=age_data,
        extra_states=mtb.get_extrastates(),
    )
    tb = mtb.TB(pars=tbpars)
    networks = [ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})]

    sim = ss.Sim(
        people=pop,
        networks=networks,
        interventions=interventions,
        diseases=[tb],
        pars=spars,
    )
    sim.label = label
    return sim


def get_scenarios():
    """BCG-only scenarios for comparison."""
    return [
        ('Baseline', None, 123),
        ('Single BCG (80%, age 0–5)', dict(
            coverage=ss.bernoulli(p=0.8),
            efficacy=ss.bernoulli(p=0.8),
            start=ss.date('1980-01-01'),
            stop=ss.date('2020-12-31'),
            age_range=[0, 5],
            delivery=ss.uniform(0, 5),
            immunity_period=ss.years(10),
            waning=ss.expon(scale=1.0),
            activation_modifier=ss.uniform(0.5, 0.65),
            clearance_modifier=ss.uniform(1.3, 1.5),
            death_modifier=ss.uniform(0.05, 0.15),
        ), 456),
        ('Single BCG (90%, age 0–2)', dict(
            coverage=ss.bernoulli(p=0.9),
            efficacy=ss.bernoulli(p=0.8),
            start=ss.date('1980-01-01'),
            stop=ss.date('2020-12-31'),
            age_range=[0, 2],
            delivery=ss.constant(0),
            immunity_period=ss.years(10),
            waning=ss.expon(scale=1.0),
            activation_modifier=ss.uniform(0.5, 0.65),
            clearance_modifier=ss.uniform(1.3, 1.5),
            death_modifier=ss.uniform(0.05, 0.15),
        ), 789),
        ('Multiple BCG (children + adolescents)', [
            dict(
                coverage=ss.bernoulli(p=0.9),
                efficacy=ss.bernoulli(p=0.8),
                start=ss.date('1980-01-01'),
                stop=ss.date('2020-12-31'),
                age_range=[0, 5],
                delivery=ss.constant(0),
                immunity_period=ss.years(10),
            ),
            dict(
                coverage=ss.bernoulli(p=0.3),
                efficacy=ss.bernoulli(p=0.8),
                start=ss.date('1985-01-01'),
                stop=ss.date('2015-12-31'),
                age_range=[6, 15],
                delivery=ss.uniform(0, 3),
                immunity_period=ss.years(10),
            ),
        ], 1011),
    ]


def run_scenarios(parallel=True, plot=True, outdir=None):
    """Run all BCG scenarios in parallel and optionally plot."""
    scenarios = get_scenarios()
    sims = []
    for label, bcg_pars, seed in scenarios:
        sim = build_sim(
            label=label,
            bcg_intervention=bcg_pars,
            spars={'rand_seed': seed},
        )
        sims.append(sim)

    print(f"\nRunning {len(sims)} BCG scenarios {'in parallel' if parallel else 'sequentially'}...")
    msim = ss.parallel(*sims, parallel=parallel, verbose=0, reseed=True)

    results = {}
    for sim in msim.sims:
        if hasattr(sim, 'results') and sim.results is not None:
            results[sim.label] = sim.results.flatten()
            flat = results[sim.label]
            n_vac = flat.get('bcgprotection_cumulative_vaccinated')
            n_vac = int(n_vac.values[-1]) if n_vac is not None else 0
            extra = f" | BCG vaccinations: {n_vac}" if n_vac else ""
            print(f"  [OK] {sim.label}{extra}")
        else:
            print(f"  [SKIP] {sim.label} (no results)")

    if plot and results:
        import tbsim.utils.plots as pl
        outdir = outdir or os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'bcg_scenarios')
        os.makedirs(outdir, exist_ok=True)
        pl.plot_combined(results, heightfold=2, outdir=outdir, dark=False, marker_size=1)
        plt.show()

    return results


if __name__ == '__main__':
    run_scenarios(parallel=True, plot=True)
