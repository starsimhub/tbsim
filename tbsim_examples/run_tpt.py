import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import starsim as ss
import tbsim
from tbsim.interventions.tpt import (
    TPTRoutine,
    TPTRegimen,
    TPTProduct,
    RegimenCategory,
    REGIMENS,
)
from tbsim.tb_lshtm import TBSL

# shared constants
N_AGENTS  = 2_000
SIM_START = ss.date('2000-01-01')
SIM_STOP  = ss.date('2020-12-31')
DT        = ss.days(7)
INIT_PREV = 0.25   # 25% latent TB at baseline


def make_sim(*interventions):
    """Build a Sim with a fixed random seed for reproducibility."""
    pop = ss.People(n_agents=N_AGENTS)
    tb  = tbsim.TB_LSHTM(name='tb', pars={
        'init_prev': INIT_PREV,
        'beta': ss.peryear(1.5),
    })
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=8), dur=0))
    return ss.Sim(
        people=pop,
        diseases=tb,
        networks=net,
        interventions=list(interventions),
        pars=dict(dt=DT, start=SIM_START, stop=SIM_STOP, rand_seed=42),
        verbose=0,
    )

def _summarize_to_dict(label, sim, itv_key=None):
    tb = sim.diseases.tb

    total_active = int(np.sum(
        np.isin(np.asarray(tb.state), [
            int(TBSL.SYMPTOMATIC),
            int(TBSL.ASYMPTOMATIC),
            int(TBSL.NON_INFECTIOUS),
        ])
    ))

    if itv_key is not None:
        itv           = sim.interventions[itv_key]
        total_started = int(sum(itv.results.n_initiated))
        peak_protected = int(max(itv.results.n_protected))
    else:
        total_started  = 0
        peak_protected = 0

    return dict(
        label=label,
        total_active=total_active,
        total_started=total_started,
        peak_protected=peak_protected,
    )

def _print_summary(summary):
    print(f'\n  {summary["label"]}')
    print(f'    Active TB at end-of-sim : {summary["total_active"]}')
    print(f'    Total TPT initiations   : {summary["total_started"]}')
    print(f'    Peak protected (agents) : {summary["peak_protected"]}')

def build_sims():
    sims = []

    # Scenario 1: Baseline (no TPT)
    sim1 = make_sim()
    sim1.label = 'Baseline — no TPT'
    sims.append(sim1)

    # Scenario 2 — routine 3HP targeting latent TB (INFECTION state)
    itv2 = TPTRoutine(
        product='3HP',
        pars={
            'coverage':        ss.bernoulli(p=0.60),
            'eligible_states': [TBSL.INFECTION],
            'start':           ss.date('2005-01-01'),   # programme starts in 2005
            'stop':            SIM_STOP,
        },
    )
    sim2 = make_sim(itv2)
    sim2.label = 'Routine 3HP — LTBI targeting'
    sims.append(sim2)

    # Scenario 3 — PLHIV-style 6H (universal, no LTBI test)
    itv3 = TPTRoutine(
        product='6H',
        pars={
            'coverage':        ss.bernoulli(p=0.50),
            'eligible_states': None,   # no LTBI test required
            'start':           ss.date('2005-01-01'),
            'stop':            SIM_STOP,
        },
    )
    sim3 = make_sim(itv3)
    sim3.label = 'PLHIV-style 6H — universal'
    sims.append(sim3)

    # Scenario 4 — custom regimen (e.g. a novel 2-month regimen under evaluation)
    # - Shows how to define a new regimen without modifying core code.
    # - Hypothetical: 2-month course, 95% completion, same efficacy as 3HP.
    novel_2m = TPTRegimen(
        name='2M-novel',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.months(2)),
        dur_protection=ss.constant(v=ss.years(1.5)),
        p_complete=ss.bernoulli(p=0.95),
        activation_modifier=ss.constant(v=0.65),
    )
    itv4 = TPTRoutine(
        product=TPTProduct(regimen=novel_2m),
        pars={
            'coverage':        ss.bernoulli(p=0.90),
            'eligible_states': [TBSL.INFECTION],
            'age_range':       [5, 65],
            'start':           ss.date('2005-01-01'),
            'stop':            SIM_STOP,
        },
    )
    sim4 = make_sim(itv4)
    sim4.label = 'Custom 2M regimen — age 5–65'
    sims.append(sim4)

    return sims


def main():
    print('Running scenarios 1–4 in parallel …')
    sims = build_sims()
    msim = ss.MultiSim(sims=sims, label='tpt_scenarios')
    msim.run(parallel=True, shrink=False, reseed=False)

    for sim in msim.sims:
        itv_key = 'tptroutine' if 'tptroutine' in sim.interventions else None
        _print_summary(_summarize_to_dict(sim.label, sim, itv_key=itv_key))

    plot_results = {
        sim.label: {str(k): v for k, v in sim.results.flatten().items()}
        for sim in msim.sims
    }
    tbsim.plot_combined(
        plot_results,
        title='TPT scenarios',
        dark=False,
        n_cols=4,
        heightfold=2,
        keywords=['tb', 'tpt'],
    )

    # Regimen catalog overview:
    print('\n\nAvailable regimens in REGIMENS catalog:')
    print(f'  {"Name":<8} {"Category":<20} {"Treatment":<16} {"p_complete":<12} {"modifier"}')

    for name, reg in REGIMENS.items():
        dur_tx    = reg.dur_treatment.pars.v
        p_comp    = reg.p_complete.pars.p
        modifier  = reg.activation_modifier.pars.v
        print(f'  {name:<8} {reg.category.value:<20} {str(dur_tx):<16} {p_comp:<12.2f} {modifier:.2f}')

if __name__ == '__main__':
    main()
