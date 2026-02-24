"""
Sample script for the revised TPT intervention (tpt2).

Demonstrates four scenarios:
  1. Baseline — no TPT
  2. Routine 3HP targeting latent TB (LTBI population)
  3. PLHIV-style routine 6H (all alive agents, no LTBI test required)
  4. Custom regimen with a user-defined completion rate

Each scenario runs a 20-year simulation and prints a brief summary of
incidence avoided, peak protection coverage, and total initiations.

Run from project root:
    python tbsim_examples/run_tpt2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import starsim as ss
import tbsim
from tbsim.interventions.tpt import (
    TPTRoutine,
    TPTProduct,
    REGIMENS,
)
from tbsim.tb_lshtm import TBSL


# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise(label, sim, itv_key=None):
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

    print(f'\n  {label}')
    print(f'    Active TB at end-of-sim : {total_active}')
    print(f'    Total TPT initiations   : {total_started}')
    print(f'    Peak protected (agents) : {peak_protected}')


# ---------------------------------------------------------------------------
# Scenario 1 — baseline
# ---------------------------------------------------------------------------

print('Running scenario 1: Baseline (no TPT) …')
sim1 = make_sim()
sim1.run()
summarise('Baseline — no TPT', sim1)


# ---------------------------------------------------------------------------
# Scenario 2 — routine 3HP targeting latent TB (INFECTION state)
#
# Uses the WHO-preferred short-course rifapentine+isoniazid regimen.
# Coverage of 60% with the default active-TB exclusion and p_complete=0.93.
# ---------------------------------------------------------------------------

print('\nRunning scenario 2: Routine 3HP (LTBI targeting, 60% coverage) …')
itv2 = TPTRoutine(
    product='3HP',
    pars={
        'coverage':        ss.bernoulli(p=0.60),
        'eligible_states': [TBSL.INFECTION],
        'start':           ss.date('2005-01-01'),   # programme starts in 2005
    },
)
sim2 = make_sim(itv2)
sim2.run()
summarise('Routine 3HP — LTBI targeting', sim2, itv_key='tptroutine')


# ---------------------------------------------------------------------------
# Scenario 3 — PLHIV-style 6H (universal, no LTBI test)
#
# WHO Recommendation 1 (2024): PLHIV receive TPT regardless of LTBI test result.
# Set eligible_states=None to target all alive, non-active-TB agents.
# Uses 6H (isoniazid) with lower completion rate (0.70) vs. 3HP.
# ---------------------------------------------------------------------------

print('\nRunning scenario 3: PLHIV-style 6H (universal, 50% coverage) …')
itv3 = TPTRoutine(
    product='6H',
    pars={
        'coverage':        ss.bernoulli(p=0.50),
        'eligible_states': None,   # no LTBI test required
        'start':           ss.date('2005-01-01'),
    },
)
sim3 = make_sim(itv3)
sim3.run()
summarise('PLHIV-style 6H — universal', sim3, itv_key='tptroutine')


# ---------------------------------------------------------------------------
# Scenario 4 — custom regimen (e.g. a novel 2-month regimen under evaluation)
#
# Shows how to define a new regimen without modifying core code.
# Hypothetical: 2-month course, 95% completion, same efficacy as 3HP.
# ---------------------------------------------------------------------------

print('\nRunning scenario 4: Custom 2-month regimen (90% coverage) …')
itv4 = TPTRoutine(
    product=TPTProduct(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(2)),
            dur_protection=ss.constant(v=ss.years(1.5)),
            p_complete=ss.bernoulli(p=0.95),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    pars={
        'coverage':        ss.bernoulli(p=0.90),
        'eligible_states': [TBSL.INFECTION],
        'age_range':       [5, 65],
        'start':           ss.date('2005-01-01'),
    },
)
sim4 = make_sim(itv4)
sim4.run()
summarise('Custom 2M regimen — age 5–65', sim4, itv_key='tptroutine')


# ---------------------------------------------------------------------------
# Regimen catalog overview
# ---------------------------------------------------------------------------

print('\n\nAvailable regimens in REGIMENS catalog:')
print(f'  {"Name":<8} {"Treatment":<16} {"p_complete":<12} {"effects"}')
print('  ' + '-' * 60)
for name, spec in REGIMENS.items():
    p        = spec['pars']
    dur_tx   = p['dur_treatment'].pars.v
    p_comp   = p['p_complete'].pars.p
    effects  = ', '.join(f'{k}={v.pars.v:.2f}' for k, v in spec['effects'].items())
    print(f'  {name:<8} {str(dur_tx):<16} {p_comp:<12.2f} {effects}')
