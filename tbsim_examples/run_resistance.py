"""
Resistance overlay demo: compare a status-quo first-line-only scenario to a
scenario adding long-acting injectable (LAI) BDQ as preventive therapy, and
report drug-resistant TB incidence by phenotype.

This is the canonical decision-relevant scenario from the TBsim resistance
tech-spec review (LAI BDQ → does it accelerate BDQ resistance?). Three
sims are run:

1. **Baseline**: first-line treatment only (INH+RIF). No resistance pressure
   from preventive therapy beyond background random acquisition.
2. **INH-TPT**: standard isoniazid preventive therapy on latent contacts.
3. **LAI-BDQ-TPT**: LAI BDQ preventive therapy. Hypothesis: this raises
   BDQ-resistance prevalence relative to INH-TPT through TPT-driven
   acquisition.

The example prints per-scenario summaries of:

- Total active TB incidence
- Active TB incidence by carried strain phenotype
- Cumulative duplicate-strain blocks (Decision 3 diagnostic)

Run with::

    python tbsim_examples/run_resistance.py
"""

import os
import sys
from collections import OrderedDict

# Ensure the workspace's local tbsim shadows any stale site-packages install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import starsim as ss

import tbsim
from tbsim.resistance import (
    DSTDelivery, DSTDx, DuplicateStrainAnalyzer, Regimen, ResistanceConnector,
    StrainAwareTPTTx, StrainAwareTx, StrainAwareTxDelivery, StrainResults,
    StrainSpec,
)


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

DRUGS = ['INH', 'RIF', 'BDQ']


def build_strains():
    """Standard four-strain registry: pan-susceptible, INH-R, RIF-R, MDR, BDQ-R."""
    return [
        StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.00, init_prev=0.04),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.005),
        StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.002),
        StrainSpec('mdr',   {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.001),
        StrainSpec('bdq_r', {'INH': 0, 'RIF': 0, 'BDQ': 1}, fitness=0.90, init_prev=0.0),
    ]


def build_tb():
    """TB module with bottleneck progression and a low background acquisition rate."""
    return tbsim.TB(
        strains=build_strains(),
        pars=dict(init_prev=ss.bernoulli(0.04)),
        progression_mode='bottleneck',
        p_multi=0.8,                         # mostly bottleneck, some carry-over
        p_random_acquisition=dict(           # tiny background random acquisition
            INH=1e-5, RIF=5e-6, BDQ=5e-6,
        ),
    )


def first_line_tx(tb):
    """Standard first-line regimen: INH + RIF (kills pan and BDQ-R; spares MDR)."""
    regimen = Regimen(
        'first_line', drugs=['INH', 'RIF'],
        per_drug_efficacy={'INH': 0.95, 'RIF': 0.95},
    )
    product = StrainAwareTx(
        regimen=regimen, registry=tb._strain_registry,
        p_selective_acquisition=dict(INH=0.05, RIF=0.02),
        adherence=0.85,
    )
    return StrainAwareTxDelivery(product=product)


def inh_tpt(tb):
    """Standard isoniazid TPT: covers pan and RIF-R; misses INH-R, MDR, BDQ-R."""
    regimen = Regimen('inh_tpt', drugs=['INH'])
    product = StrainAwareTPTTx(
        regimen=regimen, registry=tb._strain_registry,
        p_tpt_acquisition=dict(INH=0.01),
    )
    return product


def lai_bdq_tpt(tb):
    """LAI BDQ TPT: covers pan, INH-R, RIF-R, MDR; misses BDQ-R; can drive BDQ-R."""
    regimen = Regimen('lai_bdq_tpt', drugs=['BDQ'])
    product = StrainAwareTPTTx(
        regimen=regimen, registry=tb._strain_registry,
        p_tpt_acquisition=dict(BDQ=0.02),
    )
    return product


# ---------------------------------------------------------------------------
# Sim builders
# ---------------------------------------------------------------------------

def make_sim(scenario):
    """Build a sim for one scenario: 'baseline', 'inh_tpt', or 'lai_bdq_tpt'.

    The cascade in each scenario:

    1. ``HealthSeekingBehavior`` → ``Xpert`` (diagnosis) → ``StrainAwareTxDelivery`` (first line).
    2. For 'inh_tpt' / 'lai_bdq_tpt', a ``TPTSimple`` delivery layers preventive
       therapy on latently-infected agents.

    DST is added after diagnosis so the observed resistance phenotype is
    populated for downstream analysis even though Phase 2 does not yet
    auto-route by phenotype.
    """
    tb = build_tb()

    hsb     = tbsim.HealthSeekingBehavior()
    confirm = tbsim.DxDelivery(
        name='confirm', product=tbsim.Xpert(), coverage=0.8,
        result_state='diagnosed',
    )
    treat   = first_line_tx(tb)
    dst     = DSTDelivery(
        product=DSTDx(tb._strain_registry, drugs=DRUGS,
                      sensitivity=0.95, specificity=0.99),
    )

    interventions = [hsb, confirm, treat, dst]
    if scenario == 'inh_tpt':
        interventions.append(tbsim.TPTSimple(product=inh_tpt(tb)))
    elif scenario == 'lai_bdq_tpt':
        interventions.append(tbsim.TPTSimple(product=lai_bdq_tpt(tb)))
    elif scenario != 'baseline':
        raise ValueError(f'unknown scenario {scenario!r}')

    sim = tbsim.Sim(
        n_agents      = 2000,
        start         = ss.date('2000-01-01'),
        stop          = ss.date('2010-01-01'),
        dt            = ss.days(14),
        tb_model      = tb,
        interventions = interventions,
        connectors    = ResistanceConnector(),
        analyzers     = [StrainResults(), DuplicateStrainAnalyzer()],
    )
    sim.pars.verbose = 0
    return sim


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarize(sim, scenario):
    """Compute per-scenario summary metrics from the StrainResults analyzer."""
    tb = tbsim.get_tb(sim)
    strain_res = next(a for a in sim.analyzers.values() if isinstance(a, StrainResults))
    dup_res    = next(a for a in sim.analyzers.values() if isinstance(a, DuplicateStrainAnalyzer))

    summary = OrderedDict()
    summary['scenario'] = scenario
    for uid in tb._strain_registry.uids:
        new_carriers = strain_res.results[f'new_carriers_{uid}'][:]
        n_active = strain_res.results[f'n_active_{uid}'][:]
        summary[f'cum_new_carriers_{uid}'] = int(new_carriers.sum())
        summary[f'final_active_{uid}']     = int(n_active[-1])
    summary['cum_duplicate_blocked'] = int(
        dup_res.results['cum_duplicate_blocked'][-1]
    )
    # Treatment cascade summary
    try:
        tx_r = sim.results['strainawaretxdelivery']
        summary['cum_treated'] = int(tx_r['n_treated'].values.sum())
        summary['cum_success'] = int(tx_r['cum_success'].values[-1])
        summary['cum_failure'] = int(tx_r['cum_failure'].values[-1])
    except (KeyError, AttributeError):
        summary['cum_treated'] = 0
        summary['cum_success'] = 0
        summary['cum_failure'] = 0
    return summary


def print_summary(rows):
    """Print a side-by-side scenario comparison."""
    if not rows:
        return
    keys = list(rows[0].keys())
    col_w = max(max(len(k) for k in keys), 22)
    sce_w = max(12, max(len(str(r['scenario'])) for r in rows) + 2)
    header = f'{"metric":<{col_w}}'
    for r in rows:
        header += f'{str(r["scenario"]):>{sce_w}}'
    print(header)
    print('-' * len(header))
    for k in keys:
        if k == 'scenario':
            continue
        line = f'{k:<{col_w}}'
        for r in rows:
            line += f'{r[k]:>{sce_w}}'
        print(line)
    return


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    scenarios = ['baseline', 'inh_tpt', 'lai_bdq_tpt']
    summaries = []
    for sce in scenarios:
        print(f'... running {sce}')
        sim = make_sim(sce)
        sim.run()
        summaries.append(summarize(sim, sce))
    print()
    print_summary(summaries)
    return summaries


if __name__ == '__main__':
    main()
