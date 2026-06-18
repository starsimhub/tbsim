
import os
import sys
import pandas as pd
import sciris as sc
import starsim as ss

import tbsim
from tbsim.resistance import ( DSTDelivery, DSTDx, DuplicateStrainAnalyzer, Regimen, ResistanceConnector,
                               StrainAwareTPTTx, StrainAwareTx, StrainAwareTxDelivery, StrainResults, StrainSpec)

DRUGS = ['INH', 'RIF', 'BDQ']

DEFAULT_SPARS = dict(
    n_agents = 8000,                       # large enough for resistance events
    start    = ss.date('2000-01-01'),
    stop     = ss.date('2020-01-01'),
    dt       = ss.days(14),
    verbose  = 0,
)

# Per-TPT-dose acquisition probability for a strain that *survives* TPT.
# NOTE: in the current ``StrainAwareTPTTx`` flow these never fire (see write-up
# in chat) — selective_acquisition runs *after* sterilization, by which point
# all surviving strains are already resistant to the regimen's drugs and the
# acquisition resolver has nothing susceptible to mutate. Left here as the
# intended tuning knob; revisit once the TPT resistance path is wired.
TPT_ACQ_INH = 0.10
TPT_ACQ_BDQ = 0.15


def build_strains():
    """Standard five-strain registry: pan-susceptible, INH-R, RIF-R, MDR, BDQ-R."""
    return [
        StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.00, init_prev=0.040), # pan-susceptible
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.010), # INH-mono-resistant
        StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.005), # RIF-mono-resistant
        StrainSpec('mdr',   {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.003), # MDR (INH+RIF)
        StrainSpec('bdq_r', {'INH': 0, 'RIF': 0, 'BDQ': 1}, fitness=0.90, init_prev=0.001), # BDQ-resistant seed
    ]


def build_tb():
    """TB module with bottleneck strain progression and a tiny background acquisition rate."""
    return tbsim.TB(
        strains=build_strains(),
        pars=dict(
            init_prev=ss.bernoulli(0.06),     # ~6% initial latent/active TB
            beta=ss.permonth(0.3),            # boosted from default 0.2 for clearer dynamics
        ),
        progression_mode='bottleneck',
        p_multi=0.8,
        p_random_acquisition=dict(INH=1e-5, RIF=5e-6, BDQ=5e-6),
    )


def first_line_tx(tb):
    """Standard first-line regimen INH+RIF (combine='max'): kills pan, INH-R, RIF-R, BDQ-R; spares MDR."""
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
    """Standard isoniazid TPT: covers any INH-susceptible strain (pan, RIF-R, BDQ-R); misses INH-R and MDR."""
    regimen = Regimen('inh_tpt', drugs=['INH'])
    return StrainAwareTPTTx(
        regimen=regimen, registry=tb._strain_registry,
        p_tpt_acquisition=dict(INH=TPT_ACQ_INH),
    )


def lai_bdq_tpt(tb):
    """LAI BDQ TPT: covers pan, INH-R, RIF-R, MDR; misses BDQ-R; can drive BDQ-R."""
    regimen = Regimen('lai_bdq_tpt', drugs=['BDQ'])
    return StrainAwareTPTTx(
        regimen=regimen, registry=tb._strain_registry,
        p_tpt_acquisition=dict(BDQ=TPT_ACQ_BDQ),
    )


def get_scenarios():
    """Scenarios for the LAI-BDQ-TPT question. ``tpt`` selects the preventive layer (None / 'inh' / 'lai_bdq')."""
    return {
        'Baseline': {
            'name': 'Baseline',
            'tpt': None,
        },
        'INH-TPT': {
            'name': 'INH-TPT',
            'tpt': 'inh',
        },
        'LAI-BDQ-TPT': {
            'name': 'LAI-BDQ-TPT',
            'tpt': 'lai_bdq',
        },
    }


def build_sim(scenario=None, spars=None):
    """Assemble one scenario sim: HSB -> Xpert -> first-line tx -> DST, optionally + TPT."""
    scenario = scenario or {}
    spars = sc.objdict({**DEFAULT_SPARS, **(spars or {})})

    tb = build_tb()

    hsb     = tbsim.HealthSeekingBehavior()
    
    confirm = tbsim.DxDelivery(
        name='confirm', product=tbsim.Xpert(), coverage=0.8,
        result_state='diagnosed')
    
    treat   = first_line_tx(tb)
    
    dst     = DSTDelivery(
                product=DSTDx(tb._strain_registry, drugs=DRUGS,
                            sensitivity=0.95, specificity=0.99))

    interventions = [hsb, confirm, treat, dst]
    tpt = scenario.get('tpt')
    # Coverage 0.9 (vs default 0.5) so the TPT arms actually deliver enough
    # doses for the resistance contrast to be visible in a single seed.
    tpt_pars = dict(coverage=ss.bernoulli(p=0.9))
    if tpt == 'inh':
        interventions.append(tbsim.TPTSimple(product=inh_tpt(tb), pars=tpt_pars))
    elif tpt == 'lai_bdq':
        interventions.append(tbsim.TPTSimple(product=lai_bdq_tpt(tb), pars=tpt_pars))
    elif tpt is not None:
        raise ValueError(f'unknown tpt {tpt!r}')

    sim = tbsim.Sim(
        label         = scenario.get('name', 'scenario'),
        sim_pars      = spars,
        tb_model      = tb,
        interventions = interventions,
        connectors    = ResistanceConnector(),
        analyzers     = [StrainResults(), DuplicateStrainAnalyzer()],
    )
    return sim


def _find(sim_analyzers, cls):
    return next(a for a in sim_analyzers.values() if isinstance(a, cls))


def summarize(sim):
    """One-row scenario summary: per-strain incidence + active, plus tx cascade."""
    tb         = tbsim.get_tb(sim)
    strain_res = _find(sim.analyzers, StrainResults).results
    dup_res    = _find(sim.analyzers, DuplicateStrainAnalyzer).results
    tx_res     = sim.results.get('strainawaretxdelivery')

    row = {'scenario': sim.label}
    for uid in tb._strain_registry.uids:
        row[f'cum_new_carriers_{uid}'] = int(strain_res[f'new_carriers_{uid}'][:].sum())
        row[f'final_active_{uid}']     = int(strain_res[f'n_active_{uid}'][-1])
    row['cum_duplicate_blocked'] = int(dup_res['cum_duplicate_blocked'][-1])
    row['cum_treated'] = int(tx_res['n_treated'].values.sum())   if tx_res else 0
    row['cum_success'] = int(tx_res['cum_success'].values[-1])   if tx_res else 0
    row['cum_failure'] = int(tx_res['cum_failure'].values[-1])   if tx_res else 0
    return row


def print_summary(rows):
    """Print a side-by-side scenario comparison (metrics as rows, scenarios as columns)."""
    if not rows:
        return
    print(pd.DataFrame(rows).set_index('scenario').T.to_string())


def run_scenarios(do_plot=False, savefig=False, fig_path='results/resistance_multisim.png'):
    """Run resistance scenarios and optionally plot outputs."""
    scenarios = get_scenarios()
    sims = []
    summaries = []

    for scenario in scenarios.values():
        print(f'... running {scenario["name"]}')
        sim = build_sim(scenario=scenario)
        sim.run()
        sims.append(sim)
        summaries.append(summarize(sim))

    print()
    print_summary(summaries)
    summary_df = pd.DataFrame(summaries)

    msim = ss.MultiSim(sims=sims)

    return msim, summary_df


if __name__ == '__main__':
    print('Running resistance scenarios...')
    msim, summary = run_scenarios()

    tbsim.plot(
        msim,
        title='Resistance scenarios (TBsim)',
        select=dict(items=[
            # Headline TB curves
            'n_alive', 'n_infectious', 'prevalence_active', 'incidence_kpy', 'new_deaths',
            # Per-strain incidence (cumulative is what the summary reports)
            'new_carriers_pan', 'new_carriers_inh_r', 'new_carriers_rif_r',
            'new_carriers_mdr', 'new_carriers_bdq_r',
            # Per-strain active prevalence
            'n_active_pan', 'n_active_inh_r', 'n_active_rif_r',
            'n_active_mdr', 'n_active_bdq_r',
            # Duplicate-strain diagnostic + treatment cascade
            'cum_duplicate_blocked', 'n_treated', 'cum_success', 'cum_failure',
        ]),
        show=True,
    )
