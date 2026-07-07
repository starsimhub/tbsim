"""INH/BDQ TPT trade-off example using MultiStrainTB and strain-aware interventions."""

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import starsim as ss

import tbsim
from tbsim.plots import _normalize_results
from tbsim.resistance import ( DSTDelivery, DSTDx, DuplicateStrainAnalyzer, MultiStrainTB, Regimen, ResistanceConnector,
                               StrainAwareTPTTx, StrainAwareTx, StrainAwareTxDelivery, StrainResults, StrainSpec)

DRUGS = ['INH', 'RIF', 'BDQ']

DEFAULT_SPARS = dict(
    n_agents = 8000,                       # large enough for resistance events
    start    = ss.date('2000-01-01'),
    stop     = ss.date('2020-01-01'),
    dt       = ss.days(14),
    verbose  = 0,
)


TPT_ACQ_INH = 0.20
TPT_ACQ_BDQ = 0.30

TPT_STATE_MODIFIERS = dict(infection=1.0, non_infectious=1.0,
                           asymptomatic=1.0, symptomatic=1.0,
                           treatment=0.0, cleared=0.0)


def build_strains():
    """Standard five-strain catalog: pan-susceptible, INH-R, RIF-R, MDR, BDQ-R."""
    return [
        StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}, fitness=1.00, init_prev=0.040), # pan-susceptible
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95, init_prev=0.010), # INH-mono-resistant
        StrainSpec('rif_r', {'INH': 0, 'RIF': 1, 'BDQ': 0}, fitness=0.90, init_prev=0.005), # RIF-mono-resistant
        StrainSpec('mdr',   {'INH': 1, 'RIF': 1, 'BDQ': 0}, fitness=0.85, init_prev=0.003), # MDR (INH+RIF)
        StrainSpec('bdq_r', {'INH': 0, 'RIF': 0, 'BDQ': 1}, fitness=0.90, init_prev=0.001), # BDQ-resistant seed
    ]


def build_tb():
    """TB module with bottleneck strain progression and a tiny background acquisition rate."""
    return MultiStrainTB(
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
        regimen=regimen, catalog=tb._strain_catalog,
        p_selective_acquisition=dict(INH=0.05, RIF=0.02),
        adherence=0.85,
    )
    return StrainAwareTxDelivery(product=product)


def inh_tpt(tb):
    """Standard isoniazid TPT: covers any INH-susceptible strain (pan, RIF-R, BDQ-R); misses INH-R and MDR."""
    regimen = Regimen('inh_tpt', drugs=['INH'])
    return StrainAwareTPTTx(
        regimen=regimen, catalog=tb._strain_catalog,
        p_tpt_acquisition=dict(INH=TPT_ACQ_INH),
        acq_state_modifiers=TPT_STATE_MODIFIERS,
    )


def lai_bdq_tpt(tb):
    """LAI BDQ TPT: covers pan, INH-R, RIF-R, MDR; misses BDQ-R; can drive BDQ-R."""
    regimen = Regimen('lai_bdq_tpt', drugs=['BDQ'])
    return StrainAwareTPTTx(
        regimen=regimen, catalog=tb._strain_catalog,
        p_tpt_acquisition=dict(BDQ=TPT_ACQ_BDQ),
        acq_state_modifiers=TPT_STATE_MODIFIERS,
    )


def get_scenarios():
    """Scenarios for the LAI-BDQ-TPT question. ``tpt`` selects the preventive layer (None / 'inh' / 'lai_bdq')."""
    return {
        'Baseline': {
            'name': 'Baseline (No TPT)',
            'tpt': None,
        },
        'INH-TPT': {
            'name': 'Intervention: Isoniazid TPT',
            'tpt': 'inh',
        },
        'LAI-BDQ-TPT': {
            'name': 'Intervention: Long-Acting Injectable Bedaquiline',
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
                product=DSTDx(tb._strain_catalog, drugs=DRUGS,
                            sensitivity=0.95, specificity=0.99))

    interventions = [hsb, confirm, treat, dst]
    tpt = scenario.get('tpt')
    # Coverage 0.9 (vs default 0.5) so TPT arms deliver enough doses for the
    # resistance contrast to show up in a single seed.
    tpt_pars = dict(coverage=ss.bernoulli(p=0.9))
    if tpt is not None:
        # Force every effective TPT dose down the sterilization path
        # (default p_sterilize=0.0 routes everything to suppression, which
        # never exercises the strain-aware mutation / clearance logic).
        product = inh_tpt(tb) if tpt == 'inh' else lai_bdq_tpt(tb) if tpt == 'lai_bdq' else None
        if product is None:
            raise ValueError(f'unknown tpt {tpt!r}')
        product.pars.p_sterilize = ss.bernoulli(p=1.0)
        interventions.append(tbsim.TPTSimple(product=product, pars=tpt_pars))

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
    for analyzer in sim_analyzers.values():
        if isinstance(analyzer, cls) or analyzer.__class__.__name__ == cls.__name__:
            return analyzer
    raise RuntimeError(f'Could not find analyzer {cls.__name__!r}')


def summarize(sim):
    """One-row scenario summary: per-strain incidence + active, plus tx cascade."""
    tb         = tbsim.get_tb(sim)
    strain_res = _find(sim.analyzers, StrainResults).results
    dup_res    = _find(sim.analyzers, DuplicateStrainAnalyzer).results
    tx_res     = sim.results.get('strainawaretxdelivery')

    row = {'scenario': sim.label}
    for uid in tb._strain_catalog.uids:
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


STRAIN_LABELS = {
    'pan': 'Susceptible',
    'inh_r': 'INH-resistant',
    'rif_r': 'RIF-resistant',
    'mdr': 'MDR',
    'bdq_r': 'BDQ-resistant',
}

ABREV = (
    'Lines: StrainResults via _normalize_results after MultiSim.run(). '
    'Bars: summarize() sums new_carriers_* by strain.\n'
    'INH = isoniazid; RIF = rifampicin; BDQ = bedaquiline; MDR = multidrug-resistant (INH+RIF); '
    'TPT = tuberculosis preventive treatment; LAI = long-acting injectable bedaquiline; DST = drug susceptibility testing.'
)

PLOT_SCENARIOS = [
    ('Baseline (No TPT)', '--', '#666666'),
    ('Intervention: Isoniazid TPT', '-', '#C44E52'),
    ('Intervention: Long-Acting Injectable Bedaquiline', '-', '#4C72B0'),
]
PLOT_SCENARIO_LABELS = {
    'Baseline (No TPT)': 'No TPT',
    'Intervention: Isoniazid TPT': 'Isoniazid TPT',
    'Intervention: Long-Acting Injectable Bedaquiline': 'LAI Bedaquiline TPT',
}


def _xy(res):
    if res is None or not hasattr(res, 'timevec'):
        return None, None
    return np.asarray(res.timevec), np.asarray(res.values).ravel()


def _finish(fig, filename=None, show=True, bottom=0.22):
    fig.text(
        0.5, 0.01, ABREV, ha='center', va='bottom', fontsize=7.5, transform=fig.transFigure,
        bbox=dict(boxstyle='square,pad=0.35', facecolor='#f7f7f7', edgecolor='#ddd'),
    )
    fig.subplots_adjust(bottom=bottom)
    if filename:
        sc.savefig(sc.makefilepath(filename, makedirs=True), fig=fig)
    if show:
        pl.show()
    return fig


def plot_results(msim, summary_df, tradeoff_fig=None, summary_fig=None, show=True):
    """TPT trade-off panels and end-of-run carrier summary."""
    flat = _normalize_results(msim)
    panels = [
        ('n_carriers_pan', 'Susceptible carriers'),
        ('n_carriers_inh_r', 'INH-resistant carriers'),
        ('n_carriers_bdq_r', 'BDQ-resistant carriers'),
        ('n_active_mdr', 'Active MDR cases'),
    ]
    fig, axs = pl.subplots(2, 2, figsize=(8, 5.5), sharex=True)
    for ax, (key, title) in zip(axs.flat, panels):
        for scen, ls, color in PLOT_SCENARIOS:
            x, y = _xy(flat.get(scen, {}).get(key))
            if x is None:
                continue
            ax.plot(x, y, ls, color=color, lw=1.8, label=PLOT_SCENARIO_LABELS.get(scen, scen))
        ax.set(title=title, ylabel='People')
        ax.grid(True, alpha=0.25, linestyle=':')
    axs[1, 0].set_xlabel('Year')
    axs[1, 1].set_xlabel('Year')
    fig.suptitle('TPT trade-off: benefit vs resistance risk', fontsize=11, y=0.98)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.22), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    _finish(fig, tradeoff_fig, show, bottom=0.28)

    cols = [c for c in summary_df.columns if c.startswith('cum_new_carriers_')]
    if not cols:
        return
    strains = [c.removeprefix('cum_new_carriers_') for c in cols]
    plot_df = summary_df.set_index('scenario')[cols].T
    plot_df.index = [STRAIN_LABELS.get(s, s) for s in strains]
    plot_df.columns = [PLOT_SCENARIO_LABELS.get(c, c) for c in plot_df.columns]
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_df.plot(kind='bar', ax=ax, rot=0, width=0.8, color=[c for _, _, c in PLOT_SCENARIOS])
    ax.set(title='New carriers by strain (end of simulation)', xlabel='', ylabel='People')
    ax.legend(title='Scenario', fontsize=8)
    fig.tight_layout()
    _finish(fig, summary_fig, show)


def run_scenarios(do_plot=False, savefig=False,
                  tradeoff_fig_path='results/resistance_tradeoff.png',
                  summary_fig_path='results/resistance_summary.png'):
    """Run resistance scenarios and optionally plot outputs."""
    scenarios = get_scenarios()
    sims = []

    for scenario in scenarios.values():
        print(f'... building {scenario["name"]}')
        sim = build_sim(scenario=scenario)
        sims.append(sim)

    print(f'... running {len(sims)} scenarios with ss.parallel()')
    msim = ss.parallel(*sims, verbose=0)
    summaries = [summarize(sim) for sim in msim.sims]

    print()
    print_summary(summaries)
    summary_df = pd.DataFrame(summaries)

    if do_plot or savefig:
        plot_results(
            msim,
            summary_df,
            tradeoff_fig=sc.makefilepath(tradeoff_fig_path, makedirs=True) if savefig else None,
            summary_fig=sc.makefilepath(summary_fig_path, makedirs=True) if savefig else None,
            show=do_plot,
        )

    return msim, summary_df


if __name__ == '__main__':
    print('Running resistance scenarios...')
    run_scenarios(do_plot=True, savefig=True)
