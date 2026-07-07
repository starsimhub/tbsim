"""Spec-validation scenarios, summaries, and reporting for resistance testing."""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

from ..tb import TBS, get_tb

from .analyzers import StrainResults
from .strains import StrainSpec

__all__ = [
    'build_spec_sim',
    'compute_spec_directional_checks',
    'format_spec_report',
    'get_spec_scenario_configs',
    'save_spec_report',
    'summarize_spec_sim',
    'strain_preset_spec',
    'SPEC_SCENARIO_LABELS',
    'SPEC_SCENARIO_META',
]


def strain_preset_spec(resistant_fitness=0.95, resistant_init_prev=0.0, init_prev_pan=0.05):
    """Pan + INH-R catalog used in the updated resistance tech-spec test matrix."""
    return [
        StrainSpec('pan', {'INH': 0, 'RIF': 0}, fitness=1.0, init_prev=init_prev_pan),
        StrainSpec('inh_r', {'INH': 1, 'RIF': 0}, fitness=resistant_fitness, init_prev=resistant_init_prev),
    ]


SPEC_SCENARIO_LABELS = {
    'A_baseline_no_resistance': 'No resistance',
    'B_resistance_overlay_enabled': 'With resistance',
    'C1_low_acquisition': 'Low acquisition',
    'C2_high_acquisition': 'High acquisition',
    'D1_low_fitness_cost': 'Low fitness cost',
    'D2_high_fitness_cost': 'High fitness cost',
    'E1_no_treatment_pressure': 'No treatment',
    'E2_with_treatment_pressure': 'With treatment',
    'F1_uniform_short_pan_tb_no_dst': 'No DST',
    'F2_dst_routed_alternate_regimen': 'DST routing',
}

_SPEC_META_ROWS = [
    # (key, group, spec_ref, knob)
    ('A_baseline_no_resistance',        'overlay_burden',    'PDF Testing §1 / CP24 steps 1–3', 'Base TB without resistance overlay'),
    ('B_resistance_overlay_enabled',    'overlay_burden',    'PDF Testing §1 / CP24 steps 2–4', 'Matched sim with resistance overlay, minimal pressure'),
    ('C1_low_acquisition',              'random_acquisition','PDF Testing §2 / CP24 step 5',    'p_random_acquisition(INH)=0 (de-novo/endogenous acquisition)'),
    ('C2_high_acquisition',             'random_acquisition','PDF Testing §2 / CP24 step 5',    'p_random_acquisition(INH)=0.6'),
    ('D1_low_fitness_cost',             'fitness_cost',      'PDF Testing §2 / CP24 step 6',    'INH-R fitness=1.0 (low cost)'),
    ('D2_high_fitness_cost',            'fitness_cost',      'PDF Testing §2 / CP24 step 6',    'INH-R fitness=0.3 (high cost)'),
    ('E1_no_treatment_pressure',        'treatment_efficacy','PDF Testing §2 / CP24 step 7',    'No treatment (differential efficacy off)'),
    ('E2_with_treatment_pressure',      'treatment_efficacy','PDF Testing §2 / CP24 step 7',    'Low-efficacy INH Tx on active TB (selective pressure on pan-susceptible strains)'),
    ('F1_uniform_short_pan_tb_no_dst',  'dst_routing',       'PDF Testing §2 (treatment rate) / CP24 step 7', 'Uniform short INH course without DST'),
    ('F2_dst_routed_alternate_regimen', 'dst_routing',       'PDF Testing §2 (treatment rate) / CP20',        'DST-routed INH first-line vs RIF second-line'),
]

SPEC_SCENARIO_META = {
    key: dict(group=group, spec_ref=spec_ref, knob=knob)
    for key, group, spec_ref, knob in _SPEC_META_ROWS
}

SPEC_TEST_GROUPS = {
    'overlay_burden': 'Burden before vs after resistance overlay',
    'random_acquisition': 'Random/endogenous acquisition sensitivity',
    'fitness_cost': 'Resistant-strain fitness cost sensitivity',
    'treatment_efficacy': 'Treatment efficacy / selective pressure',
    'dst_routing': 'Treatment program: uniform vs DST-routed',
}

# Scenario pairs used by reporting / directional checks
_SCENARIO_PAIRS = [
    ('A_baseline_no_resistance', 'B_resistance_overlay_enabled', 'overlay_burden', 'Overlay burden'),
    ('C1_low_acquisition', 'C2_high_acquisition', 'random_acquisition', 'Random acquisition'),
    ('D1_low_fitness_cost', 'D2_high_fitness_cost', 'fitness_cost', 'Fitness cost'),
    ('E1_no_treatment_pressure', 'E2_with_treatment_pressure', 'treatment_efficacy', 'Treatment pressure'),
    ('F1_uniform_short_pan_tb_no_dst', 'F2_dst_routed_alternate_regimen', 'dst_routing', 'DST routing'),
]

_BURDEN_COLS = [
    ('prevalence_active_per_100k', 'Active TB prevalence / 100k'),
    ('annual_incidence_per_100k', 'Annual active TB incidence / 100k'),
    ('annual_mortality_per_100k', 'Annual TB mortality / 100k'),
    ('pct_active_tb_resistant', 'Active TB that is resistant (%)'),
    ('cum_new_inh_r', 'New INH-R carriers (cum)'),
]


# -- Scenario config builder -------------------------------------------------

def _spec_network():
    return ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))


def _spec_sim_pars(**overrides):
    base = dict(n_agents=1200, start=ss.date('2000-01-01'), stop=ss.date('2010-01-01'), dt=ss.days(14), verbose=0)
    base.update(overrides)
    return base


def _cfg(name, *, resistance=True, seed=7, n_agents=1200, stop='2010-01-01',
         strains_kw=None, cascade=False, cascade_pars=None,
         resistance_pars=None, analyzers=None):
    """Compact helper to build one scenario config dict."""
    spec_tb = dict(init_prev=ss.bernoulli(0.05), beta=ss.permonth(0.22))
    cfg = dict(
        name=name,
        sim_pars=_spec_sim_pars(rand_seed=seed, n_agents=n_agents, stop=ss.date(stop)),
        tb_pars=spec_tb,
        networks=[_spec_network()],
    )
    if not resistance:
        cfg['resistance'] = False
        cfg['analyzers'] = False
        return cfg
    if strains_kw is not None:
        cfg['strains'] = strain_preset_spec(**strains_kw)
    else:
        cfg['strains'] = strain_preset_spec()
    cfg['cascade'] = cascade
    if cascade_pars:
        cfg['cascade_pars'] = cascade_pars
    if resistance_pars:
        cfg['resistance_pars'] = resistance_pars
    cfg['analyzers'] = analyzers or [StrainResults()]
    return cfg


def get_spec_scenario_configs():
    """
    Return validation scenarios for the tech-spec **Testing** section (PDF p.10).

    These operationalize ``resistance_spec_pseudocode.md`` **Critical Path 24**
    (overlay burden + directional sensitivity).

    For the full **Critical Path 23** intervention matrix, use
    ``run_resistance_critical_paths.py`` instead.
    """
    return [
        _cfg('A_baseline_no_resistance', resistance=False, seed=7),
        _cfg('B_resistance_overlay_enabled', seed=7, resistance_pars=dict(p_random_acquisition=None)),
        _cfg('C1_low_acquisition', seed=11, n_agents=1000, stop='2008-01-01',
             strains_kw=dict(resistant_init_prev=0.0), resistance_pars=dict(p_random_acquisition={'INH': 0.0})),
        _cfg('C2_high_acquisition', seed=11, n_agents=1000, stop='2008-01-01',
             strains_kw=dict(resistant_init_prev=0.0), resistance_pars=dict(p_random_acquisition={'INH': 0.6})),
        _cfg('D1_low_fitness_cost', seed=13, n_agents=1400, stop='2015-01-01',
             strains_kw=dict(resistant_fitness=1.0, resistant_init_prev=0.02)),
        _cfg('D2_high_fitness_cost', seed=13, n_agents=1400, stop='2015-01-01',
             strains_kw=dict(resistant_fitness=0.3, resistant_init_prev=0.02)),
        _cfg('E1_no_treatment_pressure', seed=17, n_agents=1000,
             strains_kw=dict(resistant_fitness=1.0, resistant_init_prev=0.005)),
        _cfg('E2_with_treatment_pressure', seed=17, n_agents=1000,
             strains_kw=dict(resistant_fitness=1.0, resistant_init_prev=0.005),
             cascade='tx_pressure', cascade_pars=dict(p_selective_acquisition={'INH': 0.0})),
        _cfg('F1_uniform_short_pan_tb_no_dst', seed=19, n_agents=1400, stop='2014-01-01',
             strains_kw=dict(resistant_fitness=1.0, resistant_init_prev=0.01), cascade='uniform_no_dst'),
        _cfg('F2_dst_routed_alternate_regimen', seed=19, n_agents=1400, stop='2014-01-01',
             strains_kw=dict(resistant_fitness=1.0, resistant_init_prev=0.01), cascade='dst_routed_inh'),
    ]


def build_spec_sim(name, **overrides):
    """Build one spec-matrix :class:`ResistanceSim` by scenario key."""
    from .sim import ResistanceSim  # avoid circular import
    configs = {cfg['name']: cfg for cfg in get_spec_scenario_configs()}
    if name not in configs:
        raise ValueError(f"Unknown spec scenario {name!r}; available: {', '.join(sorted(configs))}")
    cfg = sc.mergedicts(configs[name], overrides)
    label = cfg.pop('name', name)
    return ResistanceSim(label=label, **cfg)


# -- Summarize & directional checks ------------------------------------------

def _cum_new_inh_r(sim):
    analyzer = next((a for a in sim.analyzers.values() if isinstance(a, StrainResults)), None)
    if analyzer is None or 'new_carriers_inh_r' not in analyzer.results:
        return np.nan
    return float(np.asarray(analyzer.results['new_carriers_inh_r'][:]).sum())


def _final_active_resistant_share(sim):
    tb = get_tb(sim)
    profile = getattr(tb, 'agent_strains', None)
    if profile is None:
        return np.nan
    active = (
        (tb.state == TBS.NON_INFECTIOUS)
        | (tb.state == TBS.ASYMPTOMATIC)
        | (tb.state == TBS.SYMPTOMATIC)
    ).uids
    if len(active) == 0:
        return 0.0
    n_r = len(profile.carriers('inh_r').intersect(active))
    n_pan = len(profile.carriers('pan').intersect(active))
    den = n_r + n_pan
    return (n_r / den) if den else 0.0


def _sim_years(sim):
    delta = sim.pars.stop - sim.pars.start
    if hasattr(delta, 'days'):
        return max(float(delta.days) / 365.25, 1e-6)
    return max(float(delta) / 365.25, 1e-6)


def summarize_spec_sim(sim):
    """Summarize one validation scenario using PDF Testing burden metrics."""
    tb = get_tb(sim)
    profile = getattr(tb, 'agent_strains', None)
    years = _sim_years(sim)
    n = float(sim.pars.n_agents)
    mean_prev = float(np.mean(tb.results['prevalence_active'][-5:]))
    cum_active = float(tb.results['cum_active'][-1])
    cum_deaths = float(tb.results['cum_deaths'][-1])
    meta = SPEC_SCENARIO_META.get(sim.label, {})
    return {
        'scenario': sim.label,
        'label': SPEC_SCENARIO_LABELS.get(sim.label, sim.label),
        'spec_group': meta.get('group', ''),
        'spec_ref': meta.get('spec_ref', ''),
        'knob': meta.get('knob', ''),
        'with_resistance': profile is not None,
        'sim_years': years,
        'n_agents': int(n),
        'prevalence_active_per_100k': mean_prev * 1e5,
        'annual_incidence_per_100k': (cum_active / n / years) * 1e5,
        'annual_mortality_per_100k': (cum_deaths / n / years) * 1e5,
        'pct_active_tb_resistant': _final_active_resistant_share(sim) * 100.0,
        'cum_new_inh_r': _cum_new_inh_r(sim),
        'mean_prevalence_active_last5': mean_prev,
        'cum_active': cum_active,
        'cum_deaths': cum_deaths,
        'final_active_resistant_share': _final_active_resistant_share(sim),
    }


def compute_spec_directional_checks(summary_df):
    """Directional checks from the updated resistance tech-spec testing section."""
    by = summary_df.set_index('scenario')

    def _ratio(num, den, col):
        if num in by.index and den in by.index:
            d = float(by.loc[den, col])
            return (float(by.loc[num, col]) / d) if d > 0 else np.nan
        return np.nan

    def _delta(a, b, col):
        if a in by.index and b in by.index:
            return float(by.loc[a, col]) - float(by.loc[b, col])
        return np.nan

    checks = {
        'burden_prev_ratio_vs_no_resistance': _ratio('B_resistance_overlay_enabled', 'A_baseline_no_resistance', 'mean_prevalence_active_last5'),
        'burden_inc_ratio_vs_no_resistance': _ratio('B_resistance_overlay_enabled', 'A_baseline_no_resistance', 'cum_active'),
        'burden_mort_ratio_vs_no_resistance': _ratio('B_resistance_overlay_enabled', 'A_baseline_no_resistance', 'cum_deaths'),
        'delta_new_inh_r_high_minus_low_acq': _delta('C2_high_acquisition', 'C1_low_acquisition', 'cum_new_inh_r'),
        'delta_resistant_share_low_minus_high_fitness_cost': _delta('D1_low_fitness_cost', 'D2_high_fitness_cost', 'final_active_resistant_share'),
        'delta_resistant_share_with_minus_without_treatment': _delta('E2_with_treatment_pressure', 'E1_no_treatment_pressure', 'final_active_resistant_share'),
        'delta_resistant_share_dst_minus_no_dst': _delta('F2_dst_routed_alternate_regimen', 'F1_uniform_short_pan_tb_no_dst', 'final_active_resistant_share'),
        'delta_cum_active_dst_minus_no_dst': _delta('F2_dst_routed_alternate_regimen', 'F1_uniform_short_pan_tb_no_dst', 'cum_active'),
    }
    return checks


# -- Reporting ----------------------------------------------------------------

def _fmt_val(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'NA'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f'{float(x):.{nd}f}'


def _pair_table(summary_df, left, right, metrics):
    by = summary_df.set_index('scenario')
    rows = []
    for col, label in metrics:
        lval = by.loc[left, col] if left in by.index else np.nan
        rval = by.loc[right, col] if right in by.index else np.nan
        delta = rval - lval if pd.notna(lval) and pd.notna(rval) else np.nan
        rows.append(dict(metric=label, left=left, right=right,
                         value_left=lval, value_right=rval, delta_right_minus_left=delta))
    return pd.DataFrame(rows)


def format_spec_report(summary_df, checks):
    """Return a human-readable report aligned to the PDF Testing section / CP24."""
    lines = [
        'Resistance tech-spec validation (PDF Testing section / Critical Path 24)',
        '=' * 72, '',
        'Scope: overlay burden check + directional sensitivity on random acquisition,',
        'fitness cost, treatment efficacy/rate, and DST routing.', '',
    ]

    for group_key, group_title in SPEC_TEST_GROUPS.items():
        sub = summary_df[summary_df['spec_group'] == group_key]
        if sub.empty:
            continue
        lines += [group_title, '-' * len(group_title)]
        for _, row in sub.iterrows():
            lines.append(f"  [{row['scenario']}] {row['knob']}")
            lines.append(f"    ref: {row['spec_ref']}")
            for col, label in _BURDEN_COLS:
                if col in row and pd.notna(row[col]):
                    lines.append(f"    {label:42s} {_fmt_val(row[col])}")
        lines.append('')

    pair_metrics = [(c, l.split('/')[-1].strip() if '/' in l else l) for c, l in _BURDEN_COLS]
    lines += ['Paired comparisons', '-' * 18]
    for left, right, _, title in _SCENARIO_PAIRS:
        if left not in summary_df['scenario'].values or right not in summary_df['scenario'].values:
            continue
        lines.append(f'{title}: {SPEC_SCENARIO_LABELS.get(left, left)} vs {SPEC_SCENARIO_LABELS.get(right, right)}')
        for _, r in _pair_table(summary_df, left, right, pair_metrics).iterrows():
            lines.append(
                f"  {r['metric']:28s}  {_fmt_val(r['value_left']):>10s}  ->  {_fmt_val(r['value_right']):>10s}"
                f"  (delta {_fmt_val(r['delta_right_minus_left'], nd=3)})"
            )
        lines.append('')

    lines += ['Directional expectations (PDF Testing §2 / CP24)', '-' * 44]
    for key, desc, ok in [
        ('delta_new_inh_r_high_minus_low_acq', 'Random acquisition: high > low new INH-R', lambda v: v > 0),
        ('delta_resistant_share_low_minus_high_fitness_cost', 'Fitness: low-cost share >= high-cost share', lambda v: v >= 0),
        ('delta_resistant_share_with_minus_without_treatment', 'Treatment: with-Tx resistant share >= without', lambda v: v >= 0),
        ('delta_resistant_share_dst_minus_no_dst', 'DST routing: resistant share <= uniform no-DST', lambda v: v <= 0),
    ]:
        val = checks.get(key, np.nan)
        if np.isnan(val):
            lines.append(f'  {desc}: NA')
        else:
            lines.append(f"  {desc}: {_fmt_val(val, nd=4)}  [{'PASS' if ok(val) else 'FAIL'}]")

    overlay = checks.get('burden_inc_ratio_vs_no_resistance', np.nan)
    if not np.isnan(overlay):
        lines.append(f"  Overlay incidence ratio (B/A): {_fmt_val(overlay, nd=3)}"
                      '  (expect ~1 if overlay does not dominate burden)')
    return '\n'.join(lines)


def save_spec_report(summary_df, checks, outdir='results/resistance_simple'):
    """Write summary table, pair comparisons, and checks to CSV files."""
    outdir = sc.makefilepath(outdir, makedirs=True)
    summary_path = sc.makefilepath(f'{outdir}/summary.csv')
    checks_path = sc.makefilepath(f'{outdir}/directional_checks.csv')
    pairs_path = sc.makefilepath(f'{outdir}/pairs.csv')

    summary_df.to_csv(summary_path, index=False)
    pd.DataFrame([checks]).to_csv(checks_path, index=False)

    pair_metrics = [
        ('prevalence_active_per_100k', 'prevalence_per_100k'),
        ('annual_incidence_per_100k', 'incidence_per_100k'),
        ('annual_mortality_per_100k', 'mortality_per_100k'),
        ('pct_active_tb_resistant', 'pct_resistant'),
        ('cum_new_inh_r', 'cum_new_inh_r'),
    ]
    frames = [
        _pair_table(summary_df, left, right, pair_metrics).assign(comparison=tag)
        for left, right, tag, _ in _SCENARIO_PAIRS
    ]
    pd.concat(frames, ignore_index=True).to_csv(pairs_path, index=False)
    return dict(summary=summary_path, checks=checks_path, pairs=pairs_path)
