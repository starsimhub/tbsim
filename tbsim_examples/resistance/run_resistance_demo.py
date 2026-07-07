"""Full RIF/BDQ/FQ resistance program demo with MultiStrainTB, DST routing, and parallel arms."""

import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import starsim as ss

import tbsim
from tbsim import TBS
from tbsim.resistance import (
    DSTDx,
    DSTDelivery,
    DuplicateStrainAnalyzer,
    MultiStrainTB,
    Regimen,
    ResistanceConnector,
    StrainAwareTPTTx,
    StrainAwareTx,
    StrainAwareTxDelivery,
    StrainResults,
    StrainSpec,
    treatment_monitoring_eligibility,
)


DRUGS = ['RIF', 'BDQ', 'FQ']
R_FITNESS = dict(RIF=0.50, BDQ=0.80, FQ=0.85)
P_RANDOM_ACQ = dict(RIF=0.0, BDQ=1e-5, FQ=1e-5)

DEFAULT_SPARS = dict(
    n_agents=2000,
    start=ss.date('2015-01-01'),
    stop=ss.date('2030-01-01'),
    dt=ss.days(28),
    verbose=0,
)

DEFAULT_SCENARIO = dict(
    no_resistance=False,
    tpt_coverage=0.35,
    tpt_acq_rif=0.02,
    tpt_start='2024-01-01',
    dst_coverage=0.85,
    bpal=False,
    p_random_acquisition=P_RANDOM_ACQ,
    p_multi=1.0,
    p_strain_obs=None,
    p_sample=1.0,
    p_culture=1.0,
    first_line_acq=dict(RIF=0.01, FQ=0.01),
    bpal_acq=dict(BDQ=0.01, FQ=0.005),
    bpal_adherence=0.90,
    fitness=R_FITNESS,
    monitoring=False,
    monitor_after_steps=4,
    monitor_every_steps=None,
    monitor_coverage=0.90,
)

STRAIN_LABELS = {
    'x1_pan':        'DS-TB',
    'x2_rif':        'RR-TB',
    'x3_rif_fq':     'RR+FQ-R',
    'x4_rif_bdq':    'RR+BDQ-R',
    'x5_rif_bdq_fq': 'RR+BDQ+FQ-R',
    'x6_bdq':        'BDQ-R',
    'x7_fq':         'FQ-R',
    'x8_bdq_fq':     'BDQ+FQ-R',
}

STRAIN_COLORS = {
    'DS-TB':        '#4C72B0',
    'RR-TB':        '#DD8452',
    'RR+FQ-R':      '#C44E52',
    'RR+BDQ-R':     '#8172B3',
    'RR+BDQ+FQ-R':  '#937860',
    'BDQ-R':        '#55A868',
    'FQ-R':         '#64B5CD',
    'BDQ+FQ-R':     '#DA8BC3',
}

# Critical Path 23 scenario vocabulary (``resistance_spec_pseudocode.md``).
LABEL_NO_RESISTANCE = 'No-resistance comparator'
LABEL_BASELINE = 'Baseline program'
LABEL_TPT_SCALEUP = 'TPT scale-up'
LABEL_DST_PLUS_SECOND_LINE = 'DST + second-line'
LABEL_COMBINED_TPT_SECOND_LINE = 'Combined TPT + second-line'
LABEL_HIGH_ACQUISITION = 'High acquisition pressure'
LABEL_LOWER_FITNESS = 'Lower fitness cost'
LABEL_HIGHER_FITNESS = 'Higher fitness cost'
LABEL_PROGRESSION_BOTTLENECK = 'Progression bottleneck'
LABEL_DST_DROPOUT = 'DST strain-dropout'
LABEL_TREATMENT_MONITORING = 'Treatment monitoring'


def label_tpt_scaleup_sensitivity(coverage):
    return f'TPT scale-up sensitivity ({coverage * 100:.0f}% coverage)'


def label_tpt_acquisition_sensitivity(level):
    return f'TPT-driven acquisition sensitivity ({level} RIF)'


def label_dst_scaleup_sensitivity(coverage):
    return f'DST scale-up sensitivity ({coverage * 100:.0f}% coverage)'


def label_dst_full_observability():
    return 'DST full profile observability'


def label_dst_regimen_routing():
    return 'DST-based regimen routing'


def label_second_line_adherence_sensitivity():
    return 'Second-line regimen sensitivity (lower adherence)'


def label_selective_acquisition_sensitivity(regimen, drug, level):
    return f'Selective acquisition sensitivity ({level} {drug} on {regimen} failure)'


def label_random_acquisition_sensitivity(drugs, level='high'):
    drug_text = ' and '.join(drugs)
    return f'Random endogenous acquisition sensitivity ({level} {drug_text})'


# Display-layer short labels (≤30 chars) for y-axis tick text on bar charts.
# Keys are full scenario labels; values are the abbreviated versions.
_SHORT_LABEL_MAP = {
    label_dst_full_observability():            'DST full observability',
    label_dst_regimen_routing():               'DST regimen routing',
    label_second_line_adherence_sensitivity(): '2nd-line: lower adherence',
}

# Add dynamic labels that depend on parameter values.
for _cov in (0.40, 0.50, 0.70, 0.90, 0.95):
    _SHORT_LABEL_MAP[label_tpt_scaleup_sensitivity(_cov)] = f'TPT scale-up ({_cov * 100:.0f}% cov.)'
    _SHORT_LABEL_MAP[label_dst_scaleup_sensitivity(_cov)] = f'DST scale-up ({_cov * 100:.0f}% cov.)'
for _lvl in ('low', 'high'):
    _SHORT_LABEL_MAP[label_tpt_acquisition_sensitivity(_lvl)] = f'TPT acq. ({_lvl} RIF)'
for _reg, _reg_short in (('first-line', '1st-line'), ('second-line', '2nd-line')):
    for _drug in ('RIF', 'BDQ', 'FQ'):
        for _lvl in ('low', 'high'):
            full = label_selective_acquisition_sensitivity(_reg, _drug, _lvl)
            _SHORT_LABEL_MAP[full] = f'{_lvl.title()} {_drug} acq. ({_reg_short})'
for _drugs in (['BDQ'], ['FQ'], ['BDQ', 'FQ']):
    full = label_random_acquisition_sensitivity(_drugs)
    drug_short = '+'.join(_drugs)
    _SHORT_LABEL_MAP[full] = f'Random acq.: {drug_short}'


def short_label(label):
    """Return a ≤30-char display label for use on plot axes."""
    sl = _SHORT_LABEL_MAP.get(label, label)
    if len(sl) > 30:
        sl = sl[:29] + '…'
    return sl


# Cohen / Ryckman plotting conventions: baseline dashed gray, interventions colored solid.
PLOT_BASELINE_COLOR = '#555555'
PLOT_COMPARATOR_COLOR = '#222222'
PLOT_INTERVENTION_COLOR = '#4C72B0'
PLOT_GRID_ALPHA = 0.25
PLOT_UNCERTAINTY_ALPHA = 0.15
INTERVENTION_PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B3', '#DD8452', '#64B5CD', '#937860', '#DA8BC3']
INTERVENTION_MARKERS = ['s', '^', 'D', 'v', 'P', 'X', '*', 'h']

PANEL_SPECS = {
    'prevalence_active': ('Active TB prevalence', 'Proportion of population'),
    'incidence_per_100k': ('TB notification incidence', 'Per 100,000 person-years'),
    'RIF': ('Share of active carriers with rifampicin resistance', 'Percent (%)'),
    'BDQ': ('Share of active carriers with bedaquiline resistance', 'Percent (%)'),
    'FQ': ('Share of active carriers with fluoroquinolone resistance', 'Percent (%)'),
    'superinfection': ('Agents carrying two or more strain profiles', 'Percent (%)'),
}

# Related metrics are plotted together in one figure (one row per group).
TIMESERIES_GROUPS = [
    ('TB burden', ['prevalence_active', 'incidence_per_100k']),
    ('Drug-resistant active carrier shares', ['RIF', 'BDQ', 'FQ']),
    ('Superinfection', ['superinfection']),
]

# ``tbsim.plot`` panels: filename suffix, title, select spec, subplot columns.
STARSIM_PLOT_SELECTORS = [
    ('starsim_tb_burden.png', 'TB disease burden', dict(like='prevalence'), 3),
    ('starsim_tb_flows.png', 'TB flows and infectious pool', dict(regex=r'^(incidence|new_active|n_infectious|new_infections|n_symptomatic)'), 3),
    ('starsim_strain_active.png', 'Active strain carriers', dict(regex=r'^n_active_'), 4),
    ('starsim_strain_carriers.png', 'Infected strain carriers', dict(regex=r'^n_carriers_'), 4),
    ('starsim_strain_new.png', 'New strain carriers', dict(regex=r'^new_carriers_'), 4),
    ('starsim_program.png', 'DST, monitoring, duplicate blocking, and superinfection', dict(regex=r'(cum_tested_dst|cum_obs_|cum_duplicate|prop_superinfected|cum_positive|n_duplicate|n_superinfected)'), 3),
]

SUMMARY_BAR_GROUPS = [
    ('summary_burden.png', 'Final TB burden by scenario', ['final_incidence_per_100k', 'final_prevalence_active', 'final_active_tb']),
    ('summary_resistance.png', 'Drug-resistant burden by scenario', [
        'final_active_rif_resistant_n', 'final_active_rif_resistant_pct',
        'final_active_bdq_resistant_pct', 'final_active_fq_resistant_pct',
    ]),
    ('summary_program.png', 'Program and acquisition outcomes by scenario', [
        'final_superinfection_pct', 'cum_duplicate_blocked', 'cum_dst_tested',
        'cum_dst_rif_resistant', 'cum_monitor_positive', 'cum_new_rif_resistant', 'cum_new_bdq_resistant',
    ]),
]

METRIC_LABELS = {
    'final_incidence_per_100k': 'TB incidence (per 100,000 person-years)',
    'final_prevalence_active': 'Active TB prevalence (proportion)',
    'final_active_tb': 'Active TB cases (count)',
    'final_active_rif_resistant_n': 'Active RR-TB carriers (count)',
    'final_active_rif_resistant_pct': 'Active RR-TB share (%)',
    'final_carriers_bdq_resistant': 'BDQ-resistant carriers (count)',
    'final_carriers_fq_resistant': 'FQ-resistant carriers (count)',
    'cum_new_bdq_resistant': 'Cumulative new BDQ-resistant carriers',
    'cum_new_rif_resistant': 'Cumulative new RR carriers',
    'final_active_bdq_resistant_pct': 'Active BDQ-R share (%)',
    'final_active_fq_resistant_pct': 'Active FQ-R share (%)',
    'final_superinfection_pct': 'Superinfection prevalence (%)',
    'cum_duplicate_blocked': 'Duplicate-profile reinfections blocked',
    'cum_dst_tested': 'DST tests performed',
    'cum_dst_rif_resistant': 'Observed RIF-resistant DST results',
    'cum_monitor_positive': 'Treatment-monitoring positives',
}

COMPARISON_METRICS = list(METRIC_LABELS.keys())

DELTA_METRIC_BATCHES = [
    ('delta_burden.png', ['final_incidence_per_100k', 'final_prevalence_active', 'final_active_tb']),
    ('delta_resistance.png', ['final_active_rif_resistant_n', 'final_active_rif_resistant_pct', 'final_active_bdq_resistant_pct', 'final_active_fq_resistant_pct']),
    ('delta_carriers.png', ['final_carriers_bdq_resistant', 'final_carriers_fq_resistant', 'cum_new_bdq_resistant', 'cum_new_rif_resistant']),
    ('delta_program.png', ['final_superinfection_pct', 'cum_duplicate_blocked', 'cum_dst_tested', 'cum_dst_rif_resistant', 'cum_monitor_positive']),
]


def _intervention_style_index(label):
    """Stable palette index for intervention scenarios."""
    intervention_order = [
        f'{LABEL_TPT_SCALEUP} to 70%',
        LABEL_DST_PLUS_SECOND_LINE,
        LABEL_COMBINED_TPT_SECOND_LINE,
        LABEL_HIGH_ACQUISITION,
        LABEL_LOWER_FITNESS,
        LABEL_PROGRESSION_BOTTLENECK,
        LABEL_DST_DROPOUT,
        LABEL_TREATMENT_MONITORING,
    ]
    if label in intervention_order:
        return intervention_order.index(label)
    return abs(hash(label)) % len(INTERVENTION_PALETTE)


def scenario_line_style(label, npts=None):
    """Return matplotlib kwargs aligned with Cohen/Ryckman scenario comparisons."""
    if label == LABEL_BASELINE:
        style = dict(color=PLOT_BASELINE_COLOR, linestyle='--', marker='o')
    elif label == LABEL_NO_RESISTANCE:
        style = dict(color=PLOT_COMPARATOR_COLOR, linestyle=(0, (3, 1, 1, 1)), marker=None)
    else:
        idx = _intervention_style_index(label)
        style = dict(
            color=INTERVENTION_PALETTE[idx % len(INTERVENTION_PALETTE)],
            linestyle='-',
            marker=INTERVENTION_MARKERS[idx % len(INTERVENTION_MARKERS)],
        )
    markevery = None
    if style.get('marker') is not None and npts is not None:
        markevery = max(int(npts // 8), 1)
    style.update(lw=1.8, markevery=markevery, markersize=4, markerfacecolor='white', markeredgewidth=0.8)
    return style


def apply_plot_style():
    """Apply shared matplotlib defaults for resistance example figures."""
    pl.rcParams.update({
        'axes.grid': True,
        'grid.alpha': PLOT_GRID_ALPHA,
        'grid.linestyle': ':',
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'font.size': 9,
    })
    return


# Backward-compatible lookups used by older example code.
SCENARIO_COLORS = {
    LABEL_NO_RESISTANCE: PLOT_COMPARATOR_COLOR,
    LABEL_BASELINE: PLOT_BASELINE_COLOR,
    f'{LABEL_TPT_SCALEUP} to 70%': INTERVENTION_PALETTE[0],
    LABEL_DST_PLUS_SECOND_LINE: INTERVENTION_PALETTE[1],
    LABEL_COMBINED_TPT_SECOND_LINE: INTERVENTION_PALETTE[2],
    LABEL_HIGH_ACQUISITION: INTERVENTION_PALETTE[3],
    LABEL_LOWER_FITNESS: INTERVENTION_PALETTE[4],
    LABEL_PROGRESSION_BOTTLENECK: INTERVENTION_PALETTE[5],
    LABEL_DST_DROPOUT: INTERVENTION_PALETTE[6],
    LABEL_TREATMENT_MONITORING: INTERVENTION_PALETTE[7],
}
SCENARIO_LINESTYLES = {k: scenario_line_style(k)['linestyle'] for k in SCENARIO_COLORS}
SCENARIO_MARKERS = {k: scenario_line_style(k).get('marker') for k in SCENARIO_COLORS}


class SuperinfectionPrevalence(ss.Analyzer):
    """Track agents carrying at least two strains."""

    def __init__(self, disease='tb', **kwargs):
        super().__init__(**kwargs)
        self.disease = disease
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_superinfected', dtype=int),
            ss.Result('prop_superinfected', dtype=float),
        )
        return

    def step(self):
        tb = self.sim.diseases[self.disease]
        profile = tb.agent_strains
        counts = profile.n_strains_per_agent()
        n_super = int(np.count_nonzero(counts >= 2))
        n_alive = self.sim.people.alive.count()
        self.results['n_superinfected'][self.ti] = n_super
        self.results['prop_superinfected'][self.ti] = n_super / n_alive if n_alive else 0.0
        return


def strain_fitness(bits, fitness=None):
    """Product-form fitness cost from the spec."""
    fitness = fitness or R_FITNESS
    fit = 1.0
    for drug, resistant in zip(DRUGS, bits):
        if resistant:
            fit *= fitness[drug]
    return fit


def build_strains(fitness=None):
    """Eight RIF/BDQ/FQ phenotypes from the IDM ML Studio demo."""
    specs = [
        ('x1_pan',        (0, 0, 0), 0.040),
        ('x2_rif',        (1, 0, 0), 0.004),
        ('x3_rif_fq',     (1, 0, 1), 0.002),
        ('x4_rif_bdq',    (1, 1, 0), 0.001),
        ('x5_rif_bdq_fq', (1, 1, 1), 0.0005),
        ('x6_bdq',        (0, 1, 0), 0.0005),
        ('x7_fq',         (0, 0, 1), 0.001),
        ('x8_bdq_fq',     (0, 1, 1), 0.0005),
    ]
    return [
        StrainSpec(
            uid=uid,
            resistance=dict(zip(DRUGS, bits)),
            fitness=strain_fitness(bits, fitness=fitness),
            init_prev=init_prev,
        )
        for uid, bits, init_prev in specs
    ]


def build_tb(scenario=None):
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    return MultiStrainTB(
        strains=build_strains(fitness=scenario.fitness),
        pars=dict(
            init_prev=ss.bernoulli(0.21),
            beta=ss.permonth(0.30),
        ),
        progression_mode='bottleneck',
        p_multi=scenario.p_multi,
        p_random_acquisition=scenario.p_random_acquisition,
        alpha_super=None,  # default: rr_reinfection_rec
        alpha_act=dict(asymptomatic=0.0, symptomatic=0.0),
    )


def build_plain_tb():
    """Build matched TB without the resistance overlay."""
    return tbsim.TB(
        pars=dict(
            init_prev=ss.bernoulli(0.21),
            beta=ss.permonth(0.30),
        ),
    )


def build_plain_interventions(scenario=None):
    """Matched care cascade for the no-resistance comparator (base Tx/TPT, no DST)."""
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    tpt_product = tbsim.TPTTx(
        pars=dict(
            efficacy=ss.bernoulli(p=0.60),
            p_sterilize=ss.bernoulli(p=1.0),
            dur_treatment=ss.constant(v=ss.months(3)),
        ),
    )
    return [
        tbsim.HealthSeekingBehavior(),
        tbsim.DxDelivery(
            name='confirm',
            product=tbsim.Xpert(),
            coverage=0.80,
            result_state='diagnosed',
        ),
        tbsim.TxDelivery(
            name='first_line_tx',
            product=tbsim.FirstLine(adherence=0.85),
        ),
        tbsim.TPTSimple(
            name='three_hp_tpt',
            product=tpt_product,
            pars=dict(
                start=ss.date(scenario.tpt_start),
                coverage=ss.bernoulli(p=float(scenario.tpt_coverage)),
            ),
        ),
    ]


def rif_route(dst_name, want_resistant):
    """Select diagnosed, DST-tested agents by observed RIF phenotype."""
    def _eligible(sim):
        dx = sim.get_dx(result_state='diagnosed')
        if dx is None:
            return ss.uids()
        dst = sim.interventions[dst_name]
        tb = tbsim.get_tb(sim)
        eligible = sim.people.alive.uids
        eligible = eligible.intersect(dx.diagnosed.uids)
        eligible = eligible.intersect(dst.tested_dst.uids)
        eligible = eligible.intersect(tb.on_treatment.false())
        rif_match = dst.observed_RIF_resistant.uids if want_resistant else dst.observed_RIF_resistant.false()
        return eligible.intersect(rif_match)
    return _eligible


def first_line_tx(tb, dst, scenario=None):
    """Current rifampicin-susceptible program arm."""
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    regimen = Regimen(
        'first_line_rif_fq',
        drugs=['RIF', 'FQ'],
        per_drug_efficacy={'RIF': 0.90, 'FQ': 0.80},
        combine='parallel',
    )
    product = StrainAwareTx(
        regimen=regimen,
        catalog=tb._strain_catalog,
        p_selective_acquisition=scenario.first_line_acq,
        adherence=0.85,
    )
    return StrainAwareTxDelivery(
        name='first_line_tx',
        product=product,
        eligibility=rif_route(dst.name, want_resistant=False),
    )


def bpal_tx(tb, dst, scenario=None):
    """BPaL-like second-line arm, represented by BDQ/FQ activity."""
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    regimen = Regimen(
        'bpal_like',
        drugs=['BDQ', 'FQ'],
        per_drug_efficacy={'BDQ': 0.85, 'FQ': 0.80},
        combine='parallel',
    )
    product = StrainAwareTx(
        regimen=regimen,
        catalog=tb._strain_catalog,
        p_selective_acquisition=scenario.bpal_acq,
        adherence=scenario.bpal_adherence,
    )
    return StrainAwareTxDelivery(
        name='bpal_tx',
        product=product,
        eligibility=rif_route(dst.name, want_resistant=True),
    )


def tpt_3hp(tb, scenario=None):
    """3HP proxy within RIF/BDQ/FQ space: RIF-class preventive pressure."""
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    regimen = Regimen('three_hp_proxy', drugs=['RIF'])
    product = StrainAwareTPTTx(
        regimen=regimen,
        catalog=tb._strain_catalog,
        p_tpt_acquisition=dict(RIF=scenario.tpt_acq_rif),
        acq_state_modifiers=dict(
            infection=0.05,
            non_infectious=0.50,
            asymptomatic=1.00,
            symptomatic=1.00,
            treatment=0.00,
            cleared=0.00,
        ),
    )
    product.pars.p_sterilize = ss.bernoulli(p=1.0)
    return tbsim.TPTSimple(
        name='three_hp_tpt',
        product=product,
        pars=dict(
            start=ss.date(scenario.tpt_start),
            coverage=ss.bernoulli(p=float(scenario.tpt_coverage)),
        ),
    )


def monitor_switch_tx(tb, scenario=None):
    """BPaL-like switch for monitoring positives; cancels in-flight first-line Tx."""
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario or {}))
    regimen = Regimen(
        'bpal_like',
        drugs=['BDQ', 'FQ'],
        per_drug_efficacy={'BDQ': 0.85, 'FQ': 0.80},
        combine='parallel',
    )
    product = StrainAwareTx(
        regimen=regimen,
        catalog=tb._strain_catalog,
        p_selective_acquisition=scenario.bpal_acq,
        adherence=scenario.bpal_adherence,
    )
    return StrainAwareTxDelivery(
        name='monitor_switch_tx',
        product=product,
        cancel_delivery='first_line_tx',
        eligibility=lambda sim: sim.interventions['monitor'].still_positive.uids,
    )


def treatment_monitor(scenario):
    """Treatment-monitoring diagnostic gated by time on first-line treatment."""
    return tbsim.DxDelivery(
        name='monitor',
        product=tbsim.Xpert(),
        coverage=scenario.monitor_coverage,
        result_state='still_positive',
        eligibility=treatment_monitoring_eligibility(
            'first_line_tx',
            after_steps=scenario.monitor_after_steps,
            every_steps=scenario.monitor_every_steps,
        ),
    )


def get_scenarios():
    return {
        'no_resistance': dict(
            label=LABEL_NO_RESISTANCE,
            no_resistance=True,
        ),
        'baseline': dict(
            label=LABEL_BASELINE,
        ),
        'tpt_scaleup': dict(
            label=f'{LABEL_TPT_SCALEUP} to 70%',
            tpt_coverage=0.70,
        ),
        'dst_bpal': dict(
            label=LABEL_DST_PLUS_SECOND_LINE,
            dst_coverage=0.95,
            bpal=True,
        ),
        'bpal_tpt': dict(
            label=LABEL_COMBINED_TPT_SECOND_LINE,
            tpt_coverage=0.70,
            dst_coverage=0.95,
            bpal=True,
        ),
        'high_acquisition_pressure': dict(
            label=LABEL_HIGH_ACQUISITION,
            tpt_coverage=0.70,
            tpt_acq_rif=0.08,
            p_random_acquisition=dict(RIF=0.0, BDQ=5e-5, FQ=5e-5),
            first_line_acq=dict(RIF=0.05, FQ=0.03),
            bpal_acq=dict(BDQ=0.05, FQ=0.02),
            bpal=True,
        ),
        'lower_fitness_cost': dict(
            label=LABEL_LOWER_FITNESS,
            fitness=dict(RIF=0.90, BDQ=0.95, FQ=0.95),
        ),
        'progression_bottleneck': dict(
            label=LABEL_PROGRESSION_BOTTLENECK,
            p_multi=0.0,
        ),
        'dst_dropout': dict(
            label=LABEL_DST_DROPOUT,
            dst_coverage=0.95,
            p_sample=0.85,
            p_culture=0.70,
            p_strain_obs=0.35,
            bpal=True,
        ),
        'treatment_monitoring': dict(
            label=LABEL_TREATMENT_MONITORING,
            dst_coverage=0.95,
            bpal=True,
            monitoring=True,
            monitor_after_steps=4,
            monitor_every_steps=4,
        ),
    }


def build_sim(scenario, spars=None, seed=1):
    scenario = sc.objdict(sc.mergedicts(DEFAULT_SCENARIO, scenario))
    spars = sc.objdict(sc.mergedicts(DEFAULT_SPARS, spars))

    if scenario.no_resistance:
        tb = build_plain_tb()
        return tbsim.Sim(
            label=scenario.label,
            sim_pars=sc.mergedicts(spars, dict(rand_seed=seed)),
            tb_model=tb,
            interventions=build_plain_interventions(scenario),
        )

    tb = build_tb(scenario)

    hsb = tbsim.HealthSeekingBehavior()
    confirm = tbsim.DxDelivery(
        name='confirm',
        product=tbsim.Xpert(),
        coverage=0.80,
        result_state='diagnosed',
    )
    dst = DSTDelivery(
        name='dst',
        product=DSTDx(
            tb._strain_catalog,
            drugs=DRUGS,
            sensitivity=0.95,
            specificity=0.99,
            p_strain_obs=scenario.p_strain_obs,
            p_sample=scenario.p_sample,
            p_culture=scenario.p_culture,
        ),
        coverage=scenario.dst_coverage,
    )
    interventions = [
        hsb,
        confirm,
        dst,
        first_line_tx(tb, dst, scenario),
        tpt_3hp(tb, scenario),
    ]
    if scenario.monitoring:
        interventions.append(treatment_monitor(scenario))
        interventions.append(monitor_switch_tx(tb, scenario))
    if scenario.bpal:
        interventions.append(bpal_tx(tb, dst, scenario))

    return tbsim.Sim(
        label=scenario.label,
        sim_pars=sc.mergedicts(spars, dict(rand_seed=seed)),
        tb_model=tb,
        interventions=interventions,
        connectors=ResistanceConnector(),
        analyzers=[StrainResults(), DuplicateStrainAnalyzer(), SuperinfectionPrevalence()],
    )


def _find(sim, cls):
    for analyzer in sim.analyzers.values():
        if isinstance(analyzer, cls) or analyzer.__class__.__name__ == cls.__name__:
            return analyzer
    raise RuntimeError(f'Could not find analyzer {cls.__name__!r} in sim {sim.label!r}')


def has_resistance(sim):
    """Return whether this sim has the resistance overlay enabled."""
    tb = tbsim.get_tb(sim)
    return getattr(tb, 'agent_strains', None) is not None


def _active_uids(tb):
    return (
        (tb.state == TBS.NON_INFECTIOUS)
        | (tb.state == TBS.ASYMPTOMATIC)
        | (tb.state == TBS.SYMPTOMATIC)
    ).uids


def summarize(sim):
    tb = tbsim.get_tb(sim)
    active = _active_uids(tb)
    dst = sim.interventions.get('dst', None)
    monitor = sim.interventions.get('monitor', None)
    resistance = has_resistance(sim)
    dup = _find(sim, DuplicateStrainAnalyzer).results if resistance else None
    superinf = _find(sim, SuperinfectionPrevalence).results if resistance else None
    strain_res = _find(sim, StrainResults).results if resistance else None

    row = {
        'scenario': sim.label,
        'final_incidence_per_100k': float(tb.results.incidence_kpy[-1] * 100.0),
        'final_prevalence_active': float(tb.results.prevalence_active[-1]),
        'final_active_tb': int(len(active)),
        'final_superinfection_pct': float(superinf['prop_superinfected'][-1] * 100.0) if superinf else 0.0,
        'cum_duplicate_blocked': int(dup['cum_duplicate_blocked'][-1]) if dup else 0,
        'cum_dst_tested': int(dst.results['cum_tested_dst'][-1]) if dst else 0,
        'cum_dst_rif_resistant': int(dst.results['cum_obs_RIF_resistant'][-1]) if dst else 0,
        'cum_monitor_positive': int(monitor.results['cum_positive'][-1]) if monitor else 0,
    }

    active_carriers = ss.uids()
    infected_carriers = ss.uids()
    resistant_active = {drug: ss.uids() for drug in DRUGS}
    resistant_infected = {drug: ss.uids() for drug in DRUGS}
    if resistance:
        catalog = tb._strain_catalog
        profile = tb.agent_strains
        infected = tb.infected.uids
        for s_idx, uid in enumerate(catalog.uids):
            active_strain = getattr(profile._tb, profile.names[s_idx]).uids.intersect(active)
            infected_strain = getattr(profile._tb, profile.names[s_idx]).uids.intersect(infected)
            active_carriers = active_carriers.union(active_strain)
            infected_carriers = infected_carriers.union(infected_strain)
            for drug_idx, drug in enumerate(catalog.drugs):
                if catalog.resistance[s_idx, drug_idx]:
                    resistant_active[drug] = resistant_active[drug].union(active_strain)
                    resistant_infected[drug] = resistant_infected[drug].union(infected_strain)
            row[f'final_active_{uid}'] = int(len(active_strain))
            row[f'final_carriers_{uid}'] = int(len(infected_strain))
    else:
        for uid in STRAIN_LABELS.keys():
            row[f'final_active_{uid}'] = 0
            row[f'final_carriers_{uid}'] = 0

    active_denom = len(active_carriers)
    for drug in DRUGS:
        drug_key = drug.lower()
        row[f'final_active_{drug_key}_resistant_n'] = int(len(resistant_active[drug]))
        row[f'final_carriers_{drug_key}_resistant'] = int(len(resistant_infected[drug]))
        row[f'final_active_{drug_key}_resistant_pct'] = (
            100.0 * len(resistant_active[drug]) / active_denom if active_denom else 0.0
        )

    if resistance and strain_res is not None:
        catalog = tb._strain_catalog
        bdq_idx = catalog.drugs.index('BDQ')
        cum_new = {drug: 0 for drug in DRUGS}
        for s_idx, uid in enumerate(catalog.uids):
            new_total = int(np.sum(strain_res[f'new_carriers_{uid}'][:]))
            for drug_idx, drug in enumerate(catalog.drugs):
                if catalog.resistance[s_idx, drug_idx]:
                    cum_new[drug] += new_total
        for drug in DRUGS:
            row[f'cum_new_{drug.lower()}_resistant'] = cum_new[drug]
    else:
        for drug in DRUGS:
            row[f'cum_new_{drug.lower()}_resistant'] = 0
    return row


def _values(res):
    return np.asarray(res.values if hasattr(res, 'values') else res[:], dtype=float).ravel()


def _time(res):
    if hasattr(res, 'timevec'):
        return np.asarray(res.timevec)
    return np.arange(len(_values(res)))


def _strain_results(sim):
    return _find(sim, StrainResults).results


def _tb_time(sim):
    res = sim.get_tb().results['prevalence_active']
    return _time(res)


def active_resistant_share(sim, drug):
    """Return time series of active strain-carrier share resistant to one drug."""
    if not has_resistance(sim):
        return np.zeros(len(_tb_time(sim)), dtype=float)
    catalog = tbsim.get_tb(sim)._strain_catalog
    res = _strain_results(sim)
    num = None
    den = None
    for s_idx, uid in enumerate(catalog.uids):
        y = _values(res[f'n_active_{uid}'])
        den = y.copy() if den is None else den + y
        if catalog.resistance[s_idx, catalog.drugs.index(drug)]:
            num = y.copy() if num is None else num + y
    if num is None:
        num = np.zeros_like(den)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0) * 100.0


def superinfection_share(sim):
    """Return percent superinfected time series, or zeros for non-overlay sims."""
    if not has_resistance(sim):
        return np.zeros(len(_tb_time(sim)), dtype=float)
    res = _find(sim, SuperinfectionPrevalence).results['prop_superinfected']
    return _values(res) * 100.0


def panel_series(sim, key):
    """Return x/y arrays for the named comparison panel."""
    if key == 'prevalence_active':
        res = sim.get_tb().results['prevalence_active']
        return _time(res), _values(res)
    if key == 'incidence_per_100k':
        res = sim.get_tb().results['incidence_kpy']
        return _time(res), _values(res) * 100.0
    if key in DRUGS:
        return _tb_time(sim), active_resistant_share(sim, key)
    if key == 'superinfection':
        return _tb_time(sim), superinfection_share(sim)
    raise ValueError(f'unknown panel key {key!r}')


def compare_to_baseline(baseline_row, scenario_row, metrics):
    """Return baseline, scenario, delta, and percent-change columns."""
    out = {
        'scenario': scenario_row.get('scenario'),
        'baseline': baseline_row.get('scenario'),
    }
    if 'key' in scenario_row:
        out['key'] = scenario_row['key']
    for metric in metrics:
        base = float(baseline_row.get(metric, 0.0))
        val = float(scenario_row.get(metric, 0.0))
        out[f'{metric}_baseline'] = base
        out[f'{metric}_scenario'] = val
        out[f'{metric}_delta'] = val - base
        out[f'{metric}_pct_change'] = (100.0 * (val - base) / base) if base else np.nan
    return out


def format_comparison_summary(compare_df, metrics, baseline_name=LABEL_BASELINE):
    """Build a compact table for console output (Cohen/Ryckman reporting style)."""
    rows = []
    for _, row in compare_df.iterrows():
        entry = {'scenario': row['scenario']}
        for metric in metrics:
            label = METRIC_LABELS.get(metric, metric)
            delta = row.get(f'{metric}_delta', np.nan)
            pct = row.get(f'{metric}_pct_change', np.nan)
            if np.isfinite(pct):
                entry[label] = f'{delta:+.4g} ({pct:+.1f}% vs {baseline_name})'
            else:
                entry[label] = f'{delta:+.4g}'
        rows.append(entry)
    return pd.DataFrame(rows)


def _panel_has_data(sims, key):
    """Return True if at least one sim has a nonzero value for this panel key."""
    for sim in sims:
        _, y = panel_series(sim, key)
        if np.any(np.asarray(y, dtype=float) != 0):
            return True
    return False


def plot_grouped_timeseries(sims, figpath, title=None, show=False, savefig=True):
    """Plot time series in metric groups (one figure, one row per group)."""
    apply_plot_style()

    # Filter each group's keys to non-empty panels, then drop empty groups.
    filtered_groups = []
    for group_name, keys in TIMESERIES_GROUPS:
        active_keys = [k for k in keys if _panel_has_data(sims, k)]
        if active_keys:
            filtered_groups.append((group_name, active_keys))
    if not filtered_groups:
        return None

    n_groups = len(filtered_groups)
    max_panels = max(len(keys) for _, keys in filtered_groups)
    fig, axs = pl.subplots(n_groups, max_panels, figsize=(4.2 * max_panels, 3.6 * n_groups), sharex=True, squeeze=False)

    for row, (group_name, keys) in enumerate(filtered_groups):
        for col, key in enumerate(keys):
            ax = axs[row, col]
            title_panel, ylabel = PANEL_SPECS[key]
            for sim in sims:
                x, y = panel_series(sim, key)
                ax.plot(x, y, label=sim.label, **scenario_line_style(sim.label, len(x)))
            ax.set_title(title_panel, fontsize=10)
            ax.set_ylabel(ylabel)
            if row == n_groups - 1:
                ax.set_xlabel('Year')
        for col in range(len(keys), max_panels):
            axs[row, col].set_visible(False)
        axs[row, 0].annotate(
            group_name,
            xy=(-0.22, 0.5),
            xycoords='axes fraction',
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            fontweight='bold',
        )

    if title is None:
        stop_year = sims[0].t.yearvec[-1].year if hasattr(sims[0].t.yearvec[-1], 'year') else ''
        title = f'Projected TB outcomes by scenario, 2015–{stop_year}'
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7, frameon=False)
    fig.suptitle(title, y=0.98, fontsize=12)
    fig.tight_layout(rect=[0.06, 0.14, 1, 0.95])
    if savefig:
        sc.savefig(sc.makefilepath(figpath, makedirs=True), fig=fig)
    if show:
        pl.show()
    else:
        pl.close(fig)
    return fig


def plot_baseline_comparison(baseline, scenario, outpath=None, show=False):
    """Baseline vs one intervention using grouped metric panels."""
    sims = [baseline, scenario]
    title = f'{scenario.label} compared with {baseline.label}'
    return plot_grouped_timeseries(
        sims,
        figpath=outpath or 'baseline_comparison.png',
        title=title,
        show=show,
        savefig=outpath is not None,
    )


def build_compare_df(summary_df, baseline_label=LABEL_BASELINE, metrics=None):
    """Build baseline comparison rows from a scenario summary table."""
    metrics = metrics or COMPARISON_METRICS
    baseline_rows = summary_df.loc[summary_df['scenario'] == baseline_label]
    if baseline_rows.empty:
        return None
    baseline_row = baseline_rows.iloc[0].to_dict()
    rows = []
    for _, row in summary_df.iterrows():
        if row['scenario'] == baseline_label:
            continue
        rows.append(compare_to_baseline(baseline_row, row.to_dict(), metrics))
    return pd.DataFrame(rows) if rows else None


def plot_starsim_panels(msim, figdir, show=False, savefig=True, prefix='resistance'):
    """Plot result time series using ``tbsim.plot`` (Starsim built-in panels)."""
    if not (savefig or show):
        return
    for suffix, title, select, n_cols in STARSIM_PLOT_SELECTORS:
        filename = f'{figdir}/{prefix}_{suffix}' if savefig else None
        try:
            fig = tbsim.plot(
                msim,
                select=select,
                title=title,
                filename=filename,
                n_cols=n_cols,
                show=show,
            )
            if fig is not None and not show:
                pl.close(fig)
        except Exception as err:
            print(f'... skipping starsim plot {suffix}: {err}')
    return


def plot_summary_bars(summary_df, figdir, show=False, savefig=True, prefix='resistance'):
    """Horizontal bar charts of final summary metrics grouped by topic."""
    if not (savefig or show):
        return
    apply_plot_style()
    df = summary_df.set_index('scenario')
    n_scenarios = len(df)
    bar_height = max(0.3, min(0.7, 4.0 / n_scenarios))
    row_height = 0.45
    label_pad = 3.5  # inches on the left for scenario labels

    for suffix, group_title, metrics in SUMMARY_BAR_GROUPS:
        metrics = [m for m in metrics if m in df.columns and df[m].abs().max() > 0]
        if not metrics:
            continue
        n = len(metrics)
        fig_h = max(4.0, n_scenarios * row_height + 1.5)
        fig, axs = pl.subplots(1, n, figsize=(label_pad + 3.5 * n, fig_h), sharey=True)
        if n == 1:
            axs = [axs]
        short_labels = [short_label(idx) for idx in df.index]
        for col_idx, (ax, metric) in enumerate(zip(axs, metrics)):
            title = METRIC_LABELS.get(metric, metric)
            s = df[metric].dropna()
            colors = [PLOT_BASELINE_COLOR if idx == LABEL_BASELINE else PLOT_INTERVENTION_COLOR
                      for idx in s.index]
            y_pos = range(len(s))
            ax.barh(y_pos, s.values, height=bar_height, color=colors, alpha=0.85)
            ax.set_title(title, fontsize=9, wrap=True)
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelsize=7)
            if col_idx == 0:
                ax.set_yticks(list(y_pos))
                ax.set_yticklabels(short_labels, fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            ax.invert_yaxis()
        fig.suptitle(group_title, y=1.01, fontsize=12)
        fig.tight_layout()
        if savefig:
            sc.savefig(sc.makefilepath(f'{figdir}/{prefix}_{suffix}', makedirs=True), fig=fig)
        if show:
            pl.show()
        else:
            pl.close(fig)
    return


def plot_comparison_deltas(compare_df, figdir, metric_batches=None, baseline_name=LABEL_BASELINE, show=False, prefix='resistance'):
    """Save one delta bar chart per metric batch."""
    if compare_df is None or compare_df.empty:
        return
    metric_batches = metric_batches or DELTA_METRIC_BATCHES
    for suffix, metrics in metric_batches:
        metrics = [m for m in metrics if f'{m}_delta' in compare_df.columns]
        if not metrics:
            continue
        plot_scenario_deltas(
            compare_df,
            metrics=metrics,
            figpath=f'{figdir}/{prefix}_{suffix}',
            baseline_name=baseline_name,
            show=show,
        )
    return


def plot_all_results(msim, summary_df, figdir='results/resistance_sa_demo', compare_df=None,
                     show=False, savefig=True, prefix='resistance'):
    """Plot curated panels, summary bars, Starsim engine plots, and baseline deltas."""
    if not (savefig or show):
        return

    sims = msim.sims if hasattr(msim, 'sims') else [msim]
    plot_grouped_timeseries(
        sims,
        figpath=f'{figdir}/{prefix}_scenario_timeseries.png',
        show=show,
        savefig=savefig,
    )

    strain_cols = [
        c for c in summary_df.columns
        if c.startswith('final_active_') and c != 'final_active_tb' and '_resistant' not in c
    ]
    if strain_cols:
        apply_plot_style()
        strain_df = summary_df.set_index('scenario')[strain_cols]
        strain_df.columns = [STRAIN_LABELS.get(c.removeprefix('final_active_'), c) for c in strain_cols]
        colors = [STRAIN_COLORS.get(c, '#999999') for c in strain_df.columns]
        fig, ax = pl.subplots(figsize=(10, 5.5))
        strain_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, color=colors, legend=False)
        ax.set_title('Final active TB strain composition by scenario')
        ax.set_xlabel('')
        ax.set_ylabel('Active strain carriers (count)')
        handles = [pl.Rectangle((0, 0), 1, 1, color=c) for c in colors]
        ax.legend(handles, strain_df.columns, title='Resistance phenotype', fontsize=7, title_fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1.0))
        fig.tight_layout()
        if savefig:
            sc.savefig(sc.makefilepath(f'{figdir}/{prefix}_final_strains.png', makedirs=True), fig=fig)
        if show:
            pl.show()
        else:
            pl.close(fig)

    plot_summary_bars(summary_df, figdir=figdir, show=show, savefig=savefig, prefix=prefix)
    plot_starsim_panels(msim, figdir=figdir, show=show, savefig=savefig, prefix=prefix)

    if compare_df is None:
        compare_df = build_compare_df(summary_df)
    plot_comparison_deltas(compare_df, figdir=figdir, show=show, prefix=prefix)
    return


def plot_scenario_deltas(compare_df, metrics, figpath, baseline_name=LABEL_BASELINE, show=False):
    """Horizontal bar chart of scenario deltas vs baseline."""
    apply_plot_style()
    plot_df = compare_df.set_index('scenario')
    plot_df = plot_df.drop(LABEL_NO_RESISTANCE, errors='ignore')

    # Drop metrics with no variation across all scenarios.
    metrics = [m for m in metrics
               if f'{m}_delta' in plot_df.columns and plot_df[f'{m}_delta'].abs().max() > 0]
    if not metrics:
        return None

    n = len(metrics)
    row_height = 0.45
    label_pad = 3.5
    fig_h = max(4.0, len(plot_df) * row_height + 1.5)
    fig, axs = pl.subplots(1, n, figsize=(label_pad + 4.0 * n, fig_h), sharey=True)
    if n == 1:
        axs = [axs]
    for col_idx, (ax, metric) in enumerate(zip(axs, metrics)):
        col = f'{metric}_delta'
        title = METRIC_LABELS.get(metric, metric)
        series = plot_df[col].sort_values()
        short_labels = [short_label(idx) for idx in series.index]
        y_pos = range(len(series))
        colors = np.where(series.values >= 0, '#C44E52', '#55A868')
        ax.barh(list(y_pos), series.values, color=colors, alpha=0.85, height=0.65)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_title(f'{title}\ndelta vs {baseline_name}', fontsize=9, wrap=True)
        ax.set_xlabel('Change from baseline', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
        if col_idx == 0:
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(short_labels, fontsize=8)
        else:
            ax.tick_params(labelleft=False)
    fig.suptitle('Intervention impacts compared with baseline resistance program', y=1.01, fontsize=12)
    fig.tight_layout()
    sc.savefig(sc.makefilepath(figpath, makedirs=True), fig=fig)
    if show:
        pl.show()
    else:
        pl.close(fig)
    return fig


def plot_results(msim, summary_df, figdir='results/resistance_sa_demo', show=False, savefig=True, compare_df=None):
    """Plot all resistance demo outputs (curated panels plus Starsim engine plots)."""
    plot_all_results(msim, summary_df, figdir=figdir, compare_df=compare_df, show=show, savefig=savefig)
    return


def scenario_items(keys=None):
    """Return selected scenarios as ``(key, scenario)`` pairs."""
    scenarios = get_scenarios()
    if keys is None:
        keys = list(scenarios.keys())
    return [(key, scenarios[key]) for key in keys]


def run_scenarios(spars=None, seed=1, do_plot=False, savefig=True,
                  figdir='results/resistance_sa_demo', scenario_keys=None):
    sims = []
    for _, scenario in scenario_items(scenario_keys):
        print(f'... building {scenario["label"]}')
        sims.append(build_sim(scenario, spars=spars, seed=seed))

    print(f'... running {len(sims)} scenarios with ss.parallel()')
    msim = ss.parallel(*sims, verbose=0)
    rows = [summarize(sim) for sim in msim.sims]
    df = pd.DataFrame(rows)
    print()
    print(df.set_index('scenario').T.to_string())
    if do_plot or savefig:
        plot_results(msim, df, figdir=figdir, show=do_plot, savefig=savefig)
    return msim, df


def plot_uncertainty(sims_by_label, figdir='results/resistance_sa_demo', show=False, savefig=True):
    """Plot median and 10th–90th percentile bands using grouped metric panels."""
    apply_plot_style()
    all_sims_flat = [s for sims in sims_by_label.values() for s in sims]
    filtered_groups = [
        (gname, [k for k in keys if _panel_has_data(all_sims_flat, k)])
        for gname, keys in TIMESERIES_GROUPS
    ]
    filtered_groups = [(g, ks) for g, ks in filtered_groups if ks]
    if not filtered_groups:
        return None
    n_groups = len(filtered_groups)
    max_panels = max(len(keys) for _, keys in filtered_groups)
    fig, axs = pl.subplots(n_groups, max_panels, figsize=(4.2 * max_panels, 3.6 * n_groups), sharex=True, squeeze=False)

    for row, (group_name, keys) in enumerate(filtered_groups):
        for col, key in enumerate(keys):
            ax = axs[row, col]
            title_panel, ylabel = PANEL_SPECS[key]
            for label, sims in sims_by_label.items():
                arr = []
                x = None
                for sim in sims:
                    x, y = panel_series(sim, key)
                    arr.append(y)
                if not arr:
                    continue
                vals = np.asarray(arr, dtype=float)
                q10, med, q90 = np.quantile(vals, [0.10, 0.50, 0.90], axis=0)
                style = scenario_line_style(label, len(x))
                color = style.pop('color')
                ax.plot(x, med, label=label, color=color, **style)
                ax.fill_between(x, q10, q90, color=color, alpha=PLOT_UNCERTAINTY_ALPHA)
            ax.set_title(title_panel, fontsize=10)
            ax.set_ylabel(ylabel)
            if row == n_groups - 1:
                ax.set_xlabel('Year')
        for col in range(len(keys), max_panels):
            axs[row, col].set_visible(False)
        axs[row, 0].annotate(
            group_name,
            xy=(-0.22, 0.5),
            xycoords='axes fraction',
            rotation=90,
            va='center',
            ha='center',
            fontsize=10,
            fontweight='bold',
        )

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=7, frameon=False)
    fig.suptitle('Projected TB outcomes with uncertainty across random seeds (10th–90th percentile)', y=0.98, fontsize=12)
    fig.tight_layout(rect=[0.06, 0.14, 1, 0.95])
    if savefig:
        sc.savefig(sc.makefilepath(f'{figdir}/resistance_uncertainty_bands.png', makedirs=True), fig=fig)
    if show:
        pl.show()
    else:
        pl.close(fig)
    return


def run_uncertainty(spars=None, seed=1, n_runs=3, do_plot=False, savefig=True,
                    figdir='results/resistance_sa_demo', scenario_keys=None):
    """Run selected scenarios across seeds and summarize uncertainty."""
    meta = []
    sims = []
    for run in range(n_runs):
        run_seed = seed + run
        for _, scenario in scenario_items(scenario_keys):
            label = scenario['label']
            print(f'... building uncertainty run {run + 1}/{n_runs}: {label}')
            meta.append((run_seed, label))
            sims.append(build_sim(scenario, spars=spars, seed=run_seed))

    print(f'... running {len(sims)} uncertainty scenarios with ss.parallel()')
    msim = ss.parallel(*sims, verbose=0)

    sims_by_label = {}
    rows = []
    for sim, (run_seed, label) in zip(msim.sims, meta):
        rows.append(sc.mergedicts(dict(seed=run_seed), summarize(sim)))
        sims_by_label.setdefault(label, []).append(sim)
    df = pd.DataFrame(rows)
    metrics = ['final_active_tb', 'final_active_rif_resistant_pct', 'final_incidence_per_100k']
    summary = df.groupby('scenario')[metrics].quantile([0.1, 0.5, 0.9]).unstack()
    print()
    print(summary.to_string())
    if do_plot or savefig:
        plot_uncertainty(sims_by_label, figdir=figdir, show=do_plot, savefig=savefig)
    return sims_by_label, df


def main():
    n_agents = 2000
    stop = ss.date('2030-01-01')
    seed = 1
    figdir = 'results/resistance_sa_demo'
    do_plot = True
    savefig = True
    do_uncertainty = False
    uncertainty_runs = 3

    spars = dict(n_agents=n_agents, stop=stop)
    run_scenarios(
        spars=spars,
        seed=seed,
        do_plot=do_plot,
        savefig=savefig,
        figdir=figdir,
    )
    if do_uncertainty:
        run_uncertainty(
            spars=spars,
            seed=seed,
            n_runs=uncertainty_runs,
            do_plot=do_plot,
            savefig=savefig,
            figdir=figdir,
        )
    return


if __name__ == '__main__':
    main()
