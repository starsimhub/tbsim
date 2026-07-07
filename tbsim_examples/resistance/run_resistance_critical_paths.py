"""One-scenario-at-a-time resistance critical-path comparisons (MultiStrainTB).

Scenario names and parameterizations follow ``resistance_spec_pseudocode.md``
Critical Paths 11–21 and 23 (baseline, intervention, sensitivity, comparator).
Plot and summary style follows Cohen/Ryckman TB modeling conventions via
``run_resistance_demo``.
"""

import re

import pandas as pd
import sciris as sc
import starsim as ss

import run_resistance_demo as demo


CRITICAL_PATH_SCENARIOS = [
    dict(key='no_resistance', label=demo.LABEL_NO_RESISTANCE, no_resistance=True),
    dict(key='tpt_50', label=demo.label_tpt_scaleup_sensitivity(0.50), tpt_coverage=0.50),
    dict(key='tpt_70', label=demo.label_tpt_scaleup_sensitivity(0.70), tpt_coverage=0.70),
    dict(key='tpt_90', label=demo.label_tpt_scaleup_sensitivity(0.90), tpt_coverage=0.90),
    dict(key='tpt_low_acq', label=demo.label_tpt_acquisition_sensitivity('low'), tpt_acq_rif=0.005),
    dict(key='tpt_high_acq', label=demo.label_tpt_acquisition_sensitivity('high'), tpt_acq_rif=0.08),
    dict(key='dst_low_coverage', label=demo.label_dst_scaleup_sensitivity(0.40), dst_coverage=0.40, bpal=True),
    dict(key='dst_high_coverage', label=demo.label_dst_scaleup_sensitivity(0.95), dst_coverage=0.95, bpal=True),
    dict(
        key='dst_full_observation',
        label=demo.label_dst_full_observability(),
        p_strain_obs=1.00,
        dst_coverage=0.95,
        bpal=True,
    ),
    dict(
        key='dst_dropout',
        label=demo.LABEL_DST_DROPOUT,
        p_strain_obs=0.35,
        dst_coverage=0.95,
        bpal=True,
    ),
    dict(
        key='bpal_routing',
        label=demo.label_dst_regimen_routing(),
        dst_coverage=0.95,
        bpal=True,
    ),
    dict(
        key='bpal_low_adherence',
        label=demo.label_second_line_adherence_sensitivity(),
        dst_coverage=0.95,
        bpal=True,
        bpal_adherence=0.65,
    ),
    dict(
        key='bpal_high_acq',
        label=demo.label_selective_acquisition_sensitivity('second-line', 'BDQ', 'high'),
        dst_coverage=0.95,
        bpal=True,
        bpal_acq=dict(BDQ=0.08, FQ=0.02),
    ),
    dict(
        key='bpal_tpt',
        label=demo.LABEL_COMBINED_TPT_SECOND_LINE,
        tpt_coverage=0.70,
        dst_coverage=0.95,
        bpal=True,
    ),
    dict(
        key='random_bdq_high',
        label=demo.label_random_acquisition_sensitivity(['BDQ']),
        p_random_acquisition=dict(RIF=0.0, BDQ=5e-4, FQ=1e-5),
    ),
    dict(
        key='random_fq_high',
        label=demo.label_random_acquisition_sensitivity(['FQ']),
        p_random_acquisition=dict(RIF=0.0, BDQ=1e-5, FQ=5e-4),
    ),
    dict(
        key='random_bdq_fq_high',
        label=demo.label_random_acquisition_sensitivity(['BDQ', 'FQ']),
        p_random_acquisition=dict(RIF=0.0, BDQ=5e-4, FQ=5e-4),
    ),
    dict(
        key='first_line_rif_acq',
        label=demo.label_selective_acquisition_sensitivity('first-line', 'RIF', 'high'),
        first_line_acq=dict(RIF=0.08, FQ=0.01),
    ),
    dict(
        key='first_line_fq_acq',
        label=demo.label_selective_acquisition_sensitivity('first-line', 'FQ', 'high'),
        first_line_acq=dict(RIF=0.01, FQ=0.08),
    ),
    dict(key='progression_bottleneck', label=demo.LABEL_PROGRESSION_BOTTLENECK, p_multi=0.0),
    dict(
        key='lower_fitness_cost',
        label=demo.LABEL_LOWER_FITNESS,
        fitness=dict(RIF=0.90, BDQ=0.95, FQ=0.95),
    ),
    dict(
        key='severe_fitness_cost',
        label=demo.LABEL_HIGHER_FITNESS,
        fitness=dict(RIF=0.25, BDQ=0.50, FQ=0.55),
    ),
    dict(
        key='treatment_monitoring',
        label=demo.LABEL_TREATMENT_MONITORING,
        dst_coverage=0.95,
        bpal=True,
        monitoring=True,
        monitor_after_steps=4,
        monitor_every_steps=4,
    ),
]

COMPARISON_METRICS = [
    'final_incidence_per_100k',
    'final_prevalence_active',
    'final_active_tb',
    'final_active_rif_resistant_n',
    'final_active_rif_resistant_pct',
    'final_carriers_bdq_resistant',
    'final_carriers_fq_resistant',
    'cum_new_bdq_resistant',
    'cum_new_rif_resistant',
    'final_active_bdq_resistant_pct',
    'final_active_fq_resistant_pct',
    'final_superinfection_pct',
    'cum_duplicate_blocked',
    'cum_dst_tested',
    'cum_dst_rif_resistant',
    'cum_monitor_positive',
]

SUMMARY_METRICS = [
    'final_prevalence_active',
    'final_active_rif_resistant_n',
    'cum_new_bdq_resistant',
]


def select_overview_scenarios(baseline, simdict, compare_df, max_scenarios=6):
    """Pick baseline plus the most divergent scenarios for a readable overview plot."""
    keys = []
    if 'no_resistance' in simdict:
        keys.append('no_resistance')
    ranked = compare_df.iloc[compare_df['final_prevalence_active_delta'].abs().argsort()[::-1]]
    for key in ranked['key']:
        if key in simdict and key not in keys:
            keys.append(key)
        if len(keys) >= max_scenarios:
            break
    return [baseline] + [simdict[k] for k in keys if k in simdict]


def plot_summary(compare_df, figdir, show=False):
    """Delta bar charts for all comparison metric batches."""
    demo.plot_comparison_deltas(
        compare_df,
        figdir=figdir,
        metric_batches=demo.DELTA_METRIC_BATCHES,
        baseline_name=demo.LABEL_BASELINE,
        show=show,
        prefix='critical_path',
    )
    return


def plot_overview(baseline, simdict, compare_df, figdir, show=False):
    """One grouped time-series figure for baseline and key scenarios."""
    sims = select_overview_scenarios(baseline, simdict, compare_df)
    demo.plot_grouped_timeseries(
        sims,
        figpath=f'{figdir}/critical_path_overview_timeseries.png',
        title='Critical-path overview: baseline and selected scenarios',
        show=show,
    )
    return


def slugify(text):
    """Return a filesystem-safe scenario name."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def scenario_dict(row):
    """Convert a scenario definition into the shape expected by the demo."""
    return {k: v for k, v in row.items() if k != 'key'}


def make_one(row, spars, seed):
    """Build one labelled scenario sim without running it."""
    return demo.build_sim(scenario_dict(row), spars=spars, seed=seed)


def summarize_one(row, sim):
    """Summarize one completed scenario sim."""
    summary = demo.summarize(sim)
    summary['key'] = row['key']
    return summary


def run_critical_paths(spars=None, seed=1, figdir='results/resistance_critical_paths', show=False):
    """Run baseline once, then run and compare each explicit scenario."""
    spars = sc.objdict(sc.mergedicts(demo.DEFAULT_SPARS, spars))

    baseline_spec = dict(key='baseline', label=demo.LABEL_BASELINE)
    specs = [baseline_spec] + CRITICAL_PATH_SCENARIOS
    sims = []
    for row in specs:
        print(f'... building scenario: {row["label"]}')
        sims.append(make_one(row, spars=spars, seed=seed))

    print(f'... running {len(sims)} scenarios with ss.parallel()')
    msim = ss.parallel(*sims, verbose=0)
    run_sims = msim.sims

    rows = []
    compare_rows = []
    simdict = {}
    baseline = run_sims[0]
    baseline_row = summarize_one(baseline_spec, baseline)
    for row, sim in zip(CRITICAL_PATH_SCENARIOS, run_sims[1:]):
        print(f'... summarizing scenario: {row["label"]}')
        summary = summarize_one(row, sim)
        rows.append(summary)
        compare_rows.append(demo.compare_to_baseline(baseline_row, summary, COMPARISON_METRICS))
        simdict[row['key']] = sim

    summary_df = pd.DataFrame([baseline_row] + rows)
    compare_df = pd.DataFrame(compare_rows)
    summary_path = sc.makefilepath(f'{figdir}/critical_path_summaries.csv', makedirs=True)
    compare_path = sc.makefilepath(f'{figdir}/critical_path_vs_baseline.csv', makedirs=True)
    summary_df.to_csv(summary_path, index=False)
    compare_df.to_csv(compare_path, index=False)
    plot_overview(baseline, simdict, compare_df, figdir=figdir, show=show)
    demo.plot_all_results(
        msim,
        summary_df,
        figdir=figdir,
        compare_df=compare_df,
        show=show,
        savefig=True,
        prefix='critical_path',
    )

    print()
    print(demo.format_comparison_summary(compare_df, SUMMARY_METRICS).to_string(index=False))
    print()
    print(f'Saved scenario summaries to {summary_path}')
    print(f'Saved baseline comparisons to {compare_path}')
    return baseline, simdict, summary_df, compare_df


def main():
    n_agents = 2000
    stop = ss.date('2030-01-01')
    seed = 1
    figdir = 'results/resistance_critical_paths'
    show = True

    spars = dict(n_agents=n_agents, stop=stop)
    run_critical_paths(spars=spars, seed=seed, figdir=figdir, show=show)
    return


if __name__ == '__main__':
    main()
