
"""Scenario-driven resistance example aligned to updated technical spec.

This script intentionally mirrors the scenario classes used in the updated
resistance technical specification testing guidance:

1. Burden comparison before/after enabling resistance.
2. Directional sensitivity: acquisition risk.
3. Directional sensitivity: fitness cost.
4. Directional sensitivity: treatment pressure / differential efficacy.

It is a runnable example (not a unit test) that produces a concise summary
table and optional figures to inspect these scenario dynamics.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

import tbsim
from tbsim import TBS
from tbsim.plots import _normalize_results
from tbsim.resistance import (
    DSTDx,
    DSTDelivery,
    Regimen,
    RegimenRouter,
    ResistanceConnector,
    StrainResults,
    StrainAwareTx,
    StrainAwareTxDelivery,
    StrainSpec,
)


DEFAULT_SPARS = dict(
    n_agents=1200,
    start=ss.date('2000-01-01'),
    stop=ss.date('2010-01-01'),
    dt=ss.days(14),
    verbose=0,
)


SCENARIO_LABELS = {
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

ABREV = (
    'Bars: summarize_sim() after MultiSim.run(). Lines: StrainResults via _normalize_results.\n'
    'INH = isoniazid; RIF = rifampicin; BDQ = bedaquiline; MDR = multidrug-resistant (INH+RIF); '
    'DST = drug susceptibility testing; pan = pan-susceptible; inh_r = INH-monoresistant.'
)


def _sl(name):
    return SCENARIO_LABELS.get(name, name.replace('_', ' '))


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
        plt.show()
    return fig


def plot_results(msim, summary_df, summary_fig=None, dynamics_fig=None, show=True):
    """Summary bar chart and directional dynamics figure."""
    df = summary_df.set_index('scenario')
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title, color, ylab in (
        (axs[0], 'final_active_resistant_share', 'Share of active TB that is resistant', '#4C72B0', 'Share'),
        (axs[1], 'cum_new_inh_r', 'New INH-resistant carriers (total)', '#C44E52', 'People'),
    ):
        s = df[col].dropna()
        s.plot(kind='bar', ax=ax, color=color)
        ax.set(title=title, ylabel=ylab, xlabel='')
        ax.set_xticklabels([_sl(n) for n in s.index], rotation=45, ha='right')
    fig.tight_layout()
    _finish(fig, summary_fig, show)

    flat = _normalize_results(msim)
    pairs = [
        ('C1_low_acquisition', 'C2_high_acquisition', 'Acquisition rate'),
        ('E1_no_treatment_pressure', 'E2_with_treatment_pressure', 'Treatment'),
        ('F1_uniform_short_pan_tb_no_dst', 'F2_dst_routed_alternate_regimen', 'DST routing'),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for ax, (a, b, title) in zip(axs, pairs):
        for scen, color in ((a, '#4C72B0'), (b, '#C44E52')):
            _, y_p = _xy(flat.get(scen, {}).get('n_active_pan'))
            x, y_r = _xy(flat.get(scen, {}).get('n_active_inh_r'))
            if x is None:
                continue
            share = np.divide(y_r, y_p + y_r, out=np.zeros_like(y_r, dtype=float), where=(y_p + y_r) > 0)
            ax.plot(x, share, lw=1.9, color=color, label=_sl(scen))
        ax.set(title=title, xlabel='Year')
        ax.grid(True, alpha=0.25, linestyle=':')
    axs[0].set_ylabel('Share of active TB\nthat is resistant')
    h, l = axs[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 0.18), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    _finish(fig, dynamics_fig, show, bottom=0.28)


def _build_sim(
    *,
    with_resistance=False,
    rand_seed=1,
    p_selective=0.0,
    resistant_fitness=0.95,
    resistant_init_prev=0.0,
    include_treatment=False,
    treatment_mode=None,
    n_agents=None,
    stop=None,
    p_random_acquisition=None,
):
    """Build one scenario sim.

    This helper intentionally follows the same structure as the scientific
    scenario tests so the example and validation logic stay aligned.
    """
    if with_resistance:
        strains = [
            StrainSpec('pan', {'INH': 0, 'RIF': 0}, fitness=1.0, init_prev=0.05),
            StrainSpec(
                'inh_r',
                {'INH': 1, 'RIF': 0},
                fitness=resistant_fitness,
                init_prev=resistant_init_prev,
            ),
        ]
        tb = tbsim.MultiStrainTB(
            strains=strains,
            pars=dict(init_prev=ss.bernoulli(0.05), beta=ss.permonth(0.22)),
            p_random_acquisition=p_random_acquisition,
        )
        interventions = []
        if include_treatment and treatment_mode is None:
            regimen = Regimen(
                'inh_first_line',
                drugs=['INH'],
                per_drug_efficacy={'INH': 0.05},
            )
            tx_product = StrainAwareTx(
                regimen=regimen,
                registry=tb._strain_registry,
                p_selective_acquisition={'INH': p_selective},
                acq_state_modifiers={
                    'infection': 1.0,
                    'non_infectious': 1.0,
                    'asymptomatic': 1.0,
                    'symptomatic': 1.0,
                    'treatment': 0.0,
                    'cleared': 0.0,
                },
                adherence=1.0,
            )

            def _tx_elig(sim):
                tb_local = tbsim.get_tb(sim)
                active = (
                    (tb_local.state == TBS.NON_INFECTIOUS)
                    | (tb_local.state == TBS.ASYMPTOMATIC)
                    | (tb_local.state == TBS.SYMPTOMATIC)
                ).uids
                return active.intersect(sim.people.alive.uids).intersect(
                    tb_local.on_treatment.false()
                )

            interventions = [
                StrainAwareTxDelivery(
                    product=tx_product,
                    name='tx_first',
                    eligibility=_tx_elig,
                ),
            ]

        # Scenario family requested by stakeholders:
        # compare no-DST uniform short pan-TB treatment vs DST-routed
        # alternate regimen when resistance is detected.
        if treatment_mode == 'uniform_no_dst':
            hsb = tbsim.HealthSeekingBehavior()
            confirm = tbsim.DxDelivery(
                name='confirm',
                product=tbsim.Xpert(),
                coverage=0.95,
                result_state='diagnosed',
            )

            uniform_regimen = Regimen(
                'uniform_short_pan_tb',
                drugs=['INH'],
                per_drug_efficacy={'INH': 0.95},
            )
            uniform_tx = StrainAwareTx(
                regimen=uniform_regimen,
                registry=tb._strain_registry,
                p_selective_acquisition={'INH': 0.0},
                adherence=1.0,
            )

            def _uniform_elig(sim):
                tb_local = tbsim.get_tb(sim)
                dx_local = sim.get_dx(result_state='diagnosed')
                if dx_local is None:
                    return ss.uids()
                return dx_local.diagnosed.uids.intersect(sim.people.alive.uids).intersect(
                    tb_local.on_treatment.false()
                )

            interventions = [
                hsb,
                confirm,
                StrainAwareTxDelivery(product=uniform_tx, name='uniform_tx', eligibility=_uniform_elig),
            ]

        if treatment_mode == 'dst_router':
            hsb = tbsim.HealthSeekingBehavior()
            confirm = tbsim.DxDelivery(
                name='confirm',
                product=tbsim.Xpert(),
                coverage=0.95,
                result_state='diagnosed',
            )

            dst = DSTDelivery(
                name='dst',
                product=DSTDx(
                    tb._strain_registry,
                    drugs=['INH'],
                    sensitivity=0.98,
                    specificity=0.99,
                    p_strain_obs=1.0,
                ),
                coverage=0.95,
            )
            router = RegimenRouter(dst, diagnosed_state='diagnosed', require_dst_tested=True)

            first_line_regimen = Regimen(
                'first_line_inh',
                drugs=['INH'],
                per_drug_efficacy={'INH': 0.95},
            )
            first_line_tx = StrainAwareTx(
                regimen=first_line_regimen,
                registry=tb._strain_registry,
                p_selective_acquisition={'INH': 0.0},
                adherence=1.0,
            )

            second_line_regimen = Regimen(
                'second_line_rif',
                drugs=['RIF'],
                per_drug_efficacy={'RIF': 0.95},
            )
            second_line_tx = StrainAwareTx(
                regimen=second_line_regimen,
                registry=tb._strain_registry,
                p_selective_acquisition={'RIF': 0.0},
                adherence=1.0,
            )

            # Important: second-line (resistant) tier first, then susceptible tier.
            interventions = [
                hsb,
                confirm,
                dst,
                StrainAwareTxDelivery(
                    product=second_line_tx,
                    name='tx_second_line',
                    eligibility=router.matches(INH=True),
                ),
                StrainAwareTxDelivery(
                    product=first_line_tx,
                    name='tx_first_line',
                    eligibility=router.matches(INH=False),
                ),
            ]
        analyzers = [StrainResults()]
    else:
        tb = tbsim.TB(pars=dict(init_prev=ss.bernoulli(0.05), beta=ss.permonth(0.22)))
        interventions = []
        analyzers = []

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim = tbsim.Sim(
        tb_model=tb,
        sim_pars=dict(
            n_agents=n_agents if n_agents is not None else DEFAULT_SPARS['n_agents'],
            start=DEFAULT_SPARS['start'],
            stop=ss.date(stop) if stop is not None else DEFAULT_SPARS['stop'],
            dt=DEFAULT_SPARS['dt'],
            rand_seed=rand_seed,
            verbose=DEFAULT_SPARS['verbose'],
        ),
        networks=[net],
        connectors=ResistanceConnector(),
        interventions=interventions,
        analyzers=analyzers,
    )
    sim.pars.verbose = 0
    return sim


def get_scenarios():
    """Return spec-aligned scientific scenarios for the resistance demo."""
    return [
        {
            'name': 'A_baseline_no_resistance',
            'with_resistance': False,
            'rand_seed': 7,
        },
        {
            'name': 'B_resistance_overlay_enabled',
            'with_resistance': True,
            'rand_seed': 7,
            'p_selective': 0.0,
            'include_treatment': False,
        },
        {
            'name': 'C1_low_acquisition',
            'with_resistance': True,
            'rand_seed': 11,
            'include_treatment': False,
            'resistant_init_prev': 0.0,
            'p_random_acquisition': {'INH': 0.0},
            'n_agents': 1000,
            'stop': '2008-01-01',
        },
        {
            'name': 'C2_high_acquisition',
            'with_resistance': True,
            'rand_seed': 11,
            'include_treatment': False,
            'resistant_init_prev': 0.0,
            'p_random_acquisition': {'INH': 0.6},
            'n_agents': 1000,
            'stop': '2008-01-01',
        },
        {
            'name': 'D1_low_fitness_cost',
            'with_resistance': True,
            'rand_seed': 13,
            'include_treatment': False,
            'resistant_fitness': 1.0,
            'resistant_init_prev': 0.02,
            'n_agents': 1400,
            'stop': '2015-01-01',
        },
        {
            'name': 'D2_high_fitness_cost',
            'with_resistance': True,
            'rand_seed': 13,
            'include_treatment': False,
            'resistant_fitness': 0.3,
            'resistant_init_prev': 0.02,
            'n_agents': 1400,
            'stop': '2015-01-01',
        },
        {
            'name': 'E1_no_treatment_pressure',
            'with_resistance': True,
            'rand_seed': 17,
            'include_treatment': False,
            'resistant_fitness': 1.0,
            'resistant_init_prev': 0.005,
            'n_agents': 1000,
            'stop': '2010-01-01',
        },
        {
            'name': 'E2_with_treatment_pressure',
            'with_resistance': True,
            'rand_seed': 17,
            'include_treatment': True,
            'p_selective': 0.0,
            'resistant_fitness': 1.0,
            'resistant_init_prev': 0.005,
            'n_agents': 1000,
            'stop': '2010-01-01',
        },
        {
            'name': 'F1_uniform_short_pan_tb_no_dst',
            'with_resistance': True,
            'rand_seed': 19,
            'treatment_mode': 'uniform_no_dst',
            'resistant_fitness': 1.0,
            'resistant_init_prev': 0.01,
            'n_agents': 1400,
            'stop': '2014-01-01',
        },
        {
            'name': 'F2_dst_routed_alternate_regimen',
            'with_resistance': True,
            'rand_seed': 19,
            'treatment_mode': 'dst_router',
            'resistant_fitness': 1.0,
            'resistant_init_prev': 0.01,
            'n_agents': 1400,
            'stop': '2014-01-01',
        },
    ]


def _cum_new_inh_r(sim):
    """Cumulative new inh_r carrier events from StrainResults."""
    if not sim.analyzers:
        return np.nan
    analyzer = next((a for a in sim.analyzers.values() if isinstance(a, StrainResults)), None)
    if analyzer is None:
        return np.nan
    key = 'new_carriers_inh_r'
    if key not in analyzer.results:
        return np.nan
    return float(np.asarray(analyzer.results[key][:]).sum())


def _final_active_resistant_share(sim):
    """Final active-state resistant share among pan+inh_r agents."""
    tb = tbsim.get_tb(sim)
    profile = getattr(tb, 'strain_profile', None)
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


def summarize_sim(sim, scenario_name):
    """Return one row of high-level scientific outputs for one scenario."""
    tb = tbsim.get_tb(sim)
    profile = getattr(tb, 'strain_profile', None)
    row = {
        'scenario': scenario_name,
        'with_resistance': bool(profile is not None),
        'mean_prevalence_active_last5': float(np.mean(tb.results['prevalence_active'][-5:])),
        'cum_active': float(tb.results['cum_active'][-1]),
        'cum_deaths': float(tb.results['cum_deaths'][-1]),
        'cum_new_inh_r': _cum_new_inh_r(sim),
        'final_active_resistant_share': _final_active_resistant_share(sim),
    }
    return row


def _compute_directional_checks(summary_df):
    """Compute directional checks requested by updated spec testing guidance."""
    by_name = summary_df.set_index('scenario')

    def _ratio(num_name, den_name, col):
        if num_name in by_name.index and den_name in by_name.index:
            den = float(by_name.loc[den_name, col])
            num = float(by_name.loc[num_name, col])
            return (num / den) if den > 0 else np.nan
        return np.nan

    burden_prev_ratio = _ratio(
        'B_resistance_overlay_enabled',
        'A_baseline_no_resistance',
        'mean_prevalence_active_last5',
    )
    burden_inc_ratio = _ratio('B_resistance_overlay_enabled', 'A_baseline_no_resistance', 'cum_active')
    burden_mort_ratio = _ratio('B_resistance_overlay_enabled', 'A_baseline_no_resistance', 'cum_deaths')

    acq_delta = np.nan
    if 'C2_high_acquisition' in by_name.index and 'C1_low_acquisition' in by_name.index:
        acq_delta = float(by_name.loc['C2_high_acquisition', 'cum_new_inh_r']) - float(
            by_name.loc['C1_low_acquisition', 'cum_new_inh_r']
        )

    fitness_delta = np.nan
    if 'D1_low_fitness_cost' in by_name.index and 'D2_high_fitness_cost' in by_name.index:
        fitness_delta = float(by_name.loc['D1_low_fitness_cost', 'final_active_resistant_share']) - float(
            by_name.loc['D2_high_fitness_cost', 'final_active_resistant_share']
        )

    tx_delta = np.nan
    if 'E2_with_treatment_pressure' in by_name.index and 'E1_no_treatment_pressure' in by_name.index:
        tx_delta = float(by_name.loc['E2_with_treatment_pressure', 'final_active_resistant_share']) - float(
            by_name.loc['E1_no_treatment_pressure', 'final_active_resistant_share']
        )

    checks = {
        'burden_prev_ratio_vs_no_resistance': burden_prev_ratio,
        'burden_inc_ratio_vs_no_resistance': burden_inc_ratio,
        'burden_mort_ratio_vs_no_resistance': burden_mort_ratio,
        'delta_new_inh_r_high_minus_low_acq': acq_delta,
        'delta_resistant_share_low_minus_high_fitness_cost': fitness_delta,
        'delta_resistant_share_with_minus_without_treatment': tx_delta,
    }

    # DST strategy comparison requested by stakeholders.
    if (
        'F1_uniform_short_pan_tb_no_dst' in by_name.index
        and 'F2_dst_routed_alternate_regimen' in by_name.index
    ):
        checks['delta_resistant_share_dst_minus_no_dst'] = float(
            by_name.loc['F2_dst_routed_alternate_regimen', 'final_active_resistant_share']
        ) - float(by_name.loc['F1_uniform_short_pan_tb_no_dst', 'final_active_resistant_share'])
        checks['delta_cum_active_dst_minus_no_dst'] = float(
            by_name.loc['F2_dst_routed_alternate_regimen', 'cum_active']
        ) - float(by_name.loc['F1_uniform_short_pan_tb_no_dst', 'cum_active'])
    else:
        checks['delta_resistant_share_dst_minus_no_dst'] = np.nan
        checks['delta_cum_active_dst_minus_no_dst'] = np.nan

    return checks


def print_summary(summary_df, checks):
    """Print scenario metrics in a compact table."""
    cols = [
        'scenario',
        'mean_prevalence_active_last5',
        'cum_active',
        'cum_deaths',
        'cum_new_inh_r',
        'final_active_resistant_share',
    ]
    print(summary_df[cols].to_string(index=False))
    print('\nDirectional checks (updated spec scientific scenarios):')

    def _fmt(x):
        return 'NA' if np.isnan(x) else f'{x:.4f}'

    b_prev = checks['burden_prev_ratio_vs_no_resistance']
    b_inc = checks['burden_inc_ratio_vs_no_resistance']
    b_mort = checks['burden_mort_ratio_vs_no_resistance']
    acq = checks['delta_new_inh_r_high_minus_low_acq']
    fit = checks['delta_resistant_share_low_minus_high_fitness_cost']
    tx = checks['delta_resistant_share_with_minus_without_treatment']
    dst_share = checks['delta_resistant_share_dst_minus_no_dst']
    dst_burden = checks['delta_cum_active_dst_minus_no_dst']

    print(f"- Burden prevalence ratio (with/without resistance): {_fmt(b_prev)}")
    print(f"- Burden incidence ratio (with/without resistance): {_fmt(b_inc)}")
    print(f"- Burden mortality ratio (with/without resistance): {_fmt(b_mort)}")
    print(f"- Acquisition direction (high - low new inh_r): {_fmt(acq)}")
    print(f"- Fitness-cost direction (low - high resistant share): {_fmt(fit)}")
    print(f"- Treatment-pressure direction (with - without resistant share): {_fmt(tx)}")
    print(f"- DST strategy resistant-share delta (DST - no DST): {_fmt(dst_share)}")
    print(f"- DST strategy burden delta cum_active (DST - no DST): {_fmt(dst_burden)}")

    # Quick pass/fail style indicators for directional scenarios.
    if not np.isnan(acq):
        print(f"  -> acquisition directional expectation (high > low): {'PASS' if acq > 0 else 'FAIL'}")
    if not np.isnan(fit):
        print(f"  -> fitness-cost directional expectation (low_cost >= high_cost): {'PASS' if fit >= 0 else 'FAIL'}")
    if not np.isnan(tx):
        print(f"  -> treatment-pressure directional expectation (with >= without): {'PASS' if tx >= 0 else 'FAIL'}")
    if not np.isnan(dst_share):
        print(f"  -> DST strategy expectation (DST <= no-DST resistant share): {'PASS' if dst_share <= 0 else 'FAIL'}")


def run_scenarios(
    do_plot=False,
    savefig=False,
    summary_fig_path='results/resistance_spec_scenarios.png',
    dynamics_fig_path='results/resistance_dynamics.png',
):
    """Run all spec-aligned scientific scenarios and summarize outputs."""
    scenario_specs = get_scenarios()
    sims = []
    names = []

    for spec in scenario_specs:
        name = spec['name']
        print(f'... building {name}')
        kwargs = dict(spec)
        kwargs.pop('name')
        sim = _build_sim(**kwargs)
        sim.label = name
        sims.append(sim)
        names.append(name)

    print(f'... running {len(sims)} scenarios via MultiSim')
    msim = ss.MultiSim(sims=sims)
    msim.run()

    rows = [summarize_sim(sim, name) for sim, name in zip(msim.sims, names)]
    summary_df = pd.DataFrame(rows)
    checks = _compute_directional_checks(summary_df)

    print()
    print_summary(summary_df, checks)

    if do_plot or savefig:
        plot_results(
            msim,
            summary_df,
            summary_fig=sc.makefilepath(summary_fig_path, makedirs=True) if savefig else None,
            dynamics_fig=sc.makefilepath(dynamics_fig_path, makedirs=True) if savefig else None,
            show=do_plot,
        )

    return msim, summary_df


if __name__ == '__main__':
    print('Running updated-spec resistance scenarios...')
    run_scenarios(do_plot=True, savefig=True)
