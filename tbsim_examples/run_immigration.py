"""
Run TB with ongoing immigration and optional TPT / DOTS interventions.

Uses the same scenario-dict pattern as ``run_tb_interventions.py``: each
scenario is a dict of optional component overrides (``immigration``,
``tptintervention``, ``dotsintervention``, ``tbpars``), and ``build_sim``
assembles a fully configured ``tbsim.Sim`` from them. ``run_scenarios``
runs all scenarios together as a single ``MultiSim`` so they can be
compared on one figure.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim


# Module-level defaults (override per scenario via ``spars``/``tbpars``)
DEFAULT_SPARS = dict(
    n_agents=2_000,
    start=ss.date('2000-01-01'),
    stop=ss.date('2005-01-01'),
    dt=ss.days(30),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    init_prev=ss.bernoulli(0.03),
    beta=ss.peryear(0.06),
)

DEFAULT_IMMIGRATION_PARS = dict(
    immigration_rate=ss.freqperyear(200),
    tb_state_distribution=dict(
        SUSCEPTIBLE=0.85,
        INFECTION=0.13,
        ASYMPTOMATIC=0.01,
        SYMPTOMATIC=0.01,
    ),
)


def _make_household_dhs_data(n_agents, rand_seed):
    """Create a synthetic household table for ``ss.HouseholdNet``."""
    rng = np.random.default_rng(rand_seed)
    hh_id, ages, i, h = [], [], 0, 0
    while i < n_agents:
        hh_size = int(rng.integers(1, 7))
        hh_size = min(hh_size, n_agents - i)
        hh_ages = rng.integers(1, 75, size=hh_size)
        hh_id.append(h)
        ages.append(sc.strjoin(hh_ages))
        i += hh_size
        h += 1
    return sc.dataframe(hh_id=hh_id, ages=ages)


def build_sim(scenario=None, spars=None):

    scenario = scenario or {}

    spars = sc.objdict({**DEFAULT_SPARS, **(spars or {})})
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}

    # Demographics: always include births/deaths; immigration is optional
    demographics = [ss.Births(), ss.Deaths()]
    imm_pars = scenario.get('immigration')
    if imm_pars is not None:
        merged_imm = {**DEFAULT_IMMIGRATION_PARS, **imm_pars}
        demographics.append(tbsim.Immigration(pars=merged_imm))

    # Interventions: TPT (single dict or list)
    interventions = []
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        if isinstance(tpt_params, dict):
            interventions.append(tbsim.TPTSimple(pars=tpt_params))
        elif isinstance(tpt_params, list):
            for i, params in enumerate(tpt_params):
                params['name'] = f'TPT_{i}'
                interventions.append(tbsim.TPTSimple(pars=params))

    # DOTS treatment cascade: HSB -> DxDelivery(Xpert) -> TxDelivery(DOTS).
    # TxDelivery's default eligibility is "diagnosed & active TB", so an upstream
    # health-seeking + diagnostic step is required for anyone to get treated.
    dots_params = scenario.get('dotsintervention')
    if dots_params is not None:
        dx_coverage = dots_params.get('coverage', 0.8)
        interventions.append(tbsim.HealthSeekingBehavior())
        interventions.append(tbsim.DxDelivery(product=tbsim.Xpert(), coverage=dx_coverage))
        interventions.append(tbsim.TxDelivery(product=tbsim.DOTS()))

    # Networks: random community contacts + DHS-derived household network
    dhs_data = _make_household_dhs_data(n_agents=spars.n_agents, rand_seed=spars.rand_seed)
    networks = [
        ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
        ss.HouseholdNet(dhs_data=dhs_data, dynamic=False),
    ]

    return tbsim.Sim(
        label=scenario.get('name', 'scenario'),
        sim_pars=spars,
        tb_pars=tbpars,
        networks=networks,
        demographics=demographics,
        interventions=interventions,
    )


def get_scenarios():
    """Define scenarios for evaluating immigration x TPT combinations."""
    return {
        'Baseline': {
            'name': 'Baseline',
        },
        'Immigration only': {
            'name': 'Immigration only',
            'immigration': {},
        },
        'TPT only': {
            'name': 'TPT only',
            'tptintervention': dict(
                coverage=0.7,
                start=ss.date('2001-01-01'),
                stop=ss.date('2005-01-01'),
            ),
        },
        'Immigration + TPT': {
            'name': 'Immigration + TPT',
            'immigration': {},
            'tptintervention': dict(
                coverage=0.7,
                start=ss.date('2001-01-01'),
                stop=ss.date('2005-01-01'),
            ),
        },
        'Immigration + TPT + DOTS': {
            'name': 'Immigration + TPT + DOTS',
            'immigration': {},
            'tptintervention': dict(
                coverage=0.7,
                start=ss.date('2001-01-01'),
                stop=ss.date('2005-01-01'),
            ),
            'dotsintervention': dict(coverage=0.8),
        },
    }


def _print_scenario_summary(sim):
    """Print a one-line summary of an immigration/TPT scenario sim."""
    tb_mod = tbsim.get_tb(sim)
    n_imm = int(sim.results.immigration.n_immigrants.sum()) if 'immigration' in sim.results else 0
    n_imm_current = 0
    n_hh_assigned = 0
    if 'immigration' in sim.demographics:
        imm = sim.demographics.immigration
        immigrant_uids = imm.is_immigrant.uids
        n_imm_current = len(immigrant_uids)
        n_hh_assigned = int(np.count_nonzero(imm.hhid[immigrant_uids] >= 0))
    if 'tptsimple' in sim.results:
        tpt_r = sim.results.tptsimple
        n_tpt_init = int(tpt_r.n_newly_initiated.sum())
        n_tpt_protected = int(tpt_r.n_protected[-1])
    else:
        n_tpt_init = 0
        n_tpt_protected = 0
    if 'txdelivery' in sim.results:
        tx_r = sim.results.txdelivery
        n_dots_treated = int(tx_r.n_treated.sum())
        n_dots_cured = int(tx_r.cum_success[-1])
    else:
        n_dots_treated = 0
        n_dots_cured = 0
    print(
        f'{sim.label}: '
        f'final_pop={len(sim.people)} '
        f'total_immigrants={n_imm} current_immigrants={n_imm_current} '
        f'hh_assigned={n_hh_assigned} '
        f'tpt_initiated={n_tpt_init} tpt_protected={n_tpt_protected} '
        f'dots_treated={n_dots_treated} dots_cured={n_dots_cured} '
        f'final_infectious={int(tb_mod.results.n_infectious[-1])}'
    )


def run_scenarios(
    do_plot=True,
    savefig=True,
    fig_path='tbsim_examples/figures/immigration_multisim.png',
    hh_fig_prefix='tbsim_examples/figures/households',
):
    """Run all scenarios as a single MultiSim and optionally plot results."""
    scenarios = get_scenarios()
    sims = [build_sim(scenario=s) for s in scenarios.values()]

    msim = ss.MultiSim(sims=sims)
    msim.run()

    for sim in msim.sims:
        _print_scenario_summary(sim)

    if do_plot or savefig:
        if savefig:
            fig_path = sc.makefilepath(fig_path, makedirs=True)
        tbsim.plot(
            msim,
            title='TB: immigration x TPT x DOTS scenarios',
            select=['~acute', '~None'],
            n_cols=6,
            row_height=1.5,
            filename=fig_path if savefig else None,
            show=do_plot,
        )

        # for sim in msim.sims:
        #     hh_file = None
        #     if savefig:
        #         hh_file = sc.makefilepath(f'{hh_fig_prefix}_{sim.label}.png', makedirs=True)
        #     tbsim.plot_household(
        #         sim,
        #         title=f'Household structure: {sim.label} \n(top 25 households sizes)',
        #         max_households=25,
        #         show=do_plot,
        #         filename=hh_file,
        #         show_labels=True,
        #     )

    return msim


if __name__ == '__main__':
    run_scenarios(do_plot=True)
