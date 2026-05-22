

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim


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
    rel_immigration=1.0,
    tb_state_distribution=dict(
        SUSCEPTIBLE=0.85,
        INFECTION=0.13,
        ASYMPTOMATIC=0.01,
        SYMPTOMATIC=0.01,
    ),
)

# Starsim-style age histogram (counts need not sum to 1; Immigration normalizes)
DEMO_AGE_DATA = pd.DataFrame({
    'age': [0, 5, 15, 30, 50, 65],
    'value': [240, 320, 500, 440, 300, 100],
})


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


def get_imm(sim):
    """Return the Immigration demographics module."""
    for dem in sim.demographics.values():
        if isinstance(dem, tbsim.Immigration):
            return dem
    return None


def build_sim(scenario=None, spars=None):
    """Build a ``tbsim.Sim`` from a scenario dict (see ``get_scenarios()``)."""
    scenario = scenario or {}
    spars = sc.objdict({**DEFAULT_SPARS, **(spars or {})})
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}
    use_acute = bool(scenario.get('use_acute', False))

    demographics = [ss.Births(), ss.Deaths()]
    imm_pars = scenario.get('immigration')
    if imm_pars is not None:
        demographics.append(tbsim.Immigration(pars={**DEFAULT_IMMIGRATION_PARS, **imm_pars}))

    interventions = []
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        items = tpt_params if isinstance(tpt_params, list) else [tpt_params]
        for i, params in enumerate(items):
            p = dict(params)
            p.setdefault('name', f'TPT_{i}')
            interventions.append(tbsim.TPTSimple(pars=p))

    dhs_data = _make_household_dhs_data(n_agents=spars.n_agents, rand_seed=spars.rand_seed)
    networks = [
        ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
        ss.HouseholdNet(dhs_data=dhs_data, dynamic=False),
    ]

    return tbsim.Sim(
        label=scenario.get('name', 'scenario'),
        sim_pars=spars,
        tb_pars=tbpars,
        tb_model='acute' if use_acute else 'default',
        networks=networks,
        demographics=demographics,
        interventions=interventions,
    )


def get_scenarios():
    tpt_window = dict(coverage=0.7, start=ss.date('2001-01-01'), stop=ss.date('2005-01-01'))
    return {
        'Baseline': {
            'name': 'Baseline',
        },
        'Immigration (defaults)': {
            'name': 'Immigration (defaults)',
            'immigration': {},
        },
        'Immigration + age_distribution': {
            'name': 'Immigration + age_distribution',
            'immigration': dict(
                immigration_rate=ss.freqperyear(220),
                age_distribution={0: 0.15, 5: 0.20, 15: 0.30, 30: 0.20, 50: 0.10, 65: 0.05},
            ),
        },
        'Immigration + age_data': {
            'name': 'Immigration + age_data',
            'immigration': dict(
                immigration_rate=ss.freqperyear(220),
                age_data=DEMO_AGE_DATA,
            ),
        },
        'Immigration + latent-heavy imports': {
            'name': 'Immigration + latent-heavy imports',
            'immigration': dict(
                tb_state_distribution=dict(SUSCEPTIBLE=0.55, INFECTION=0.40, ASYMPTOMATIC=0.05),
            ),
        },
        'TPT only': {
            'name': 'TPT only',
            'tptintervention': tpt_window,
        },
        'Immigration + TPT': {
            'name': 'Immigration + TPT',
            'immigration': dict(
                tb_state_distribution=dict(SUSCEPTIBLE=0.55, INFECTION=0.40, ASYMPTOMATIC=0.05),
            ),
            'tptintervention': tpt_window,
        },
        'Immigration + TBAcute (ACUTE imports)': {
            'name': 'Immigration + TBAcute (ACUTE imports)',
            'use_acute': True,
            'immigration': dict(
                tb_state_distribution=dict(
                    SUSCEPTIBLE=0.70,
                    INFECTION=0.15,
                    ACUTE=0.10,
                    ASYMPTOMATIC=0.05,
                ),
            ),
        },
    }


def _print_scenario_summary(sim):
    """Print one-line summary of immigration, households, and optional interventions."""
    tb_mod = tbsim.get_tb(sim)
    imm = get_imm(sim)
    n_imm = int(sim.results.immigration.n_immigrants.sum()) if imm is not None else 0
    n_imm_current = len(imm.is_immigrant.uids) if imm is not None else 0
    n_hh_assigned = 0
    mean_age_imm = np.nan
    if imm is not None and n_imm_current:
        uids = imm.is_immigrant.uids
        n_hh_assigned = int(np.count_nonzero(imm.hhid[uids] >= 0))
        mean_age_imm = float(np.nanmean(imm.age_at_immigration[uids]))
    n_tpt_init = int(sim.results.tptsimple.n_newly_initiated.sum()) if 'tptsimple' in sim.results else 0
    n_tpt_protected = int(sim.results.tptsimple.n_protected[-1]) if 'tptsimple' in sim.results else 0
    print(
        f'{sim.label}: '
        f'final_pop={len(sim.people)} '
        f'total_immigrants={n_imm} current_immigrants={n_imm_current} '
        f'mean_age_at_immigration={mean_age_imm:.1f} hh_assigned={n_hh_assigned} '
        f'tpt_initiated={n_tpt_init} tpt_protected={n_tpt_protected} '
        f'final_infectious={int(tb_mod.results.n_infectious[-1])}'
    )


def run_scenarios(do_plot=False, savefig=False, fig_path='tbsim_examples/figures/immigration_multisim.png'):
    """Run all scenarios as a MultiSim; optionally plot aggregate results."""
    scenarios = get_scenarios()
    msim = ss.MultiSim(sims=[build_sim(scenario=s) for s in scenarios.values()])
    msim.run()

    for sim in msim.sims:
        _print_scenario_summary(sim)

    if do_plot or savefig:
        if savefig:
            fig_path = sc.makefilepath(fig_path, makedirs=True)
        tbsim.plot(
            msim,
            title='TB: Immigration features x TPT',
            select=[
                '~None',
                '~n_multiplier_applied',
                '~ACUTE', '~acute', '~deaths',
            ],
            n_cols=6,
            row_height=1.6,
            filename=fig_path if savefig else None,
            show=do_plot,
        )
    return msim


if __name__ == '__main__':
    run_scenarios(do_plot=True, savefig=True)
