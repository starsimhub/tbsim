
import os
import sys
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim


DEFAULT_SPARS = dict(
    n_agents=2_000,
    start=ss.date('2000-01-01'),
    stop=ss.date('2005-01-01'),
    dt=ss.days(14),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = {} # Will use default TB parameters

DEFAULT_IMMIGRATION_PARS = dict(
    immigration_rate=ss.freqperyear(200),
    tb_state_distribution=dict(   # randomly chosen for demo purposes
        SUSCEPTIBLE=0.85,
        INFECTION=0.13,
        ASYMPTOMATIC=0.01,
        SYMPTOMATIC=0.01,
    ),
)

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


def generate_scenario_profiles(profilers, csv_path='immigration_scenario_profiles.csv'):
    summary_rows = []
    for scenario, prof in profilers.items():
        sim = prof.sim
        wall_time_s = float(sim.elapsed) if sim.elapsed is not None else float('nan')
        prof_df = prof.df.copy() if prof.df is not None else pd.DataFrame()
        hotspot_df = prof_df[prof_df['name'] != 'sim.Sim.run'] if len(prof_df) else prof_df
        if len(prof_df):
            if len(hotspot_df):
                top_row = hotspot_df.sort_values('time', ascending=False).iloc[0]
            else:
                top_row = prof_df.sort_values('time', ascending=False).iloc[0]
            top_label = str(top_row['name'])
            top_time_s = float(top_row['time'])
        else:
            top_label = 'none'
            top_time_s = 0.0

        summary_rows.append(dict(
            scenario=scenario,
            wall_time_s=round(wall_time_s, 4),
            top_hotspot=top_label,
            top_hotspot_s=round(top_time_s, 4),
            final_population=int(len(sim.people)),
        ))

    profiles = pd.DataFrame(summary_rows).sort_values('scenario').reset_index(drop=True)
    if csv_path:
        csv_path = sc.makefilepath(csv_path, makedirs=True)
        profiles.to_csv(csv_path, index=False)
    return profiles


def run_scenarios(do_plot=False, savefig=False, profile=False, save_profiles=True, 
                  fig_path='results/immigration_multisim.png',
                  profiles_path='results/imm_sce_prof.csv'):
    """Run all immigration scenarios; optionally profile each sim with ``sim.profile()``."""
    scenarios = get_scenarios()
    profilers = {}
    sims = []
    for scenario in scenarios.values():
        sim = build_sim(scenario=scenario)
        if profile:
            prof = sim.profile(do_run=True, plot=False, verbose=False)
            profilers[sim.label] = prof
            sims.append(prof.sim)
        else:
            sim.run()
            sims.append(sim)

    msim = ss.MultiSim(sims=sims)
    profiles = None
    if profile:
        profiles = generate_scenario_profiles(
            profilers,
            csv_path=profiles_path if save_profiles else None,
        )
        print('\nScenario performance profiles\n-----------------------------')
        print(profiles.to_string(index=False))

    if do_plot or savefig:
        if savefig:
            fig_path = sc.makefilepath(fig_path, makedirs=True)
        tbsim.plot(
            msim,
            title='TB, Immigration, age distribution and household scenarios',
            select=[
                '~None',
                '~n_multiplier_applied',
                '~ACUTE', '~acute', '~deaths',
            ],
            # n_cols=6,
            # row_height=1.6,
            filename=fig_path if savefig else None,
            show=do_plot,
        )
    return msim, profiles


def get_scenarios():
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
                immigration_rate=ss.freqperyear(120),
                age_distribution={0: 0.15, 5: 0.20, 15: 0.30, 30: 0.20, 50: 0.10, 65: 0.05},
            ),
        },
        'Immigration + age_data': {
            'name': 'Immigration + age_data',
            'immigration': dict(
                immigration_rate=ss.freqperyear(120),
                age_data=DEMO_AGE_DATA,
            ),
        },
    }

if __name__ == '__main__':
    print('Running scenarios...')
    run_scenarios(do_plot=True, savefig=True, profile=False, save_profiles=False)
