"""Migration example: scenarios for growth, shrinkage, turnover, and intervention compatibility."""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim


DEFAULT_SPARS = dict(
    n_agents=1_500,
    start=ss.date('2000-01-01'),
    stop=ss.date('2004-01-01'),
    dt=ss.days(14),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    init_prev=ss.bernoulli(0.05),
    beta=ss.peryear(0.02),
)

DEMO_AGE_DATA = pd.DataFrame({
    'age': [0, 5, 15, 30, 50, 65],
    'value': [220, 280, 420, 300, 180, 90],
})

DEFAULT_MIGRATION_PARS = dict(
    immigration_rate=ss.freqperyear(80),
    emigration_rate=ss.freqperyear(80),
)


def _make_household_dhs_data(n_agents, rand_seed):
    """Create a synthetic DHS household table for ``ss.library.HouseholdNet``."""
    rng = np.random.default_rng(rand_seed)
    hh_id = []
    ages = []
    n_assigned = 0
    h = 0
    while n_assigned < n_agents:
        hh_size = int(rng.integers(2, 7))
        hh_size = min(hh_size, n_agents - n_assigned)
        hh_ages = rng.integers(1, 75, size=hh_size)
        hh_id.append(h)
        ages.append(sc.strjoin(hh_ages))
        n_assigned += hh_size
        h += 1
    return sc.dataframe(hh_id=hh_id, ages=ages)


def build_sim(scenario=None, spars=None):
    """Build a tbsim.Sim for a migration scenario."""
    scenario = scenario or {}
    spars = sc.objdict({**DEFAULT_SPARS, **(spars or {})})
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}

    demographics = [ss.Births(), ss.Deaths()]
    migration_pars = scenario.get('migration')
    if migration_pars is not None:
        demographics.append(tbsim.Migration(pars={**DEFAULT_MIGRATION_PARS, **migration_pars}))

    interventions = []
    tpt_pars = scenario.get('tptintervention')
    if tpt_pars is not None:
        interventions.append(tbsim.TPTSimple(pars=tpt_pars))

    include_households = bool(scenario.get('use_households', True))
    networks = [ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))]
    if include_households:
        networks.append(ss.library.HouseholdNet(dhs_data=_make_household_dhs_data(n_agents=spars.n_agents, rand_seed=spars.rand_seed), dynamic=False))

    return tbsim.Sim(
        label=scenario.get('name', 'scenario'),
        sim_pars=spars,
        tb_pars=tbpars,
        networks=networks,
        demographics=demographics,
        interventions=interventions,
    )


def summarize(sim):
    """Return a one-row summary for a completed scenario."""
    household_net = getattr(sim.networks, 'householdnet', None)
    if household_net is not None:
        hh_ids = np.asarray(household_net.household_ids, dtype=float)
        valid = hh_ids[~np.isnan(hh_ids)].astype(int)
        if valid.size:
            _, counts = np.unique(valid, return_counts=True)
            n_households = int(counts.size)
            mean_household_size = float(counts.mean())
        else:
            n_households = 0
            mean_household_size = float('nan')
    else:
        n_households = np.nan
        mean_household_size = np.nan

    row = dict(
        scenario=sim.label,
        final_population=int(sim.results.n_alive[-1]),
        n_households=n_households,
        mean_household_size=mean_household_size,
        cum_tb_deaths=int(sim.results.tb.cum_deaths[-1]),
    )

    migration = next((m for m in sim.demographics.values() if isinstance(m, tbsim.Migration)), None)
    if migration is None:
        row.update(dict(total_immigrants=0, total_emigrants=0, net_migration=0))
    else:
        row.update(dict(
            total_immigrants=int(migration.results.n_immigrants[:].sum()),
            total_emigrants=int(migration.results.n_emigrants[:].sum()),
            net_migration=int(migration.results.net_migration[:].sum()),
        ))

    return row


def get_scenarios():
    """Scenarios highlighting different Migration usage patterns."""
    return {
        'Baseline (no migration)': {
            'name': 'Baseline (no migration)',
            'migration': None,
        },
        'Balanced Turnover': {
            'name': 'Balanced Turnover',
            'migration': dict(
                immigration_rate=ss.freqperyear(100),
                emigration_rate=ss.freqperyear(100),
            ),
        },
        'Maintain Population': {
            'name': 'Maintain Population',
            'migration': dict(
                immigration_rate=ss.freqperyear(50),
                emigration_rate=ss.freqperyear(50),
                maintain_population=True,
            ),
        },
        'Age Data + Imported Burden': {
            'name': 'Age Data + Imported Burden',
            'migration': dict(
                immigration_rate=ss.freqperyear(120),
                emigration_rate=ss.freqperyear(70),
                age_data=DEMO_AGE_DATA,
                tb_state_distribution=dict(SUSCEPTIBLE=0.7, INFECTION=0.2, ASYMPTOMATIC=0.1),
            ),
        },
        'Migration + TPTSimple': {
            'name': 'Migration + TPTSimple',
            'migration': dict(
                immigration_rate=ss.freqperyear(120),
                emigration_rate=ss.freqperyear(90),
            ),
            'tptintervention': dict(
                coverage=ss.bernoulli(p=0.5),
                start=ss.date('2000-01-01'),
                stop=ss.date('2004-01-01'),
            ),
        }
    }


def run_scenarios(do_plot=False, savefig=False, fig_path='results/migration_multisim.png'):
    """Run migration scenarios and optionally plot outputs."""
    scenarios = get_scenarios()
    sims = []
    summaries = []

    for scenario in scenarios.values():
        sim = build_sim(scenario=scenario)
        sim.run()
        sims.append(sim)
        summaries.append(summarize(sim))

    summary_df = pd.DataFrame(summaries).sort_values('scenario').reset_index(drop=True)
    print('\nMigration scenario summary\n--------------------------')
    print(summary_df.to_string(index=False))

    msim = ss.MultiSim(sims=sims)
    if do_plot or savefig:
        if savefig:
            fig_path = sc.makefilepath(fig_path, makedirs=True)
        tbsim.plot(
            msim,
            title='Migration scenarios (TBsim)',
            select=['~None', '~n_multiplier_applied', '~15+'],
            filename=fig_path if savefig else None,
            show=do_plot,
            # style='dark_background',
        )

    return msim, summary_df


if __name__ == '__main__':
    print('Running migration scenarios...')
    run_scenarios(do_plot=True, savefig=True)
