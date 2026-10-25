"""
Migration example: scenarios for population turnover, growth, imported TB, and migration with an intervention.

Migration is part of the TB model: `immigration` is the number of people arriving per year,
into the states given by `immigrants` (by default, all susceptible), and `emigration` is the
rate at which people leave, per person per year. The sims run as agents in households;
people who arrive do not join a household.
"""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import starsim.library as ssl
import tbsim


sim_pars = dict(
    n_agents = 1_500,
    total_pop = 10_000,
    start = '2000-01-01',
    stop = '2010-01-01',
    dt = ss.days(14),
    rand_seed = 123,
    verbose = 0,
)


def make_households(n_agents, rand_seed):
    """ Create a synthetic DHS household table for ``ssl.HouseholdNet`` """
    rng = np.random.default_rng(rand_seed)
    hh_id = []
    ages = []
    n_assigned = 0
    while n_assigned < n_agents:
        hh_size = min(int(rng.integers(2, 7)), n_agents - n_assigned)
        hh_id.append(len(hh_id))
        ages.append(sc.strjoin(rng.integers(1, 75, size=hh_size)))
        n_assigned += hh_size
    return sc.dataframe(hh_id=hh_id, ages=ages)


def build_sim(name, migration=None, tpt=False):
    """ Build a tbsim.Sim for a migration scenario """
    tb = tbsim.TB(beta=3, init=dict(infection=0.2, symptomatic=0.005), **sc.mergedicts(migration))
    households = ssl.HouseholdNet(dhs_data=make_households(sim_pars['n_agents'], sim_pars['rand_seed']), dynamic=False)
    interventions = ss.treat(tbsim.TPT(), where=tb.infection, coverage=0.5, interval=ss.years(1), name='tpt_rounds') if tpt else None
    return tbsim.Sim(tb=tb, networks=households, interventions=interventions, label=name, **sim_pars)


def summarize(sim):
    """ Return a one-row summary of a completed scenario """
    res = sim.results.tb
    hh_ids = sim.networks.householdnet.household_ids[sim.people.auids] # The households of everyone alive
    _, counts = np.unique(hh_ids[~np.isnan(hh_ids)], return_counts=True)
    immigrants = sum(res[key].sum() for key in res.keys() if key.startswith('immigrants_'))
    emigrants = res.emigrants.sum() if 'emigrants' in res else 0
    return dict(
        scenario = sim.label,
        final_population = sim.results.alive[-1],
        n_households = len(counts),
        mean_household_size = counts.mean(),
        tb_deaths = res.tb_death.cum[-1],
        immigrants = immigrants,
        emigrants = emigrants,
        net_migration = immigrants - emigrants,
        tpt_courses = sim.results.tpt_rounds.delivered.sum() if 'tpt_rounds' in sim.results else 0,
    )


# Each scenario: its name, and the arguments of build_sim()
scenarios = {
    'Baseline (no migration)': dict(),
    'Balanced turnover': dict(migration=dict(immigration=300, emigration=0.03)), # 300 people arrive and 3% of 10,000 leave each year
    'Growth': dict(migration=dict(immigration=600, emigration=0.03)),
    'Imported TB': dict(migration=dict(immigration=600, emigration=0.03, immigrants=dict(susceptible=0.7, infection=0.25, asymptomatic=0.05))),
    'Imported TB + TPT': dict(migration=dict(immigration=600, emigration=0.03, immigrants=dict(susceptible=0.7, infection=0.25, asymptomatic=0.05)), tpt=True),
}


def run_scenarios(do_plot=False, savefig=False, fig_path='results/migration_multisim.png'):
    """ Run the migration scenarios, and optionally plot them """
    sims = [build_sim(name, **kwargs) for name,kwargs in scenarios.items()]
    msim = ss.MultiSim(sims)
    msim.run(parallel=False)

    summary = pd.DataFrame([summarize(sim) for sim in msim.sims]).round(1)
    print('\nMigration scenario summary\n--------------------------')
    print(summary.to_string(index=False))

    if do_plot or savefig:
        tbsim.plot(
            msim,
            title = 'Migration scenarios (TBsim)',
            select = ['alive', 'susceptible', 'infection', 'infectious', 'tb_death', 'emigrants'],
            filename = fig_path if savefig else None,
            show = do_plot,
        )
    return msim, summary


if __name__ == '__main__':
    print('Running migration scenarios...')
    run_scenarios(do_plot=True, savefig=True)
