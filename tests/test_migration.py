"""Scientific tests for the Migration demographics module.

These tests check the population-level consequences of migration, not object
internals: net-positive migration should grow the population and net-negative
migration should shrink it, relative to a no-migration baseline.
"""

import starsim as ss
import tbsim

# Disable transmission and background TB so population change is driven purely by migration.
QUIET_TB = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0))


def make_sim(immigration_rate=0, emigration_rate=0, migration=True, n_agents=500, rand_seed=2, stop='2004-01-01'):
    """Build a minimal TB sim, optionally with a Migration module."""
    demographics = []
    if migration:
        demographics.append(tbsim.Migration(pars=dict(
            immigration_rate=ss.freqperyear(immigration_rate),
            emigration_rate=ss.freqperyear(emigration_rate),
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        )))
    return ss.Sim(
        n_agents=n_agents,
        start='2000-01-01',
        stop=stop,
        dt=ss.days(30),
        rand_seed=rand_seed,
        verbose=0,
        diseases=tbsim.TB(pars=QUIET_TB),
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0)),
        demographics=demographics,
    )


def final_pop(sim):
    return int(sim.results.n_alive[-1])


def test_population_grows_when_immigration_exceeds_emigration():
    """Net-positive migration grows the population above a no-migration baseline."""
    baseline = make_sim(migration=False)
    grow = make_sim(immigration_rate=100, emigration_rate=20)
    baseline.run()
    grow.run()
    assert final_pop(grow) > final_pop(baseline)


def test_population_shrinks_when_emigration_exceeds_immigration():
    """Net-negative migration shrinks the population below a no-migration baseline."""
    baseline = make_sim(migration=False)
    shrink = make_sim(immigration_rate=20, emigration_rate=100)
    baseline.run()
    shrink.run()
    assert final_pop(shrink) < final_pop(baseline)


def test_balanced_migration_keeps_population_near_baseline():
    """Equal immigration and emigration leaves the population close to baseline."""
    baseline = make_sim(migration=False)
    balanced = make_sim(immigration_rate=80, emigration_rate=80)
    baseline.run()
    balanced.run()
    # Allow for stochastic flow imbalance over the run.
    assert abs(final_pop(balanced) - final_pop(baseline)) < 0.15 * final_pop(baseline)


if __name__ == '__main__':
    test_population_grows_when_immigration_exceeds_emigration()
    test_population_shrinks_when_emigration_exceeds_immigration()
    test_balanced_migration_keeps_population_near_baseline()
