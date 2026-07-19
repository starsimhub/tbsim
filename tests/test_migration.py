"""Scientific tests for the Migration demographics module.

Migration covers both immigration (new agents arriving) and emigration
(existing agents leaving); immigration-only behavior is simply
``emigration_rate=0``. These tests check population-level consequences rather
than object internals: net-positive migration grows the population,
net-negative migration shrinks it, and importing from a higher-prevalence
source raises active-TB prevalence -- all relative to a no-migration baseline.
"""

import starsim as ss
import tbsim


def make_sim(immigration_rate=0, emigration_rate=0, tb_state_distribution=None, init_prev=0.0,
             migration=True, n_agents=500, rand_seed=2, stop='2004-01-01'):
    """Build a minimal TB sim, optionally with a Migration module.

    Transmission is disabled (``beta=0``) so that changes in population and
    prevalence are attributable to migration rather than within-sim spread.
    """
    demographics = []
    if migration:
        demographics.append(tbsim.Migration(pars=dict(
            immigration_rate=ss.freqperyear(immigration_rate),
            emigration_rate=ss.freqperyear(emigration_rate),
            tb_state_distribution=tb_state_distribution or dict(SUSCEPTIBLE=1.0),
        )))
    return tbsim.Sim(
        n_agents=n_agents,
        start='2000-01-01',
        stop=stop,
        dt=ss.days(30),
        rand_seed=rand_seed,
        verbose=0,
        diseases=tbsim.TB(init_prev=ss.bernoulli(init_prev), beta=ss.peryear(0.0)),
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


def test_immigration_increases_population():
    """Immigration with no emigration grows the population above a no-migration baseline."""
    baseline = make_sim(migration=False)
    imm = make_sim(immigration_rate=100, emigration_rate=0)
    baseline.run()
    imm.run()
    assert final_pop(imm) > final_pop(baseline)


def test_prevalence_increases_with_high_prevalence_source():
    """Importing from a higher-prevalence source raises active-TB prevalence above baseline."""
    # Baseline: low-prevalence resident population, no migration.
    baseline = make_sim(migration=False, init_prev=0.02)
    # Immigration from a source where arrivals already have active TB.
    high = make_sim(
        immigration_rate=200,
        emigration_rate=0,
        init_prev=0.02,
        tb_state_distribution=dict(ASYMPTOMATIC=0.5, SYMPTOMATIC=0.5),
    )
    baseline.run()
    high.run()
    base_prev = float(baseline.get_tb().results.prevalence_active[-1])
    high_prev = float(high.get_tb().results.prevalence_active[-1])
    assert high_prev > base_prev


if __name__ == '__main__':
    test_population_grows_when_immigration_exceeds_emigration()
    test_population_shrinks_when_emigration_exceeds_immigration()
    test_balanced_migration_keeps_population_near_baseline()
    test_immigration_increases_population()
    test_prevalence_increases_with_high_prevalence_source()
