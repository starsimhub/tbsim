"""Scientific tests for immigration, modeled via the Migration module.

Immigration is the immigration-only configuration of ``tbsim.Migration``
(``emigration_rate=0``). These tests check the population-level effects of
immigration relative to a no-migration baseline: more people, and higher
active-TB prevalence when arrivals come from a higher-prevalence source.
"""

import starsim as ss
import tbsim


def make_sim(immigration_rate=0, tb_state_distribution=None, init_prev=0.0, migration=True,
             n_agents=500, rand_seed=2, stop='2004-01-01'):
    """Build a minimal TB sim, optionally with immigration (no emigration).

    Transmission is disabled (``beta=0``) so that any change in prevalence is
    attributable to imported cases rather than within-sim spread.
    """
    demographics = []
    if migration:
        demographics.append(tbsim.Migration(pars=dict(
            immigration_rate=ss.freqperyear(immigration_rate),
            emigration_rate=ss.freqperyear(0),
            tb_state_distribution=tb_state_distribution or dict(SUSCEPTIBLE=1.0),
        )))
    return ss.Sim(
        n_agents=n_agents,
        start='2000-01-01',
        stop=stop,
        dt=ss.days(30),
        rand_seed=rand_seed,
        verbose=0,
        diseases=tbsim.TB(pars=dict(init_prev=ss.bernoulli(init_prev), beta=ss.peryear(0.0))),
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0)),
        demographics=demographics,
    )


def test_immigration_increases_population():
    """Immigration with no emigration grows the population above a no-migration baseline."""
    baseline = make_sim(migration=False)
    imm = make_sim(immigration_rate=100)
    baseline.run()
    imm.run()
    assert int(imm.results.n_alive[-1]) > int(baseline.results.n_alive[-1])


def test_prevalence_increases_with_high_prevalence_source():
    """Importing from a higher-prevalence source raises active-TB prevalence above baseline."""
    # Baseline: low-prevalence resident population, no migration.
    baseline = make_sim(migration=False, init_prev=0.02)
    # Immigration from a source where arrivals already have active TB.
    high = make_sim(
        immigration_rate=200,
        init_prev=0.02,
        tb_state_distribution=dict(ASYMPTOMATIC=0.5, SYMPTOMATIC=0.5),
    )
    baseline.run()
    high.run()
    base_prev = float(tbsim.get_tb(baseline).results.prevalence_active[-1])
    high_prev = float(tbsim.get_tb(high).results.prevalence_active[-1])
    assert high_prev > base_prev


if __name__ == '__main__':
    test_immigration_increases_population()
    test_prevalence_increases_with_high_prevalence_source()
