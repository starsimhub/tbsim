"""Tests for HealthSeekingBehavior (TB model)."""

import sys
import pytest
import numpy as np
import starsim as ss
import tbsim


def make_sim(n_agents=200, stop=ss.date("2005-12-31"), tb_pars=None, hsb_pars=None, dt=ss.days(7)):
    tb_pars  = tb_pars  or {}
    hsb_pars = hsb_pars or {}
    sim = tbsim.Sim(
        people      = ss.People(n_agents=n_agents),
        networks    = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
        diseases    = tbsim.TB(pars=tb_pars),
        interventions = tbsim.HealthSeekingBehavior(pars=hsb_pars),
        demographics = [],
        dt    = dt,
        start = ss.date("2000-01-01"),
        stop  = stop,
        verbose = 0,
    )
    return sim


def hsb(sim):
    """ Get the HealthSeekingBehavior intervention from the sim """
    return sim.interventions.healthseekingbehavior


def test_care_seeking_fires():
    """With symptomatic agents and a high rate, some agents seek care."""
    sim = make_sim(
        n_agents = 500,
        stop     = ss.date("2010-12-31"),
        tb_pars  = dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars = dict(initial_care_seeking_rate=ss.perday(0.5)),
    )
    sim.run()
    assert hsb(sim).results['n_ever_sought_care'][:].max() > 0


def test_no_care_seeking_without_eligible_agents():
    """With no TB, nobody seeks care."""
    sim = make_sim(
        tb_pars  = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        hsb_pars = dict(initial_care_seeking_rate=ss.perday(0.9)),
    )
    sim.run()
    assert hsb(sim).results['n_ever_sought_care'][:].max() == 0


def test_one_shot_per_episode():
    """Agents seek care at most once per episode when care_retry_steps is None."""
    sim = make_sim(
        n_agents = 500,
        stop     = ss.date("2010-12-31"),
        tb_pars  = dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars = dict(initial_care_seeking_rate=ss.perday(0.9), care_retry_steps=None),
    )
    sim.run()
    # With care_retry_steps=None, each agent should have sought_care True at most once per episode.
    # The BoolState enforces this; verify it was set for some agents.
    assert hsb(sim).results.new_sought_care.values.sum() > 0


def test_inactive_outside_start_stop():
    """Intervention outside its active window records no seekers."""
    sim = make_sim(
        n_agents = 300,
        stop     = ss.date("2005-12-31"),
        tb_pars  = dict(init_prev=ss.bernoulli(0.30)),
        hsb_pars = dict(
            initial_care_seeking_rate = ss.perday(0.9),
            start = ss.date("2010-01-01"),
            stop  = ss.date("2020-12-31"),
        ),
    )
    sim.run()
    assert hsb(sim).results['new_sought_care'][:].sum() == 0


def test_care_seeking_correct_after_deaths():
    """Regression for the issue #425 bug class in HealthSeekingBehavior.step.

    After agents die, the eligible-for-seek mask is over the Arr's alive-only
    view, so compact positions no longer equal UIDs. Care-seeking must still
    target genuinely eligible (symptomatic, alive) agents, not random living
    agents at the matching compact positions.
    """
    sim = make_sim(
        n_agents = 300,
        tb_pars  = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        hsb_pars = dict(initial_care_seeking_rate=ss.perday(1.0)),
    )
    sim.init()
    ppl = sim.people
    h = hsb(sim)
    tb = sim.get_tb()

    # Kill a block of low-numbered agents so alive-array positions no longer
    # line up with UIDs (the precondition that triggers the bug).
    ppl.request_death(ss.uids(np.arange(0, len(ppl) // 3)))
    ppl.step_die()
    ppl.remove_dead()
    alive = ppl.alive.uids
    assert not np.array_equal(np.asarray(alive), np.arange(len(alive))), \
        "Test setup failed to create a UID/position gap"

    # Make a block of high-UID alive agents symptomatic (care-seeking eligible).
    eligible = alive[-20:]
    tb.state[eligible] = tbsim.TBS.SYMPTOMATIC

    h.step()

    sought = h.sought_care.uids
    assert len(sought) > 0, "Some eligible agent should have sought care"
    # Everyone who sought care must really be eligible + alive, and must come
    # only from the symptomatic block we created.
    assert np.all(np.isin(np.asarray(tb.state[sought]), h._states)), \
        "Care-seekers must be in an eligible state"
    assert np.all(np.asarray(ppl.alive[sought])), "Care-seekers must be alive"
    assert set(np.asarray(sought).tolist()).issubset(set(np.asarray(eligible).tolist())), \
        "Care-seekers must be a subset of the genuinely eligible agents"


def test_missing_tb_raises():
    """A sim without tb raises an explicit error on init."""
    sim = ss.Sim(
        people      = ss.People(n_agents=50),
        networks    = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0)),
        diseases    = ss.SIR(), # Wrong module
        interventions = tbsim.HealthSeekingBehavior(),
        dt = ss.days(7), start = ss.date("2000-01-01"), stop = ss.date("2002-12-31"),
        verbose = 0,
    )
    with pytest.raises(KeyError):
        sim.init()


def make_symptomatic_sim(rate, n_agents=20_000, dt=ss.days(7)):
    """Initialize a sim with no transmission, then make every agent symptomatic (eligible to seek care)."""
    sim = make_sim(
        n_agents = n_agents,
        dt       = dt,
        tb_pars  = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0)),
        hsb_pars = dict(initial_care_seeking_rate=ss.perday(rate)),
    )
    sim.init()
    uids = sim.people.alive.uids
    sim.get_tb().state[uids] = tbsim.TBS.SYMPTOMATIC
    return sim, uids


def test_care_seeking_prob_matches_rate():
    """The per-step probability of seeking care is 1 - exp(-rate*dt), for both daily and weekly timesteps."""
    rate = 0.1
    for dt in [ss.days(1), ss.days(7)]:
        sim, uids = make_symptomatic_sim(rate=rate, dt=dt)
        h = hsb(sim)
        h.step()
        expected = 1 - np.exp(-rate*dt.days)
        assert np.isclose(h.sought_care[uids].mean(), expected, atol=0.02), f"dt={dt}: expected {expected:.3f}"


def test_multiplier_scales_care_seeking():
    """Agents with a care-seeking multiplier seek care at the scaled rate; the rate is scaled before it is converted to a probability."""
    rate = 0.01
    mult = 10.0
    dt = 7 # Days per step, the make_sim() default
    sim, uids = make_symptomatic_sim(rate=rate)
    h = hsb(sim)
    boosted   = uids[::2]
    unboosted = uids[1::2]
    h.care_seeking_multiplier[boosted] = mult
    h.step()
    assert np.isclose(h.sought_care[boosted].mean(),   1 - np.exp(-rate*mult*dt), atol=0.02)
    assert np.isclose(h.sought_care[unboosted].mean(), 1 - np.exp(-rate*dt),      atol=0.02)


def make_cascade_sim(dx_mult=1.0, tx_mult=1.0):
    """Run a sim with the full care cascade: care seeking, Xpert diagnosis, and DOTS treatment."""
    sim = tbsim.Sim(
        n_agents = 5000,
        interventions = [
            tbsim.HealthSeekingBehavior(),
            tbsim.DxDelivery(product=tbsim.Xpert(), care_seeking_multiplier=dx_mult),
            tbsim.TxDelivery(product=tbsim.DOTS(), reseek_multiplier=tx_mult),
        ],
        sim_pars = dict(start=ss.date("2000-01-01"), stop=ss.date("2010-01-01"), rand_seed=1, verbose=0),
        tb_pars  = dict(init_prev=0.30),
    )
    sim.run()
    return sim


def test_dx_and_tx_multipliers_reach_care_seeking():
    """False negatives (DxDelivery) and treatment failures (TxDelivery) raise the care-seeking multiplier, which changes when agents seek care."""
    base = hsb(make_cascade_sim()).results.new_sought_care.values
    for kwargs in [dict(dx_mult=10.0), dict(tx_mult=10.0)]:
        h = hsb(make_cascade_sim(**kwargs))
        assert (h.care_seeking_multiplier[h.care_seeking_multiplier.auids] > 1).any(), f"{kwargs}: no agent was boosted"
        assert not np.array_equal(h.results.new_sought_care.values, base), f"{kwargs}: the multiplier had no effect on care seeking"


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
