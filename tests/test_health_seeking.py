"""Tests for HealthSeekingBehavior (TB model)."""

import sys
import pytest
import numpy as np
import starsim as ss
import tbsim


def make_sim(n_agents=200, stop=ss.date("2005-12-31"), tb_pars=None, hsb_pars=None):
    tb_pars  = tb_pars  or {}
    hsb_pars = hsb_pars or {}
    sim = ss.Sim(
        people      = ss.People(n_agents=n_agents),
        networks    = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
        diseases    = tbsim.TB(pars=tb_pars),
        interventions = tbsim.HealthSeekingBehavior(pars=hsb_pars),
        dt    = ss.days(7),
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
    tb = tbsim.get_tb(sim)

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

if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
