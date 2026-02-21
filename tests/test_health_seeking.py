"""Tests for HealthSeekingBehavior (TB_LSHTM model)."""

import numpy as np
import starsim as ss
import tbsim 
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior


def make_sim(n_agents=200, stop=ss.date("2005-12-31"), tb_pars=None, hsb_pars=None):
    tb_pars  = tb_pars  or {}
    hsb_pars = hsb_pars or {}
    return ss.Sim(
        people      = ss.People(n_agents=n_agents),
        networks    = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0)),
        diseases    = tbsim.tb_lshtm.TB_LSHTM(pars=tb_pars),
        interventions = HealthSeekingBehavior(pars=hsb_pars),
        dt    = ss.days(7),
        start = ss.date("2000-01-01"),
        stop  = stop,
        verbose = 0,
    )


def hsb(sim):
    return sim.interventions[0]


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
    assert hsb(sim).n_care_sought[:].max() <= 1


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


def test_missing_tb_lshtm_raises():
    """A sim without tb_lshtm raises an explicit error on init."""
    import pytest
    sim = ss.Sim(
        people      = ss.People(n_agents=50),
        networks    = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0)),
        diseases    = ss.SIR(),
        interventions = HealthSeekingBehavior(),
        dt = ss.days(7), start = ss.date("2000-01-01"), stop = ss.date("2002-12-31"),
        verbose = 0,
    )
    with pytest.raises(ValueError, match="tb_lshtm"):
        sim.init()


if __name__ == '__main__':
    import pytest, sys
    sys.exit(pytest.main([__file__, '-v']))
