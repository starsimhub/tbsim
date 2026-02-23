"""
Tests for the Immigration demographics module (LSHTM-only).

These tests are scoped to TB_LSHTM / TB_LSHTM_Acute and the current Starsim API.
"""

import numpy as np
import pytest
import starsim as ss

from tbsim.tb_lshtm import TB_LSHTM, TB_LSHTM_Acute, TBSL
from tbsim.interventions.immigration import Immigration


def _make_sim(*, acute: bool, immigration_pars: dict, dt=ss.days(1), start="2020-01-01", stop="2020-01-15"):
    # Keep this small. We only need the module wiring to work.
    tb = TB_LSHTM_Acute() if acute else TB_LSHTM()
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=2), "dur": 0})
    imm = Immigration(pars=immigration_pars)
    sim = ss.Sim(
        n_agents=200,
        diseases=tb,
        networks=[net],
        demographics=[imm],
        start=ss.date(start),
        stop=ss.date(stop),
        dt=dt,
        verbose=0,
    )
    return sim


def test_expected_immigrants_matches_starsim_to_events():
    """expected_immigrants_per_timestep matches Starsim's Rate.to_events(dt)."""
    sim = _make_sim(
        acute=False,
        immigration_pars=dict(
            immigration_rate=ss.freqperyear(365.0),
            rel_immigration=1.0,
            tb_state_distribution={"SUSCEPTIBLE": 1.0},
        ),
        dt=ss.days(1),
        stop="2020-01-03",
    )
    sim.init()
    imm = sim.demographics[0]

    expected = float(ss.freqperyear(365.0).to_events(ss.days(1)))
    got = float(imm.expected_immigrants_per_timestep())
    assert got == expected


def test_tb_state_distribution_rejects_acute_without_acute_model():
    """ACUTE in tb_state_distribution is only valid for TB_LSHTM_Acute."""
    sim = _make_sim(
        acute=False,
        immigration_pars=dict(
            immigration_rate=ss.freqperyear(1.0),
            tb_state_distribution={"ACUTE": 1.0},
        ),
        stop="2020-01-03",
    )
    sim.init()
    imm = sim.demographics[0]
    tb = sim.diseases[0]

    new_uids = sim.people.grow(1)
    with pytest.raises(ValueError, match="includes ACUTE"):
        imm._init_tb_lshtm(tb, new_uids)


def test_sim_run_adds_immigrants_and_initializes_fields():
    """Run a short sim; immigrants get flags, TB state codes, and HHIDs."""
    sim = _make_sim(
        acute=False,
        immigration_pars=dict(
            # Make this effectively deterministic: P(no arrivals) is ~0 over this run.
            immigration_rate=ss.freqperyear(36_500.0),  # ~100/day
            rel_immigration=1.0,
            age_distribution={0: 1.0},
            tb_state_distribution={"INFECTION": 1.0},
        ),
        dt=ss.days(1),
        stop="2020-01-11",
    )
    sim.run()
    imm = sim.demographics[0]
    tb = sim.diseases[0]

    n_added = int(np.sum(imm.results["n_immigrants"]))
    assert n_added > 0

    imm_uids = np.where(imm.is_immigrant)[0]
    assert len(imm_uids) == n_added

    # These should be set on arrival and never revert.
    assert np.isfinite(imm.immigration_time[imm_uids]).all()
    assert (imm.age_at_immigration[imm_uids] >= 0).all()
    assert (imm.age_at_immigration[imm_uids] <= 85).all()

    # TB state codes: stored at arrival, and valid for LSHTM.
    valid = set(int(s) for s in TBSL)
    codes = imm.immigration_tb_status[imm_uids]
    assert (codes >= 0).all()
    assert all(int(c) in valid for c in codes)

    # The live TB module should also have valid LSHTM states for immigrants.
    states = tb.state[imm_uids]
    assert all(int(s) in valid for s in states)

    # No household network is provided by Starsim 3.1.1. This should still run.
    assert (imm.hhid[imm_uids] == -1).all()

