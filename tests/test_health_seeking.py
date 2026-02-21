"""Tests for HealthSeekingBehavior intervention with TB and LSHTM models."""

import pytest
import tbsim as mtb
import starsim as ss
import numpy as np


def _run_health_seeking_sim(tb_module, n_agents=500, dur_years=0.5):
    """Run a sim with HealthSeekingBehavior."""
    pop = mtb.TBPeople(n_agents=n_agents)
    tb = tb_module(pars={"init_prev": ss.bernoulli(0.1)})
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})
    intervention = mtb.HealthSeekingBehavior(pars={"initial_care_seeking_rate": ss.perday(0.2)})
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=intervention,
        pars={"dt": ss.days(7), "dur": ss.years(dur_years), "verbose": 0},
    )
    sim.run()
    return sim


def test_health_seeking_with_legacy_tb():
    """HealthSeekingBehavior works with legacy TB model."""
    sim = _run_health_seeking_sim(mtb.TB)
    assert "healthseekingbehavior" in sim.results
    res = sim.results["healthseekingbehavior"]
    assert "n_sought_care" in res
    assert "n_eligible" in res
    assert res["n_sought_care"][-1] >= 0


def test_health_seeking_with_lshtm():
    """HealthSeekingBehavior works with TB_LSHTM model."""
    sim = _run_health_seeking_sim(mtb.TB_LSHTM)
    assert "healthseekingbehavior" in sim.results
    res = sim.results["healthseekingbehavior"]
    assert res["n_sought_care"][-1] >= 0


def test_health_seeking_with_lshtm_acute():
    """HealthSeekingBehavior works with TB_LSHTM_Acute model."""
    sim = _run_health_seeking_sim(mtb.TB_LSHTM_Acute)
    assert "healthseekingbehavior" in sim.results
    res = sim.results["healthseekingbehavior"]
    assert res["n_sought_care"][-1] >= 0


def test_health_seeking_rejects_unsupported_disease():
    """HealthSeekingBehavior raises TypeError when no TB/LSHTM disease module exists."""
    from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior

    class EmptySim:
        diseases = []

    intv = HealthSeekingBehavior()
    intv.setattribute("sim", EmptySim())
    with pytest.raises(TypeError, match="HealthSeekingBehavior requires TB or TB_LSHTM"):
        intv._ensure_tb_resolved()
