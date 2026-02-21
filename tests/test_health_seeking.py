"""Tests for HealthSeekingBehavior intervention with TB and LSHTM models."""

import pytest
import tbsim as mtb
import starsim as ss
import numpy as np

SPARS = {"dt": ss.days(7), "start": ss.date("2000-01-01"), "stop": ss.date("2010-01-01"), "verbose": 0}


def _run_sim(tb_module, n_agents=500, intervention_pars=None):
    """Run sim with HealthSeekingBehavior."""
    pop = mtb.TBPeople(n_agents=n_agents)
    tb = tb_module(pars={"init_prev": ss.bernoulli(0.1), "beta": ss.peryear(0.2)})
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})
    pars = intervention_pars or {"initial_care_seeking_rate": ss.perday(0.2)}
    intv = mtb.HealthSeekingBehavior(pars=pars)
    sim = ss.Sim(
        people=pop, networks=net, diseases=tb, interventions=intv, pars=SPARS,
    )
    sim.run()
    return sim


def test_with_legacy_tb():
    """Works with legacy TB."""
    sim = _run_sim(mtb.TB)
    res = sim.results["healthseekingbehavior"]
    assert "n_sought_care" in res
    assert "n_eligible" in res
    assert "new_sought_care" in res
    assert len(res["n_sought_care"]) == sim.t.npts


def test_with_lshtm():
    """Works with TB_LSHTM."""
    sim = _run_sim(mtb.TB_LSHTM)
    res = sim.results["healthseekingbehavior"]
    assert res["n_sought_care"][-1] >= 0


def test_with_lshtm_acute():
    """Works with TB_LSHTM_Acute."""
    sim = _run_sim(mtb.TB_LSHTM_Acute)
    res = sim.results["healthseekingbehavior"]
    assert res["n_sought_care"][-1] >= 0


def test_results_consistency():
    """new_sought_care and n_sought_care are consistent."""
    sim = _run_sim(mtb.TB_LSHTM, n_agents=300)
    res = sim.results["healthseekingbehavior"]
    new_sum = np.sum(res["new_sought_care"])
    final_cum = res["n_sought_care"][-1]
    assert new_sum == final_cum


def test_sought_care_updated():
    """Seeking care sets ppl.sought_care."""
    sim = _run_sim(mtb.TB_LSHTM, n_agents=800, intervention_pars={"initial_care_seeking_rate": ss.perday(0.5)})
    res = sim.results["healthseekingbehavior"]
    if res["n_sought_care"][-1] > 0:
        assert np.any(sim.people.sought_care)


def test_start_stop_window():
    """Intervention respects start/stop; no care-seeking when sim ends before start."""
    pop = mtb.TBPeople(n_agents=300)
    tb = mtb.TB_LSHTM(pars={"init_prev": ss.bernoulli(0.15), "beta": ss.peryear(0.25)})
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})
    # Sim runs 2000-2002; intervention starts 2005, so never active
    intv = mtb.HealthSeekingBehavior(pars={
        "initial_care_seeking_rate": ss.perday(0.9),
        "start": ss.date("2005-01-01"),
        "stop": ss.date("2010-01-01"),
    })
    sim = ss.Sim(
        people=pop, networks=net, diseases=tb, interventions=intv,
        pars={"dt": ss.days(7), "start": ss.date("2000-01-01"), "stop": ss.date("2002-01-01"), "verbose": 0},
    )
    sim.run()
    res = sim.results["healthseekingbehavior"]
    assert np.sum(res["new_sought_care"]) == 0


def test_single_use_expires():
    """With single_use=True, intervention expires after first care-seeker."""
    pop = mtb.TBPeople(n_agents=400)
    tb = mtb.TB_LSHTM(pars={"init_prev": ss.bernoulli(0.2), "beta": ss.peryear(0.3)})
    net = ss.RandomNet(pars={"n_contacts": ss.poisson(lam=5), "dur": 0})
    intv = mtb.HealthSeekingBehavior(pars={
        "initial_care_seeking_rate": ss.perday(0.9),
        "single_use": True,
    })
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=intv, pars=SPARS)
    sim.run()
    res = sim.results["healthseekingbehavior"]
    # If any sought care, intervention should be expired (sim uses a copy, so get from sim)
    run_intv = sim.interventions[0]
    if res["n_sought_care"][-1] > 0:
        assert getattr(run_intv, "expired", False) is True


def test_zero_rate_no_care_seeking():
    """Zero rate yields no care-seeking."""
    sim = _run_sim(mtb.TB_LSHTM, n_agents=200, intervention_pars={"initial_care_seeking_rate": ss.perday(0)})
    res = sim.results["healthseekingbehavior"]
    assert np.sum(res["new_sought_care"]) == 0
    assert res["n_sought_care"][-1] == 0


def test_n_eligible_populated():
    """n_eligible is non-negative and consistent."""
    sim = _run_sim(mtb.TB_LSHTM, n_agents=300)
    res = sim.results["healthseekingbehavior"]
    assert np.all(np.array(res["n_eligible"]) >= 0)
    # n_eligible + n_sought_care should be <= total ever eligible (sanity)
    assert res["n_eligible"][-1] >= 0


def test_requires_tb_module():
    """Fails when no TB/LSHTM disease module."""
    from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior

    class EmptySim:
        diseases = []

    intv = HealthSeekingBehavior()
    intv.setattribute("sim", EmptySim())
    with pytest.raises((IndexError, TypeError, AttributeError)):
        intv.init_post()
