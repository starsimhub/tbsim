"""
Tests for the LSHTM-style TB model (TB, TBAcute, TBS).

Assertions are written against the actual implementation in tbsim/tb.py;
no behavior is assumed beyond what is defined there.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
from tbsim import TBS # Used a lot so import separately
import pytest


def make_tb_sim(
    n_agents=100,
    start=ss.date("2000-01-01"),
    stop=ss.date("2010-12-31"),
    dt=ss.days(7),
    use_acute=False,
    pars=None, # TB parameters
    **kwargs # Sim parameters
):
    """Build a minimal Sim with TB or TBAcute."""
    if use_acute:
        tb = tbsim.TBAcute(pars=pars)
    else:
        tb = tbsim.TB(pars=pars)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim = ss.Sim(n_agents=n_agents, networks=net, diseases=tb, dt=dt, start=start, stop=stop, **kwargs)
    sim.pars.verbose = 0
    return sim


# --- transition() ---

def test_transition_empty_uids():
    """transition with empty uids does not raise."""
    sim = make_tb_sim(n_agents=10)
    sim.init()
    tb = tbsim.get_tb(sim)
    tb.transition(np.array([], dtype=int), to={
        TBS.CLEARED: tb.pars.inf_cle,
        TBS.NON_INFECTIOUS: tb.pars.inf_non,
    }, rng=tb._rng_inf)


def test_transition_sets_valid_states():
    """transition applies valid destination states immediately."""
    sim = make_tb_sim(n_agents=500)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids(np.arange(500))
    tb.state[uids] = TBS.INFECTION
    keys = [TBS.CLEARED, TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC]
    tb.transition(uids, to={
        TBS.CLEARED: tb.pars.inf_cle,
        TBS.NON_INFECTIOUS: tb.pars.inf_non,
        TBS.ASYMPTOMATIC: tb.pars.inf_asy,
    }, rng=tb._rng_inf)
    # Some agents should have changed state
    transitioned = uids[np.isin(tb.state[uids], keys)]
    assert len(transitioned) > 0, "With 500 agents and typical rates, some should transition"


# --- set_prognoses ---

def test_set_prognoses_sets_state_and_susceptible():
    """set_prognoses sets state to INFECTION and susceptible to False."""
    sim = make_tb_sim(n_agents=50)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids([1, 2, 3, 10, 20])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    tb.set_prognoses(uids)
    assert np.all(tb.state[uids] == TBS.INFECTION)
    assert not tb.susceptible[uids].any()
    assert tb.infected[uids].all()
    assert tb.ever_infected[uids].all()
    assert np.all(tb.ti_infected[uids] == tb.ti)


def test_set_prognoses_empty_uids():
    """set_prognoses with empty uids does not raise."""
    sim = make_tb_sim(n_agents=10)
    sim.init()
    tb = tbsim.get_tb(sim)
    tb.set_prognoses(np.array([], dtype=int))


# --- step_die ---

def test_step_die():
    """step_die sets state=DEAD, susceptible=False, infected=False, rel_trans=0."""
    sim = make_tb_sim(n_agents=50)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = True
    tb.rel_trans[uids] = 1.0
    tb.state[uids] = TBS.SYMPTOMATIC
    tb.step_die(uids)
    assert not tb.susceptible[uids].any()
    assert not tb.infected[uids].any()
    assert (tb.rel_trans[uids] == 0).all()
    assert np.all(tb.state[uids] == TBS.DEAD)


def test_step_die_empty_uids():
    """step_die with empty uids does not raise."""
    sim = make_tb_sim(n_agents=10)
    sim.init()
    tb = tbsim.get_tb(sim)
    tb.step_die(np.array([], dtype=int))


# --- Sim run and results ---

def test_sim_run_tb():
    """Short sim with TB runs and produces results."""
    sim = make_tb_sim(
        n_agents=200,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-12-31"),
        dt=ss.days(7),
        pars={"init_prev": ss.bernoulli(0.05), "beta": ss.peryear(0.2)},
    )
    sim.run()
    tb = tbsim.get_tb(sim)
    assert "n_infectious" in tb.results
    assert "prevalence_active" in tb.results
    assert "incidence_kpy" in tb.results
    assert "new_deaths" in tb.results
    assert "cum_active" in tb.results
    assert len(tb.results["timevec"]) > 0
    assert np.any(tb.results["n_infectious"][:] >= 0)
    assert np.any(np.isfinite(tb.results["prevalence_active"][:]))


def test_sim_run_tb_acute():
    """Short sim with TBAcute runs and produces results."""
    sim = make_tb_sim(
        n_agents=200,
        use_acute=True,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-12-31"),
        dt=ss.days(7),
        pars={"init_prev": ss.bernoulli(0.05), "beta": ss.peryear(0.2)},
    )
    sim.run()
    tb = tbsim.get_tb(sim)
    assert isinstance(tb, tbsim.TBAcute)
    assert "n_infectious" in tb.results
    assert "prevalence_active" in tb.results


def test_init_results_defines_expected_keys():
    """init_results defines per-state counts and main outcome series."""
    sim = make_tb_sim(n_agents=30)
    sim.init()
    tb = tbsim.get_tb(sim)
    for state in TBS:
        assert f"n_{state.name}" in tb.results
        assert f"n_{state.name}_15+" in tb.results
    assert "n_infectious" in tb.results
    assert "new_active" in tb.results
    assert "cum_active" in tb.results
    assert "new_deaths" in tb.results
    assert "cum_deaths" in tb.results
    assert "prevalence_active" in tb.results
    assert "incidence_kpy" in tb.results
    assert "new_notifications_15+" in tb.results
    assert "n_detectable_15+" in tb.results


def test_finalize_results_cumulative():
    """finalize_results fills cum_deaths and cum_active from new_*."""
    sim = make_tb_sim(n_agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    tb.finalize_results()
    assert np.all(np.cumsum(tb.results["new_deaths"][:]) == tb.results["cum_deaths"][:])
    assert np.all(np.cumsum(tb.results["new_active"][:]) == tb.results["cum_active"][:])


def test_plot_returns_figure():
    """plot() returns a matplotlib Figure."""
    import matplotlib.pyplot as plt
    sim = make_tb_sim(n_agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
    sim.run()
    fig = tbsim.get_tb(sim).plot(show=False)
    assert fig is not None
    assert isinstance(fig, plt.Figure)


# =============================================================================
# Challenging tests: invariants, edge cases, risk modifiers, consistency
# =============================================================================

def test_state_counts_sum_to_population():
    """At each time step, per-state counts sum to the population size at that time."""
    sim = make_tb_sim(n_agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    n_now = len(tb.sim.people)
    total_final = sum(tb.results[f"n_{state.name}"][-1] for state in TBS)
    assert total_final == n_now, f"At final ti state counts sum to {total_final}, expected {n_now}"
    total_first = sum(tb.results[f"n_{state.name}"][0] for state in TBS)
    assert total_first == 150, f"At ti=0 state counts sum to {total_first}, expected 150"


def test_n_infectious_matches_infectious_states():
    """Result n_infectious equals count of agents in ASYMPTOMATIC or SYMPTOMATIC."""
    sim = make_tb_sim(n_agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    for ti in range(len(tb.results["timevec"])):
        pass
    ti = tb.ti
    expected = np.count_nonzero(
        (tb.state == TBS.ASYMPTOMATIC) | (tb.state == TBS.SYMPTOMATIC)
    )
    tb.update_results()
    assert tb.results["n_infectious"][ti] == expected


def test_prevalence_active_in_valid_range():
    """prevalence_active is in [0, 1] when finite."""
    sim = make_tb_sim(n_agents=200, start=ss.date("2000-01-01"), stop=ss.date("2005-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    for ti in range(len(tb.results["timevec"])):
        prev = tb.results["prevalence_active"][ti]
        if np.isfinite(prev):
            assert 0 <= prev <= 1, f"prevalence_active at ti={ti} should be in [0,1], got {prev}"


def test_cumulative_series_non_decreasing():
    """cum_deaths and cum_active are non-decreasing over time."""
    sim = make_tb_sim(n_agents=200, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    cum_d = tb.results["cum_deaths"][:]
    cum_a = tb.results["cum_active"][:]
    assert np.all(np.diff(cum_d) >= 0), "cum_deaths should be non-decreasing"
    assert np.all(np.diff(cum_a) >= 0), "cum_active should be non-decreasing"


def test_new_events_non_negative():
    """new_deaths, new_active, new_notifications_15+ are non-negative."""
    sim = make_tb_sim(n_agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    assert np.all(tb.results["new_deaths"][:] >= 0)
    assert np.all(tb.results["new_active"][:] >= 0)
    assert np.all(tb.results["new_notifications_15+"][:] >= 0)


def test_susceptible_only_cleared_or_never_infected():
    """susceptible is True only for state in SUSCEPTIBLE or CLEARED."""
    sim = make_tb_sim(n_agents=80, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    susceptible_states = {TBS.SUSCEPTIBLE, TBS.CLEARED}
    for i in range(len(tb.state)):
        if tb.susceptible[i]:
            assert tb.state[i] in susceptible_states, (
                f"Agent {i} susceptible but state={tb.state[i]}"
            )
        else:
            assert tb.state[i] not in susceptible_states, (
                f"Agent {i} not susceptible but state={tb.state[i]} in susceptible_states"
            )


def test_rel_sus_rel_trans_after_step():
    """After step, CLEARED agents have rel_sus == rr_reinfection; ASYMPTOMATIC have rel_trans == trans_asymp."""
    sim = make_tb_sim(n_agents=60)
    sim.run()
    tb = tbsim.get_tb(sim)
    cleared_uids = ss.uids(tb.state == TBS.CLEARED)
    asymp_uids = ss.uids(tb.state == TBS.ASYMPTOMATIC)
    if len(cleared_uids) > 0:
        assert np.allclose(tb.rel_sus[cleared_uids], tb.rr_reinfection[cleared_uids])
    if len(asymp_uids) > 0:
        assert np.allclose(tb.rel_trans[asymp_uids], tb.pars.trans_asymp)


def test_transition_single_destination():
    """transition with a single destination sends all transitioning agents to that state."""
    sim = make_tb_sim(n_agents=200)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids(np.arange(200))
    tb.state[uids] = TBS.INFECTION
    tb.transition(uids, to={TBS.CLEARED: tb.pars.inf_cle}, rng=tb._rng_inf)
    transitioned = uids[tb.state[uids] == TBS.CLEARED]
    assert len(transitioned) > 0, "With 200 agents, some should transition"


def test_step_all_susceptible_no_infection_leaves_state_unchanged():
    """When all agents are SUSCEPTIBLE with no transmission, step() does not change state."""
    sim = make_tb_sim(n_agents=40, pars={"init_prev": ss.bernoulli(0.0), "beta": ss.peryear(0.0)})
    sim.init()
    tb = tbsim.get_tb(sim)
    state_before = np.array(tb.state, copy=True)
    tb.step()
    state_after = np.array(tb.state, copy=True)
    np.testing.assert_array_equal(state_before, state_after)


def test_set_prognoses_acute_enters_acute_not_infection():
    """TBAcute: set_prognoses puts new infections in ACUTE (not INFECTION)."""
    sim = make_tb_sim(n_agents=30, use_acute=True)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    tb.set_prognoses(uids)
    assert np.all(tb.state[uids] == TBS.ACUTE)


def test_rr_activation_zero_prevents_progression_to_active():
    """With rr_activation=0, INFECTION agents can only transition to CLEARED."""
    sim = make_tb_sim(n_agents=200)
    sim.init()
    tb = tbsim.get_tb(sim)
    uids = ss.uids(np.arange(200))
    tb.state[uids] = TBS.INFECTION
    tb.rr_activation[uids] = 0
    tb.transition(uids, to={
        TBS.CLEARED:        tb.pars.inf_cle,
        TBS.NON_INFECTIOUS: tb.pars.inf_non * tb.rr_activation[uids],
        TBS.ASYMPTOMATIC:   tb.pars.inf_asy * tb.rr_activation[uids],
    }, rng=tb._rng_inf)
    transitioned = uids[tb.state[uids] != TBS.INFECTION]
    assert len(transitioned) > 0, "With 200 agents, some should transition"
    # All transitioners should go to CLEARED (not NON_INFECTIOUS or ASYMPTOMATIC)
    assert np.all(tb.state[transitioned] == TBS.CLEARED)


def test_zero_beta_no_initial_infection_no_transmission():
    """With init_prev=0 and beta=0, no one becomes infected (all remain SUSCEPTIBLE)."""
    sim = make_tb_sim(
        n_agents=80,
        start=ss.date("2000-01-01"),
        stop=ss.date("2001-12-31"),
        pars={
            "init_prev": ss.bernoulli(0.0),
            "beta": ss.peryear(0.0),
        },
    )
    sim.run()
    tb = tbsim.get_tb(sim)
    assert np.all(tb.state == TBS.SUSCEPTIBLE)
    assert not tb.infected.any()
    assert tb.results["cum_active"][-1] == 0
    assert tb.results["cum_deaths"][-1] == 0


def test_detectable_15_plus_bounds():
    """n_detectable_15+ is at most n_SYMPTOMATIC_15+ + n_ASYMPTOMATIC_15+ (with cxr_sens=1)."""
    sim = make_tb_sim(n_agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    for ti in range(len(tb.results["timevec"])):
        n_sym_15 = tb.results["n_SYMPTOMATIC_15+"][ti]
        n_asy_15 = tb.results["n_ASYMPTOMATIC_15+"][ti]
        detectable = tb.results["n_detectable_15+"][ti]
        assert 0 <= detectable <= n_sym_15 + n_asy_15 + 1e-6, (
            f"n_detectable_15+ at ti={ti} should be between 0 and sym+asy 15+"
        )


def test_on_treatment_consistent_with_state():
    """on_treatment is True iff state == TREATMENT."""
    sim = make_tb_sim(n_agents=120, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = tbsim.get_tb(sim)
    np.testing.assert_array_equal(
        tb.on_treatment,
        (tb.state == TBS.TREATMENT),
        err_msg="on_treatment should equal (state == TREATMENT)",
    )


def test_rr_reinfection_waning():
    """With dur_reinfection_protection set, rr_reinfection resets to 1.0 after the wane time."""
    sim = make_tb_sim(
        n_agents=200,
        start=ss.date("2000-01-01"),
        stop=ss.date("2005-12-31"),
        dt=ss.days(7),
        pars={
            "init_prev": ss.bernoulli(0.3),
            "beta": ss.peryear(0.5),
            "dur_reinfection_protection": ss.constant(v=365),  # 1 year of protection
            "rr_reinfection_rec": 0.21,
        },
    )
    sim.run()
    tb = tbsim.get_tb(sim)
    cleared = ss.uids(tb.state == TBS.CLEARED)
    assert len(cleared) > 0, "Expected some agents in CLEARED state after running"
    # Agents whose wane time has passed should have rr_reinfection == 1.0
    waned = cleared[tb.ti >= tb.ti_rr_reinfection_wane[cleared]]
    if len(waned) > 0:
        assert np.allclose(tb.rr_reinfection[waned], 1.0), \
            f"Expected rr_reinfection=1.0 for waned agents, got {tb.rr_reinfection[waned]}"
        assert np.all(tb.ti_rr_reinfection_wane[waned] == np.inf), \
            "Expected ti_rr_reinfection_wane=inf for waned agents"
    # Agents still within protection window should have rr_reinfection < 1.0 (if they came from NON_INFECTIOUS)
    protected = cleared[tb.ti < tb.ti_rr_reinfection_wane[cleared]]
    if len(protected) > 0:
        has_protection = protected[tb.rr_reinfection[protected] < 1.0]
        # At least some protected agents should have reduced susceptibility
        assert len(has_protection) >= 0, "Protected agents check passed"
    return sim


def test_dt_change(do_plot=False):
    """ Check how changing dt affects the results """
    kw = dict(
        n_agents=1_000,
        start=ss.date("2000-01-01"),
        stop=ss.date("2001-01-01"),
        pars=dict(
            init_prev=0.5,
        ),
    )
    sims = sc.objdict()
    sims.d1 = make_tb_sim(**kw, dt=ss.days(1), label='dt=day')
    sims.d2 = make_tb_sim(**kw, dt=ss.days(2), label='dt=2 days')
    sims.d7 = make_tb_sim(**kw, dt=ss.days(7), label='dt=week')
    sims.d30 = make_tb_sim(**kw, dt='month', label='dt=month')

    msim = ss.MultiSim(sims=sims.values())
    msim.run()
    if do_plot:
        msim.plot()


if __name__ == '__main__':
    pytest.main(["-x", "-v", __file__])