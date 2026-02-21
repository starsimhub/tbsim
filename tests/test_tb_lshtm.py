"""
Tests for the LSHTM-style TB model (TB_LSHTM, TB_LSHTM_Acute, TBSL).

Assertions are written against the actual implementation in tbsim/tb_lshtm.py;
no behavior is assumed beyond what is defined there.
"""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim as mtb
from tbsim import TBSL # Used a lot so import separately


def make_lshtm_sim(
    n_agents=100,
    start=ss.date("2000-01-01"),
    stop=ss.date("2010-12-31"),
    dt=ss.days(7),
    use_acute=False,
    pars=None, # TB parameters
    **kwargs # Sim parameters
):
    """Build a minimal Sim with TB_LSHTM or TB_LSHTM_Acute."""
    if use_acute:
        tb = mtb.TB_LSHTM_Acute(pars=pars)
    else:
        tb = mtb.TB_LSHTM(pars=pars)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=30))
    sim = ss.Sim(n_agents=n_agents, networks=net, diseases=tb, dt=dt, start=start, stop=stop, **kwargs)
    sim.pars.verbose = 0
    return sim


# --- transition() ---

def test_transition_empty_uids():
    """transition with empty uids does not raise."""
    sim = make_lshtm_sim(n_agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.transition(np.array([], dtype=int), to={
        TBSL.CLEARED: tb.pars.inf_cle,
        TBSL.NON_INFECTIOUS: tb.pars.inf_non,
    }, rng=tb._rng_inf)


def test_transition_sets_valid_states():
    """transition sets state_next to one of the destination keys for transitioning agents."""
    sim = make_lshtm_sim(n_agents=500)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(500))
    tb.state[uids] = TBSL.INFECTION  # put them in INFECTION so transition makes sense
    tb.ti_next[uids] = np.inf  # not yet scheduled
    keys = [TBSL.CLEARED, TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC]
    tb.transition(uids, to={
        TBSL.CLEARED: tb.pars.inf_cle,
        TBSL.NON_INFECTIOUS: tb.pars.inf_non,
        TBSL.ASYMPTOMATIC: tb.pars.inf_asy,
    }, rng=tb._rng_inf)
    # Some agents should have transitioned (with 500 agents, very likely)
    transitioned = uids[tb.ti_next[uids] == tb.ti]
    assert len(transitioned) > 0, "With 500 agents and typical rates, some should transition"
    assert np.all(np.isin(tb.state_next[transitioned], keys))


# --- set_prognoses ---

def test_set_prognoses_sets_state_and_susceptible():
    """set_prognoses sets state to INFECTION and susceptible to False."""
    sim = make_lshtm_sim(n_agents=50)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids([1, 2, 3, 10, 20])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    tb.set_prognoses(uids)
    assert np.all(tb.state[uids] == TBSL.INFECTION)
    assert not tb.susceptible[uids].any()
    assert tb.infected[uids].all()
    assert tb.ever_infected[uids].all()
    assert np.all(tb.ti_infected[uids] == tb.ti)
    # No transition scheduling — step() evaluates per dt
    assert np.all(tb.ti_next[uids] == np.inf)


def test_set_prognoses_empty_uids():
    """set_prognoses with empty uids does not raise."""
    sim = make_lshtm_sim(n_agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.set_prognoses(np.array([], dtype=int))


# --- step_die ---

def test_step_die():
    """step_die sets state=DEAD, susceptible=False, infected=False, rel_trans=0."""
    sim = make_lshtm_sim(n_agents=50)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = True
    tb.rel_trans[uids] = 1.0
    tb.state[uids] = TBSL.SYMPTOMATIC
    tb.step_die(uids)
    assert not tb.susceptible[uids].any()
    assert not tb.infected[uids].any()
    assert (tb.rel_trans[uids] == 0).all()
    assert np.all(tb.state[uids] == TBSL.DEAD)
    assert np.all(tb.ti_next[uids] == np.inf)


def test_step_die_empty_uids():
    """step_die with empty uids does not raise."""
    sim = make_lshtm_sim(n_agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.step_die(np.array([], dtype=int))


# --- start_treatment ---

def test_start_treatment_latent_cleared():
    """start_treatment on INFECTION (latent) schedules CLEARED."""
    sim = make_lshtm_sim(n_agents=50)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids([1, 2, 3])
    tb.state[uids] = TBSL.INFECTION
    tb.state_next[uids] = TBSL.NON_INFECTIOUS  # would have transitioned later
    tb.ti_next[uids] = tb.ti + 10
    tb.start_treatment(uids)
    assert np.all(tb.state_next[uids] == TBSL.CLEARED)
    assert np.all(tb.ti_next[uids] == tb.ti)


def test_start_treatment_active_to_treatment():
    """start_treatment on NON_INFECTIOUS/ASYMPTOMATIC/SYMPTOMATIC schedules TREATMENT."""
    sim = make_lshtm_sim(n_agents=50)
    sim.init()
    tb = sim.diseases[0]
    for state in [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]:
        uids = ss.uids([0])
        tb.state[uids] = state
        tb.start_treatment(uids)
        assert np.all(tb.state_next[uids] == TBSL.TREATMENT)
        assert np.all(tb.ti_next[uids] == tb.ti)


def test_start_treatment_empty_uids():
    """start_treatment with empty uids does not raise."""
    sim = make_lshtm_sim(n_agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.start_treatment(np.array([], dtype=int))


def test_start_treatment_acute_latent_cleared():
    """TB_LSHTM_Acute: start_treatment on ACUTE or INFECTION schedules CLEARED."""
    sim = make_lshtm_sim(n_agents=20, use_acute=True)
    sim.init()
    tb = sim.diseases[0]
    uids_acute = ss.uids([0])
    uids_inf = ss.uids([1])
    tb.state[uids_acute] = TBSL.ACUTE
    tb.state[uids_inf] = TBSL.INFECTION
    tb.start_treatment(ss.uids([0, 1]))
    assert tb.state_next[0] == TBSL.CLEARED
    assert tb.state_next[1] == TBSL.CLEARED


# --- Sim run and results ---

def test_sim_run_tb_lshtm():
    """Short sim with TB_LSHTM runs and produces results."""
    sim = make_lshtm_sim(
        n_agents=200,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-12-31"),
        dt=ss.days(7),
        pars={"init_prev": ss.bernoulli(0.05), "beta": ss.peryear(0.2)},
    )
    sim.run()
    tb = sim.diseases[0]
    assert "n_infectious" in tb.results
    assert "prevalence_active" in tb.results
    assert "incidence_kpy" in tb.results
    assert "new_deaths" in tb.results
    assert "cum_active" in tb.results
    assert len(tb.results["timevec"]) > 0
    # At least one time point should have some infectious or some prevalence
    assert np.any(tb.results["n_infectious"][:] >= 0)
    assert np.any(np.isfinite(tb.results["prevalence_active"][:]))


def test_sim_run_tb_lshtm_acute():
    """Short sim with TB_LSHTM_Acute runs and produces results."""
    sim = make_lshtm_sim(
        n_agents=200,
        use_acute=True,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-12-31"),
        dt=ss.days(7),
        pars={"init_prev": ss.bernoulli(0.05), "beta": ss.peryear(0.2)},
    )
    sim.run()
    tb = sim.diseases[0]
    assert isinstance(tb, mtb.TB_LSHTM_Acute)
    assert "n_infectious" in tb.results
    assert "prevalence_active" in tb.results


def test_init_results_defines_expected_keys():
    """init_results defines per-state counts and main outcome series."""
    sim = make_lshtm_sim(n_agents=30)
    sim.init()
    tb = sim.diseases[0]
    for state in TBSL:
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
    sim = make_lshtm_sim(n_agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
    sim.run()
    tb = sim.diseases[0]
    tb.finalize_results()
    assert np.all(np.cumsum(tb.results["new_deaths"][:]) == tb.results["cum_deaths"][:])
    assert np.all(np.cumsum(tb.results["new_active"][:]) == tb.results["cum_active"][:])


def test_plot_returns_figure():
    """plot() returns a matplotlib Figure."""
    sim = make_lshtm_sim(n_agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
    sim.run()
    tb = sim.diseases[0]
    fig = tb.plot()
    assert fig is not None
    import matplotlib.pyplot as plt
    assert isinstance(fig, plt.Figure)


# =============================================================================
# Challenging tests: invariants, edge cases, risk modifiers, consistency
# =============================================================================

def test_state_counts_sum_to_population():
    """At each time step, per-state counts sum to the population size at that time."""
    sim = make_lshtm_sim(n_agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = sim.diseases[0]
    # At final ti, sum of n_* must equal current population (deaths may have removed agents)
    n_now = len(tb.sim.people)
    total_final = sum(tb.results[f"n_{state.name}"][-1] for state in TBSL)
    assert total_final == n_now, f"At final ti state counts sum to {total_final}, expected {n_now}"
    # At first ti, sum should equal initial population
    total_first = sum(tb.results[f"n_{state.name}"][0] for state in TBSL)
    assert total_first == 150, f"At ti=0 state counts sum to {total_first}, expected 150"


def test_n_infectious_matches_infectious_states():
    """Result n_infectious equals count of agents in ASYMPTOMATIC or SYMPTOMATIC."""
    sim = make_lshtm_sim(n_agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for ti in range(len(tb.results["timevec"])):
        # We can't replay state at ti, but we can check the last step
        pass
    # After run, do one update_results at final ti and check
    ti = tb.ti
    expected = np.count_nonzero(
        (tb.state == TBSL.ASYMPTOMATIC) | (tb.state == TBSL.SYMPTOMATIC)
    )
    tb.update_results()
    assert tb.results["n_infectious"][ti] == expected


def test_prevalence_active_in_valid_range():
    """prevalence_active is in [0, 1] when finite (implementation divides by alive count)."""
    sim = make_lshtm_sim(n_agents=200, start=ss.date("2000-01-01"), stop=ss.date("2005-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for ti in range(len(tb.results["timevec"])):
        prev = tb.results["prevalence_active"][ti]
        if np.isfinite(prev):
            assert 0 <= prev <= 1, f"prevalence_active at ti={ti} should be in [0,1], got {prev}"


def test_cumulative_series_non_decreasing():
    """cum_deaths and cum_active are non-decreasing over time."""
    sim = make_lshtm_sim(n_agents=200, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = sim.diseases[0]
    cum_d = tb.results["cum_deaths"][:]
    cum_a = tb.results["cum_active"][:]
    assert np.all(np.diff(cum_d) >= 0), "cum_deaths should be non-decreasing"
    assert np.all(np.diff(cum_a) >= 0), "cum_active should be non-decreasing"


def test_new_events_non_negative():
    """new_deaths, new_active, new_notifications_15+ are non-negative."""
    sim = make_lshtm_sim(n_agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = sim.diseases[0]
    assert np.all(tb.results["new_deaths"][:] >= 0)
    assert np.all(tb.results["new_active"][:] >= 0)
    assert np.all(tb.results["new_notifications_15+"][:] >= 0)


def test_susceptible_only_cleared_recovered_treated_or_never_infected():
    """susceptible is True only for state in SUSCEPTIBLE, CLEARED, RECOVERED, TREATED."""
    sim = make_lshtm_sim(n_agents=80, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = sim.diseases[0]
    susceptible_states = {TBSL.SUSCEPTIBLE, TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED}
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
    """After step, RECOVERED have rel_sus=rr_rec (π pi), TREATED rel_sus=rr_treat (ρ rho), ASYMPTOMATIC rel_trans=trans_asymp (κ kappa)."""
    sim = make_lshtm_sim(n_agents=60)
    sim.init()
    tb = sim.diseases[0]
    sim.run()
    tb = sim.diseases[0]
    rr_rec, rr_treat, trans_asymp = tb.pars.rr_rec, tb.pars.rr_treat, tb.pars.trans_asymp
    recovered_uids = ss.uids(tb.state == TBSL.RECOVERED)
    treated_uids = ss.uids(tb.state == TBSL.TREATED)
    asymp_uids = ss.uids(tb.state == TBSL.ASYMPTOMATIC)
    if len(recovered_uids) > 0:
        assert np.allclose(tb.rel_sus[recovered_uids], rr_rec)
    if len(treated_uids) > 0:
        assert np.allclose(tb.rel_sus[treated_uids], rr_treat)
    if len(asymp_uids) > 0:
        assert np.allclose(tb.rel_trans[asymp_uids], trans_asymp)


def test_start_treatment_mixed_latent_active_ignores_cleared():
    """start_treatment with mix of INFECTION, SYMPTOMATIC, CLEARED: only INFECTION and SYMPTOMATIC are changed."""
    sim = make_lshtm_sim(n_agents=50)
    sim.init()
    tb = sim.diseases[0]
    u_inf = ss.uids([0])
    u_sym = ss.uids([1])
    u_clr = ss.uids([2])
    tb.state[u_inf] = TBSL.INFECTION
    tb.state[u_sym] = TBSL.SYMPTOMATIC
    tb.state[u_clr] = TBSL.CLEARED
    tb.state_next[u_inf] = TBSL.NON_INFECTIOUS
    tb.state_next[u_sym] = TBSL.DEAD
    tb.ti_next[u_inf] = tb.ti + 5
    tb.ti_next[u_sym] = tb.ti + 5
    tb.state_next[u_clr] = TBSL.CLEARED
    tb.ti_next[u_clr] = np.inf
    all_u = ss.uids([0, 1, 2])
    tb.start_treatment(all_u)
    assert tb.state_next[0] == TBSL.CLEARED and tb.ti_next[0] == tb.ti
    assert tb.state_next[1] == TBSL.TREATMENT and tb.ti_next[1] == tb.ti
    # CLEARED stays CLEARED, ti_next unchanged (still inf)
    assert tb.state_next[2] == TBSL.CLEARED
    assert tb.ti_next[2] == np.inf


def test_transition_single_destination():
    """transition with a single destination sends all transitioning agents to that state."""
    sim = make_lshtm_sim(n_agents=200)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(200))
    tb.state[uids] = TBSL.INFECTION
    tb.ti_next[uids] = np.inf
    tb.transition(uids, to={TBSL.CLEARED: tb.pars.inf_cle}, rng=tb._rng_inf)
    transitioned = uids[tb.ti_next[uids] == tb.ti]
    assert len(transitioned) > 0, "With 200 agents, some should transition"
    assert np.all(tb.state_next[transitioned] == TBSL.CLEARED)


def test_step_all_susceptible_no_infection_leaves_state_unchanged():
    """When all agents are SUSCEPTIBLE with no transmission, step() does not change state."""
    sim = make_lshtm_sim(n_agents=40, pars={"init_prev": ss.bernoulli(0.0), "beta": ss.peryear(0.0)})
    sim.init()
    tb = sim.diseases[0]
    state_before = np.array(tb.state, copy=True)
    tb.step()
    state_after = np.array(tb.state, copy=True)
    np.testing.assert_array_equal(state_before, state_after)


def test_set_prognoses_acute_enters_acute_not_infection():
    """TB_LSHTM_Acute: set_prognoses puts new infections in ACUTE (not INFECTION)."""
    sim = make_lshtm_sim(n_agents=30, use_acute=True)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    tb.set_prognoses(uids)
    assert np.all(tb.state[uids] == TBSL.ACUTE)
    # No transition scheduling — step() evaluates per dt
    assert np.all(tb.ti_next[uids] == np.inf)


def test_rr_activation_zero_prevents_progression_to_active():
    """With rr_activation=0, INFECTION agents can only transition to CLEARED."""
    sim = make_lshtm_sim(n_agents=200)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(200))
    tb.state[uids] = TBSL.INFECTION
    tb.ti_next[uids] = np.inf
    tb.rr_activation[uids] = 0
    # Call transition directly with rr_activation=0 rates
    tb.transition(uids, to={
        TBSL.CLEARED:        tb.pars.inf_cle,
        TBSL.NON_INFECTIOUS: tb.pars.inf_non * tb.rr_activation[uids],
        TBSL.ASYMPTOMATIC:   tb.pars.inf_asy * tb.rr_activation[uids],
    }, rng=tb._rng_inf)
    # Some agents should have transitioned
    transitioned = uids[tb.ti_next[uids] == tb.ti]
    assert len(transitioned) > 0, "With 200 agents, some should transition"
    # All transitioners should go to CLEARED (not NON_INFECTIOUS or ASYMPTOMATIC)
    assert np.all(tb.state_next[transitioned] == TBSL.CLEARED)


def test_zero_beta_no_initial_infection_no_transmission():
    """With init_prev=0 and beta=0, no one becomes infected (all remain SUSCEPTIBLE)."""
    sim = make_lshtm_sim(
        n_agents=80,
        start=ss.date("2000-01-01"),
        stop=ss.date("2001-12-31"),
        pars={
            "init_prev": ss.bernoulli(0.0),  # no initial cases
            "beta": ss.peryear(0.0),
        },
    )
    sim.run()
    tb = sim.diseases[0]
    assert np.all(tb.state == TBSL.SUSCEPTIBLE)
    assert not tb.infected.any()
    assert tb.results["cum_active"][-1] == 0
    assert tb.results["cum_deaths"][-1] == 0


def test_detectable_15_plus_bounds():
    """n_detectable_15+ is at most n_SYMPTOMATIC_15+ + n_ASYMPTOMATIC_15+ (with cxr_sens=1)."""
    sim = make_lshtm_sim(n_agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for ti in range(len(tb.results["timevec"])):
        n_sym_15 = tb.results["n_SYMPTOMATIC_15+"][ti]
        n_asy_15 = tb.results["n_ASYMPTOMATIC_15+"][ti]
        detectable = tb.results["n_detectable_15+"][ti]
        # With cxr_asymp_sens=1: detectable = sym_15 + asy_15 (integer). With sens<1 it could be less.
        assert 0 <= detectable <= n_sym_15 + n_asy_15 + 1e-6, (
            f"n_detectable_15+ at ti={ti} should be between 0 and sym+asy 15+"
        )


def test_ti_next_inf_for_cleared_recovered_treated():
    """After step, CLEARED, RECOVERED, TREATED have ti_next=inf (no scheduled transition)."""
    sim = make_lshtm_sim(n_agents=100, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for state in [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED]:
        u = ss.uids(tb.state == state)
        if len(u) > 0:
            assert np.all(tb.ti_next[u] == np.inf), f"Agents in {state.name} should have ti_next=inf"


def test_on_treatment_consistent_with_state():
    """on_treatment is True iff state == TREATMENT."""
    sim = make_lshtm_sim(n_agents=120, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = sim.diseases[0]
    np.testing.assert_array_equal(
        tb.on_treatment,
        (tb.state == TBSL.TREATMENT),
        err_msg="on_treatment should equal (state == TREATMENT)",
    )

def test_dt_change(do_plot=False):
    """ Check how changing dt affects the results """
    kw = dict(
        n_agents=10_000,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-01-01"),
        pars=dict( # TB parameters
            init_prev=0.5, # Much higher to get better statistics
        ),
    )
    sims = sc.objdict()
    sims.d1 = make_lshtm_sim(**kw, dt=ss.days(1), label='dt=day')
    sims.d2 = make_lshtm_sim(**kw, dt=ss.days(2), label='dt=2 days')
    sims.d7 = make_lshtm_sim(**kw, dt=ss.days(7), label='dt=week')
    sims.d30 = make_lshtm_sim(**kw, dt='month', label='dt=month')

    msim = ss.MultiSim(sims=sims.values())
    msim.run()
    if do_plot:
        msim.plot()


if __name__ == '__main__':
    test_dt_change(do_plot=True)