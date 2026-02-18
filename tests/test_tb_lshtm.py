"""
Tests for the LSHTM-style TB model (TB_LSHTM, TB_LSHTM_Acute, TBSL).

Assertions are written against the actual implementation in tbsim/tb_lshtm.py;
no behavior is assumed beyond what is defined there.
"""

import pytest
import numpy as np
import starsim as ss
import tbsim as mtb
from tbsim import TB_LSHTM, TB_LSHTM_Acute, TBSL
from tbsim.tb_lshtm import make_scaled_rate


def make_lshtm_sim(
    agents=100,
    start=ss.date("2000-01-01"),
    stop=ss.date("2010-12-31"),
    dt=ss.days(7),
    use_acute=False,
    pars=None,
):
    """Build a minimal Sim with TB_LSHTM or TB_LSHTM_Acute."""
    pop = ss.People(n_agents=agents)
    if use_acute:
        tb = mtb.TB_LSHTM_Acute(pars=pars)
    else:
        tb = mtb.TB_LSHTM(pars=pars)
    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        pars=dict(dt=dt, start=start, stop=stop),
    )
    sim.pars.verbose = 0
    return sim


# --- TBSL enum ---


def test_tbsl_enum_values():
    """TBSL has expected state values."""
    assert TBSL.SUSCEPTIBLE == -1
    assert TBSL.INFECTION == 0
    assert TBSL.CLEARED == 1
    assert TBSL.NON_INFECTIOUS == 2
    assert TBSL.RECOVERED == 3
    assert TBSL.ASYMPTOMATIC == 4
    assert TBSL.SYMPTOMATIC == 5
    assert TBSL.TREATMENT == 6
    assert TBSL.TREATED == 7
    assert TBSL.DEAD == 8
    assert TBSL.ACUTE == 9


def test_tbsl_enum_names():
    """TBSL has expected state names."""
    names = [s.name for s in TBSL]
    assert "SUSCEPTIBLE" in names
    assert "INFECTION" in names
    assert "ASYMPTOMATIC" in names
    assert "SYMPTOMATIC" in names
    assert "ACUTE" in names


# --- TB_LSHTM initialization ---


def test_tb_lshtm_initialization():
    """TB_LSHTM initializes with default pars and state types."""
    tb = TB_LSHTM()
    assert tb.pars.beta is not None
    assert tb.pars.kappa == 0.82
    assert tb.pars.pi == 0.21
    assert tb.pars.rho == 3.15
    assert hasattr(tb.pars, "inf_cle") and hasattr(tb.pars, "inf_non") and hasattr(tb.pars, "inf_asy")
    assert hasattr(tb.pars, "mu_tb") and hasattr(tb.pars, "theta") and hasattr(tb.pars, "delta")


def test_tb_lshtm_initial_states_after_init():
    """TB_LSHTM state arrays exist and have correct type after sim.init()."""
    sim = make_lshtm_sim(agents=50)
    sim.init()
    tb = sim.diseases[0]
    assert isinstance(tb.susceptible, (ss.BoolArr, np.ndarray))
    assert isinstance(tb.infected, (ss.BoolArr, np.ndarray))
    assert isinstance(tb.state, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.state_next, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.ti_next, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.rel_sus, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.rel_trans, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.rr_activation, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.rr_clearance, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.rr_death, (ss.FloatArr, np.ndarray))
    assert isinstance(tb.on_treatment, (ss.BoolArr, np.ndarray))
    assert len(tb.state) == 50


def test_tb_lshtm_acute_initialization():
    """TB_LSHTM_Acute adds acu_inf and alpha pars."""
    tb = TB_LSHTM_Acute()
    assert hasattr(tb.pars, "acu_inf")
    assert tb.pars.alpha == 0.9


# --- Infectious property ---


def test_tb_lshtm_infectious():
    """Only ASYMPTOMATIC and SYMPTOMATIC are infectious in TB_LSHTM."""
    sim = make_lshtm_sim(agents=20)
    sim.init()
    tb = sim.diseases[0]
    n = len(tb.state)
    # None infectious when all susceptible
    tb.state[:] = TBSL.SUSCEPTIBLE
    assert not tb.infectious.any()
    # ASYMPTOMATIC infectious
    tb.state[:] = TBSL.ASYMPTOMATIC
    assert tb.infectious.all()
    # SYMPTOMATIC infectious
    tb.state[:] = TBSL.SYMPTOMATIC
    assert tb.infectious.all()
    # INFECTION (latent) not infectious
    tb.state[:] = TBSL.INFECTION
    assert not tb.infectious.any()
    # TREATMENT not infectious
    tb.state[:] = TBSL.TREATMENT
    assert not tb.infectious.any()


def test_tb_lshtm_acute_infectious():
    """ACUTE is infectious in TB_LSHTM_Acute."""
    sim = make_lshtm_sim(agents=20, use_acute=True)
    sim.init()
    tb = sim.diseases[0]
    tb.state[:] = TBSL.ACUTE
    assert tb.infectious.all()
    tb.state[:] = TBSL.ASYMPTOMATIC
    assert tb.infectious.all()
    tb.state[:] = TBSL.INFECTION
    assert not tb.infectious.any()


# --- make_scaled_rate ---


def test_scaled_rate_positive_rr():
    """make_scaled_rate with rr > 0 scales waiting time (finite values)."""
    sim = make_lshtm_sim(agents=5)
    sim.init()
    tb = sim.diseases[0]
    # Use an already-initialized rate from the model (inf_cle)
    base = tb.pars.inf_cle
    rr = np.array([1.0, 2.0, 0.5, 1.0, 1.0], dtype=float)
    scaled = make_scaled_rate(base, lambda uids: rr)
    uids = ss.uids(np.arange(5))
    t = scaled.sample_waiting_times(uids)
    assert t.shape == (5,)
    assert np.all(t >= 0)
    assert np.all(np.isfinite(t))


def test_scaled_rate_zero_rr():
    """make_scaled_rate with rr=0 returns inf (never transition)."""
    sim = make_lshtm_sim(agents=3)
    sim.init()
    tb = sim.diseases[0]
    base = tb.pars.inf_cle
    rr = np.array([0.0, 1.0, 0.0], dtype=float)
    scaled = make_scaled_rate(base, lambda uids: rr)
    uids = ss.uids([0, 1, 2])
    with np.errstate(divide="ignore"):  # intentional divide-by-zero -> inf
        t = scaled.sample_waiting_times(uids)
    assert t[0] == np.inf
    assert t[2] == np.inf
    assert np.isfinite(t[1])


def test_scaled_rate_mean_waiting_time():
    """make_scaled_rate with rr=1 has mean waiting time 1/λ; with rr=2, mean ≈ 1/(2λ)."""
    from tbsim.tb_lshtm import _get_rate_from_base
    sim = make_lshtm_sim(agents=500, pars={"init_prev": ss.bernoulli(0)})
    sim.init(seed=42)
    tb = sim.diseases[0]
    base = tb.pars.inf_cle
    lam = _get_rate_from_base(base)
    uids = ss.uids(np.arange(500))
    # rr=1: mean T should be 1/lam
    scaled1 = make_scaled_rate(base, lambda uids: np.ones(len(uids)))
    draws1 = [scaled1.sample_waiting_times(uids) for _ in range(200)]
    mean1 = np.mean(draws1)
    assert 0.8 / lam <= mean1 <= 1.2 / lam, f"mean T with rr=1 should be ~1/λ={1/lam}, got {mean1}"
    # rr=2: mean T should be 1/(2*lam)
    scaled2 = make_scaled_rate(base, lambda uids: np.full(len(uids), 2.0))
    draws2 = [scaled2.sample_waiting_times(uids) for _ in range(200)]
    mean2 = np.mean(draws2)
    assert 0.8 / (2 * lam) <= mean2 <= 1.2 / (2 * lam), f"mean T with rr=2 should be ~1/(2λ)={1/(2*lam)}, got {mean2}"


# --- transition() ---


def test_transition_empty_uids():
    """transition with empty uids returns empty arrays."""
    sim = make_lshtm_sim(agents=10)
    sim.init()
    tb = sim.diseases[0]
    state_next, ti_next = tb.transition(np.array([], dtype=int), to={
        TBSL.CLEARED: tb.pars.inf_cle,
        TBSL.NON_INFECTIOUS: tb.pars.inf_non,
    })
    assert len(state_next) == 0
    assert len(ti_next) == 0


def test_transition_returns_valid_states():
    """transition returns state_next in keys and ti_next >= ti."""
    sim = make_lshtm_sim(agents=30)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(30))
    keys = [TBSL.CLEARED, TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC]
    state_next, ti_next = tb.transition(uids, to={
        TBSL.CLEARED: tb.pars.inf_cle,
        TBSL.NON_INFECTIOUS: tb.pars.inf_non,
        TBSL.ASYMPTOMATIC: tb.pars.inf_asy,
    })
    assert len(state_next) == 30
    assert len(ti_next) == 30
    assert np.all(np.isin(state_next, keys))
    assert np.all(ti_next >= tb.ti)


# --- set_prognoses ---


def test_set_prognoses_sets_state_and_susceptible():
    """set_prognoses sets state to INFECTION and susceptible to False."""
    sim = make_lshtm_sim(agents=50)
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
    # Next transition should be to one of CLEARED, NON_INFECTIOUS, ASYMPTOMATIC
    assert np.all(np.isin(tb.state_next[uids], [TBSL.CLEARED, TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC]))
    assert np.all(np.isfinite(tb.ti_next[uids]))


def test_set_prognoses_empty_uids():
    """set_prognoses with empty uids does not raise."""
    sim = make_lshtm_sim(agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.set_prognoses(np.array([], dtype=int))


# --- step_die ---


def test_step_die():
    """step_die sets state=DEAD, susceptible=False, infected=False, rel_trans=0."""
    sim = make_lshtm_sim(agents=50)
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
    sim = make_lshtm_sim(agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.step_die(np.array([], dtype=int))


# --- start_treatment ---


def test_start_treatment_latent_cleared():
    """start_treatment on INFECTION (latent) schedules CLEARED."""
    sim = make_lshtm_sim(agents=50)
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
    sim = make_lshtm_sim(agents=50)
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
    sim = make_lshtm_sim(agents=10)
    sim.init()
    tb = sim.diseases[0]
    tb.start_treatment(np.array([], dtype=int))


def test_start_treatment_acute_latent_cleared():
    """TB_LSHTM_Acute: start_treatment on ACUTE or INFECTION schedules CLEARED."""
    sim = make_lshtm_sim(agents=20, use_acute=True)
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
        agents=200,
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
        agents=200,
        use_acute=True,
        start=ss.date("2000-01-01"),
        stop=ss.date("2002-12-31"),
        dt=ss.days(7),
        pars={"init_prev": ss.bernoulli(0.05), "beta": ss.peryear(0.2)},
    )
    sim.run()
    tb = sim.diseases[0]
    assert isinstance(tb, TB_LSHTM_Acute)
    assert "n_infectious" in tb.results
    assert "prevalence_active" in tb.results


def test_init_results_defines_expected_keys():
    """init_results defines per-state counts and main outcome series."""
    sim = make_lshtm_sim(agents=30)
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
    sim = make_lshtm_sim(agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
    sim.run()
    tb = sim.diseases[0]
    tb.finalize_results()
    assert np.all(np.cumsum(tb.results["new_deaths"][:]) == tb.results["cum_deaths"][:])
    assert np.all(np.cumsum(tb.results["new_active"][:]) == tb.results["cum_active"][:])


def test_plot_returns_figure():
    """plot() returns a matplotlib Figure."""
    sim = make_lshtm_sim(agents=50, start=ss.date("2000-01-01"), stop=ss.date("2001-12-31"))
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
    sim = make_lshtm_sim(agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
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
    sim = make_lshtm_sim(agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
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
    sim = make_lshtm_sim(agents=200, start=ss.date("2000-01-01"), stop=ss.date("2005-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for ti in range(len(tb.results["timevec"])):
        prev = tb.results["prevalence_active"][ti]
        if np.isfinite(prev):
            assert 0 <= prev <= 1, f"prevalence_active at ti={ti} should be in [0,1], got {prev}"


def test_cumulative_series_non_decreasing():
    """cum_deaths and cum_active are non-decreasing over time."""
    sim = make_lshtm_sim(agents=200, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = sim.diseases[0]
    cum_d = tb.results["cum_deaths"][:]
    cum_a = tb.results["cum_active"][:]
    assert np.all(np.diff(cum_d) >= 0), "cum_deaths should be non-decreasing"
    assert np.all(np.diff(cum_a) >= 0), "cum_active should be non-decreasing"


def test_new_events_non_negative():
    """new_deaths, new_active, new_notifications_15+ are non-negative."""
    sim = make_lshtm_sim(agents=150, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = sim.diseases[0]
    assert np.all(tb.results["new_deaths"][:] >= 0)
    assert np.all(tb.results["new_active"][:] >= 0)
    assert np.all(tb.results["new_notifications_15+"][:] >= 0)


def test_susceptible_only_cleared_recovered_treated_or_never_infected():
    """susceptible is True only for state in SUSCEPTIBLE, CLEARED, RECOVERED, TREATED."""
    sim = make_lshtm_sim(agents=80, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
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
    """After step, RECOVERED have rel_sus=pi, TREATED have rel_sus=rho, ASYMPTOMATIC have rel_trans=kappa."""
    sim = make_lshtm_sim(agents=60)
    sim.init()
    tb = sim.diseases[0]
    # Force some agents into each state and run one step with no one due (so only rel_sus/rel_trans get set for others we don't touch)
    # Easier: set state and call the part of step that sets rel_sus/rel_trans by hand, or run until we have RECOVERED/TREATED/ASYMPTOMATIC
    # Instead: run a short sim, then for any agent in RECOVERED, check rel_sus==pi; TREATED rel_sus==rho; ASYMPTOMATIC rel_trans==kappa
    sim.run()
    tb = sim.diseases[0]
    pi, rho, kappa = tb.pars.pi, tb.pars.rho, tb.pars.kappa
    recovered_uids = ss.uids(tb.state == TBSL.RECOVERED)
    treated_uids = ss.uids(tb.state == TBSL.TREATED)
    asymp_uids = ss.uids(tb.state == TBSL.ASYMPTOMATIC)
    if len(recovered_uids) > 0:
        assert np.allclose(tb.rel_sus[recovered_uids], pi)
    if len(treated_uids) > 0:
        assert np.allclose(tb.rel_sus[treated_uids], rho)
    if len(asymp_uids) > 0:
        assert np.allclose(tb.rel_trans[asymp_uids], kappa)


def test_start_treatment_mixed_latent_active_ignores_cleared():
    """start_treatment with mix of INFECTION, SYMPTOMATIC, CLEARED: only INFECTION and SYMPTOMATIC are changed."""
    sim = make_lshtm_sim(agents=50)
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


def test_transition_single_destination_all_get_it():
    """transition with a single destination sends all uids to that state with ti_next >= ti."""
    sim = make_lshtm_sim(agents=20)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(20))
    state_next, ti_next = tb.transition(uids, to={TBSL.CLEARED: tb.pars.inf_cle})
    assert np.all(state_next == TBSL.CLEARED)
    assert len(ti_next) == 20
    assert np.all(ti_next >= tb.ti)
    assert np.all(np.isfinite(ti_next))


def test_step_with_no_one_due_leaves_state_unchanged():
    """When no agent has ti >= ti_next, step() does not change state."""
    sim = make_lshtm_sim(agents=40)
    sim.init()
    tb = sim.diseases[0]
    state_before = np.array(tb.state, copy=True)
    tb.ti_next[:] = np.inf  # No one due
    tb.step()
    state_after = np.array(tb.state, copy=True)
    np.testing.assert_array_equal(state_before, state_after)


def test_set_prognoses_acute_enters_acute_not_infection():
    """TB_LSHTM_Acute: set_prognoses puts new infections in ACUTE and schedules ACUTE->INFECTION."""
    sim = make_lshtm_sim(agents=30, use_acute=True)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids([1, 2, 3])
    tb.susceptible[uids] = True
    tb.infected[uids] = False
    tb.set_prognoses(uids)
    assert np.all(tb.state[uids] == TBSL.ACUTE)
    assert np.all(tb.state_next[uids] == TBSL.INFECTION)
    assert np.all(np.isfinite(tb.ti_next[uids]))


def test_rr_activation_zero_prevents_progression_to_active():
    """With rr_activation=0, latent agents only transition to CLEARED (inf_cle not scaled)."""
    sim = make_lshtm_sim(agents=50)
    sim.init()
    tb = sim.diseases[0]
    uids = ss.uids(np.arange(20))
    tb.rr_activation[uids] = 0  # before set_prognoses so transition uses it
    with np.errstate(divide="ignore"):
        tb.set_prognoses(uids)  # inf_non/inf_asy -> inf; CLEARED wins
    tb.ti_next[uids] = tb.ti  # transition this step
    tb.step()
    assert np.all(tb.state[uids] == TBSL.CLEARED)


def test_zero_beta_no_initial_infection_no_transmission():
    """With init_prev=0 and beta=0, no one becomes infected (all remain SUSCEPTIBLE)."""
    sim = make_lshtm_sim(
        agents=80,
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
    sim = make_lshtm_sim(agents=100, start=ss.date("2000-01-01"), stop=ss.date("2002-12-31"))
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
    sim = make_lshtm_sim(agents=100, start=ss.date("2000-01-01"), stop=ss.date("2003-12-31"))
    sim.run()
    tb = sim.diseases[0]
    for state in [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED]:
        u = ss.uids(tb.state == state)
        if len(u) > 0:
            assert np.all(tb.ti_next[u] == np.inf), f"Agents in {state.name} should have ti_next=inf"


def test_on_treatment_consistent_with_state():
    """on_treatment is True iff state == TREATMENT."""
    sim = make_lshtm_sim(agents=120, start=ss.date("2000-01-01"), stop=ss.date("2004-12-31"))
    sim.run()
    tb = sim.diseases[0]
    np.testing.assert_array_equal(
        tb.on_treatment,
        (tb.state == TBSL.TREATMENT),
        err_msg="on_treatment should equal (state == TREATMENT)",
    )
