"""Unit tests for the Migration demographics module.

These tests target Migration internals and boundary behavior — age sampling,
household assignment, emigration weighting, TB-state derivation, rate parsing,
and population-maintenance — beyond the population-level scientific checks in
``tests/test_migration.py``.

Run with::

    pytest tests/devtests/test_migration_unit.py -q
"""

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import tbsim
from tbsim.tb import TBS


QUIET_TB = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dhs(n_hh=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_hh):
        sz = int(rng.integers(2, 6))
        ages = rng.integers(1, 75, size=sz)
        rows.append((h, sc.strjoin(ages)))
    return sc.dataframe(hh_id=[r[0] for r in rows], ages=[r[1] for r in rows])


def _make_sim(migration_pars=None, networks=None, n_agents=200, stop='2000-02-01',
              rand_seed=1, init_prev=0.0):
    demographics = []
    if migration_pars is not None:
        demographics.append(tbsim.Migration(pars=migration_pars))
    return ss.Sim(
        n_agents=n_agents,
        start='2000-01-01',
        stop=stop,
        dt=ss.days(30),
        rand_seed=rand_seed,
        verbose=0,
        diseases=tbsim.TB(pars=dict(init_prev=ss.bernoulli(init_prev), beta=ss.peryear(0.0))),
        networks=networks if networks is not None else ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0)),
        demographics=demographics,
    )


def _migration(sim):
    return sim.demographics.migration


# ---------------------------------------------------------------------------
# tb_state_distribution validation + derivation
# ---------------------------------------------------------------------------

def test_validate_tb_state_distribution_rejects_empty():
    with pytest.raises(ValueError):
        tbsim.Migration._validate_tb_state_distribution({})


def test_validate_tb_state_distribution_drops_unknown_and_terminal_states():
    with pytest.warns(UserWarning):
        out = tbsim.Migration._validate_tb_state_distribution(
            dict(SUSCEPTIBLE=1.0, NOT_A_STATE=2.0, DEAD=3.0, REMOVED=1.0)
        )
    assert set(out.keys()) == {'SUSCEPTIBLE'}
    assert pytest.approx(out['SUSCEPTIBLE']) == 1.0


def test_validate_tb_state_distribution_normalizes_weights():
    out = tbsim.Migration._validate_tb_state_distribution(
        dict(SUSCEPTIBLE=2.0, INFECTION=2.0)
    )
    assert pytest.approx(sum(out.values())) == 1.0
    assert pytest.approx(out['SUSCEPTIBLE']) == 0.5


def test_default_tb_state_distribution_uses_init_prev_and_inf_rates():
    tb = tbsim.TB(pars=dict(init_prev=ss.bernoulli(0.5), beta=ss.peryear(0.0)))
    sim = ss.Sim(n_agents=10, start='2000-01-01', stop='2000-02-01', dt=ss.days(30),
                 verbose=0, diseases=tb)
    sim.init()
    dist = tbsim.Migration._derive_default_tb_state_distribution(sim.diseases.tb)
    assert pytest.approx(dist['SUSCEPTIBLE']) == 0.5
    assert pytest.approx(dist['INFECTION'] + dist['ASYMPTOMATIC']) == 0.5


def test_init_pre_raises_without_tb_module():
    sim = ss.Sim(n_agents=10, start='2000-01-01', stop='2000-02-01', dt=ss.days(30),
                 verbose=0, demographics=[tbsim.Migration(pars=dict(tb_state_distribution=dict(SUSCEPTIBLE=1.0)))])
    with pytest.raises(RuntimeError):
        sim.init()


def test_init_tb_states_rejects_acute_for_plain_tb():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=0.5, ACUTE=0.5),
    ))
    sim.init()
    new_uids = sim.people.grow(2)
    with pytest.raises(ValueError):
        _migration(sim)._init_tb_states(new_uids)


# ---------------------------------------------------------------------------
# Rate parsing
# ---------------------------------------------------------------------------

def test_expected_events_per_timestep_handles_none_zero_and_negatives():
    dt = ss.days(30)
    assert tbsim.Migration._expected_events_per_timestep(None, dt) == 0.0
    assert tbsim.Migration._expected_events_per_timestep(ss.freqperyear(0), dt) == 0.0
    assert tbsim.Migration._expected_events_per_timestep(-1, dt) == 0.0


def test_expected_events_per_timestep_warns_on_peryear():
    dt = ss.days(30)
    with pytest.warns(UserWarning):
        tbsim.Migration._expected_events_per_timestep(ss.peryear(0.5), dt)


# ---------------------------------------------------------------------------
# Age sampling (immigration)
# ---------------------------------------------------------------------------

def test_age_dict_bins_constructed_correctly():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        max_age=80,
        immigration_age_distribution={0: 0.5, 20: 0.3, 50: 0.2},
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    assert np.array_equal(mig._age_lows, np.array([0.0, 20.0, 50.0]))
    assert np.array_equal(mig._age_highs, np.array([20.0, 50.0, 80.0]))


def test_invalid_immigration_age_bins_fall_back_without_breaking_tb_init():
    """Empty or all-zero age bins must not skip TB state setup in init_pre (P0)."""
    sim = ss.Sim(
        n_agents=200,
        start='2000-01-01',
        stop='2001-01-01',
        dt=ss.days(30),
        rand_seed=3,
        verbose=0,
        diseases=tbsim.TB(pars=QUIET_TB),
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0)),
        demographics=[
            tbsim.Migration(pars=dict(
                immigration_rate=ss.freqperyear(200),
                emigration_rate=ss.freqperyear(0),
                max_age=10,
                immigration_age_distribution={15: 1.0, 30: 1.0},  # all edges >= max_age
                tb_state_distribution=dict(SUSCEPTIBLE=1.0),
            )),
        ],
    )
    sim.run()
    mig = sim.demographics.migration
    assert int(TBS.SUSCEPTIBLE) in mig._dist_tbstate.pars.a
    imm = ss.uids(mig.is_immigrant)
    assert len(imm) > 0
    ages_at_arrival = np.asarray(mig.age_at_immigration[imm], dtype=float)
    assert np.all(ages_at_arrival >= 0)
    assert ages_at_arrival.max() < 10
    tb = tbsim.get_tb(sim)
    assert np.all(tb.state[imm] == TBS.SUSCEPTIBLE)


def test_invalid_max_age_falls_back_to_default():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        max_age=-5,
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    with pytest.warns(UserWarning):
        sim.init()
    assert _migration(sim).pars.max_age == 85.0


def test_sample_ages_respects_bounds_and_negative_input():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        max_age=50,
        immigration_age_distribution={0: 1.0},
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    assert mig._sample_ages(0).shape == (0,)
    ages = mig._sample_ages(200)
    assert ages.min() >= 0
    assert ages.max() < 50


def test_bound_ages_with_invalid_max_only_clips_negative():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    mig.pars.max_age = float('nan')
    out = mig._bound_ages(np.array([-1.0, 5.0, 200.0]))
    assert out[0] == 0.0
    assert out[1] == 5.0
    assert out[2] == 200.0


# ---------------------------------------------------------------------------
# Age sampling (emigration weighting)
# ---------------------------------------------------------------------------

def test_emig_weights_match_loop_reference():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        max_age=60,
        emigration_age_distribution={0: 0.1, 10: 0.4, 30: 0.5},
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    uids = sim.people.auids
    w_fast = mig._emig_weights_for_uids(uids)
    ages = np.asarray(sim.people.age[uids], dtype=float)
    w_ref = np.zeros(len(uids))
    for lo, hi, w in zip(mig._emig_age_lows, mig._emig_age_highs, mig._emig_age_weights):
        w_ref[(ages >= lo) & (ages < hi)] = w
    assert np.allclose(w_fast, w_ref)


def test_emig_weights_zero_when_no_spec_or_empty_uids():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    assert mig._emig_weights_for_uids(sim.people.auids) is None
    assert mig._emig_weights_for_uids(ss.uids()) is None


def test_emig_age_distribution_warns_on_empty():
    with pytest.warns(UserWarning):
        sim = _make_sim(migration_pars=dict(
            immigration_rate=ss.freqperyear(0),
            emigration_rate=ss.freqperyear(0),
            emigration_age_distribution={},
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ))
        sim.init()
    assert _migration(sim)._emig_age_lows is None


def test_emig_age_distribution_warns_when_all_above_max_age():
    with pytest.warns(UserWarning):
        sim = _make_sim(migration_pars=dict(
            immigration_rate=ss.freqperyear(0),
            emigration_rate=ss.freqperyear(0),
            max_age=20,
            emigration_age_distribution={50: 1.0, 60: 1.0},
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ))
        sim.init()
    assert _migration(sim)._emig_age_lows is None


# ---------------------------------------------------------------------------
# Household assignment
# ---------------------------------------------------------------------------

def test_members_by_household_id_matches_naive_scan():
    dhs = _make_dhs(n_hh=10, seed=1)
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ), networks=[hh_net], n_agents=40)
    sim.init()
    mig = _migration(sim)
    hh = sim.networks.householdnet
    hh_ids_arr = np.asarray(hh.household_ids, dtype=float)
    targets = [int(h) for h in np.unique(hh_ids_arr[~np.isnan(hh_ids_arr)])[:4]]
    fast = mig._members_by_household_id(hh, targets)
    for hid in targets:
        a = np.sort(np.asarray(ss.uids(hh.household_ids == hid), dtype=int))
        b = np.sort(np.asarray(fast.get(hid, ss.uids()), dtype=int))
        assert np.array_equal(a, b)


def test_assign_immigrants_returns_none_without_household_net():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    new_uids = sim.people.grow(3)
    assert _migration(sim).assign_immigrants_to_households(new_uids) is None


def test_assign_immigrants_creates_singletons_when_no_live_household_members():
    dhs = _make_dhs(n_hh=4)
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ), networks=[hh_net], n_agents=10)
    sim.init()
    mig = _migration(sim)
    hh = sim.networks.householdnet
    hh.household_ids[:] = np.nan
    starting_n = hh.n_households
    new_uids = sim.people.grow(3)
    assigned = mig.assign_immigrants_to_households(new_uids)
    assert assigned is not None
    assert hh.n_households == starting_n + 3
    assert set(np.asarray(assigned, dtype=int)) == set(np.asarray(hh.household_ids[new_uids], dtype=int))


def test_assign_immigrants_adds_complete_edges_to_existing_members():
    dhs = sc.dataframe(hh_id=[0], ages=[sc.strjoin([20, 30, 40])])
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ), networks=[hh_net], n_agents=3)
    sim.init()
    mig = _migration(sim)
    hh = sim.networks.householdnet
    edges_before = len(hh.p1)
    new_uids = sim.people.grow(1)
    assigned = mig.assign_immigrants_to_households(new_uids)
    assert assigned.shape == (1,)
    assert len(hh.p1) == edges_before + 3  # one new uid connected to 3 existing members


def test_perform_immigration_zero_arrivals_is_noop():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    out = mig._perform_immigration(0)
    assert len(out) == 0
    assert mig.n_immigrants == 0
    assert mig._fresh_import_uids is None


# ---------------------------------------------------------------------------
# Emigration mechanics
# ---------------------------------------------------------------------------

def test_active_pop_excludes_terminal_states():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    tb = tbsim.get_tb(sim)
    before = len(mig._active_pop_uids())
    victim = sim.people.auids[0]
    tb.state[ss.uids(victim)] = TBS.DEAD
    after = len(mig._active_pop_uids())
    assert after == before - 1


def test_sample_emigrants_caps_at_eligible_count():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ), n_agents=5)
    sim.init()
    mig = _migration(sim)
    sample = mig._sample_emigrants(50)
    assert len(sample) == 5


def test_sample_emigrants_zero_request_returns_empty():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    assert len(_migration(sim)._sample_emigrants(0)) == 0


def test_apply_emigration_removes_from_household_and_marks_state():
    dhs = _make_dhs(n_hh=4, seed=3)
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ), networks=[hh_net], n_agents=20)
    sim.init()
    mig = _migration(sim)
    tb = tbsim.get_tb(sim)
    targets = sim.people.auids[:3]
    mig._apply_emigration(targets)
    assert mig.n_emigrants == 3
    assert np.all(np.asarray(mig.is_emigrant[targets]))
    assert np.all(np.asarray(tb.state[targets]) == TBS.REMOVED)
    assert np.all(np.asarray(mig.hhid[targets]) == -1)


# ---------------------------------------------------------------------------
# maintain_population behavior
# ---------------------------------------------------------------------------

def test_adjust_arrivals_is_noop_when_disabled():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    assert mig._adjust_arrivals_for_pop_target(7, 99) == 7


def test_adjust_arrivals_compensates_excess_emigration():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        maintain_population=True,
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    baseline = mig._baseline_population
    # No actual emigration applied; ask for hypothetical 10 emigrants and 1 immigrant.
    adjusted = mig._adjust_arrivals_for_pop_target(1, 10)
    assert adjusted == max(0, 1 + (baseline - (baseline + 1 - 10)))
    assert adjusted == 10


def test_adjust_arrivals_clamped_at_zero():
    sim = _make_sim(migration_pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        maintain_population=True,
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim.init()
    mig = _migration(sim)
    # Pretend big surplus already exists.
    mig._baseline_population = 1
    adjusted = mig._adjust_arrivals_for_pop_target(0, 0)
    assert adjusted == max(0, adjusted)
    assert adjusted >= 0


# ---------------------------------------------------------------------------
# Integration smoke test (immigration + emigration + households)
# ---------------------------------------------------------------------------

def test_full_step_smoke_run_with_households():
    dhs = _make_dhs(n_hh=20, seed=7)
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    sim = _make_sim(
        migration_pars=dict(
            immigration_rate=ss.freqperyear(120),
            emigration_rate=ss.freqperyear(80),
            max_age=70,
            immigration_age_distribution={0: 0.2, 15: 0.5, 30: 0.3},
            emigration_age_distribution={0: 0.2, 20: 0.8},
            tb_state_distribution=dict(SUSCEPTIBLE=0.9, INFECTION=0.1),
        ),
        networks=[hh_net], n_agents=80, stop='2001-01-01', rand_seed=11,
    )
    sim.run()
    mig = _migration(sim)
    assert int(mig.results['n_immigrants'][:].sum()) > 0
    assert int(mig.results['n_emigrants'][:].sum()) > 0
    imm = ss.uids(mig.is_immigrant)
    if len(imm):
        ages = np.asarray(mig.age_at_immigration[imm], dtype=float)
        assert ages.min() >= 0
        assert ages.max() < 70

def test_emigration_age_bins_respect_max_age_upper_bound():
    """Emigration weights use [low, high) bins; ages at/above max_age are not age-weighted."""
    sim = ss.Sim(
        n_agents=50,
        start='2000-01-01',
        stop='2000-02-01',
        dt=ss.days(30),
        rand_seed=1,
        verbose=0,
        diseases=tbsim.TB(pars=QUIET_TB),
        networks=ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0)),
        demographics=[
            tbsim.Migration(pars=dict(
                immigration_rate=ss.freqperyear(0),
                emigration_rate=ss.freqperyear(0),
                max_age=40,
                emigration_age_distribution={0: 0.2, 20: 0.8},
                tb_state_distribution=dict(SUSCEPTIBLE=1.0),
            )),
        ],
    )
    sim.init()
    mig = sim.demographics.migration
    assert mig._emig_age_lows is not None
    assert np.allclose(mig._emig_age_lows, [0, 20])
    assert np.allclose(mig._emig_age_highs, [20, 40])

    uids = sim.people.auids
    w = mig._emig_weights_for_uids(uids)
    ages = np.asarray(sim.people.age[uids], dtype=float)
    assert w[ages < 0].sum() == 0  # no negative ages
    assert np.all(w[(ages >= 0) & (ages < 20)] == 0.2)
    assert np.all(w[(ages >= 20) & (ages < 40)] == 0.8)
    assert np.all(w[ages >= 40] == 0)


def test_household_assignment_handles_sparse_household_ids():
    """Immigrants should join existing sparse household IDs, not bincount indices."""
    dhs = sc.dataframe(
        hh_id=np.arange(5),
        ages=[sc.strjoin([25, 30]), sc.strjoin([10, 35]), sc.strjoin([40, 45]), sc.strjoin([20, 50]), sc.strjoin([5, 60])],
    )
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    mig = tbsim.Migration(pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim = ss.Sim(
        n_agents=10,
        start='2000-01-01',
        stop='2000-02-01',
        dt=ss.days(30),
        rand_seed=4,
        verbose=0,
        diseases=tbsim.TB(pars=QUIET_TB),
        networks=[hh_net],
        demographics=[mig],
    )
    sim.init()
    mig = sim.demographics.migration
    hh = sim.networks.householdnet
    # Force sparse IDs to emulate non-contiguous household numbering.
    hh.household_ids[:] = np.array([0, 0, 2, 2, 5, 5, 7, 7, 10, 10], dtype=float)
    new_uids = sim.people.grow(3)
    assigned = mig.assign_immigrants_to_households(new_uids)
    assert len(assigned) == 3
    assert set(np.asarray(assigned, dtype=int)).issubset({0, 2, 5, 7, 10})


def test_batched_household_assignment_connects_new_immigrants_to_each_other():
    """Batch assignment should preserve complete-graph edges among same-HH arrivals."""
    dhs = sc.dataframe(hh_id=[0], ages=[sc.strjoin([20, 30, 40, 50])])
    hh_net = ss.HouseholdNet(dhs_data=dhs, dynamic=False)
    mig = tbsim.Migration(pars=dict(
        immigration_rate=ss.freqperyear(0),
        emigration_rate=ss.freqperyear(0),
        tb_state_distribution=dict(SUSCEPTIBLE=1.0),
    ))
    sim = ss.Sim(
        n_agents=4,
        start='2000-01-01',
        stop='2000-02-01',
        dt=ss.days(30),
        rand_seed=5,
        verbose=0,
        diseases=tbsim.TB(pars=QUIET_TB),
        networks=[hh_net],
        demographics=[mig],
    )
    sim.init()
    mig = sim.demographics.migration
    hh = sim.networks.householdnet
    new_uids = sim.people.grow(3)
    assigned = mig.assign_immigrants_to_households(new_uids)
    assert np.all(assigned == 0)

    edges = {tuple(sorted(pair)) for pair in zip(np.asarray(hh.p1, dtype=int), np.asarray(hh.p2, dtype=int))}
    for i, uid in enumerate(np.asarray(new_uids, dtype=int)):
        for other in np.asarray(new_uids[:i], dtype=int):
            assert tuple(sorted((uid, other))) in edges


if __name__ == '__main__':
    pytest.main([__file__, '-q'])
