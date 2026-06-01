"""Tests for the Migration demographics module."""

import numpy as np
import pandas as pd
import starsim as ss
import tbsim


AGE_BINS = np.array([0, 5, 15, 30, 50, 65, 85], dtype=float)
AGE_WEIGHTS = np.array([180, 220, 260, 170, 110, 60], dtype=float)
AGE_DISTRIBUTION = {0: 0.18, 5: 0.22, 15: 0.26, 30: 0.17, 50: 0.11, 65: 0.06}
QUIET_TB_PARS = dict(init_prev=ss.bernoulli(0.0), beta=ss.peryear(0.0))


def make_age_data():
    """Create a reproducible age histogram in Starsim People format."""
    return pd.DataFrame({'age': AGE_BINS[:-1], 'value': AGE_WEIGHTS})


def make_households(n_agents, seed=1):
    """Create synthetic households with moderate size variation."""
    rng = np.random.default_rng(seed)
    households = []
    uid = 0
    while uid < n_agents:
        hh_size = int(rng.integers(2, 6))
        hh_size = min(hh_size, n_agents - uid)
        households.append(list(range(uid, uid + hh_size)))
        uid += hh_size
    return households


def get_migration(sim):
    """Return the Migration module from a simulation."""
    for dem in sim.demographics.values():
        if isinstance(dem, tbsim.Migration):
            return dem
    raise RuntimeError('Migration module not found')


def make_sim(
    n_agents=500,
    migration_pars=None,
    analyzers=None,
    interventions=None,
    rand_seed=2,
    beta=0.0,
    tb_pars=None,
    stop='2004-01-01',
):
    """Build a compact sim for migration validation."""
    tb_defaults = dict(init_prev=ss.bernoulli(0.02), beta=ss.peryear(beta))
    tb_defaults.update(tb_pars or {})

    demographics = []
    if migration_pars is not None:
        demographics.append(tbsim.Migration(pars=migration_pars))

    sim = ss.Sim(
        people=ss.People(n_agents=n_agents, age_data=make_age_data()),
        start='2000-01-01',
        stop=stop,
        dt=ss.days(30),
        rand_seed=rand_seed,
        verbose=0,
        diseases=tbsim.TB(pars=tb_defaults),
        networks=[
            ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0)),
            tbsim.HouseholdNet(hhs=make_households(n_agents=n_agents, seed=rand_seed)),
        ],
        demographics=demographics,
        analyzers=analyzers or [],
        interventions=interventions or [],
    )
    return sim


def normalized_hist(values, bins):
    """Return a normalized histogram over fixed bins."""
    counts, _ = np.histogram(values, bins=bins)
    total = counts.sum()
    return counts / total if total else counts.astype(float)


def household_size_hist(hh_sizes, max_size=8):
    """Return a normalized histogram for household sizes."""
    hh_sizes = np.asarray(hh_sizes, dtype=int)
    counts = np.array([(hh_sizes == size).sum() for size in range(1, max_size + 1)], dtype=float)
    total = counts.sum()
    return counts / total if total else counts


def assert_no_removed_household_edges(sim):
    """Removed agents should not remain connected on the household network."""
    net = sim.networks.householdnet
    removed = np.where(sim.people.ti_removed.raw <= sim.ti)[0]
    if len(removed) == 0 or len(net.edges.p1) == 0:
        return
    pairs = np.stack([np.asarray(net.edges.p1, dtype=int), np.asarray(net.edges.p2, dtype=int)], axis=1)
    assert not np.isin(pairs, removed).any(), 'Removed agents should not remain in household edges'


def test_population_growth_shrinkage_and_stability():
    """Migration rates should move population in the expected direction and magnitude."""
    initial_n = 500
    years = 4.0
    scenarios = [
        (80, 20, 'grow'),
        (20, 80, 'shrink'),
        (60, 60, 'stable'),
    ]

    for immigration_rate, emigration_rate, expected_direction in scenarios:
        sim = make_sim(
            n_agents=initial_n,
            migration_pars=dict(
                immigration_rate=ss.freqperyear(immigration_rate),
                emigration_rate=ss.freqperyear(emigration_rate),
                immigration_age_distribution=AGE_DISTRIBUTION,
                    tb_state_distribution=dict(SUSCEPTIBLE=1.0),
            ),
            rand_seed=3,
                tb_pars=QUIET_TB_PARS,
        )
        sim.run()
        migration = get_migration(sim)
        final_n = int(sim.results.n_alive[-1])
        observed_delta = final_n - initial_n
        total_immigrants = int(migration.results.n_immigrants[:].sum())
        total_emigrants = int(migration.results.n_emigrants[:].sum())
        expected_immigrants = immigration_rate * years
        expected_emigrants = emigration_rate * years
        flow_tolerance = int(np.ceil(4 * np.sqrt((immigration_rate + emigration_rate) * years) + 15))
        assert abs(total_immigrants - expected_immigrants) <= flow_tolerance, (
            f'Immigration count drift too large: expected about {expected_immigrants}, got {total_immigrants}'
        )
        assert abs(total_emigrants - expected_emigrants) <= flow_tolerance, (
            f'Emigration count drift too large: expected about {expected_emigrants}, got {total_emigrants}'
        )
        net_flow = total_immigrants - total_emigrants
        end_tolerance = int(np.ceil(0.35 * max(total_emigrants, 1) + 10))
        assert abs(observed_delta - net_flow) <= end_tolerance, (
            f'Population delta should be broadly consistent with observed net migration; '
            f'got delta {observed_delta}, net flow {net_flow}, tol {end_tolerance}'
        )
        if expected_direction == 'grow':
            assert final_n > initial_n, 'Population should grow when immigration exceeds emigration'
        elif expected_direction == 'shrink':
            assert final_n < initial_n, 'Population should shrink when emigration exceeds immigration'
        else:
            stable_tolerance = int(np.ceil(4 * np.sqrt(expected_immigrants + expected_emigrants) + 35))
            assert abs(final_n - initial_n) <= stable_tolerance, 'Population should remain approximately stable'


def test_age_distribution_stability():
    """Balanced migration should preserve the population age histogram within stochastic tolerance."""
    baseline = make_sim(n_agents=700, migration_pars=None, rand_seed=5)
    migration = make_sim(
        n_agents=700,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(80),
            emigration_rate=ss.freqperyear(80),
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ),
        rand_seed=5,
        tb_pars=QUIET_TB_PARS,
    )
    baseline.run()
    migration.run()

    baseline_hist = normalized_hist(np.asarray(baseline.people.age.values, dtype=float), AGE_BINS)
    migration_hist = normalized_hist(np.asarray(migration.people.age.values, dtype=float), AGE_BINS)
    max_diff = float(np.max(np.abs(baseline_hist - migration_hist)))
    assert max_diff <= 0.10, f'Age histogram drift too large under balanced migration: {max_diff:.3f}'


def test_household_size_distribution_stability():
    """Balanced migration should preserve realistic household size statistics."""
    baseline_az = tbsim.HouseholdStats()
    migration_az = tbsim.HouseholdStats()
    baseline = make_sim(n_agents=700, analyzers=[baseline_az], rand_seed=7)
    migration = make_sim(
        n_agents=700,
        analyzers=[migration_az],
        migration_pars=dict(
            immigration_rate=ss.freqperyear(70),
            emigration_rate=ss.freqperyear(70),
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ),
        rand_seed=7,
        tb_pars=QUIET_TB_PARS,
    )
    baseline.run()
    migration.run()
    baseline_az = baseline.analyzers.householdstats
    migration_az = migration.analyzers.householdstats

    mean_diff = abs(float(baseline_az.results.mean_hh_size[-1]) - float(migration_az.results.mean_hh_size[-1]))
    median_diff = abs(float(baseline_az.results.median_hh_size[-1]) - float(migration_az.results.median_hh_size[-1]))
    end_hist_diff = np.max(np.abs(household_size_hist(baseline_az.hh_size_hists[-1]) - household_size_hist(migration_az.hh_size_hists[-1])))
    within_run_diff = np.max(np.abs(household_size_hist(migration_az.hh_size_hists[0]) - household_size_hist(migration_az.hh_size_hists[-1])))

    assert mean_diff <= 0.75, f'Mean household size drift too large: {mean_diff:.3f}'
    assert median_diff <= 1.0, f'Median household size drift too large: {median_diff:.3f}'
    assert end_hist_diff <= 0.20, f'Final household size histogram drift too large: {end_hist_diff:.3f}'
    assert within_run_diff <= 0.20, f'Within-run household size histogram drift too large: {within_run_diff:.3f}'


def test_household_age_distribution_stability():
    """Balanced migration should not systematically distort household age structure."""
    baseline_az = tbsim.HouseholdStats()
    migration_az = tbsim.HouseholdStats()
    baseline = make_sim(n_agents=650, analyzers=[baseline_az], rand_seed=11)
    migration = make_sim(
        n_agents=650,
        analyzers=[migration_az],
        migration_pars=dict(
            immigration_rate=ss.freqperyear(65),
            emigration_rate=ss.freqperyear(65),
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ),
        rand_seed=11,
        tb_pars=QUIET_TB_PARS,
    )
    baseline.run()
    migration.run()
    baseline_az = baseline.analyzers.householdstats
    migration_az = migration.analyzers.householdstats

    mean_diff = abs(float(baseline_az.results.mean_hh_age[-1]) - float(migration_az.results.mean_hh_age[-1]))
    median_diff = abs(float(baseline_az.results.median_hh_age[-1]) - float(migration_az.results.median_hh_age[-1]))
    within_mean_diff = abs(float(migration_az.results.mean_hh_age[-1]) - float(migration_az.results.mean_hh_age[0]))
    within_median_diff = abs(float(migration_az.results.median_hh_age[-1]) - float(migration_az.results.median_hh_age[0]))

    assert mean_diff <= 4.0, f'Household mean age drift too large: {mean_diff:.3f}'
    assert median_diff <= 5.0, f'Household median age drift too large: {median_diff:.3f}'
    assert within_mean_diff <= 5.0, f'Within-run household mean age drift too large: {within_mean_diff:.3f}'
    assert within_median_diff <= 6.0, f'Within-run household median age drift too large: {within_median_diff:.3f}'


def test_migration_works_with_tptsimple():
    """Migration should run with TPTSimple without household corruption."""
    tpt = tbsim.TPTSimple(pars=dict(
        coverage=ss.bernoulli(p=1.0),
        start=ss.date('2000-01-01'),
        stop=ss.date('2004-01-01'),
    ))
    sim = make_sim(
        n_agents=500,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(60),
            emigration_rate=ss.freqperyear(40),
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=dict(SUSCEPTIBLE=0.7, INFECTION=0.25, ASYMPTOMATIC=0.05),
        ),
        interventions=[tpt],
        rand_seed=13,
        stop='2003-01-01',
    )
    sim.run()

    migration = get_migration(sim)
    net = sim.networks.householdnet
    alive = sim.people.alive.uids
    assert migration.results.n_immigrants.sum() > 0
    assert migration.results.n_emigrants.sum() > 0
    assert hasattr(tpt, 'results')
    assert np.all(~np.isnan(net.household_ids[alive])), 'Alive agents should remain assigned to households'
    assert_no_removed_household_edges(sim)


def test_imported_asymptomatic_contributes_to_active_results():
    """Imported asymptomatic cases should initialize ti_asymp and count toward active burden."""
    sim = make_sim(
        n_agents=200,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(120),
            emigration_rate=ss.freqperyear(0),
            tb_state_distribution=dict(ASYMPTOMATIC=1.0),
            immigration_age_distribution=AGE_DISTRIBUTION,
        ),
        rand_seed=17,
        tb_pars=dict(
            beta=ss.peryear(0.0),
            inf_cle=ss.peryear(0.0),
            inf_non=ss.peryear(0.0),
            inf_asy=ss.peryear(0.0),
            non_rec=ss.peryear(0.0),
            non_asy=ss.peryear(0.0),
            asy_non=ss.peryear(0.0),
            asy_sym=ss.peryear(0.0),
            sym_asy=ss.peryear(0.0),
            sym_dead=ss.peryear(0.0),
        ),
        stop='2001-01-01',
    )
    sim.run()

    migration = get_migration(sim)
    tb = sim.people.tb
    immigrant_uids = migration.is_immigrant.uids
    assert len(immigrant_uids) > 0, 'Expected imported asymptomatic cases'
    assert np.all(tb.state[immigrant_uids] == tbsim.TBS.ASYMPTOMATIC)
    assert np.all(tb.ti_asymp[immigrant_uids] == migration.immigration_time[immigrant_uids])
    assert int(tb.results.new_active[:].sum()) == len(immigrant_uids)
    assert int(tb.results.cum_active[-1]) == len(immigrant_uids)


def test_tb_state_distribution_defaults_from_tb_pars_and_override():
    """Migration should derive defaults from TB pars unless explicitly overridden."""
    sim_default = make_sim(
        n_agents=200,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(80),
            emigration_rate=ss.freqperyear(0),
            immigration_age_distribution=AGE_DISTRIBUTION,
        ),
        rand_seed=23,
        tb_pars=dict(
            init_prev=ss.bernoulli(0.2),
            beta=ss.peryear(0.0),
            inf_non=ss.peryear(0.30),
            inf_asy=ss.peryear(0.10),
        ),
        stop='2001-01-01',
    )
    sim_default.run()
    migration_default = get_migration(sim_default)
    derived = migration_default.pars.tb_state_distribution

    assert abs(float(derived['SUSCEPTIBLE']) - 0.8) < 1e-6
    assert abs(float(derived['INFECTION']) - 0.15) < 1e-6
    assert abs(float(derived['ASYMPTOMATIC']) - 0.05) < 1e-6

    explicit_dist = dict(SUSCEPTIBLE=0.6, INFECTION=0.3, ASYMPTOMATIC=0.1)
    sim_override = make_sim(
        n_agents=200,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(80),
            emigration_rate=ss.freqperyear(0),
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=explicit_dist,
        ),
        rand_seed=24,
        tb_pars=dict(
            init_prev=ss.bernoulli(0.5),
            beta=ss.peryear(0.0),
            inf_non=ss.peryear(0.01),
            inf_asy=ss.peryear(0.99),
        ),
        stop='2001-01-01',
    )
    sim_override.run()
    migration_override = get_migration(sim_override)
    overridden = migration_override.pars.tb_state_distribution

    assert abs(float(overridden['SUSCEPTIBLE']) - 0.6) < 1e-6
    assert abs(float(overridden['INFECTION']) - 0.3) < 1e-6
    assert abs(float(overridden['ASYMPTOMATIC']) - 0.1) < 1e-6


def test_maintain_population_uses_actual_emigrant_pool():
    """Maintain-population mode should not overshoot when emigration demand exceeds eligibility."""
    initial_n = 30
    sim = make_sim(
        n_agents=initial_n,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(0),
            emigration_rate=ss.freqperyear(300),
            maintain_population=True,
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
            immigration_age_distribution=AGE_DISTRIBUTION,
        ),
        rand_seed=31,
        tb_pars=QUIET_TB_PARS,
        stop='2001-01-01',
    )
    sim.run()

    migration = get_migration(sim)
    tb = sim.people.tb
    active = np.asarray(sim.people.alive, dtype=bool) & ~np.isin(np.asarray(tb.state), tbsim.TBS.terminal_states())
    assert int(active.sum()) == initial_n, 'Maintain-population mode should preserve active population near baseline'
    assert int(migration.results.n_immigrants[:].sum()) > 0, 'Compensating immigration should occur in this stress case'


def test_emigration_age_distribution_biases_selection():
    """Age-weighted emigration should preferentially select older agents when configured."""
    class CaptureEmigrants(ss.Analyzer):
        def __init__(self):
            super().__init__()
            self.uids = []
            return

        def step(self):
            removed_now = np.asarray(self.sim.people.ti_removed) == self.sim.ti
            if np.any(removed_now):
                uid = np.asarray(self.sim.people.uid, dtype=int)
                self.uids.extend(uid[removed_now].tolist())
            return

    analyzer = CaptureEmigrants()
    sim = make_sim(
        n_agents=600,
        migration_pars=dict(
            immigration_rate=ss.freqperyear(0),
            emigration_rate=ss.freqperyear(150),
            emigration_age_distribution={0: 0.05, 15: 0.2, 30: 1.0, 50: 1.2, 65: 0.8},
            immigration_age_distribution=AGE_DISTRIBUTION,
            tb_state_distribution=dict(SUSCEPTIBLE=1.0),
        ),
        analyzers=[analyzer],
        rand_seed=41,
        tb_pars=QUIET_TB_PARS,
        stop='2002-01-01',
    )
    sim.init()
    baseline_ages = np.asarray(sim.people.age.values, dtype=float).copy()
    sim.run()

    analyzer = sim.analyzers[0]
    emigrant_uids = np.asarray(analyzer.uids, dtype=int)
    assert len(emigrant_uids) > 0, 'Expected emigrants under positive emigration rates'

    baseline_mean_age = float(np.mean(baseline_ages))
    emigrant_mean_age = float(np.mean(baseline_ages[emigrant_uids]))
    assert emigrant_mean_age > baseline_mean_age + 2.0, (
        f'Age-weighted emigration should skew older: baseline={baseline_mean_age:.2f}, emigrants={emigrant_mean_age:.2f}'
    )