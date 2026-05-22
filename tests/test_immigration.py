"""Tests for the Immigration demographics module."""

import numpy as np
import pandas as pd
import pytest
import sciris as sc
import starsim as ss
import tbsim


def make_household_dhs_data(n_agents=300, rand_seed=2):
    """Create synthetic DHS household table for ss.HouseholdNet."""
    rng = np.random.default_rng(rand_seed)
    hh_id = []
    ages = []
    i = 0
    h = 0
    while i < n_agents:
        hh_size = int(rng.integers(2, 6))
        hh_size = min(hh_size, n_agents - i)
        hh_ages = rng.integers(1, 75, size=hh_size)
        hh_id.append(h)
        ages.append(sc.strjoin(hh_ages))
        i += hh_size
        h += 1
    return sc.dataframe(hh_id=hh_id, ages=ages)


def make_sim(
    n_agents=300,
    immigration_rate=200,
    rel_immigration=1.0,
    age_distribution=None,
    age_data=None,
    tb_state_distribution=None,
    beta=0.05,
    use_acute=False,
    use_households=False,
    start=None,
    stop=None,
):
    """Create a small TB simulation with immigration enabled."""
    tb_pars = dict(init_prev=ss.bernoulli(0.02), beta=ss.peryear(beta))
    tb = tbsim.TBAcute(pars=tb_pars) if use_acute else tbsim.TB(pars=tb_pars)
    imm_pars = dict(
        immigration_rate=ss.freqperyear(immigration_rate),
        rel_immigration=rel_immigration,
        tb_state_distribution=tb_state_distribution or dict(
            SUSCEPTIBLE=0.9,
            INFECTION=0.08,
            ASYMPTOMATIC=0.02,
        ),
    )
    if age_distribution is not None:
        imm_pars['age_distribution'] = age_distribution
    if age_data is not None:
        imm_pars['age_data'] = age_data
    imm = tbsim.Immigration(pars=imm_pars)
    networks = [ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0))]
    if use_households:
        dhs_data = make_household_dhs_data(n_agents=n_agents, rand_seed=2)
        networks.append(ss.HouseholdNet(dhs_data=dhs_data, dynamic=False))

    sim = ss.Sim(
        n_agents=n_agents,
        start=start or ss.date('2000-01-01'),
        stop=stop or ss.date('2002-01-01'),
        dt=ss.days(30),
        rand_seed=2,
        verbose=0,
        diseases=tb,
        networks=networks,
        demographics=[imm],
    )
    return sim


def immigrant_age_bin_proportions(ages, age_distribution):
    """Map immigrant ages to configured age-bin proportions (dict ``age_distribution`` path)."""
    keys = sorted(age_distribution.keys())
    weights = np.array([age_distribution[k] for k in keys], dtype=float)
    expected = weights / weights.sum()
    highs = list(keys[1:]) + [85.0]
    ages = np.asarray(ages, dtype=float)
    counts = np.zeros(len(keys), dtype=float)
    for i, (lo, hi) in enumerate(zip(keys, highs)):
        if i < len(keys) - 1:
            counts[i] = np.sum((ages >= lo) & (ages < hi))
        else:
            counts[i] = np.sum((ages >= lo) & (ages <= hi))
    observed = counts / counts.sum()
    return observed, expected


def age_data_bin_edges(age_data):
    """Bin edges for ``age_data`` using the same rules as ``ss.People.get_age_dist`` / ``ss.histogram``."""
    if isinstance(age_data, pd.DataFrame):
        bins = np.asarray(age_data['age'], dtype=float)
        weights = np.asarray(age_data['value'], dtype=float)
    else:
        raise TypeError('test helper expects a DataFrame age_data')
    if len(bins) == len(weights):
        delta = bins[-1] - bins[-2]
        bins = np.append(bins, bins[-1] + delta)
    return bins, weights / weights.sum()


def immigrant_age_bin_proportions_from_edges(ages, bin_edges, expected):
    """Map immigrant ages to histogram bins defined by ``bin_edges``."""
    ages = np.asarray(ages, dtype=float)
    n_bins = len(expected)
    counts = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < n_bins - 1:
            counts[i] = np.sum((ages >= lo) & (ages < hi))
        else:
            counts[i] = np.sum((ages >= lo) & (ages <= hi))
    observed = counts / counts.sum()
    return observed, expected


def poisson_count_tolerance(expected):
    """~3-sigma bound for Poisson count uncertainty (migration plan validation)."""
    return 3.0 * np.sqrt(max(expected, 1.0))


def assert_immigrants_have_household_edges(sim):
    """Check immigrants received household IDs and at least one household edge each."""
    imm = get_imm(sim)
    hh = sim.networks.householdnet
    immigrant_uids = imm.is_immigrant.uids
    assert len(immigrant_uids) > 0, 'Expected some immigrants in this scenario'
    assert np.all(imm.hhid[immigrant_uids] >= 0), 'Expected all immigrants to receive non-negative household IDs'

    p1 = np.asarray(hh.edges.p1, dtype=int)
    p2 = np.asarray(hh.edges.p2, dtype=int)
    max_uid = int(max(np.max(p1) if p1.size else 0, np.max(p2) if p2.size else 0, np.max(immigrant_uids)))
    deg = np.zeros(max_uid + 1, dtype=int)
    if p1.size:
        np.add.at(deg, p1, 1)
        np.add.at(deg, p2, 1)
    assert np.all(deg[np.asarray(immigrant_uids, dtype=int)] > 0), 'Expected each immigrant to have at least one household edge'


def make_treatment_sim(n_agents=400, immigration_rate=400, beta=0.06):
    """Create a TBsim simulation with immigration and active-TB treatment."""
    imm = tbsim.Immigration(pars=dict(
        immigration_rate=ss.freqperyear(immigration_rate),
        tb_state_distribution=dict(
            SUSCEPTIBLE=0.70,
            INFECTION=0.10,
            NON_INFECTIOUS=0.05,
            ASYMPTOMATIC=0.10,
            SYMPTOMATIC=0.05,
        ),
    ))

    tx = tbsim.TxDelivery(
        product=tbsim.DOTS(),
        eligibility=lambda sim: ss.uids(np.where(
            np.isin(tbsim.get_tb(sim).state, tbsim.TBS.active_tb_states()) & np.asarray(sim.people.alive)
        )[0]),
    )

    sim = tbsim.Sim(
        sim_pars=dict(
            n_agents=n_agents,
            start=ss.date('2000-01-01'),
            stop=ss.date('2003-01-01'),
            dt=ss.days(30),
            rand_seed=2,
            verbose=0,
        ),
        tb_pars=dict(
            init_prev=ss.bernoulli(0.10),
            beta=ss.peryear(beta),
        ),
        networks=[
            ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0)),
            ss.HouseholdNet(dhs_data=make_household_dhs_data(n_agents=n_agents, rand_seed=2), dynamic=False),
        ],
        demographics=[imm],
        interventions=[tx],
    )
    return sim


def get_imm(sim):
    """Get the immigration module from a simulation."""
    for dem in sim.demographics.values():
        if isinstance(dem, tbsim.Immigration):
            return dem
    raise RuntimeError('Immigration module not found')


def extract_n_infectious(sim):
    """Extract TB infectious prevalence in calibration-compatible format."""
    tb = tbsim.get_tb(sim)
    t = pd.Index(tb.results.timevec[:], name='t')
    x = np.asarray(tb.results.n_infectious[:], dtype=float)
    return pd.DataFrame(dict(x=x), index=t)


def test_immigration_arrivals_match_rate():
    """Cumulative immigrants match immigration_rate * horizon within Poisson tolerance."""
    rate = 120.0
    start = ss.date('2000-01-01')
    stop = ss.date('2004-01-01')
    sim = make_sim(n_agents=100, immigration_rate=rate, beta=0.0, start=start, stop=stop)
    sim.run()

    imm = get_imm(sim)
    total_imm = int(np.sum(imm.results.n_immigrants[:]))
    n_years = float(stop - start)
    expected = rate * n_years
    tol = poisson_count_tolerance(expected)

    assert total_imm > 0, 'Expected at least one immigrant over the test horizon'
    assert abs(total_imm - expected) <= tol, (
        f'Expected ~{expected:.0f} immigrants ({rate}/yr x {n_years:.0f} yr) within ±{tol:.0f}, got {total_imm}'
    )


def test_immigrant_age_data_matches_config():
    """Immigrant ages match Starsim age_data histogram bins (ss.People.get_age_dist)."""
    age_data = pd.DataFrame({'age': [0, 15, 30], 'value': [0.50, 0.30, 0.20]})
    sim = make_sim(
        n_agents=80,
        immigration_rate=400,
        age_data=age_data,
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2003-01-01'),
    )
    sim.run()

    imm = get_imm(sim)
    immigrant_uids = imm.is_immigrant.uids
    n_imm = len(immigrant_uids)
    assert n_imm >= 200, f'Expected enough immigrants for age_data test, got {n_imm}'
    assert imm._dist_age is not None, 'Expected age_data path to configure _dist_age histogram'

    ages = np.asarray(imm.age_at_immigration[immigrant_uids], dtype=float)
    bin_edges, expected = age_data_bin_edges(age_data)
    observed, expected = immigrant_age_bin_proportions_from_edges(ages, bin_edges, expected)
    per_bin_tol = 3.0 * np.sqrt(expected * (1.0 - expected) / n_imm)
    max_tol = max(0.08, float(np.max(per_bin_tol)))
    max_diff = float(np.max(np.abs(observed - expected)))

    assert max_diff <= max_tol, (
        f'age_data bin proportions differ by up to {max_diff:.3f} (tol {max_tol:.3f}); '
        f'observed {observed.round(3).tolist()}, expected {expected.round(3).tolist()}'
    )


def test_age_data_overrides_age_distribution():
    """age_data takes precedence over age_distribution when both are supplied."""
    age_data = pd.DataFrame({'age': [0, 40], 'value': [1.0, 0.0]})
    sim = make_sim(
        n_agents=60,
        immigration_rate=800,
        age_data=age_data,
        age_distribution={0: 0.0, 65: 1.0},
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2001-06-01'),
    )
    with pytest.warns(UserWarning, match='age_data is set; ignoring age_distribution'):
        sim.init()

    imm = get_imm(sim)
    sim.run()
    ages = np.asarray(imm.age_at_immigration[imm.is_immigrant.uids], dtype=float)
    assert len(ages) > 50, 'Expected enough immigrants to check age_data precedence'
    assert np.nanmax(ages) < 50, 'age_data should restrict immigrants to the [0, 40) histogram bin, not age_distribution seniors'


def test_immigrant_age_distribution_matches_config():
    """Immigrant ages fall into configured age bins at the expected proportions."""
    age_distribution = {0: 0.50, 15: 0.30, 30: 0.20}
    sim = make_sim(
        n_agents=80,
        immigration_rate=400,
        age_distribution=age_distribution,
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2003-01-01'),
    )
    sim.run()

    imm = get_imm(sim)
    immigrant_uids = imm.is_immigrant.uids
    n_imm = len(immigrant_uids)
    assert n_imm >= 200, f'Expected enough immigrants for age-bin test, got {n_imm}'

    ages = np.asarray(imm.age_at_immigration[immigrant_uids], dtype=float)
    observed, expected = immigrant_age_bin_proportions(ages, age_distribution)
    # Per-bin 3σ bound; floor 0.08 avoids flaky small-n edge cases (Starsim style guide).
    per_bin_tol = 3.0 * np.sqrt(expected * (1.0 - expected) / n_imm)
    max_tol = max(0.08, float(np.max(per_bin_tol)))
    max_diff = float(np.max(np.abs(observed - expected)))

    assert max_diff <= max_tol, (
        f'Age-bin proportions differ by up to {max_diff:.3f} (tol {max_tol:.3f}); '
        f'observed {observed.round(3).tolist()}, expected {expected.round(3).tolist()}'
    )


def test_migration_runs_with_tb():
    """Immigration runs with the base TB disease module."""
    sim = make_sim(n_agents=200, immigration_rate=300, beta=0.0)
    sim.run()
    imm = get_imm(sim)
    tb = tbsim.get_tb(sim)

    assert isinstance(tb, tbsim.TB), 'Expected base TB module in this scenario'
    assert int(np.sum(imm.results.n_immigrants[:])) > 0, 'Expected immigration results with TB'
    assert len(imm.is_immigrant.uids) > 0, 'Expected at least one flagged immigrant'


def test_migration_runs_with_tbacute():
    """Immigration runs with TBAcute and can seed ACUTE imports."""
    sim = make_sim(
        n_agents=150,
        immigration_rate=600,
        use_acute=True,
        beta=0.0,
        tb_state_distribution=dict(ACUTE=1.0),
        start=ss.date('2000-01-01'),
        stop=ss.date('2001-06-01'),
    )
    sim.run()
    imm = get_imm(sim)
    tb = tbsim.get_tb(sim)
    immigrant_uids = imm.is_immigrant.uids

    assert isinstance(tb, tbsim.TBAcute), 'Expected TBAcute module in this scenario'
    assert len(immigrant_uids) > 0, 'Expected immigrants when using TBAcute'
    entry_states = np.asarray(imm.immigration_tb_status[immigrant_uids], dtype=int)
    assert np.all(entry_states == int(tbsim.TBS.ACUTE)), 'Expected ACUTE at entry for ACUTE-only import distribution'
    assert np.any(tb.infected[immigrant_uids]), 'Expected at least some immigrants to remain infected after progression'


def test_immigrants_are_assigned_households_and_edges():
    """Immigrants join households and gain household-network edges (migration plan integration)."""
    sim = make_sim(n_agents=180, immigration_rate=800, use_households=True)
    sim.run()
    assert_immigrants_have_household_edges(sim)


def test_immigration_rejects_invalid_tb_state_distribution():
    """Invalid tb_state_distribution inputs fail at construction."""
    with pytest.raises(ValueError, match='must be provided'):
        tbsim.Immigration(pars=dict(tb_state_distribution={}))
    with pytest.raises(ValueError, match='at least one positive'):
        tbsim.Immigration(pars=dict(tb_state_distribution=dict(SUSCEPTIBLE=0.0)))
    with pytest.raises(KeyError, match='Unknown TB state'):
        tbsim.Immigration(pars=dict(tb_state_distribution=dict(NOT_A_STATE=1.0)))
    with pytest.raises(ValueError, match='negative'):
        tbsim.Immigration(pars=dict(tb_state_distribution=dict(SUSCEPTIBLE=-0.1)))
    with pytest.raises(ValueError, match='finite'):
        tbsim.Immigration(pars=dict(tb_state_distribution=dict(SUSCEPTIBLE=np.nan)))


def test_immigration_rejects_invalid_age_distribution():
    """Invalid age_distribution inputs fail during sim initialization."""
    tb = tbsim.TB(pars=dict(init_prev=ss.bernoulli(0), beta=ss.peryear(0)))

    def init_with_age_dist(age_distribution):
        imm = tbsim.Immigration(pars=dict(immigration_rate=ss.freqperyear(10), age_distribution=age_distribution))
        sim = ss.Sim(n_agents=50, start='2000', stop='2001', dt=0.25, diseases=tb, demographics=[imm], networks=ss.RandomNet(), verbose=0)
        sim.init()

    with pytest.raises(ValueError, match='non-negative'):
        init_with_age_dist({0: -1})
    with pytest.raises(ValueError, match='at least one positive'):
        init_with_age_dist({0: 0.0, 5: 0.0})
    with pytest.raises(ValueError, match='finite'):
        init_with_age_dist({0: np.nan})


def test_immigration_exported_and_runs():
    """Immigration is exported and updates population/results during run."""
    assert hasattr(tbsim, 'Immigration')

    sim = make_sim(n_agents=250, immigration_rate=240)
    n0 = sim.pars.n_agents
    sim.run()

    imm = get_imm(sim)
    n_added = int(np.sum(imm.results.n_immigrants[:]))
    assert n_added > 0, 'Expected at least one immigrant in this scenario'
    assert len(sim.people) == n0 + n_added, 'Population should increase by total immigrants when no other demographics are active'
    assert np.count_nonzero(np.asarray(imm.is_immigrant)) == n_added, 'is_immigrant flags should match total arrivals'


def test_imported_tb_states_are_consistent():
    """Imported agents get TB states/flags consistent with current TB implementation."""
    sim = make_sim(
        n_agents=120,
        immigration_rate=1_500,
        tb_state_distribution=dict(INFECTION=1.0),
        beta=0.0,
    )
    sim.init()
    imm = get_imm(sim)
    tb = tbsim.get_tb(sim)

    new_uids = imm.step()
    assert len(new_uids) > 0, 'Expected immigration step to add agents'
    assert np.all(tb.state[new_uids] == tbsim.TBS.INFECTION), 'Expected imported states to match configured distribution'
    assert np.all(tb.infected[new_uids]), 'INFECTION imports should be marked infected'
    assert not np.any(tb.susceptible[new_uids]), 'INFECTION imports should not be marked susceptible'
    assert np.all(np.isneginf(tb.ti_infected[new_uids])), 'Imported TB should not be counted as model incident infection'


def test_immigration_assigns_households_when_network_present():
    """Immigrants are assigned household IDs and household edges."""
    sim = make_sim(n_agents=180, immigration_rate=800, use_households=True)
    sim.run()
    assert_immigrants_have_household_edges(sim)


def test_immigration_does_not_break_calibration():
    """Calibration loop runs with Immigration in demographics."""
    pytest.importorskip('optuna')

    ref_sim = make_sim(n_agents=150, immigration_rate=120, beta=0.06)
    ref_sim.run()
    expected = extract_n_infectious(ref_sim)

    def build_fn(sim, calib_pars):
        if hasattr(sim, 'diseases'):
            tb = tbsim.get_tb(sim)
        else:
            tb = sim.pars.diseases
        tb.pars.beta = ss.peryear(calib_pars['beta']['value'])
        return sim

    comp = ss.Normal(
        name='tb_n_infectious',
        expected=expected,
        extract_fn=extract_n_infectious,
        conform='prevalent',
        sigma2=25.0,
    )

    calib_sim = make_sim(n_agents=150, immigration_rate=120, beta=0.06)
    calib = ss.Calibration(
        sim=calib_sim,
        calib_pars=dict(beta=dict(low=0.02, high=0.12, guess=0.06)),
        build_fn=build_fn,
        components=[comp],
        total_trials=2,
        n_workers=1,
        reseed=False,
        debug=True,
        verbose=False,
    )
    calib.calibrate()

    assert 'beta' in calib.best_pars, 'Calibration should return a best-fit beta parameter'
    assert np.isfinite(calib.df.mismatch.iloc[0]), 'Calibration objective should be finite'


def test_immigration_with_treatment_runs():
    """Immigration works alongside treatment delivery and treats immigrants."""
    sim = make_treatment_sim()
    sim.run()

    imm = get_imm(sim)
    tx = sim.interventions.txdelivery

    total_immigrants = int(sim.results.immigration.n_immigrants.sum())
    total_treated = int(sim.results.txdelivery.n_treated.sum())
    immigrant_uids = imm.is_immigrant.uids
    treated_immigrants = int(np.count_nonzero(np.asarray(tx.n_times_treated[immigrant_uids]) > 0))

    assert total_immigrants > 0, 'Expected immigrants to be added in treatment scenario'
    assert total_treated > 0, 'Expected treatment intervention to treat active TB agents'
    assert treated_immigrants > 0, 'Expected at least one immigrant to receive treatment'
