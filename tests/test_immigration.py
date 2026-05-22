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


def make_imm(rate=200, **pars):
    """Build an Immigration module with compact defaults."""
    return tbsim.Immigration(pars=dict(
        immigration_rate=ss.freqperyear(rate),
        tb_state_distribution=pars.pop('tb_state_distribution', dict(SUSCEPTIBLE=0.9, INFECTION=0.08, ASYMPTOMATIC=0.02)),
        **pars,
    ))


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
    demographics=None,
    interventions=None,
    start=None,
    stop=None,
):
    """Create a small TB simulation with immigration enabled."""
    tb_pars = dict(init_prev=ss.bernoulli(0.02), beta=ss.peryear(beta))
    tb = tbsim.TBAcute(pars=tb_pars) if use_acute else tbsim.TB(pars=tb_pars)
    if demographics is None:
        imm_kw = dict(rel_immigration=rel_immigration, tb_state_distribution=tb_state_distribution)
        if age_distribution is not None:
            imm_kw['age_distribution'] = age_distribution
        if age_data is not None:
            imm_kw['age_data'] = age_data
        demographics = [make_imm(immigration_rate, **{k: v for k, v in imm_kw.items() if v is not None})]
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
        demographics=demographics,
        interventions=interventions or [],
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


def assert_immigrant_age_bins_match(sim, age_spec, spec_type):
    """Check immigrant age-bin proportions against dict or age_data configuration."""
    imm = get_imm(sim)
    immigrant_uids = imm.is_immigrant.uids
    n_imm = len(immigrant_uids)
    assert n_imm >= 200, f'Expected enough immigrants for age-bin test, got {n_imm}'

    ages = np.asarray(imm.age_at_immigration[immigrant_uids], dtype=float)
    if spec_type == 'dict':
        observed, expected = immigrant_age_bin_proportions(ages, age_spec)
    elif spec_type == 'age_data':
        assert imm._dist_age is not None, 'Expected age_data path to configure _dist_age histogram'
        bin_edges, expected = age_data_bin_edges(age_spec)
        observed, expected = immigrant_age_bin_proportions_from_edges(ages, bin_edges, expected)
    else:
        raise ValueError(f'Unknown age spec type: {spec_type}')

    per_bin_tol = 3.0 * np.sqrt(expected * (1.0 - expected) / n_imm)
    max_tol = max(0.08, float(np.max(per_bin_tol)))
    max_diff = float(np.max(np.abs(observed - expected)))
    assert max_diff <= max_tol, (
        f'Age-bin proportions differ by up to {max_diff:.3f} (tol {max_tol:.3f}); '
        f'observed {observed.round(3).tolist()}, expected {expected.round(3).tolist()}'
    )


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
    if p1.size:
        assert not np.any(p1 == p2), 'Household network should not contain self-edges'
        undirected = np.sort(np.vstack([p1, p2]), axis=0).T
        assert len(undirected) == len({(int(a), int(b)) for a, b in undirected}), 'Household edges should be unique undirected pairs'


def make_treatment_sim(n_agents=400, immigration_rate=400, beta=0.06):
    """Create a TBsim simulation with immigration and active-TB treatment."""
    imm = make_imm(immigration_rate, tb_state_distribution=dict(
        SUSCEPTIBLE=0.70, INFECTION=0.10, NON_INFECTIOUS=0.05, ASYMPTOMATIC=0.10, SYMPTOMATIC=0.05,
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


def assert_immigration_smoke(sim):
    """Expect a completed run to have added immigrants."""
    imm = get_imm(sim)
    assert int(np.sum(imm.results.n_immigrants[:])) > 0, 'Expected positive per-step immigration counts'
    assert len(imm.is_immigrant.uids) > 0, 'Expected at least one flagged immigrant'


def assert_explicit_demographics_integration(sim):
    """Explicit Births + Deaths + Immigration list runs and immigration is active."""
    assert any(isinstance(d, ss.Births) for d in sim.demographics.values()), 'Expected Births in demographics list'
    assert any(isinstance(d, ss.Deaths) for d in sim.demographics.values()), 'Expected Deaths in demographics list'
    assert any(isinstance(d, tbsim.Immigration) for d in sim.demographics.values()), 'Expected Immigration in demographics list'
    assert_immigration_smoke(sim)


def assert_tpt_integration(sim):
    """Immigration coexists with TPT; both modules run without blocking arrivals."""
    assert any(isinstance(i, tbsim.TPTSimple) for i in sim.interventions.values()), 'Expected TPTSimple intervention'
    tpt = next(i for i in sim.interventions.values() if isinstance(i, tbsim.TPTSimple))
    assert hasattr(tpt, 'results'), 'Expected TPT result channels after run'
    assert_immigration_smoke(sim)


def assert_immigrant_tb_bin_proportions(sim, tb_state_distribution, min_immigrants=200):
    """Immigrant entry TB states match configured tb_state_distribution within tolerance."""
    imm = get_imm(sim)
    immigrant_uids = imm.is_immigrant.uids
    n_imm = len(immigrant_uids)
    assert n_imm >= min_immigrants, f'Expected enough immigrants for TB-state test, got {n_imm}'

    dist = {k: float(v) for k, v in tb_state_distribution.items() if v}
    total = sum(dist.values())
    expected = {k: v / total for k, v in dist.items()}
    codes = {k: int(getattr(tbsim.TBS, k)) for k in expected}
    entry = np.asarray(imm.immigration_tb_status[immigrant_uids], dtype=int)

    observed = []
    exp_props = []
    for name, prop in expected.items():
        observed.append(np.mean(entry == codes[name]))
        exp_props.append(prop)
    observed = np.asarray(observed)
    exp_props = np.asarray(exp_props)
    per_bin_tol = 3.0 * np.sqrt(exp_props * (1.0 - exp_props) / n_imm)
    max_tol = max(0.08, float(np.max(per_bin_tol)))
    max_diff = float(np.max(np.abs(observed - exp_props)))
    assert max_diff <= max_tol, (
        f'TB-state proportions differ by up to {max_diff:.3f} (tol {max_tol:.3f}); '
        f'observed {observed.round(3).tolist()}, expected {exp_props.round(3).tolist()}'
    )


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


@pytest.mark.parametrize('age_spec,spec_type', [
    ({0: 0.50, 15: 0.30, 30: 0.20}, 'dict'),
    (pd.DataFrame({'age': [0, 15, 30], 'value': [0.50, 0.30, 0.20]}), 'age_data'),
], ids=['age_distribution', 'age_data'])
def test_immigrant_age_bins_match_config(age_spec, spec_type):
    """Immigrant ages match configured bins for dict age_distribution and Starsim age_data."""
    kwargs = dict(age_distribution=age_spec) if spec_type == 'dict' else dict(age_data=age_spec)
    sim = make_sim(
        n_agents=80,
        immigration_rate=400,
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2003-01-01'),
        **kwargs,
    )
    sim.run()
    assert_immigrant_age_bins_match(sim, age_spec, spec_type)


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


def test_acute_imports_require_tbacute():
    """ACUTE imports with base TB must fail at step time."""
    sim = make_sim(immigration_rate=500, tb_state_distribution=dict(ACUTE=1.0), beta=0.0)
    sim.init()
    with pytest.raises(ValueError, match='not TBAcute'):
        get_imm(sim).step()


@pytest.mark.parametrize('demographics,interventions,assert_fn,tb_state_distribution', [
    pytest.param(
        lambda: [ss.Births(), ss.Deaths(), make_imm(300)],
        [],
        assert_explicit_demographics_integration,
        None,
        id='explicit_demographics_list',
    ),
    pytest.param(
        None,
        [tbsim.TPTSimple(pars=dict(coverage=0.5))],
        assert_tpt_integration,
        dict(INFECTION=0.4, SUSCEPTIBLE=0.6),
        id='with_tpt',
    ),
])
def test_migration_compatibility_smoke(demographics, interventions, assert_fn, tb_state_distribution):
    """Immigration runs with explicit demographics list and with TPT (plan integration smokes)."""
    dems = demographics() if callable(demographics) else demographics
    sim = make_sim(
        n_agents=200,
        immigration_rate=400,
        demographics=dems,
        interventions=interventions,
        tb_state_distribution=tb_state_distribution,
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2003-01-01'),
    )
    sim.run()
    assert_fn(sim)


def test_immigrant_tb_states_match_config():
    """Immigrant entry TB states match configured tb_state_distribution within tolerance."""
    tb_state_distribution = dict(INFECTION=0.40, SUSCEPTIBLE=0.60)
    sim = make_sim(
        n_agents=100,
        immigration_rate=500,
        tb_state_distribution=tb_state_distribution,
        beta=0.0,
        start=ss.date('2000-01-01'),
        stop=ss.date('2003-01-01'),
    )
    sim.run()
    assert_immigrant_tb_bin_proportions(sim, tb_state_distribution)


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
    """Immigration is exported, runs with TB, and updates population/results."""
    assert hasattr(tbsim, 'Immigration')

    sim = make_sim(n_agents=250, immigration_rate=240)
    n0 = sim.pars.n_agents
    sim.run()

    imm = get_imm(sim)
    tb = tbsim.get_tb(sim)
    n_added = int(np.sum(imm.results.n_immigrants[:]))
    assert isinstance(tb, tbsim.TB), 'Expected base TB module in this scenario'
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
