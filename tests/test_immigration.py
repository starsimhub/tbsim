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


def make_sim(n_agents=300, immigration_rate=200, tb_state_distribution=None, beta=0.05, use_households=False):
    """Create a small TB simulation with immigration enabled."""
    tb = tbsim.TB(pars=dict(
        init_prev=ss.bernoulli(0.02),
        beta=ss.peryear(beta),
    ))
    imm = tbsim.Immigration(pars=dict(
        immigration_rate=ss.freqperyear(immigration_rate),
        tb_state_distribution=tb_state_distribution or dict(
            SUSCEPTIBLE=0.9,
            INFECTION=0.08,
            ASYMPTOMATIC=0.02,
        ),
    ))
    networks = [ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=4), dur=0))]
    if use_households:
        dhs_data = make_household_dhs_data(n_agents=n_agents, rand_seed=2)
        networks.append(ss.HouseholdNet(dhs_data=dhs_data, dynamic=False))

    sim = ss.Sim(
        n_agents=n_agents,
        start=ss.date('2000-01-01'),
        stop=ss.date('2002-01-01'),
        dt=ss.days(30),
        rand_seed=2,
        verbose=0,
        diseases=tb,
        networks=networks,
        demographics=[imm],
    )
    return sim


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
