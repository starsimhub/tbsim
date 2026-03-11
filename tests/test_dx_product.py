"""Tests for the Dx diagnostic product."""

import pytest
import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL
from tbsim.interventions.dx_products import Dx


def make_dx_sim(n_agents=200):
    """Create a minimal sim with TB for testing Dx products."""
    pop = ss.People(n_agents=n_agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    sim = ss.Sim(people=pop, diseases=tb, networks=net, pars=pars)
    sim.init()
    return sim


def test_dx_simple_dataframe():
    """Dx with only required columns (state, result, probability) returns correct dict."""
    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=1.0),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.0),
    ])
    dx = Dx(df=df, hierarchy=['positive', 'negative'])

    sim = make_dx_sim()
    # Run a few steps so some agents become symptomatic
    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    assert 'positive' in results
    assert 'negative' in results

    # With sensitivity=1.0 for SYMPTOMATIC, all symptomatic agents should test positive
    symptomatic_uids = ss.uids(np.where(np.asarray(tb.state) == TBSL.SYMPTOMATIC)[0])
    symptomatic_alive = np.intersect1d(symptomatic_uids, all_uids)
    positive_uids = np.asarray(results['positive'])
    for uid in symptomatic_alive:
        assert uid in positive_uids, f"Symptomatic agent {uid} should test positive with sensitivity=1.0"

    # Agents not in df states should default to 'negative' (last in hierarchy)
    non_symptomatic = all_uids[~np.isin(all_uids, symptomatic_alive)]
    negative_uids = np.asarray(results['negative'])
    for uid in non_symptomatic[:10]:  # spot-check
        assert uid in negative_uids, f"Non-symptomatic agent {uid} should default to negative"


def test_dx_age_stratified():
    """Dx with age_min/age_max columns filters agents by age."""
    df = pd.DataFrame([
        # Adults: high sensitivity
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=1.0, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.0, age_min=15, age_max=np.inf),
        # Children: zero sensitivity (always negative)
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.0, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=1.0, age_min=0, age_max=15),
    ])
    dx = Dx(df=df, hierarchy=['positive', 'negative'])

    sim = make_dx_sim(n_agents=500)
    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    # Check that symptomatic children test negative
    symptomatic = np.asarray(tb.state) == TBSL.SYMPTOMATIC
    children = np.asarray(sim.people.age) < 15
    child_symptomatic = ss.uids(np.where(symptomatic & children)[0])
    child_symptomatic_alive = np.intersect1d(child_symptomatic, all_uids)

    negative_uids = np.asarray(results['negative'])
    for uid in child_symptomatic_alive:
        assert uid in negative_uids, f"Child symptomatic agent {uid} should test negative with sensitivity=0.0"


def test_dx_probabilities_sum_to_one_validation():
    """Dx raises an error if probabilities don't sum to 1.0 per group."""
    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.5),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.3),  # sums to 0.8, not 1.0
    ])
    with pytest.raises(ValueError, match="not 1.0"):
        Dx(df=df, hierarchy=['positive', 'negative'])


def test_dx_default_hierarchy():
    """Dx infers hierarchy from df.result.unique() if not provided."""
    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC, result='positive', probability=0.9),
        dict(state=TBSL.SYMPTOMATIC, result='negative', probability=0.1),
    ])
    dx = Dx(df=df)
    assert list(dx.hierarchy) == ['positive', 'negative']


def test_xpert_factory():
    """xpert() returns a Dx with age-stratified, state-stratified DataFrame."""
    from tbsim.interventions.dx_products import xpert
    dx = xpert()
    assert isinstance(dx, Dx)
    assert 'age_min' in dx.df.columns
    assert 'age_max' in dx.df.columns
    assert 'hiv' not in dx.df.columns
    assert set(dx.hierarchy) == {'positive', 'negative'}
    # Check adult symptomatic sensitivity
    adult_symp = dx.df[(dx.df.state == TBSL.SYMPTOMATIC) & (dx.df.age_min == 15) & (dx.df.result == 'positive')]
    assert np.isclose(adult_symp.probability.values[0], 0.909)


def test_oral_swab_factory():
    """oral_swab() returns a Dx with age and state stratification."""
    from tbsim.interventions.dx_products import oral_swab
    dx = oral_swab()
    assert isinstance(dx, Dx)
    assert 'hiv' not in dx.df.columns


def test_fujilam_factory():
    """fujilam() returns a Dx with HIV stratification."""
    from tbsim.interventions.dx_products import fujilam
    dx = fujilam()
    assert isinstance(dx, Dx)
    assert 'hiv' in dx.df.columns


def test_cad_cxr_factory():
    """cad_cxr() returns a Dx product."""
    from tbsim.interventions.dx_products import cad_cxr
    dx = cad_cxr()
    assert isinstance(dx, Dx)


def test_xpert_runs_in_sim():
    """xpert() product can be administered in a running sim."""
    from tbsim.interventions.dx_products import xpert
    dx = xpert()
    sim = make_dx_sim(n_agents=200)
    for _ in range(20):
        sim.run_one_step()
    results = dx.administer(sim, sim.people.alive.uids)
    assert len(results['positive']) + len(results['negative']) == len(sim.people.alive.uids)
