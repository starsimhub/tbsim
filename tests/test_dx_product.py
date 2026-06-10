"""Tests for the Dx diagnostic product."""

import pytest
import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBS


def make_dx_sim(n_agents=200, dx=None, hierarchy=None):
    """Create a minimal sim with TB and a Dx product wrapped in DxDelivery.

    Args:
        dx: Either a Dx product instance or a DataFrame to create one from.
        hierarchy: Hierarchy list (used when dx is a DataFrame).

    Returns the sim after init. Access the product via sim.interventions.dxdelivery.product.
    """
    if isinstance(dx, pd.DataFrame):
        dx = tbsim.Dx(df=dx, hierarchy=hierarchy or ['positive', 'negative'])
    interventions = []
    if dx is not None:
        interventions.append(tbsim.DxDelivery(product=dx))

    sim = tbsim.Sim(
        n_agents=n_agents,
        interventions=interventions,
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01')),
        tb_pars=dict(init_prev=0.30),
    )
    sim.init()
    return sim


def test_dx_simple_dataframe():
    """Dx with only required columns (state, result, probability) returns correct dict."""
    df = pd.DataFrame([
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=1.0),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=0.0),
    ])
    sim = make_dx_sim(dx=df)

    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    dx = sim.interventions.dxdelivery.product
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    assert 'positive' in results
    assert 'negative' in results

    # With sensitivity=1.0 for SYMPTOMATIC, all symptomatic agents should test positive
    symptomatic_uids = (tb.state == TBS.SYMPTOMATIC).uids
    symptomatic_alive = symptomatic_uids & all_uids
    positive_uids = results['positive']
    for uid in symptomatic_alive:
        assert uid in positive_uids, f"Symptomatic agent {uid} should test positive with sensitivity=1.0"

    # Agents not in df states should default to 'negative' (last in hierarchy)
    non_symptomatic = all_uids[~np.isin(all_uids, symptomatic_alive)]
    negative_uids = results['negative']
    for uid in non_symptomatic[:10]:  # spot-check
        assert uid in negative_uids, f"Non-symptomatic agent {uid} should default to negative"


def test_dx_administer_correct_after_deaths():
    """ProductMulti.administer maps states -> results -> UIDs correctly after
    deaths create a UID/position gap (issue #425 bug class).

    administer already maps results back via ``uids[results == i]``, so it is
    UID-safe; this guards that property explicitly. (It does not exercise the
    CRN-safety of ``temp_uids`` -- see starsim#1254.)
    """
    df = pd.DataFrame([
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=1.0),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=0.0),
    ])
    sim = make_dx_sim(n_agents=300, dx=df)
    ppl = sim.people
    tb = tbsim.get_tb(sim)
    dx = sim.interventions.dxdelivery.product

    # Kill a block of low-numbered agents to force a UID/position gap.
    ppl.request_death(ss.uids(np.arange(0, len(ppl) // 3)))
    ppl.step_die()
    ppl.remove_dead()
    alive = ppl.alive.uids
    assert not np.array_equal(np.asarray(alive), np.arange(len(alive))), \
        "Test setup failed to create a UID/position gap"

    # Make a block of high-UID alive agents symptomatic.
    symptomatic = alive[-20:]
    tb.state[symptomatic] = TBS.SYMPTOMATIC

    results = dx.administer(sim, alive)
    # With sensitivity=1.0, positives must be exactly the symptomatic block.
    assert set(np.asarray(results['positive']).tolist()) == set(np.asarray(symptomatic).tolist()), \
        "Positives must be exactly the genuinely symptomatic agents"


def test_dx_age_stratified():
    """Dx with age_min/age_max columns filters agents by age."""
    df = pd.DataFrame([
        # Adults: high sensitivity
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=1.0, age_min=15, age_max=np.inf),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=0.0, age_min=15, age_max=np.inf),
        # Children: zero sensitivity (always negative)
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=0.0, age_min=0, age_max=15),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=1.0, age_min=0, age_max=15),
    ])
    sim = make_dx_sim(n_agents=500, dx=df)

    for _ in range(20):
        sim.run_one_step()

    tb = tbsim.get_tb(sim)
    dx = sim.interventions.dxdelivery.product
    all_uids = sim.people.alive.uids
    results = dx.administer(sim, all_uids)

    # Check that symptomatic children test negative
    symptomatic = tb.state == TBS.SYMPTOMATIC
    children = sim.people.age < 15
    child_symptomatic_alive = (symptomatic & children).uids & all_uids

    negative_uids = results['negative']
    for uid in child_symptomatic_alive:
        assert uid in negative_uids, f"Child symptomatic agent {uid} should test negative with sensitivity=0.0"


def test_dx_probabilities_sum_to_one_validation():
    """Dx raises an error if probabilities don't sum to 1.0 per group."""
    df = pd.DataFrame([
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=0.5),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=0.3),  # sums to 0.8, not 1.0
    ])
    with pytest.raises(ValueError, match="not 1.0"):
        tbsim.Dx(df=df, hierarchy=['positive', 'negative'])


def test_dx_default_hierarchy():
    """Dx infers hierarchy from df.result.unique() if not provided."""
    df = pd.DataFrame([
        dict(state=TBS.SYMPTOMATIC, result='positive', probability=0.9),
        dict(state=TBS.SYMPTOMATIC, result='negative', probability=0.1),
    ])
    dx = tbsim.Dx(df=df)
    assert list(dx.hierarchy) == ['positive', 'negative']


def test_all_dx_products_in_sim():
    """All Dx product classes can be instantiated and administered in a sim."""
    hsb = tbsim.HealthSeekingBehavior()
    dx_intv_0 = tbsim.DxDelivery(name='dx_xpert',    product=tbsim.Xpert())
    dx_intv_1 = tbsim.DxDelivery(name='dx_oralswab', product=tbsim.OralSwab())
    dx_intv_2 = tbsim.DxDelivery(name='dx_fujilam',  product=tbsim.FujiLAM())
    dx_intv_3 = tbsim.DxDelivery(name='dx_cad',      product=tbsim.CAD())

    sim = tbsim.Sim(
        n_agents=500,
        interventions=[hsb, dx_intv_0, dx_intv_1, dx_intv_2, dx_intv_3],
        sim_pars=dict(start=ss.date('2000-01-01'), stop=ss.date('2001-01-01')),
        tb_pars=dict(init_prev=0.30),
    )
    sim.init()
    for _ in range(20):
        sim.run_one_step()

    all_uids = sim.people.alive.uids
    for intv in [sim.interventions.dx_xpert, sim.interventions.dx_oralswab,
                 sim.interventions.dx_fujilam, sim.interventions.dx_cad]:
        results = intv.product.administer(sim, all_uids)
        assert 'positive' in results
        assert 'negative' in results
        assert len(results['positive']) + len(results['negative']) == len(all_uids)
