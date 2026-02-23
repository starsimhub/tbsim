"""Tests for tbsim/interventions/tpt2.py."""

import numpy as np
import pytest
import sciris as sc
import starsim as ss
import starsim_examples
import tbsim
from tbsim.interventions.tpt2 import (
    RegimenCategory,
    TPTProduct,
    TPTRegimen,
    TPTRoutine,
    TPTHousehold,
)
from tbsim.tb_lshtm import TBSL


SIM_PARS = dict(dt=ss.days(7), start=ss.date('2000-01-01'),
                stop=ss.date('2010-12-31'), rand_seed=0, verbose=0)


def make_sim(*interventions, init_prev=0.40, n=200):
    return ss.Sim(
        people=ss.People(n_agents=n),
        diseases=tbsim.TB_LSHTM(name='tb', pars={'init_prev': init_prev}),
        networks=ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0)),
        interventions=list(interventions),
        pars=SIM_PARS,
    )


def quick_regimen(p_complete=1.0, modifier=0.5, dur_protection=ss.years(10)):
    return TPTRegimen(
        name='QUICK',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.days(0)),
        dur_protection=ss.constant(v=dur_protection),
        p_complete=ss.bernoulli(p=p_complete),
        activation_modifier=ss.constant(v=modifier),
    )


def inited_sim(regimen=None, coverage=1.0, eligible_states=(TBSL.INFECTION,)):
    prod = TPTProduct(regimen=regimen or quick_regimen())
    itv  = TPTRoutine(product=prod, pars={
        'coverage': ss.bernoulli(p=coverage),
        'eligible_states': list(eligible_states) if eligible_states else None,
    })
    sim = make_sim(itv)
    sim.init()
    return sim


def make_hh_sim(*interventions, init_prev=0.2, n_hh=60):
    age_strings = [
        sc.strjoin(np.random.randint(5, 70, np.random.randint(2, 6)).tolist())
        for _ in range(n_hh)
    ]
    dhs_data = sc.dataframe(hh_id=np.arange(n_hh), ages=age_strings)
    return ss.Sim(
        diseases=tbsim.TB_LSHTM(name='tb', pars={'init_prev': init_prev}),
        networks=starsim_examples.HouseholdDHSNet(dhs_data=dhs_data),
        interventions=list(interventions),
        pars=SIM_PARS,
    )


@pytest.mark.parametrize('state', [
    TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC, TBSL.TREATMENT,
])
def test_active_tb_excluded(state):
    sim  = inited_sim()
    tb   = sim.diseases.tb
    prod = sim.interventions['tptroutine'].product
    uids = ss.uids([0, 1, 2])
    tb.state[uids] = int(state)
    assert len(prod.administer(sim.people, uids)) == 0


def test_already_protected_skipped():
    sim  = inited_sim()
    prod = sim.interventions['tptroutine'].product
    uid  = ss.uids([0])
    sim.diseases.tb.state[uid] = int(TBSL.INFECTION)
    prod.tpt_protected[uid] = True
    assert len(prod.administer(sim.people, uid)) == 0


def test_p_complete_zero_gives_no_protection():
    sim = inited_sim(regimen=quick_regimen(p_complete=0.0), eligible_states=None)
    sim.run()
    assert sim.interventions['tptroutine'].product.n_protected == 0


def test_no_immediate_protection_during_treatment():
    reg = TPTRegimen(
        name='SLOW', category=RegimenCategory.ISONIAZID_LONG,
        dur_treatment=ss.constant(v=ss.years(1)),
        dur_protection=ss.constant(v=ss.years(5)),
        p_complete=ss.bernoulli(p=1.0),
        activation_modifier=ss.constant(v=0.68),
    )
    sim = inited_sim(regimen=reg)
    itv = sim.interventions['tptroutine']
    itv.step()

    initiated = ss.uids(~np.isnan(np.asarray(itv.product.ti_protection_starts)))
    if len(initiated):
        assert np.all(itv.product.ti_protection_starts[initiated] > itv.product.ti)
        assert np.all(~itv.product.tpt_protected[initiated])


def test_expiry_clears_offered_flag():
    reg = quick_regimen(p_complete=1.0, dur_protection=ss.days(1))
    itv = TPTRoutine(product=reg, pars={'coverage': ss.bernoulli(p=1.0)})
    sim = make_sim(itv, init_prev=0.5)
    sim.run()

    delivery  = sim.interventions['tptroutine']
    initiated = delivery.tpt_initiated.uids
    if len(initiated):
        expired = initiated[~delivery.product.tpt_protected[initiated]]
        if len(expired):
            assert np.all(~delivery.tpt_offered[expired])


def test_only_rr_activation_modified():
    sim = inited_sim(regimen=quick_regimen(modifier=0.5), eligible_states=None)
    tb  = sim.diseases.tb
    rr_clearance_before = np.array(tb.rr_clearance).copy()
    rr_death_before     = np.array(tb.rr_death).copy()

    itv = sim.interventions['tptroutine']
    itv.step()
    itv.step()

    assert np.array_equal(np.array(tb.rr_clearance), rr_clearance_before)
    assert np.array_equal(np.array(tb.rr_death), rr_death_before)

    protected = itv.product.tpt_protected.uids
    if len(protected):
        assert np.all(np.array(tb.rr_activation)[protected] < 1.0)


def test_routine_offered_gate():
    sim = inited_sim(regimen=quick_regimen(), coverage=1.0)
    itv = sim.interventions['tptroutine']
    itv.step()
    n1 = int(np.sum(itv.tpt_initiated))
    itv.step()
    n2 = int(np.sum(itv.tpt_initiated))
    assert n2 == n1


def test_hh_triggers_on_treatment_excludes_index():
    sim = make_hh_sim(TPTHousehold(pars={'hh_coverage': ss.bernoulli(p=1.0)}))
    sim.init()
    itv = sim.interventions['tpthousehold']
    tb  = sim.diseases.tb

    assert len(itv._find_candidates()) == 0

    tb.on_treatment[ss.uids([0])] = True
    candidates = itv._find_candidates()
    assert 0 not in candidates

    assert len(itv._find_candidates()) == 0


def test_hh_sentinel_triggers_rebuild():
    sim = make_hh_sim(TPTHousehold())
    sim.init()
    itv = sim.interventions['tpthousehold']

    itv._maybe_rebuild_hh_index()
    stable_id = id(itv._hh_index)

    itv._maybe_rebuild_hh_index()
    assert id(itv._hh_index) == stable_id

    itv._hh_n_assigned -= 1
    itv._maybe_rebuild_hh_index()
    assert id(itv._hh_index) != stable_id


def test_two_routines_coexist():
    itv_a = TPTRoutine(name='tpt_a', product=TPTProduct(name='prod_a', regimen='3HP'),
                       pars={'coverage': ss.bernoulli(p=0.5)})
    itv_b = TPTRoutine(name='tpt_b', product=TPTProduct(name='prod_b', regimen='6H'),
                       pars={'coverage': ss.bernoulli(p=0.3)})
    sim = make_sim(itv_a, itv_b, init_prev=0.4)
    sim.run()
    assert 'n_initiated' in sim.interventions['tpt_a'].results
    assert 'n_initiated' in sim.interventions['tpt_b'].results
