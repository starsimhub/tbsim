"""Tests for TPTTx (product), TPTSimple, and TPTHousehold delivery."""

import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_modules(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = tbsim.TB(name='tb', pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars


age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1],
})


def make_dhs_data(n_households=100):
    """Create synthetic DHS data for household network tests."""
    hh_ids = np.arange(n_households)
    age_strings = []
    for _ in range(n_households):
        hh_size = np.random.randint(2, 6)
        ages = np.random.randint(1, 70, hh_size)
        age_strings.append(sc.strjoin(ages))
    return sc.dataframe(hh_id=hh_ids, ages=age_strings)


# ---------------------------------------------------------------------------
# TPTSimple tests
# ---------------------------------------------------------------------------

def test_tpt_default_values():
    """Test TPTSimple + TPTTx with default parameters."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    itv = tbsim.TPTSimple()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    assert '0.5' in str(tpt.pars.coverage) or '0.50' in str(tpt.pars.coverage)
    assert tpt.pars.eligible_states == [tbsim.TBS.INFECTION]
    assert hasattr(tpt.product.pars, 'dur_treatment')
    assert hasattr(tpt.product.pars, 'dur_protection')
    assert tpt.product.pars.disease == 'tb'


def test_tpt_custom_values():
    """Test TPTSimple + TPTTx with custom parameters."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.months(6)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = tbsim.TPTSimple(product=product, pars={
        'coverage': 0.8,
        'age_range': [15, 65],
        'start': ss.date('2005-01-01'),
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    assert '0.8' in str(tpt.pars.coverage)
    assert tpt.pars.age_range == [15, 65]
    assert tpt.pars.start == ss.date('2005-01-01')


def test_tpt_targets_latent_only():
    """Test that TPT only targets agents in INFECTION (latent) state."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = tbsim.TPTSimple(pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()

    initiated_uids = tpt.initiated.uids
    if len(initiated_uids) > 0:
        assert np.all(~np.isnan(tpt.product.ti_protection_starts[initiated_uids]))


def test_tpt_treatment_then_protection():
    """Test the two-phase model: treatment first, then protection."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.years(1)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()
    initiated = tpt.initiated.uids
    if len(initiated) > 0:
        assert not np.all(tpt.product.tpt_protected[initiated]), \
            "Agents should not be protected immediately (treatment phase)"
        starts = tpt.product.ti_protection_starts[initiated]
        assert np.all(~np.isnan(starts)), "Protection start times should be set"
        assert np.all(starts > tpt.ti), "Protection should start after treatment completes"


def test_tpt_protection_expiry():
    """Test that protection expires after dur_protection."""
    nagents = 50
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(1)),
        'dur_protection': ss.constant(v=ss.days(1)),
    })
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    for _ in range(5):
        tpt.step()

    assert np.count_nonzero(tpt.product.tpt_protected) == 0, \
        "All protection should have expired"


def test_tpt_modifies_rr():
    """Test that TPT applies rr_* modifiers for protected agents."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(10)),
    })
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tb_disease = sim.diseases.tb
    initial_activation = np.array(tb_disease.rr_activation).copy()

    tpt = sim.interventions['tptsimple']
    tpt.step()
    tpt.step()

    current_activation = np.array(tb_disease.rr_activation)

    protected = tpt.product.tpt_protected.uids
    if len(protected) > 0:
        assert np.any(current_activation[protected] < initial_activation[protected]), \
            "TPT should reduce activation risk for protected agents"


def test_tpt_result_metrics():
    """Test that result metrics are initialized and updated correctly."""
    nagents = 50
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = tbsim.TPTSimple()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()
    tpt.update_results()

    assert 'n_newly_initiated' in tpt.results
    assert isinstance(tpt.results['n_newly_initiated'][tpt.ti], (int, np.integer))
    assert 'n_protected' in tpt.results
    assert isinstance(tpt.results['n_protected'][tpt.ti], (int, np.integer))


def test_tpt_with_age_range():
    """Test TPT delivery with age range filter."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    itv = tbsim.TPTSimple(pars={
        'coverage': 1.0,
        'age_range': [15, 50],
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    tpt.step()

    initiated_uids = tpt.initiated.uids
    if len(initiated_uids) > 0:
        ages = np.asarray(sim.people.age[initiated_uids])
        assert np.all(ages >= 15), "All initiated should be >= 15"
        assert np.all(ages <= 50), "All initiated should be <= 50"


def test_tpt_product_skips_on_treatment():
    """Test that TPTTx.administer() skips agents already on TB treatment."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tb_disease = sim.diseases.tb
    # Force some agents onto treatment
    some_uids = ss.uids([0, 1, 2])
    tb_disease.on_treatment[some_uids] = True

    tpt = sim.interventions['tptsimple']
    tpt.step()

    # Those agents should NOT have protection_starts set
    starts = tpt.product.ti_protection_starts[some_uids]
    assert np.all(np.isnan(starts)), "Agents on TB treatment should not receive TPT"


# ---------------------------------------------------------------------------
# TPTHousehold tests
# ---------------------------------------------------------------------------

def test_tpt_household_init():
    """Test TPTHousehold initializes with correct defaults."""
    dhs_data = make_dhs_data(50)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20})
    tpt_hh = tbsim.TPTHousehold()

    sim = ss.Sim(
        diseases=tb, networks=hh_net,
        interventions=tpt_hh,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')),
    )
    sim.init()

    itv = sim.interventions['tpthousehold']
    assert hasattr(itv, 'prev_on_treatment')
    assert itv.product.pars.disease == 'tb'


def test_tpt_household_traces_on_treatment_start():
    """Test that household contacts are offered TPT when an index starts treatment."""
    dhs_data = make_dhs_data(50)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20})
    tpt_hh = tbsim.TPTHousehold(pars={'coverage': 1.0})

    sim = ss.Sim(
        diseases=tb, networks=hh_net,
        interventions=tpt_hh,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')),
    )
    sim.init()

    itv = sim.interventions['tpthousehold']
    tb_disease = sim.diseases.tb

    # No one on treatment yet → no contacts should be found
    contacts = itv.check_eligibility()
    assert len(contacts) == 0, "No contacts should be found before any treatment starts"

    # Force an agent onto treatment
    index_uid = ss.uids([0])
    tb_disease.on_treatment[index_uid] = True

    # Now check_eligibility should detect the new treatment start
    contacts = itv.check_eligibility()
    # Should find household contacts of agent 0
    if len(contacts) > 0:
        # Contacts should NOT include the index case
        assert 0 not in contacts, "Index case should not be in contacts"


def test_tpt_household_no_retrigger_same_index():
    """Test that the same index case doesn't retrigger tracing on subsequent steps."""
    dhs_data = make_dhs_data(50)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20})
    tpt_hh = tbsim.TPTHousehold(pars={'coverage': 1.0})

    sim = ss.Sim(
        diseases=tb, networks=hh_net,
        interventions=tpt_hh,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')),
    )
    sim.init()

    itv = sim.interventions['tpthousehold']
    tb_disease = sim.diseases.tb

    # Force agent onto treatment
    tb_disease.on_treatment[ss.uids([0])] = True

    # First call: should find contacts
    contacts1 = itv.check_eligibility()

    # Second call (same step, on_treatment unchanged): snapshot already updated
    contacts2 = itv.check_eligibility()
    assert len(contacts2) == 0, "Same index should not retrigger tracing"


def test_tpt_household_full_sim_run():
    """Test that TPTHousehold runs without error in a full simulation."""
    dhs_data = make_dhs_data(30)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20, 'beta': ss.peryear(0.5)})
    tpt_hh = tbsim.TPTHousehold(pars={'coverage': 0.8})

    sim = ss.Sim(
        diseases=tb,
        networks=[hh_net, ss.RandomNet(dict(n_contacts=ss.poisson(lam=3), dur=0))],
        interventions=tpt_hh,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2005-12-31')),
    )
    sim.run()

    # Should complete without error; check results exist
    itv = sim.interventions['tpthousehold']
    assert 'n_newly_initiated' in itv.results
    assert 'n_protected' in itv.results


# ---------------------------------------------------------------------------
# HouseholdContactTracing tests
# ---------------------------------------------------------------------------

def test_hh_contact_tracing_identifies_contacts():
    """HouseholdContactTracing sets contact_identified on household members."""
    dhs_data = make_dhs_data(50)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20})
    hh_tracing = tbsim.HouseholdContactTracing(coverage=1.0)

    sim = ss.Sim(
        diseases=tb, networks=hh_net,
        interventions=hh_tracing,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')),
    )
    sim.init()

    ct = sim.interventions['householdcontacttracing']
    tb_disease = sim.diseases.tb

    # No contacts before treatment starts
    ct.step()
    assert ct.contact_identified.count() == 0

    # Force an agent onto treatment
    tb_disease.on_treatment[ss.uids([0])] = True
    ct.step()

    # Should have identified contacts
    n_identified = ct.contact_identified.count()
    # Agent 0 should NOT be in contacts
    if n_identified > 0:
        assert not ct.contact_identified[0], "Index case should not be identified as contact"


def test_hh_contact_tracing_no_retrigger():
    """Same index case does not retrigger contact identification."""
    dhs_data = make_dhs_data(50)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)

    tb = tbsim.TB(name='tb', pars={'init_prev': 0.20})
    hh_tracing = tbsim.HouseholdContactTracing(coverage=1.0)

    sim = ss.Sim(
        diseases=tb, networks=hh_net,
        interventions=hh_tracing,
        pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')),
    )
    sim.init()

    ct = sim.interventions['householdcontacttracing']
    tb_disease = sim.diseases.tb

    tb_disease.on_treatment[ss.uids([0])] = True
    ct.step()
    n1 = ct._n_contacts

    # Second step: same agent still on treatment, no new starts
    ct.step()
    assert ct._n_contacts == 0, "Same index should not retrigger"


# ---------------------------------------------------------------------------
# Full TPT cascade tests
# ---------------------------------------------------------------------------

def _make_cascade_sim(n_agents=500, n_households=100, rand_seed=42):
    """Build a full TPT cascade sim for testing."""
    dhs_data = make_dhs_data(n_households)
    hh_net = ss.HouseholdNet(dhs_data=dhs_data, dynamic=False)
    community_net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=3), dur=0))

    hsb = tbsim.HealthSeekingBehavior()
    screen = tbsim.DxDelivery(
        name='screen', product=tbsim.CAD(), coverage=0.9,
        result_state='screen_positive',
    )
    confirm = tbsim.DxDelivery(
        name='confirm', product=tbsim.Xpert(), coverage=0.8,
        eligibility=lambda sim: (
            sim.people.screen.screen_positive
            & ~sim.people.confirm.tested
            & sim.people.alive
        ).uids,
        result_state='diagnosed',
    )
    treat = tbsim.TxDelivery(product=tbsim.DOTS())

    hh_tracing = tbsim.HouseholdContactTracing(coverage=1.0)
    contact_screen = tbsim.DxDelivery(
        name='contact_screen', product=tbsim.Xpert(), coverage=1.0,
        eligibility=lambda sim: (
            sim.people.householdcontacttracing.contact_identified
            & ~sim.people.contact_screen.tested
            & sim.people.alive
        ).uids,
        result_state='diagnosed',
    )
    tpt = tbsim.TPTDelivery(
        product=tbsim.TPTTx(),
        contact_tracing='householdcontacttracing',
        contact_screen='contact_screen',
    )

    sim = tbsim.Sim(
        n_agents=n_agents, start='2000', stop='2010', rand_seed=rand_seed,
        init_prev=ss.bernoulli(p=0.30), beta=ss.peryear(0.05),
        networks=[hh_net, community_net],
        interventions=[hsb, screen, confirm, treat, hh_tracing, contact_screen, tpt],
    )
    return sim


def test_tpt_cascade_runs():
    """Full TPT cascade completes without error."""
    sim = _make_cascade_sim()
    sim.run()

    r = sim.results
    assert r.householdcontacttracing.n_contacts_identified.values.sum() >= 0
    assert r.contact_screen.n_tested.values.sum() >= 0
    assert r.tptdelivery.n_tpt_initiated.values.sum() >= 0


def test_tpt_cascade_contacts_screened():
    """Identified contacts are screened by contact_screen DxDelivery."""
    sim = _make_cascade_sim(n_agents=2000)
    sim.run()

    r = sim.results
    n_identified = r.householdcontacttracing.n_contacts_identified.values.sum()
    n_screened = r.contact_screen.n_tested.values.sum()

    # If contacts were identified, some should have been screened
    if n_identified > 0:
        assert n_screened > 0, \
            f"Identified {n_identified} contacts but none were screened"


def test_tpt_cascade_triage():
    """Contacts are triaged: active TB → treatment, non-active → TPT."""
    sim = _make_cascade_sim(n_agents=2000)
    sim.run()

    r = sim.results
    n_contact_positive = r.contact_screen.n_positive.values.sum()
    n_tpt = r.tptdelivery.n_tpt_initiated.values.sum()
    n_treated = r.txdelivery.n_treated.values.sum()

    # At least some contacts should get TPT or treatment
    n_contact_screened = r.contact_screen.n_tested.values.sum()
    if n_contact_screened > 0:
        assert (n_tpt + n_contact_positive) > 0, \
            f"Screened {n_contact_screened} contacts but none got TPT or treatment"


# ---------------------------------------------------------------------------
# Sterilization / suppression mechanism tests
# ---------------------------------------------------------------------------

def test_tpt_sterilization_clears_infection():
    """Agents with sterilization mechanism move from INFECTION → CLEARED after treatment."""
    nagents = 200
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(
        p_sterilization=1.0, p_suppression=0.0,
        pars={'dur_treatment': ss.constant(v=ss.days(0))},  # Instant treatment
    )
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tpt = sim.interventions['tptsimple']
    tb_disease = sim.diseases.tb

    # Run steps to initiate TPT and resolve treatment completion
    for _ in range(5):
        tpt.step()

    # Agents who were in INFECTION and got sterilized should now be CLEARED
    resolved = tpt.product.tpt_resolved.uids
    assert len(resolved) > 0, "Some agents should have been resolved"

    # Check that sterilized agents with mechanism=STERILIZE are CLEARED
    sterilize_mechs = tpt.product.tpt_mechanism[resolved] == tbsim.TPTTx.MECH_STERILIZE
    sterilized = resolved[sterilize_mechs]
    assert len(sterilized) > 0, "Some agents should have sterilization mechanism"
    states = np.asarray(tb_disease.state[sterilized])
    n_cleared = np.count_nonzero(states == tbsim.TBS.CLEARED)
    assert n_cleared > 0, "Some sterilized agents should be in CLEARED state"

    # None should be protected (suppression)
    assert tpt.product.tpt_protected.count() == 0, \
        "No agents should be in suppression protection with p_suppression=0"


def test_tpt_suppression_applies_modifiers():
    """Agents with suppression mechanism get rr_* modifiers (existing behavior)."""
    nagents = 200
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(
        p_sterilization=0.0, p_suppression=1.0,
        pars={
            'dur_treatment': ss.constant(v=ss.days(0)),
            'dur_protection': ss.constant(v=ss.years(10)),
        },
    )
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tb_disease = sim.diseases.tb
    initial_activation = np.array(tb_disease.rr_activation).copy()

    tpt = sim.interventions['tptsimple']
    tpt.step()
    tpt.step()

    current_activation = np.array(tb_disease.rr_activation)
    protected = tpt.product.tpt_protected.uids
    if len(protected) > 0:
        assert np.any(current_activation[protected] < initial_activation[protected]), \
            "Suppression should reduce activation risk for protected agents"
    # No sterilized agents
    assert tpt.product.tpt_resolved.count() == 0


def test_tpt_neither_gets_no_benefit():
    """Agents with mechanism=NONE stay in INFECTION with no modifiers."""
    nagents = 200
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(
        p_sterilization=0.0, p_suppression=0.0,
        pars={'dur_treatment': ss.constant(v=ss.days(7))},
    )
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tpt = sim.interventions['tptsimple']
    for _ in range(5):
        tpt.step()

    # No one should be protected or sterilized
    assert tpt.product.tpt_protected.count() == 0
    assert tpt.product.tpt_resolved.count() == 0


def test_tpt_mechanisms_mutually_exclusive():
    """No agent has both sterilization and suppression."""
    nagents = 200
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(
        p_sterilization=0.5, p_suppression=0.5,
        pars={'dur_treatment': ss.constant(v=ss.days(7))},
    )
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tpt = sim.interventions['tptsimple']
    for _ in range(5):
        tpt.step()

    # Check that mechanisms are 0, 1, or 2 only
    mechs = np.asarray(tpt.product.tpt_mechanism)
    initiated = mechs[mechs != 0]
    assert np.all(np.isin(initiated, [1, 2])), "Mechanisms should be 0, 1, or 2"

    # No agent should be both sterilized and protected
    both = tpt.product.tpt_resolved.uids.intersect(tpt.product.tpt_protected.uids)
    assert len(both) == 0, "No agent should have both sterilization and suppression"


def test_tpt_backward_compatible_defaults():
    """Default params (p_sterilization=0, p_suppression=1) produce suppression-only behavior."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)

    product = tbsim.TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(10)),
    })
    itv = tbsim.TPTSimple(product=product, pars={'coverage': 1.0})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()

    tpt = sim.interventions['tptsimple']
    tpt.step()
    tpt.step()

    # All initiated agents should have suppression mechanism
    initiated = tpt.initiated.uids
    if len(initiated) > 0:
        mechs = np.asarray(tpt.product.tpt_mechanism[initiated])
        assert np.all(mechs == tbsim.TPTTx.MECH_SUPPRESS), \
            "Default should assign all agents to suppression"
    # No sterilized agents
    assert tpt.product.tpt_resolved.count() == 0


def test_tpt_invalid_probabilities():
    """Raise error if p_sterilization + p_suppression > 1."""
    with pytest.raises(ValueError, match="must be <= 1.0"):
        tbsim.TPTTx(p_sterilization=0.6, p_suppression=0.6)


def test_tpt_mechanism_via_pars_dict():
    """Mechanism probabilities can be passed via pars dict."""
    product = tbsim.TPTTx(pars={'p_sterilization': 0.7, 'p_suppression': 0.3})
    assert np.isclose(product.pars.p_sterilization, 0.7)
    assert np.isclose(product.pars.p_suppression, 0.3)

    # Also works via constructor args (existing path)
    product2 = tbsim.TPTTx(p_sterilization=0.7, p_suppression=0.3)
    assert np.isclose(product2.pars.p_sterilization, 0.7)
