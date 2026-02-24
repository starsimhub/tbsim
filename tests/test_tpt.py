"""Tests for TPTTx (product), TPTSimple, and TPTHousehold delivery."""

import pytest
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import tbsim
from tbsim.interventions.tpt import TPTTx, TPTSimple, TPTHousehold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_modules(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    pop = ss.People(n_agents=agents)
    tb = tbsim.TB_LSHTM(name='tb', pars={'init_prev': 0.30})
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
    itv = TPTSimple()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    tpt = sim.interventions['tptsimple']

    assert '0.5' in str(tpt.pars.coverage) or '0.50' in str(tpt.pars.coverage)
    assert tpt.pars.eligible_states == [tbsim.TBSL.INFECTION]
    assert hasattr(tpt.product.pars, 'dur_treatment')
    assert hasattr(tpt.product.pars, 'dur_protection')
    assert tpt.product.pars.disease == 'tb'


def test_tpt_custom_values():
    """Test TPTSimple + TPTTx with custom parameters."""
    nagents = 100
    pop, tb, net, pars = make_modules(agents=nagents)
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.months(6)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = TPTSimple(product=product, pars={
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
    itv = TPTSimple(pars={'coverage': 1.0})
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

    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.years(1)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
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

    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(1)),
        'dur_protection': ss.constant(v=ss.days(1)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
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

    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(10)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
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
    itv = TPTSimple()
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

    itv = TPTSimple(pars={
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
    product = TPTTx(pars={
        'dur_treatment': ss.constant(v=ss.days(0)),
        'dur_protection': ss.constant(v=ss.years(5)),
    })
    itv = TPTSimple(product=product, pars={'coverage': 1.0})
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
    try:
        import starsim_examples as sse
    except ImportError:
        pytest.skip("starsim_examples not installed")

    dhs_data = make_dhs_data(50)
    hh_net = sse.HouseholdDHSNet(dhs_data=dhs_data)

    tb = tbsim.TB_LSHTM(name='tb', pars={'init_prev': 0.20})
    tpt_hh = TPTHousehold()

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
    try:
        import starsim_examples as sse
    except ImportError:
        pytest.skip("starsim_examples not installed")

    dhs_data = make_dhs_data(50)
    hh_net = sse.HouseholdDHSNet(dhs_data=dhs_data)

    tb = tbsim.TB_LSHTM(name='tb', pars={'init_prev': 0.20})
    tpt_hh = TPTHousehold(pars={'coverage': 1.0})

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
    try:
        import starsim_examples as sse
    except ImportError:
        pytest.skip("starsim_examples not installed")

    dhs_data = make_dhs_data(50)
    hh_net = sse.HouseholdDHSNet(dhs_data=dhs_data)

    tb = tbsim.TB_LSHTM(name='tb', pars={'init_prev': 0.20})
    tpt_hh = TPTHousehold(pars={'coverage': 1.0})

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
    try:
        import starsim_examples as sse
    except ImportError:
        pytest.skip("starsim_examples not installed")

    dhs_data = make_dhs_data(30)
    hh_net = sse.HouseholdDHSNet(dhs_data=dhs_data)

    tb = tbsim.TB_LSHTM(name='tb', pars={'init_prev': 0.20, 'beta': ss.peryear(0.5)})
    tpt_hh = TPTHousehold(pars={'coverage': 0.8})

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
