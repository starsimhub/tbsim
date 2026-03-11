"""Tests for TB interventions using TB_LSHTM / TB_LSHTM_Acute disease models."""

import pytest
import tbsim
import starsim as ss
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sim(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), dt=ss.days(7)):
    """Standard sim components using TB_LSHTM."""
    pop = ss.People(n_agents=agents)
    tb = tbsim.TB_LSHTM(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars


def make_sim_acute(agents=50, start=ss.date('2000-01-01'), stop=ss.date('2005-12-31'), dt=ss.days(7)):
    """Standard sim components using TB_LSHTM_Acute."""
    pop = ss.People(n_agents=agents)
    tb = tbsim.TB_LSHTM_Acute(pars={'init_prev': 0.30})
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars


# ---------------------------------------------------------------------------
# BetaByYear tests
# ---------------------------------------------------------------------------

def test_beta_intervention_changes_beta():
    """BetaByYear modifies beta at the specified year with TB_LSHTM."""
    initial_beta = 0.01
    x_beta = 0.5
    intervention_year = 2005
    stop_year = 2010

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start=f'{intervention_year-1}-01-01', stop=f'{stop_year}-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        interventions=[beta_intv],
        pars=sim_pars,
    )
    sim.init()

    pars = tbsim.get_tb(sim).pars
    assert np.isclose(pars.beta.value, initial_beta)

    while sim.t.now('year') < intervention_year:
        sim.run_one_step()
    assert np.isclose(pars.beta.value, initial_beta)

    sim.run_one_step()
    expected_beta = initial_beta * x_beta
    assert np.isclose(pars.beta.value, expected_beta)

    sim.run_one_step()
    assert np.isclose(pars.beta.value, expected_beta)


def test_beta_intervention_with_acute():
    """BetaByYear works with TB_LSHTM_Acute."""
    initial_beta = 0.02
    x_beta = 0.6
    intervention_year = 2005

    tb_pars = dict(beta=initial_beta, init_prev=0.25)
    sim_pars = dict(start='2004-01-01', stop='2007-01-01', dt=ss.days(7), rand_seed=42)

    pop = ss.People(n_agents=100)
    tb = tbsim.TB_LSHTM_Acute(pars=tb_pars)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': [intervention_year], 'x_beta': x_beta})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=[beta_intv], pars=sim_pars)
    sim.init()

    while sim.t.now('year') < intervention_year:
        sim.run_one_step()

    sim.run_one_step()
    expected_beta = initial_beta * x_beta
    beta_val = tbsim.get_tb(sim).pars.beta
    actual = beta_val.value if hasattr(beta_val, 'value') else float(beta_val)
    assert np.isclose(actual, expected_beta)


def test_beta_multiple_years():
    """BetaByYear applies different x_beta values at multiple years."""
    initial_beta = 0.1
    years = [2002, 2005]
    x_betas = [0.5, 0.8]

    pop = ss.People(n_agents=50)
    tb = tbsim.TB_LSHTM(name='tb', pars=dict(beta=initial_beta, init_prev=0.25))
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})

    beta_intv = tbsim.BetaByYear(pars={'years': years, 'x_beta': x_betas})
    sim = ss.Sim(people=pop, networks=net, diseases=tb, interventions=[beta_intv],
                 pars=dict(start='2001-01-01', stop='2007-01-01', dt=ss.days(7), rand_seed=42))
    sim.init()

    while sim.t.now('year') < 2002:
        sim.run_one_step()
    sim.run_one_step()
    assert np.isclose(tbsim.get_tb(sim).pars.beta.value, initial_beta * 0.5)

    while sim.t.now('year') < 2005:
        sim.run_one_step()
    sim.run_one_step()
    assert np.isclose(tbsim.get_tb(sim).pars.beta.value, initial_beta * 0.5 * 0.8)


# ---------------------------------------------------------------------------
# HealthSeekingBehavior tests (already tested separately; quick smoke test)
# ---------------------------------------------------------------------------

def test_health_seeking_runs():
    """HealthSeekingBehavior completes a full run with TB_LSHTM."""
    pop, tb, net, pars = make_sim(agents=100)
    hsb = tbsim.HealthSeekingBehavior()
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=hsb, pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# TBDiagnostic tests
# ---------------------------------------------------------------------------

def test_tb_diagnostic_runs():
    """TBDiagnostic completes a full run with TB_LSHTM + HealthSeekingBehavior."""
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.TBDiagnostic(pars={'sensitivity': 0.9, 'specificity': 0.95})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()


def test_tb_diagnostic_with_acute():
    """TBDiagnostic completes a full run with TB_LSHTM_Acute."""
    pop, tb, net, pars = make_sim_acute(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.TBDiagnostic(pars={'sensitivity': 0.85, 'specificity': 0.95})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx], pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# EnhancedTBDiagnostic tests
# ---------------------------------------------------------------------------

def test_enhanced_tb_diagnostic_runs():
    """EnhancedTBDiagnostic completes a full run with TB_LSHTM."""
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    edx = tbsim.EnhancedTBDiagnostic(pars={'use_oral_swab': True})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, edx], pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# TBTreatment tests
# ---------------------------------------------------------------------------

def test_tb_treatment_runs():
    """TBTreatment completes a full run with TB_LSHTM (requires HSB + Dx upstream)."""
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.TBDiagnostic(pars={'sensitivity': 0.9, 'specificity': 0.95})
    tx = tbsim.TBTreatment(pars={'treatment_success_prob': 0.85})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()


def test_tb_treatment_with_acute():
    """TBTreatment completes a full run with TB_LSHTM_Acute."""
    pop, tb, net, pars = make_sim_acute(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.TBDiagnostic(pars={'sensitivity': 0.9, 'specificity': 0.95})
    tx = tbsim.TBTreatment(pars={'treatment_success_prob': 0.85})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, tx], pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# EnhancedTBTreatment tests
# ---------------------------------------------------------------------------

def test_enhanced_tb_treatment_runs():
    """EnhancedTBTreatment with DOTS drug type runs with TB_LSHTM."""
    from tbsim.interventions.tb_drug_types import TBDrugType
    pop, tb, net, pars = make_sim(agents=200)
    hsb = tbsim.HealthSeekingBehavior()
    dx = tbsim.TBDiagnostic(pars={'sensitivity': 0.9, 'specificity': 0.95})
    etx = tbsim.EnhancedTBTreatment(pars={'drug_type': TBDrugType.DOTS})
    sim = ss.Sim(people=pop, diseases=tb, networks=net, interventions=[hsb, dx, etx], pars=pars)
    sim.run()


# ---------------------------------------------------------------------------
# Immigration tests
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason='Immigration class is known non-functional')
def test_immigration_runs():
    """Immigration intervention runs with TB_LSHTM."""
    from tbsim.interventions.immigration import Immigration
    pop, tb, net, pars = make_sim(agents=100)
    immig = Immigration()
    sim = ss.Sim(people=pop, diseases=tb, networks=net, demographics=immig, pars=pars)
    sim.run()


def test_simple_immigration_runs():
    """SimpleImmigration intervention runs with TB_LSHTM."""
    from tbsim.interventions.immigration import SimpleImmigration
    pop, tb, net, pars = make_sim(agents=100)
    immig = SimpleImmigration(immigration_rate=10)
    sim = ss.Sim(people=pop, diseases=tb, networks=net, demographics=immig, pars=pars)
    sim.run()


if __name__ == '__main__':
    pytest.main(["-x", "-v", __file__])
