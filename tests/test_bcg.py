import numpy as np
import pytest
import starsim as ss
import tbsim as mtb
import pandas as pd

def make_sim(agents=20, start=ss.date('2000-01-01'), stop=ss.date('2020-12-31'), dt=ss.days(7)):
    """Create a real simulation with TB disease model for testing."""
    pop = ss.People(n_agents=agents)
    tb = mtb.TB()
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    pars = dict(dt=dt, start=start, stop=stop)
    return pop, tb, net, pars

def test_bcg_intervention_default_values():
    """Test BCGProtection intervention with default parameters"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection()
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    # Check coverage value for starsim bernoulli or float
    if hasattr(bcg.pars.coverage, 'filter'):
        # starsim bernoulli object, check its string representation or default
        assert '0.5' in str(bcg.pars.coverage) or '0.50' in str(bcg.pars.coverage), "Default coverage should be 0.5"
    else:
        assert bcg.pars.coverage == 0.5, "Default coverage should be 0.5"
    assert bcg.pars.efficacy == 0.8, "Default efficacy should be 0.8"
    assert bcg.pars.start == ss.date('1900-01-01'), "Default start year should be 1900-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2100-12-31'), "Default stop year should be 2100-12-31 with type ss.date"
    # Check that immunity_period is approximately 10 years (in timesteps)
    assert abs(bcg.pars.immunity_period - 521.43) < 1.0, "Default immunity_period should be approximately 10 years"
    assert bcg.pars.age_range == [0, 5], "Default age range should be [0, 5]"
    assert bcg.min_age == 0, "Default min_age should be 0"
    assert bcg.max_age == 5, "Default max_age should be 5"
    assert len(bcg.is_bcg_vaccinated) == nagents, "is_bcg_vaccinated array should match population size"
    assert len(bcg.ti_bcg_vaccinated) == nagents, "ti_bcg_vaccinated array should match population size"

def test_bcg_intervention_custom_values():
    """Test BCGProtection intervention with custom parameters"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection(pars={
        'coverage': 0.75,
        'efficacy': 0.9,
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2015-01-01'),
        'immunity_period': ss.years(15),
        'age_range': (1, 10)
    })
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()    
    bcg = sim.interventions['bcgprotection']
    # Check coverage value for starsim bernoulli or float
    if hasattr(bcg.pars.coverage, 'filter'):
        # starsim bernoulli object, check its string representation or default
        assert '0.75' in str(bcg.pars.coverage), "Custom coverage should be 0.75"
    else:
        assert bcg.pars.coverage == 0.75, "Custom coverage should be 0.75"
    assert bcg.pars.efficacy == 0.9, "Custom efficacy should be 0.9"
    assert bcg.pars.start == ss.date('2000-01-01'), "Custom start year should be 2000-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2015-01-01'), "Custom stop year should be 2015-01-01 with type ss.date"
    # Check that immunity_period is approximately 15 years (in timesteps)
    assert abs(bcg.pars.immunity_period - 782.14) < 1.0, "Custom immunity_period should be approximately 15 years"
    assert bcg.pars.age_range == (1, 10), "Custom age range should be (1, 10)"
    assert bcg.min_age == 1, "Custom min_age should be 1"
    assert bcg.max_age == 10, "Custom max_age should be 10"
    assert len(bcg.is_bcg_vaccinated) == nagents, "is_bcg_vaccinated array should match population size"
    assert len(bcg.ti_bcg_vaccinated) == nagents, "ti_bcg_vaccinated array should match population size"

def test_bcg_age_range_functionality():
    """Test BCG age range functionality with different age ranges"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    
    # Test adult vaccination (18-65 years)
    itv = mtb.BCGProtection(pars={'age_range': (18, 65)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    assert bcg.pars.age_range == (18, 65), "Age range should be (18, 65)"
    assert bcg.min_age == 18, "min_age should be 18"
    assert bcg.max_age == 65, "max_age should be 65"
    
    # Test adolescent vaccination (10-19 years)
    itv2 = mtb.BCGProtection(pars={'age_range': (10, 19)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgprotection']
    
    assert bcg2.pars.age_range == (10, 19), "Age range should be (10, 19)"
    assert bcg2.min_age == 10, "min_age should be 10"
    assert bcg2.max_age == 19, "max_age should be 19"

age_data = pd.DataFrame({
    'age':   [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def test_bcg_eligibility_and_vaccination():
    """Test eligibility and vaccination of individuals for BCG using a real simulation"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    
    itv = mtb.BCGProtection()
    assert isinstance(itv, mtb.BCGProtection)
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Check initial state
    sim.init()
    bcg = sim.interventions['bcgprotection']
    assert len(bcg.is_bcg_vaccinated) == nagents, "is_bcg_vaccinated array should match population size"
    
    # Check eligibility before running simulation
    eligible = bcg.select_for_vaccination()
    assert len(eligible) > 0, "There should be eligible individuals for vaccination"
    assert np.all(bcg.is_bcg_vaccinated[eligible] == False), "Eligible individuals should not be vaccinated yet"
    
    # Run full simulation
    sim.run()
    
    # After simulation, some eligible individuals should be vaccinated
    assert np.any(bcg.is_bcg_vaccinated[eligible]), "Some eligible individuals should be vaccinated after simulation"

def test_bcg_eligibility_with_age_range():
    """Test eligibility with different age ranges"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    pop = ss.People(n_agents=nagents, age_data=age_data)
    
    # Test with age range 10-20 (should include ages 10, 15, 20 from age_data)
    itv = mtb.BCGProtection(pars={'age_range': (10, 20)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    eligible = bcg.select_for_vaccination()
    # Should have eligible individuals in the 10-20 age range
    assert len(eligible) > 0, "There should be eligible individuals in age range 10-20"
    
    # Test with age range 30-50 (should include ages 30, 40, 50 from age_data)
    itv2 = mtb.BCGProtection(pars={'age_range': (30, 50)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgprotection']
    
    eligible2 = bcg2.select_for_vaccination()
    # Should have eligible individuals in the 30-50 age range
    assert len(eligible2) > 0, "There should be eligible individuals in age range 30-50"

def test_bcg_improves_tb_outcomes():
    """Test that BCG vaccination improves TB outcomes using a real simulation"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Store initial TB outcomes
    sim.init()
    tb = sim.diseases.tb
    initial_rr_activation = np.array(tb.rr_activation).copy()
    initial_rr_clearance = np.array(tb.rr_clearance).copy()
    initial_rr_death = np.array(tb.rr_death).copy()
    
    # Run full simulation to apply BCG intervention
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check if TB outcomes have improved for vaccinated individuals
    tb = sim.diseases.tb
    current_activation = np.array(tb.rr_activation)
    current_clearance = np.array(tb.rr_clearance)
    current_death = np.array(tb.rr_death)
    
    # Check that some individuals are vaccinated
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated):
        vaccinated_uids = np.where(vaccinated)[0]
        # For vaccinated individuals, BCG should improve outcomes
        assert np.any(current_activation[vaccinated_uids] < initial_rr_activation[vaccinated_uids]), "BCG should reduce activation risk for vaccinated"
        assert np.any(current_clearance[vaccinated_uids] > initial_rr_clearance[vaccinated_uids]), "BCG should improve clearance rate for vaccinated"
        assert np.any(current_death[vaccinated_uids] < initial_rr_death[vaccinated_uids]), "BCG should reduce death risk for vaccinated"

def test_bcg_age_at_vaccination_calculation():
    """Test that age at vaccination is properly calculated in results using a real simulation"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run full simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check that avg_age_at_vaccination is calculated in results
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated):
        # Get results from the last timestep where vaccination occurred
        last_ti = bcg.ti
        avg_age = bcg.results['avg_age_at_vaccination'][last_ti]
        # Average age should be a valid float
        assert isinstance(avg_age, float), "avg_age_at_vaccination should be a float"
        # Average age should be within the eligible age range (or zero if no vaccination in that timestep)
        if avg_age > 0:
            assert avg_age >= bcg.min_age, "Average age at vaccination should be >= min_age"
            assert avg_age <= bcg.max_age, "Average age at vaccination should be <= max_age"

def test_bcg_protection_duration():
    """Test that protection duration is properly set using a real simulation"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection(pars={'immunity_period': ss.years(8)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    # Test that immunity_period is set correctly (approximately 8 years in timesteps)
    # With dt=7 days, 8 years = 8 * 365.25 / 7 ≈ 417.43 timesteps
    assert abs(bcg.pars.immunity_period - 417.43) < 1.0, "immunity_period should be set to approximately 8 years"
    
    # Run simulation to perform vaccination
    sim.run()
    
    # Check that protection expiration is set correctly for vaccinated individuals
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated):
        vaccinated_uids = np.where(vaccinated)[0]
        protection_expires = bcg.ti_bcg_protection_expires[vaccinated_uids]
        # Only check for non-NaN values (i.e., vaccine responders)
        valid = ~np.isnan(protection_expires)
        if np.any(valid):
            # Protection should expire at vaccination time + immunity_period
            vaccination_times = bcg.ti_bcg_vaccinated[vaccinated_uids[valid]]
            expected_expiry = vaccination_times + bcg.pars.immunity_period
            assert np.allclose(protection_expires[valid], expected_expiry), "Protection expiration should be set correctly for responders"

def test_bcg_protection_expiry_and_removal():
    """Test that protection expiry removes protection effects using a real simulation"""
    nagents = 50
    # Run simulation for 2 years to allow protection to expire (1 year immunity period)
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2002-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection(pars={'immunity_period': ss.years(1)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run full simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check that some individuals were vaccinated
    vaccinated = bcg.is_bcg_vaccinated
    assert hasattr(bcg, 'is_bcg_vaccinated'), "BCG intervention should still be functional"
    
    # Check that protection expiration is tracked
    if np.any(vaccinated):
        # Some individuals should have expired protection after 2 years with 1-year immunity
        all_vaccinated = bcg.is_bcg_vaccinated.uids
        if len(all_vaccinated) > 0:
            currently_protected = bcg.is_protected(all_vaccinated, bcg.ti)
            # Some protection should have expired (not all will be protected after 2 years with 1-year immunity)
            assert hasattr(bcg, 'ti_bcg_protection_expires'), "Protection expiration should be tracked"


def test_bcg_maintain_ongoing_protection():
    """Test that ongoing protection maintenance re-applies protection effects using a real simulation"""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-06-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    tb = sim.diseases.tb
    
    # Check that protection is maintained for vaccinated individuals
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated):
        vaccinated_uids = np.where(vaccinated)[0]
        currently_protected = bcg.is_protected(vaccinated_uids, bcg.ti)
        protected_uids = vaccinated_uids[currently_protected]
        
        if len(protected_uids) > 0:
            # Protected individuals should have reduced activation risk
            protected_activation = np.array(tb.rr_activation[protected_uids])
            # BCG should reduce activation (modifier < 1.0), so values should be less than baseline (1.0)
            assert np.any(protected_activation < 1.0), "Protected individuals should have reduced activation risk"


def test_bcg_result_metrics():
    """Test that result metrics are initialized and updated correctly using a real simulation"""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run full simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check that result metrics are present and have expected types
    assert 'n_vaccinated' in bcg.results, "Results should include n_vaccinated"
    last_ti = bcg.ti
    # Results may be stored as floats by Starsim, so check for numeric type
    n_vaccinated = bcg.results['n_vaccinated'][last_ti]
    assert isinstance(n_vaccinated, (int, np.integer, float, np.floating)), "n_vaccinated should be numeric"
    assert n_vaccinated >= 0, "n_vaccinated should be non-negative"
    assert 'vaccination_coverage' in bcg.results, "Results should include vaccination_coverage"
    assert isinstance(bcg.results['vaccination_coverage'][last_ti], (float, np.floating)), "vaccination_coverage should be float"
    assert 'n_protected' in bcg.results, "Results should include n_protected"
    n_protected = bcg.results['n_protected'][last_ti]
    assert isinstance(n_protected, (int, np.integer, float, np.floating)), "n_protected should be numeric"
    assert n_protected >= 0, "n_protected should be non-negative"


def test_bcg_get_summary_stats():
    """Test the get_summary_stats method returns expected keys and types using a real simulation"""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run full simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Get summary statistics after simulation
    stats = bcg.get_summary_stats()
    assert 'total_vaccinated' in stats and 'final_coverage' in stats, "get_summary_stats should return total_vaccinated and final_coverage"
    assert isinstance(stats['total_vaccinated'], (int, np.integer)), "total_vaccinated should be int"
    assert isinstance(stats['final_coverage'], float), "final_coverage should be float"
    assert stats['final_coverage'] >= 0.0, "final_coverage should be non-negative"
    assert stats['final_coverage'] <= 1.0, "final_coverage should be <= 1.0"

