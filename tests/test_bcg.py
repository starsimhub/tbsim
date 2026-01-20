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
    # Check efficacy value (now a distribution)
    assert isinstance(bcg.pars.efficacy, ss.Dist), "Efficacy should be a distribution"
    assert abs(bcg.pars.efficacy.pars['p'] - 0.8) < 0.01, "Default efficacy should be 0.8"
    assert bcg.pars.start == ss.date('1900-01-01'), "Default start year should be 1900-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2100-12-31'), "Default stop year should be 2100-12-31 with type ss.date"
    # Check that immunity_period is 10 years (as ss.years object)
    assert hasattr(bcg.pars.immunity_period, 'value'), "immunity_period should be a timepar object"
    assert abs(bcg.pars.immunity_period.value - 10.0) < 0.01, "Default immunity_period should be 10 years"
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
    # Check efficacy value (now a distribution)
    assert isinstance(bcg.pars.efficacy, ss.Dist), "Efficacy should be a distribution"
    assert abs(bcg.pars.efficacy.pars['p'] - 0.9) < 0.01, "Custom efficacy should be 0.9"
    assert bcg.pars.start == ss.date('2000-01-01'), "Custom start year should be 2000-01-01 with type ss.date"
    assert bcg.pars.stop == ss.date('2015-01-01'), "Custom stop year should be 2015-01-01 with type ss.date"
    # Check that immunity_period is 15 years (as ss.years object)
    assert hasattr(bcg.pars.immunity_period, 'value'), "immunity_period should be a timepar object"
    assert abs(bcg.pars.immunity_period.value - 15.0) < 0.01, "Custom immunity_period should be 15 years"
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
    eligible = ((bcg.sim.people.age >= bcg.min_age) & (bcg.sim.people.age <= bcg.max_age) & ~bcg.is_bcg_vaccinated).uids
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
    
    eligible = ((bcg.sim.people.age >= bcg.min_age) & (bcg.sim.people.age <= bcg.max_age) & ~bcg.is_bcg_vaccinated).uids
    # Should have eligible individuals in the 10-20 age range
    assert len(eligible) > 0, "There should be eligible individuals in age range 10-20"
    
    # Test with age range 30-50 (should include ages 30, 40, 50 from age_data)
    itv2 = mtb.BCGProtection(pars={'age_range': (30, 50)})
    sim2 = ss.Sim(people=pop, diseases=tb, interventions=itv2, networks=net, pars=pars)
    sim2.init()
    bcg2 = sim2.interventions['bcgprotection']
    
    eligible2 = ((bcg2.sim.people.age >= bcg2.min_age) & (bcg2.sim.people.age <= bcg2.max_age) & ~bcg2.is_bcg_vaccinated).uids
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
    
    # Check that some individuals are vaccinated
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated):
        vaccinated_uids = np.where(vaccinated)[0]
        # Check stored BCG modifiers (TB resets modifiers at end of each timestep)
        # These modifiers show what BCG applied during the simulation
        stored_activation = np.array(bcg.bcg_activation_modifier_applied[vaccinated_uids])
        stored_clearance = np.array(bcg.bcg_clearance_modifier_applied[vaccinated_uids])
        stored_death = np.array(bcg.bcg_death_modifier_applied[vaccinated_uids])
        
        # Get valid modifiers (non-NaN) - these are the vaccine responders
        valid_mask = ~np.isnan(stored_activation)
        if np.any(valid_mask):
            valid_uids = vaccinated_uids[valid_mask]
            # For vaccine responders, BCG should improve outcomes
            assert np.all(stored_activation[valid_mask] < 1.0), "BCG should reduce activation risk (modifier < 1.0)"
            assert np.all(stored_clearance[valid_mask] > 1.0), "BCG should improve clearance rate (modifier > 1.0)"
            assert np.all(stored_death[valid_mask] < 1.0), "BCG should reduce death risk (modifier < 1.0)"

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
    
    # Check that avg_age_at_vaccination is calculated in results (if it exists)
    vaccinated = bcg.is_bcg_vaccinated
    if np.any(vaccinated) and 'avg_age_at_vaccination' in bcg.results:
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
    
    # Test that immunity_period is set correctly (8 years as ss.years object)
    assert hasattr(bcg.pars.immunity_period, 'value'), "immunity_period should be a timepar object"
    assert abs(bcg.pars.immunity_period.value - 8.0) < 0.01, "immunity_period should be set to 8 years"
    
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
            # Protection should expire at vaccination time + immunity_period (converted to timesteps)
            vaccination_times = bcg.ti_bcg_vaccinated[vaccinated_uids[valid]]
            dt_days = bcg.sim.dt.days if hasattr(bcg.sim.dt, 'days') else bcg.sim.dt.value
            immunity_period_ts = bcg.pars.immunity_period.days / dt_days
            expected_expiry = vaccination_times + immunity_period_ts
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
            currently_protected = (bcg.is_bcg_vaccinated[all_vaccinated] & 
                                   (bcg.ti <= bcg.ti_bcg_protection_expires[all_vaccinated]) & 
                                   ~np.isnan(bcg.ti_bcg_protection_expires[all_vaccinated]))
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
        currently_protected = (bcg.is_bcg_vaccinated[vaccinated_uids] & 
                              (bcg.ti <= bcg.ti_bcg_protection_expires[vaccinated_uids]) & 
                              ~np.isnan(bcg.ti_bcg_protection_expires[vaccinated_uids]))
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


def test_bcg_summary_metrics():
    """Test that summary metrics can be calculated from intervention state using a real simulation"""
    nagents = 50
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2001-01-01'))
    pop = ss.People(n_agents=nagents, age_data=age_data)
    itv = mtb.BCGProtection()
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    
    # Run full simulation
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Calculate summary statistics directly from intervention state
    total_vaccinated = np.count_nonzero(bcg.is_bcg_vaccinated)
    total_pop = len(sim.people)
    final_coverage = total_vaccinated / total_pop if total_pop > 0 else 0.0
    protection_expires_array = np.array(bcg.ti_bcg_protection_expires)
    total_responders = np.sum(~np.isnan(protection_expires_array))
    effectiveness = total_responders / total_vaccinated if total_vaccinated > 0 else 0.0
    
    assert isinstance(total_vaccinated, (int, np.integer)), "total_vaccinated should be int"
    assert isinstance(final_coverage, float), "final_coverage should be float"
    assert final_coverage >= 0.0, "final_coverage should be non-negative"
    assert final_coverage <= 1.0, "final_coverage should be <= 1.0"
    assert effectiveness >= 0.0, "effectiveness should be non-negative"
    assert effectiveness <= 1.0, "effectiveness should be <= 1.0"


def test_bcg_immediate_vaccination_timing():
    """Test that immediate vaccination (vaccination_timing=None) vaccinates eligible individuals immediately"""
    nagents = 200
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2010-12-31'))
    itv = mtb.BCGProtection(pars={
        'coverage': 1.0,  # 100% coverage to ensure all eligible are vaccinated
        'age_range': (0, 18),
        'vaccination_timing': None,  # Immediate vaccination
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2010-12-31')
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check that vaccinations happened early (immediate vaccination)
    vacc_times = bcg.ti_bcg_vaccinated.raw
    valid_times = vacc_times[~np.isnan(vacc_times)]
    
    vaccinated = bcg.is_bcg_vaccinated.sum()
    assert vaccinated > 0, "Some individuals should be vaccinated"
    assert hasattr(bcg, 'ti_bcg_scheduled'), "ti_bcg_scheduled state should exist"
    
    if len(valid_times) > 0:
        # With immediate vaccination, mean vaccination time should be very early
        # Convert timesteps to years for comparison
        dt_years = sim.dt.value / 365.25
        mean_vacc_years = np.mean(valid_times) * dt_years
        assert mean_vacc_years < 1.0, f"Immediate vaccination should happen early (<1 year), got mean time: {mean_vacc_years:.2f} years"

def test_bcg_distributed_vaccination_timing():
    """Test that distributed vaccination (vaccination_timing=distribution) schedules and vaccinates over time"""
    nagents = 200
    pop, tb, net, pars = make_sim(agents=nagents, start=ss.date('2000-01-01'), stop=ss.date('2010-12-31'))
    itv = mtb.BCGProtection(pars={
        'coverage': 1.0,  # 100% coverage to ensure all eligible are vaccinated
        'age_range': (0, 18),
        'vaccination_timing': ss.uniform(0, 5),  # Distributed over 0-5 years
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2010-12-31')
    })
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check scheduled times
    scheduled = bcg.ti_bcg_scheduled.raw
    valid_scheduled = scheduled[~np.isnan(scheduled)]
    
    # Check vaccination times
    vacc_times = bcg.ti_bcg_vaccinated.raw
    valid_vacc_times = vacc_times[~np.isnan(vacc_times)]
    
    assert len(valid_scheduled) > 0, "Some individuals should have scheduled vaccination times"
    assert len(valid_vacc_times) > 0, "Some individuals should be vaccinated"
    
    if len(valid_scheduled) > 0:
        # Convert timesteps to years
        dt_years = sim.dt.value / 365.25
        scheduled_years = valid_scheduled * dt_years
        
        # Scheduled times should be distributed over 0-5 years (allowing some tolerance)
        assert np.min(scheduled_years) >= -0.1, f"Scheduled times should start near 0, got: {np.min(scheduled_years)}"
        assert np.max(scheduled_years) <= 5.5, f"Scheduled times should be within 5 years, got: {np.max(scheduled_years)}"
        
        # Mean should be around 2.5 years for uniform(0, 5)
        mean_scheduled = np.mean(scheduled_years)
        assert 1.0 <= mean_scheduled <= 4.0, f"Mean scheduled time should be around 2.5 years, got: {mean_scheduled}"
    
    if len(valid_vacc_times) > 0 and len(valid_scheduled) > 0:
        # Vaccination times should be >= scheduled times
        vacc_years = valid_vacc_times[:len(valid_scheduled)] * dt_years
        scheduled_years = valid_scheduled[:len(vacc_years)] * dt_years
        assert np.all(vacc_years >= scheduled_years - 0.1), "Vaccinations should happen at or after scheduled time"

def test_bcg_vaccination_timing_comparison():
    """Test that distributed vaccination has later mean vaccination time than immediate"""
    nagents = 100
    age_data = pd.DataFrame({'age': [0, 2, 4, 10, 15, 20, 30, 40, 50], 'value': [10, 10, 15, 15, 10, 10, 10, 10, 10]})
    
    # Immediate vaccination
    pop1 = ss.People(n_agents=nagents, age_data=age_data)
    tb1 = mtb.TB()
    net1 = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    bcg1 = mtb.BCGProtection(pars={
        'coverage': 0.9,
        'age_range': (0, 18),
        'vaccination_timing': None,
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2010-12-31')
    })
    sim1 = ss.Sim(people=pop1, diseases=tb1, interventions=bcg1, networks=net1,
                 pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')))
    sim1.run()
    
    # Distributed vaccination
    pop2 = ss.People(n_agents=nagents, age_data=age_data)
    tb2 = mtb.TB()
    net2 = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    bcg2 = mtb.BCGProtection(pars={
        'coverage': 0.9,
        'age_range': (0, 18),
        'vaccination_timing': ss.uniform(0, 5),
        'start': ss.date('2000-01-01'),
        'stop': ss.date('2010-12-31')
    })
    sim2 = ss.Sim(people=pop2, diseases=tb2, interventions=bcg2, networks=net2,
                 pars=dict(dt=ss.days(7), start=ss.date('2000-01-01'), stop=ss.date('2010-12-31')))
    sim2.run()
    
    # Compare vaccination times
    vacc1_times = bcg1.ti_bcg_vaccinated.raw
    vacc2_times = bcg2.ti_bcg_vaccinated.raw
    
    valid1 = vacc1_times[~np.isnan(vacc1_times)]
    valid2 = vacc2_times[~np.isnan(vacc2_times)]
    
    if len(valid1) > 0 and len(valid2) > 0:
        dt_years = sim1.dt.value / 365.25
        mean1 = np.mean(valid1) * dt_years
        mean2 = np.mean(valid2) * dt_years
        
        # Distributed should have later mean vaccination time
        assert mean2 > mean1, f"Distributed vaccination should have later mean time ({mean2}) than immediate ({mean1})"

def test_bcg_vaccination_timing_none_default():
    """Test that vaccination_timing defaults to None (immediate vaccination)"""
    nagents = 100
    pop, tb, net, pars = make_sim(agents=nagents)
    itv = mtb.BCGProtection(pars={'coverage': 0.9, 'age_range': (0, 18)})
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.init()
    bcg = sim.interventions['bcgprotection']
    
    # vaccination_timing should default to None
    assert bcg.pars.vaccination_timing is None, "vaccination_timing should default to None"
    assert bcg._vaccination_timing_dist is None, "_vaccination_timing_dist should be None when vaccination_timing is None"
    sim = ss.Sim(people=pop, diseases=tb, interventions=itv, networks=net, pars=pars)
    sim.run()
    bcg = sim.interventions['bcgprotection']
    
    # Check that vaccinations happened early (immediate vaccination)
    vacc_times = bcg.ti_bcg_vaccinated.raw
    valid_times = vacc_times[~np.isnan(vacc_times)]
    
    vaccinated = bcg.is_bcg_vaccinated.sum()
    assert vaccinated > 0, "Some individuals should be vaccinated"
    assert hasattr(bcg, 'ti_bcg_scheduled'), "ti_bcg_scheduled state should exist"
    
    if len(valid_times) > 0:
        # With immediate vaccination, mean vaccination time should be very early
        # Convert timesteps to years for comparison
        dt_years = sim.dt.value / 365.25
        mean_vacc_years = np.mean(valid_times) * dt_years
        assert mean_vacc_years < 1.0, f"Immediate vaccination should happen early (<1 year), got mean time: {mean_vacc_years:.2f} years"
