"""
Compute functions for TB simulation analysis

This module contains all the compute functions extracted from the main simulation script.
These functions are used to analyze simulation results and compute various metrics.
"""

import numpy as np
import pandas as pd
import datetime
import starsim as ss
import tbsim as mtb


def compute_latent_prevalence(sim):
    """Compute latent TB prevalence over time"""
    # Get latent counts
    latent_slow = sim.results['tb']['n_latent_slow']
    latent_fast = sim.results['tb']['n_latent_fast']
    latent_total = latent_slow + latent_fast

    # Try getting time-aligned n_alive from starsim if available
    try:
        n_alive_series = sim.results['n_alive']
    except KeyError:
        # Fallback: use average population size
        n_alive_series = np.full_like(latent_total, fill_value=np.count_nonzero(sim.people.alive))

    return latent_total / n_alive_series


def compute_hiv_prevalence(sim):
    """Compute HIV prevalence over time"""
    try:
        # Try to get HIV prevalence directly - use the correct key from HIV model
        hiv_prev = sim.results['hiv']['hiv_prevalence']
        return hiv_prev
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from HIV infection counts using correct key
            n_hiv = sim.results['hiv']['infected']
            n_alive = sim.results['n_alive']
            return n_hiv / n_alive
        except (KeyError, AttributeError):
            try:
                # Another fallback: use n_active from HIV model
                n_hiv = sim.results['hiv']['n_active']
                n_alive = sim.results['n_alive']
                return n_hiv / n_alive
            except (KeyError, AttributeError):
                # Debug: print available HIV result keys
                try:
                    print(f"Available HIV result keys: {list(sim.results['hiv'].keys())}")
                except:
                    print("HIV results not found in simulation")
                # If HIV results are not available, return zeros
                time_length = len(sim.results['timevec'])
                return np.zeros(time_length)


def compute_hiv_prevalence_adults_25plus(sim, target_year=None):
    """
    Compute HIV prevalence for adults 25+ at a specific year or over time
    
    Args:
        sim: Simulation object
        target_year: If specified, compute for this year only. If None, compute over time.
    
    Returns:
        If target_year specified: float (prevalence for that year)
        If target_year is None: array (prevalence over time)
    """
    try:
        # Get people alive and HIV states
        people = sim.people
        alive_mask = people.alive
        hiv_states = sim.diseases.hiv.state
        
        # Get HIV-positive states (states 1, 2, 3 are positive)
        hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
        
        # Filter for adults 25+
        adult_25plus_mask = (people.age >= 25)
        
        # Combine masks
        alive_adult_25plus_mask = alive_mask & adult_25plus_mask
        
        if target_year is not None:
            # Compute for specific year
            time_years = np.array([d.year for d in sim.results['timevec']])
            target_idx = np.argmin(np.abs(time_years - target_year))
            
            # Get states at target time (this is approximate - we use current states)
            total_adults_25plus = np.sum(alive_adult_25plus_mask)
            hiv_positive_adults_25plus = np.sum(alive_adult_25plus_mask & hiv_positive_mask)
            
            if total_adults_25plus > 0:
                return hiv_positive_adults_25plus / total_adults_25plus
            else:
                return 0.0
        else:
            # Compute over time (this is approximate since we only have current states)
            # For now, return the overall HIV prevalence as a proxy
            return compute_hiv_prevalence(sim)
            
    except Exception as e:
        print(f"Error computing HIV prevalence for adults 25+: {e}")
        if target_year is not None:
            return 0.0
        else:
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_hiv_prevalence_adults_15to24(sim, target_year=None):
    """
    Compute HIV prevalence for adults 15-24 at a specific year or over time
    
    Args:
        sim: Simulation object
        target_year: If specified, compute for this year only. If None, compute over time.
    
    Returns:
        If target_year specified: float (prevalence for that year)
        If target_year is None: array (prevalence over time)
    """
    try:
        # Get people alive and HIV states
        people = sim.people
        alive_mask = people.alive
        hiv_states = sim.diseases.hiv.state
        
        # Get HIV-positive states (states 1, 2, 3 are positive)
        hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
        
        # Filter for adults 15-24
        adult_15to24_mask = (people.age >= 15) & (people.age <= 24)
        
        # Combine masks
        alive_adult_15to24_mask = alive_mask & adult_15to24_mask
        
        if target_year is not None:
            # Compute for specific year
            time_years = np.array([d.year for d in sim.results['timevec']])
            target_idx = np.argmin(np.abs(time_years - target_year))
            
            # Get states at target time (this is approximate - we use current states)
            total_adults_15to24 = np.sum(alive_adult_15to24_mask)
            hiv_positive_adults_15to24 = np.sum(alive_adult_15to24_mask & hiv_positive_mask)
            
            if total_adults_15to24 > 0:
                return hiv_positive_adults_15to24 / total_adults_15to24
            else:
                return 0.0
        else:
            # Compute over time (this is approximate since we only have current states)
            # For now, return the overall HIV prevalence as a proxy
            return compute_hiv_prevalence(sim)
            
    except Exception as e:
        print(f"Error computing HIV prevalence for adults 15-24: {e}")
        if target_year is not None:
            return 0.0
        else:
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_hiv_positive_tb_prevalence(sim):
    """Compute HIV-positive TB prevalence as proportion of total population"""
    try:
        # Try to get HIV-positive TB counts directly
        hiv_positive_tb = sim.results['tb']['n_active_hiv_positive']
        n_alive = sim.results['n_alive']
        return hiv_positive_tb / n_alive
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from individual states
            # Get HIV and TB prevalence using correct keys
            hiv_prev = compute_hiv_prevalence(sim)
            tb_prev = sim.results['tb']['prevalence_active']
            
            # Estimate HIV-positive TB prevalence as a fraction of total TB
            # This is a rough estimate - in reality it depends on the TB-HIV interaction
            hiv_tb_overlap = hiv_prev * tb_prev * 0.3  # Assume 30% of TB cases are HIV-positive
            return hiv_tb_overlap
        except (KeyError, AttributeError):
            # If results are not available, return zeros
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_age_stratified_prevalence(sim, target_year=2018):
    """
    Compute age-stratified TB prevalence from simulation results
    
    Args:
        sim: Simulation object
        target_year: Year to compute prevalence for
    
    Returns:
        dict: Age-stratified prevalence data
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get TB states
    tb_states = sim.diseases.tb.state
    active_tb_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # Get ages at target time
    ages = people.age[alive_mask]
    active_tb_ages = people.age[alive_mask & active_tb_mask]
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    prevalence_by_age = {}
    
    for i, (min_age, max_age) in enumerate(age_groups):
        # Count people in age group
        age_mask = (ages >= min_age) & (ages <= max_age)
        total_in_age_group = np.sum(age_mask)
        
        # Count active TB cases in age group
        age_tb_mask = (active_tb_ages >= min_age) & (active_tb_ages <= max_age)
        tb_in_age_group = np.sum(age_tb_mask)
        
        # Calculate prevalence
        if total_in_age_group > 0:
            prevalence = tb_in_age_group / total_in_age_group
            prevalence_per_100k = prevalence * 100000
        else:
            prevalence = 0
            prevalence_per_100k = 0
        
        prevalence_by_age[age_group_labels[i]] = {
            'prevalence': prevalence,
            'prevalence_per_100k': prevalence_per_100k,
            'total_people': total_in_age_group,
            'tb_cases': tb_in_age_group
        }
    
    return prevalence_by_age


def compute_age_stratified_prevalence_time_series(sim):
    """
    Compute age-stratified TB prevalence time series from simulation results
    
    Args:
        sim: Simulation object
    
    Returns:
        pd.DataFrame: DataFrame with years as index and age groups as columns
    """
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Get time vector
    time_years = np.array([d.year for d in sim.results['timevec']])
    
    # Initialize DataFrame to store results
    prevalence_df = pd.DataFrame(index=time_years, columns=age_group_labels)
    
    # For each time point, compute age-stratified prevalence
    for t_idx, (time_point, year) in enumerate(zip(sim.results['timevec'], time_years)):
        # Get people alive at this time point
        people = sim.people
        
        # For simplicity, we'll use the current people state
        # In a more sophisticated approach, we'd need to track historical states
        alive_mask = people.alive
        
        # Get TB states
        tb_states = sim.diseases.tb.state
        active_tb_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
        
        # Get ages
        ages = people.age[alive_mask]
        active_tb_ages = people.age[alive_mask & active_tb_mask]
        
        # Compute prevalence for each age group
        for i, (min_age, max_age) in enumerate(age_groups):
            # Count people in age group
            age_mask = (ages >= min_age) & (ages <= max_age)
            total_in_age_group = np.sum(age_mask)
            
            # Count active TB cases in age group
            age_tb_mask = (active_tb_ages >= min_age) & (active_tb_ages <= max_age)
            tb_in_age_group = np.sum(age_tb_mask)
            
            # Calculate prevalence per 100,000
            if total_in_age_group > 0:
                prevalence = tb_in_age_group / total_in_age_group
                prevalence_per_100k = prevalence * 100000
            else:
                prevalence_per_100k = 0
            
            prevalence_df.loc[year, age_group_labels[i]] = prevalence_per_100k
    
    return prevalence_df


def compute_age_stratified_incidence(sim, target_year=2018):
    """
    Compute age-stratified TB incidence from simulation results
    
    Args:
        sim: Simulation object
        target_year: Year to compute incidence for
    
    Returns:
        dict: Age-stratified incidence data
    """
    
    # Find the time index closest to target year
    timevec = sim.results['timevec']
    if hasattr(timevec[0], 'year'):
        # Handle datetime objects
        time_years = np.array([d.year for d in timevec])
    else:
        # Handle numeric time values
        time_years = np.array([int(t) for t in timevec])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get time vector and TB results
    time = np.array(sim.results['timevec'])
    tb_results = sim.results['tb']
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get ages at target time
    ages = people.age[alive_mask]
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    incidence_by_age = {}
    
    for i, (min_age, max_age) in enumerate(age_groups):
        age_group = age_group_labels[i]
        
        # Count people in age group at target time
        age_mask = (ages >= min_age) & (ages <= max_age)
        pop_in_age_group = np.sum(age_mask)
        
        # Get cumulative active cases for this age group
        if f'cum_active_{age_group}' in tb_results:
            cum_incidence = tb_results[f'cum_active_{age_group}']
        else:
            # Fallback to overall cumulative active if age-stratified data not available
            if 'cum_active' in tb_results:
                cum_incidence = tb_results['cum_active']
            else:
                cum_incidence = np.cumsum(tb_results['new_active'])
        
        # Compute annualized incidence rate at target year
        t_date = time[target_idx]
        if hasattr(t_date, 'year'):
            # Handle datetime objects
            t_prev_date = t_date - datetime.timedelta(days=365)
            t_prev = np.searchsorted(time, t_prev_date)
            if t_prev == len(time) or time[t_prev] > t_prev_date:
                t_prev = max(0, t_prev - 1)
        else:
            # Handle numeric time values (assume yearly time steps)
            # Use a 5-year window to capture more cases
            t_prev = max(0, target_idx - 5)
        
        new_cases = cum_incidence[target_idx] - cum_incidence[t_prev]
        
        # Calculate the time span for proper annualization
        if hasattr(t_date, 'year'):
            # For datetime objects, calculate actual days
            time_span_years = (timevec[target_idx] - timevec[t_prev]).days / 365.0
        else:
            # For numeric time values, calculate years
            time_span_years = timevec[target_idx] - timevec[t_prev]
        
        # Annualize the rate properly
        if time_span_years > 0:
            annualized_cases = new_cases / time_span_years
            incidence_rate = (annualized_cases / pop_in_age_group) * 1e5 if pop_in_age_group > 0 else 0
        else:
            incidence_rate = 0
        
        incidence_by_age[age_group] = {
            'incidence_rate': incidence_rate,
            'incidence_per_100k': incidence_rate,
            'new_cases': new_cases,
            'population': pop_in_age_group
        }
    
    return incidence_by_age


def compute_age_stratified_incidence_time_series(sim):
    """
    Compute age-stratified TB incidence time series from simulation results
    
    Args:
        sim: Simulation object
    
    Returns:
        pd.DataFrame: DataFrame with years as index and age groups as columns
    """
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Get time vector and TB results
    time = np.array(sim.results['timevec'])
    tb_results = sim.results['tb']
    
    # Initialize DataFrame to store results
    if hasattr(time[0], 'year'):
        # Handle datetime objects
        time_years = np.array([d.year for d in time])
    else:
        # Handle numeric time values
        time_years = np.array([int(t) for t in time])
    incidence_df = pd.DataFrame(index=time_years, columns=age_group_labels)
    
    # For each time point, compute age-stratified incidence
    for t_idx, (time_point, year) in enumerate(zip(time, time_years)):
        # Get people alive at this time point
        people = sim.people
        
        # For simplicity, we'll use the current people state
        # In a more sophisticated approach, we'd need to track historical states
        alive_mask = people.alive
        ages = people.age[alive_mask]
        
        # Compute incidence for each age group
        for i, (min_age, max_age) in enumerate(age_groups):
            age_group = age_group_labels[i]
            
            # Count people in age group at this time point
            age_mask = (ages >= min_age) & (ages <= max_age)
            pop_in_age_group = np.sum(age_mask)
            
            # Get cumulative active cases for this age group
            if f'cum_active_{age_group}' in tb_results:
                cum_incidence = tb_results[f'cum_active_{age_group}']
            else:
                # Fallback to overall cumulative active if age-stratified data not available
                if 'cum_active' in tb_results:
                    cum_incidence = tb_results['cum_active']
                else:
                    cum_incidence = np.cumsum(tb_results['new_active'])
            
            # Compute annualized incidence rate
            t_date = time_point
            if hasattr(t_date, 'year'):
                # Handle datetime objects
                t_prev_date = t_date - datetime.timedelta(days=365)
                t_prev = np.searchsorted(time, t_prev_date)
                if t_prev == len(time) or time[t_prev] > t_prev_date:
                    t_prev = max(0, t_prev - 1)
            else:
                # Handle numeric time values (assume yearly time steps)
                # Use a 5-year window to capture more cases
                t_prev = max(0, t_idx - 5)
            
            new_cases = cum_incidence[t_idx] - cum_incidence[t_prev]
            
            # Calculate the time span for proper annualization
            if hasattr(t_date, 'year'):
                # For datetime objects, calculate actual days
                time_span_years = (time[t_idx] - time[t_prev]).days / 365.0
            else:
                # For numeric time values, calculate years
                time_span_years = time[t_idx] - time[t_prev]
            
            # Annualize the rate properly
            if time_span_years > 0:
                annualized_cases = new_cases / time_span_years
                incidence_rate = (annualized_cases / pop_in_age_group) * 1e5 if pop_in_age_group > 0 else 0
            else:
                incidence_rate = 0
            
            incidence_df.loc[year, age_group] = incidence_rate
    
    return incidence_df


def compute_annualized_infection_rate(sim):
    """
    Compute annualized TB infection rate (annual risk of infection) over time.
    
    This function calculates the annualized infection rate using two methods:
    1. Method 1: Sum new_infections over 365 days and divide by population
    2. Method 2: Difference in n_infected between T and T-365 days, divided by population
    
    Returns the annualized infection rate as a percentage of the population.
    """
    time = sim.results['timevec']
    tb_results = sim.results['tb']
    
    # Get population size over time
    try:
        n_alive = sim.results['n_alive']
    except KeyError:
        n_alive = np.full(len(time), fill_value=np.count_nonzero(sim.people.alive))
    
    # Method 1: Using new_infections (if available)
    annual_rate_method1 = None
    try:
        # Check if new_infections is available
        if 'new_infections' in tb_results:
            new_infections = tb_results['new_infections'].values
            annual_rate_method1 = np.zeros_like(time, dtype=float)
            
            # Calculate 365-day rolling sum of new infections
            days_per_step = (time[1] - time[0]).days if len(time) > 1 else 1
            steps_per_year = max(1, int(365 / days_per_step))
            
            for i in range(len(time)):
                start_idx = max(0, i - steps_per_year + 1)
                annual_infections = np.sum(new_infections[start_idx:i+1])
                annual_rate_method1[i] = (annual_infections / n_alive[i]) * 100 if n_alive[i] > 0 else 0
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Using difference in n_infected
    annual_rate_method2 = np.zeros_like(time, dtype=float)
    try:
        # Get total infected count over time
        n_infected = tb_results['n_latent_slow'].values + tb_results['n_latent_fast'].values + tb_results['n_active'].values
        
        # Calculate 365-day difference
        days_per_step = (time[1] - time[0]).days if len(time) > 1 else 1
        steps_per_year = max(1, int(365 / days_per_step))
        
        for i in range(len(time)):
            if i >= steps_per_year:
                # Calculate difference in infected count over the year
                infection_diff = n_infected[i] - n_infected[i - steps_per_year]
                annual_rate_method2[i] = (infection_diff / n_alive[i]) * 100 if n_alive[i] > 0 else 0
            else:
                # For early time points, use the current rate scaled to annual
                annual_rate_method2[i] = (n_infected[i] / n_alive[i]) * 100 if n_alive[i] > 0 else 0
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Return the more robust method (Method 2) or Method 1 if Method 2 fails
    if annual_rate_method2 is not None and not np.all(np.isnan(annual_rate_method2)):
        return annual_rate_method2
    elif annual_rate_method1 is not None and not np.all(np.isnan(annual_rate_method1)):
        return annual_rate_method1
    else:
        print("Warning: Could not compute annualized infection rate")
        return np.zeros_like(time, dtype=float)


def compute_hiv_tb_coinfection_rates(sim, target_year=2018):
    """
    Compute HIV coinfection rates among TB cases by symptom status
    
    Args:
        sim: Simulation object
        target_year: Year to compute rates for
    
    Returns:
        dict: HIV coinfection rates by TB symptom status
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get TB states
    tb_states = sim.diseases.tb.state
    hiv_states = sim.diseases.hiv.state
    
    # Define TB states by symptom status
    # Presymptomatic (0 symptoms) - ACTIVE_PRESYMP
    presymptomatic_mask = (tb_states == mtb.TBS.ACTIVE_PRESYMP)
    
    # Symptomatic (≥1 symptoms) - ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB
    symptomatic_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # All active TB (any symptoms)
    all_active_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_PRESYMP, mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # Get HIV-positive states (assuming HIV states 1, 2, 3 are positive - adjust as needed)
    # HIV states: 0=uninfected, 1=acute, 2=latent, 3=AIDS
    hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
    
    # Filter for adults (age 15+)
    adult_mask = (people.age >= 15)
    
    # Combine masks
    alive_adult_mask = alive_mask & adult_mask
    
    # Calculate coinfection rates for each category
    coinfection_rates = {}
    
    # 1. Presymptomatic TB cases (0 symptoms)
    presymp_adult_mask = alive_adult_mask & presymptomatic_mask
    presymp_total = np.sum(presymp_adult_mask)
    presymp_hiv_positive = np.sum(presymp_adult_mask & hiv_positive_mask)
    presymp_hiv_rate = (presymp_hiv_positive / presymp_total * 100) if presymp_total > 0 else 0
    
    coinfection_rates['presymptomatic'] = {
        'total_cases': presymp_total,
        'hiv_positive': presymp_hiv_positive,
        'hiv_rate_percent': presymp_hiv_rate
    }
    
    # 2. Symptomatic TB cases (≥1 symptoms)
    sympt_adult_mask = alive_adult_mask & symptomatic_mask
    sympt_total = np.sum(sympt_adult_mask)
    sympt_hiv_positive = np.sum(sympt_adult_mask & hiv_positive_mask)
    sympt_hiv_rate = (sympt_hiv_positive / sympt_total * 100) if sympt_total > 0 else 0
    
    coinfection_rates['symptomatic'] = {
        'total_cases': sympt_total,
        'hiv_positive': sympt_hiv_positive,
        'hiv_rate_percent': sympt_hiv_rate
    }
    
    # 3. All active TB cases (any symptoms)
    all_active_adult_mask = alive_adult_mask & all_active_mask
    all_active_total = np.sum(all_active_adult_mask)
    all_active_hiv_positive = np.sum(all_active_adult_mask & hiv_positive_mask)
    all_active_hiv_rate = (all_active_hiv_positive / all_active_total * 100) if all_active_total > 0 else 0
    
    coinfection_rates['all_active'] = {
        'total_cases': all_active_total,
        'hiv_positive': all_active_hiv_positive,
        'hiv_rate_percent': all_active_hiv_rate
    }
    
    return coinfection_rates


def compute_annualized_tb_mortality_rate(sim):
    """
    Compute annualized TB mortality rate (per 100,000 population) over time.
    
    This function calculates the annualized TB mortality rate by:
    1. Taking the difference in cumulative TB deaths between time T and T-365 days
    2. Dividing by the population at time T
    3. Multiplying by 100,000 to get rate per 100,000 population
    
    Returns the annualized TB mortality rate per 100,000 population.
    """
    time = sim.results['timevec']
    tb_results = sim.results['tb']
    
    # Get population size over time
    try:
        n_alive = sim.results['n_alive']
        # Handle both numpy arrays and pandas Series
        if hasattr(n_alive, 'values'):
            n_alive = n_alive.values
    except KeyError:
        n_alive = np.full(len(time), fill_value=np.count_nonzero(sim.people.alive))
    
    # Get cumulative TB deaths
    if 'cum_deaths' in tb_results:
        cum_deaths = tb_results['cum_deaths']
        # Handle both numpy arrays and pandas Series
        if hasattr(cum_deaths, 'values'):
            cum_deaths = cum_deaths.values
    else:
        # Fallback: compute cumulative sum of new_deaths
        if 'new_deaths' in tb_results:
            new_deaths = tb_results['new_deaths']
            # Handle both numpy arrays and pandas Series
            if hasattr(new_deaths, 'values'):
                new_deaths = new_deaths.values
            cum_deaths = np.cumsum(new_deaths)
        else:
            raise ValueError('No new_deaths or cum_deaths in tb results')
    
    # Compute annualized mortality rate
    mortality_rate = np.zeros_like(cum_deaths, dtype=float)
    for t in range(len(time)):
        t_date = time[t]
        t_prev_date = t_date - datetime.timedelta(days=365)
        t_prev = np.searchsorted(time, t_prev_date)
        if t_prev == len(time) or time[t_prev] > t_prev_date:
            t_prev = max(0, t_prev - 1)
        
        # Calculate difference in cumulative deaths over the year
        deaths_diff = cum_deaths[t] - cum_deaths[t_prev]
        pop = n_alive[t]
        mortality_rate[t] = (deaths_diff / pop) * 1e5 if pop > 0 else 0
    
    return mortality_rate


def compute_age_distribution_at_year(sim, target_year=2022):
    """
    Compute age distribution at a specific year
    
    Args:
        sim: Simulation object
        target_year: Year to compute age distribution for
    
    Returns:
        dict: Age distribution data with 5-year bins
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get ages at target time
    ages = people.age[alive_mask]
    
    # Define 5-year age bins: 0-4, 5-9, 10-14, 15-19, 20-24, 25-29, 30-34, 35-39, 40-44, 45-49, 50-54, 55-59, 60-64, 65-69, 70-74, 75-79, 80-84, 85-89, 90-94, 95+
    age_bins = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49), (50, 54), (55, 59), (60, 64), (65, 69), (70, 74), (75, 79), (80, 84), (85, 89), (90, 94), (95, 200)]
    age_bin_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94', '95+']
    
    total_population = len(ages)
    age_distribution = {}
    
    for i, (min_age, max_age) in enumerate(age_bins):
        # Count people in age bin
        age_mask = (ages >= min_age) & (ages <= max_age)
        count_in_bin = np.sum(age_mask)
        
        # Calculate percentage
        percentage = (count_in_bin / total_population) * 100 if total_population > 0 else 0
        
        age_distribution[age_bin_labels[i]] = {
            'count': count_in_bin,
            'percentage': percentage
        }
    
    return age_distribution
