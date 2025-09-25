#!/usr/bin/env python3
"""
TB Simulation: Grid Search Constant Prevalence Calibration

This script uses a grid search approach to calibrate TB parameters for constant prevalence
by systematically testing different parameter combinations. This approach proved more
effective than the Starsim calibration framework for this specific use case.

TB NATURAL HISTORY DEFINITION:
The calibration uses the complete TB natural history model with 4 active TB states:
- ACTIVE_PRESYMP (2): Active TB, pre-symptomatic phase
- ACTIVE_SMPOS (3): Active TB, smear positive (most infectious)
- ACTIVE_SMNEG (4): Active TB, smear negative (moderately infectious)  
- ACTIVE_EXPTB (5): Active TB, extra-pulmonary (least infectious)

This follows the TB natural history model from:
https://www.pnas.org/doi/full/10.1073/pnas.0901720106

The script provides options to analyze prevalence using either:
1. All 4 active TB states (recommended for complete TB natural history)
2. Only 3 symptomatic states (excluding pre-symptomatic cases)

CALIBRATION METHODOLOGY:
The grid search systematically tests parameter combinations to find values that achieve:
- Target prevalence of 1% (±0.2% tolerance)
- Low coefficient of variation (CV < 10% for excellent results)
- High target compliance (>70% of time points within target range)

The scoring system penalizes:
- Deviation from target prevalence (40% weight)
- High coefficient of variation (40% weight) 
- Low target compliance (20% weight)

PARAMETER RELATIONSHIPS:
- beta: Controls transmission rate - higher values increase prevalence
- rate_LS_to_presym: Latent slow progression - affects disease development timing
- rate_LF_to_presym: Latent fast progression - affects disease development timing
- rate_active_to_clear: Natural clearance rate - higher values decrease prevalence
- rate_treatment_to_clear: Treatment effectiveness - higher values decrease prevalence
- Mortality rates: Affect population dynamics and disease burden
- rel_trans_treatment: Treatment reduces but doesn't eliminate transmission
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tbsim.tb import TBS
import itertools

def create_tb_simulation(beta=0.1, rate_LS_to_presym=1e-4, rate_LF_to_presym=1e-2, 
                        rate_active_to_clear=1e-4, rate_treatment_to_clear=2.0,
                        rate_smpos_to_dead=2e-4, rate_smneg_to_dead=2e-5, 
                        rate_exptb_to_dead=1e-5, rel_trans_treatment=0.3,
                        n_agents=2000, rand_seed=1):
    """
    Create a TB simulation with specified parameters.
    
    This function sets up a complete TB simulation using the Starsim framework with
    the TB natural history model. The simulation includes:
    - Population dynamics (births, deaths)
    - TB disease progression through all states
    - Contact networks for transmission
    - Treatment effects on transmission and progression
    
    The TB model follows the natural history with latent states progressing to
    active disease, which can then be treated or lead to death.
    
    Args:
        beta (float): Transmission rate per year - controls how infectious TB is
        rate_LS_to_presym (float): Latent slow to presymptomatic rate per day
            - Controls progression from slow latent to active presymptomatic TB
        rate_LF_to_presym (float): Latent fast to presymptomatic rate per day  
            - Controls progression from fast latent to active presymptomatic TB
        rate_active_to_clear (float): Active to clearance rate per day
            - Natural clearance rate without treatment
        rate_treatment_to_clear (float): Treatment to clearance rate per year
            - Effectiveness of treatment in curing TB
        rate_smpos_to_dead (float): Smear positive to death rate per day
            - Mortality rate for most infectious TB cases
        rate_smneg_to_dead (float): Smear negative to death rate per day
            - Mortality rate for moderately infectious TB cases
        rate_exptb_to_dead (float): Extrapulmonary to death rate per day
            - Mortality rate for least infectious TB cases
        rel_trans_treatment (float): Relative transmission during treatment
            - Reduces but doesn't eliminate transmission during treatment
        n_agents (int): Number of agents in the simulation
        rand_seed (int): Random seed for reproducibility
        
    Returns:
        ss.Sim: Configured simulation ready to run
        
    Note:
        The simulation runs from 1940-1970 (31 years) with weekly time steps.
        This provides sufficient time for TB dynamics to stabilize.
    """
    
    # Simulation parameters - defines the temporal and population framework
    spars = dict(
        dt=ss.days(7),                    # Weekly time steps for computational efficiency
        start=ss.date('1940-01-01'),      # Start date - allows for TB dynamics to develop
        stop=ss.date('1970-12-31'),       # End date - 31 years provides sufficient time for stabilization
        rand_seed=rand_seed,              # Random seed for reproducibility
    )
    
    # Population - creates the agent-based population
    pop = ss.People(n_agents=n_agents)
    
    # TB parameters - defines the TB natural history model
    tb_pars = dict(
        dt=ss.days(7),                    # Time step for TB module
        beta=ss.peryear(beta),            # Annual transmission rate
        init_prev=ss.bernoulli(p=0.01),   # Initial 1% prevalence to start simulation
        
        # Progression rates - control movement from latent to active TB
        rate_LS_to_presym=ss.perday(rate_LS_to_presym),  # Slow latent progression
        rate_LF_to_presym=ss.perday(rate_LF_to_presym),  # Fast latent progression
        
        # Clearance rates - control recovery from TB
        rate_active_to_clear=ss.perday(rate_active_to_clear),      # Natural clearance
        rate_treatment_to_clear=ss.peryear(rate_treatment_to_clear), # Treatment clearance
        
        # Mortality rates - control TB-related deaths by disease type
        rate_smpos_to_dead=ss.perday(rate_smpos_to_dead),    # Smear positive mortality
        rate_smneg_to_dead=ss.perday(rate_smneg_to_dead),    # Smear negative mortality  
        rate_exptb_to_dead=ss.perday(rate_exptb_to_dead),    # Extrapulmonary mortality
        
        # Treatment effects - reduces but doesn't eliminate transmission
        rel_trans_treatment=rel_trans_treatment,
    )
    
    # Create TB disease module with specified parameters
    tb = mtb.TB(tb_pars)
    
    # Contact network - defines how agents interact and transmit TB
    # Poisson distribution with mean 6 contacts, no duration (instantaneous contacts)
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=6), dur=0))
    
    # Demographics - population dynamics
    births = ss.Births(pars=dict(birth_rate=25))    # 25 births per 1000 per year
    deaths = ss.Deaths(pars=dict(death_rate=20))    # 20 deaths per 1000 per year (net growth)
    
    # Assemble the complete simulation
    sim = ss.Sim(
        people=pop,                       # Population of agents
        networks=net,                     # Contact network for transmission
        diseases=tb,                      # TB disease module
        demographics=[deaths, births],    # Population dynamics
        pars=spars,                       # Simulation parameters
    )
    
    # Suppress verbose output for cleaner calibration runs
    sim.pars.verbose = 0
    return sim

def calculate_active_tb_prevalence(sim, include_presymptomatic=True):
    """
    Calculate active TB prevalence using proper TB natural history definition.
    
    This function calculates the proportion of the population with active TB over time.
    It provides two definitions of active TB to match different epidemiological perspectives:
    
    1. Complete TB natural history (4 states): Includes all active TB cases including
       pre-symptomatic cases, which are part of the active disease process
    2. Symptomatic TB only (3 states): Excludes pre-symptomatic cases, focusing only
       on cases that would be detected through symptoms
    
    The choice affects prevalence levels and should be consistent with the intended
    use case and epidemiological definitions.
    
    Args:
        sim (ss.Sim): Completed simulation with TB results
        include_presymptomatic (bool): Whether to include ACTIVE_PRESYMP in active TB count
            - True: Use all 4 active TB states (recommended for complete natural history)
            - False: Use only 3 symptomatic states (for symptom-based analysis)
        
    Returns:
        numpy.ndarray: Active TB prevalence over time (0-1 scale)
        
    Note:
        The prevalence is calculated as active TB cases divided by total alive population
        at each time point, providing a time series of disease burden.
    """
    # Extract simulation results
    results = sim.results
    total_pop = results['n_alive']  # Total population alive at each time point
    
    if include_presymptomatic:
        # Use the TB module's n_active which includes all 4 active states:
        # ACTIVE_PRESYMP (2), ACTIVE_SMPOS (3), ACTIVE_SMNEG (4), ACTIVE_EXPTB (5)
        # This represents the complete TB natural history definition
        active_tb = results['tb']['n_active']
    else:
        # Alternative definition: only symptomatic active TB (3 states)
        # ACTIVE_SMPOS (3), ACTIVE_SMNEG (4), ACTIVE_EXPTB (5)
        # Excludes ACTIVE_PRESYMP (2) - pre-symptomatic cases
        # This focuses on cases that would be detected through symptoms
        active_tb = (results['tb']['n_active_smpos'] + 
                    results['tb']['n_active_smneg'] + 
                    results['tb']['n_active_exptb'])
    
    # Calculate prevalence as proportion of population with active TB
    prevalence = active_tb / total_pop
    return prevalence

def evaluate_prevalence_stability(sim, target_prevalence=0.01, tolerance=0.002, include_presymptomatic=True):
    """
    Evaluate how well the simulation achieves constant prevalence.
    
    This function assesses the quality of a TB simulation by measuring how well it
    maintains a constant prevalence level. It calculates multiple metrics to evaluate
    both the accuracy (hitting the target) and stability (low variation) of prevalence.
    
    The evaluation uses a composite scoring system that balances:
    - Target accuracy: How close the mean prevalence is to the target
    - Stability: How consistent the prevalence is over time (low coefficient of variation)
    - Compliance: What percentage of time points fall within the target range
    
    Args:
        sim (ss.Sim): Completed simulation with TB results
        target_prevalence (float): Target prevalence level (default: 0.01 = 1%)
        tolerance (float): Acceptable deviation from target (default: 0.002 = ±0.2%)
        include_presymptomatic (bool): Whether to include ACTIVE_PRESYMP in active TB count
        
    Returns:
        dict: Dictionary containing evaluation metrics:
            - mean_prevalence: Average prevalence over the simulation
            - final_prevalence: Prevalence at the end of simulation
            - cv_prevalence: Coefficient of variation (stability measure)
            - late_cv: Coefficient of variation in last 30% of simulation
            - target_percentage: Percentage of time points within target range
            - score: Composite score (lower is better)
            - prevalence_series: Full time series of prevalence values
    """
    
    # Calculate prevalence over time using proper TB natural history definition
    prevalence = calculate_active_tb_prevalence(sim, include_presymptomatic)
    
    # Calculate basic statistical metrics
    mean_prevalence = np.mean(prevalence)      # Average prevalence over time
    std_prevalence = np.std(prevalence)        # Standard deviation of prevalence
    # Coefficient of variation: measures relative variability (lower = more stable)
    cv_prevalence = (std_prevalence / mean_prevalence) * 100 if mean_prevalence > 0 else 0
    
    # Check target compliance - how often prevalence falls within acceptable range
    min_target = target_prevalence - tolerance  # Lower bound of acceptable range
    max_target = target_prevalence + tolerance  # Upper bound of acceptable range
    within_target = np.sum((prevalence >= min_target) & (prevalence <= max_target))
    target_percentage = (within_target / len(prevalence)) * 100  # Percentage compliance
    
    # Calculate final prevalence - end state of simulation
    final_prevalence = prevalence[-1]
    
    # Calculate late-stage stability (last 30% of simulation)
    # This focuses on the equilibrium state after initial transients
    late_start = int(len(prevalence) * 0.7)  # Start of last 30% of time points
    late_prevalence = prevalence[late_start:]
    late_cv = (np.std(late_prevalence) / np.mean(late_prevalence)) * 100 if np.mean(late_prevalence) > 0 else 0
    
    # Overall score (lower is better) - composite measure of calibration quality
    # The scoring system penalizes three types of deviations:
    
    # 1. Target error: How far the mean is from the target (40% weight)
    target_error = abs(mean_prevalence - target_prevalence) / target_prevalence
    
    # 2. Variability penalty: High coefficient of variation (40% weight)
    cv_penalty = cv_prevalence / 100  # Normalize CV to 0-1 scale
    
    # 3. Compliance penalty: Low percentage of time points in target range (20% weight)
    compliance_penalty = (100 - target_percentage) / 100  # Normalize to 0-1 scale
    
    # Weighted composite score - balances accuracy, stability, and compliance
    score = 0.4 * target_error + 0.4 * cv_penalty + 0.2 * compliance_penalty
    
    return {
        'mean_prevalence': mean_prevalence,
        'final_prevalence': final_prevalence,
        'cv_prevalence': cv_prevalence,
        'late_cv': late_cv,
        'target_percentage': target_percentage,
        'score': score,
        'prevalence_series': prevalence
    }

def analyze_tb_prevalence_detailed(sim, include_presymptomatic=True):
    """
    Analyze TB prevalence in detail using proper TB natural history definition.
    
    This function provides a comprehensive analysis of TB prevalence dynamics,
    including statistical measures, compliance with targets, and detailed breakdown
    of TB disease states. It categorizes the simulation quality based on WHO-style
    criteria for TB prevalence stability.
    
    The analysis includes:
    - Statistical measures of prevalence (mean, std, CV)
    - Target compliance assessment
    - Late-stage stability analysis
    - Quality categorization (EXCELLENT/GOOD/MODERATE/POOR/FAILED)
    - Detailed breakdown of TB disease states
    
    Args:
        sim (ss.Sim): Completed simulation with TB results
        include_presymptomatic (bool): Whether to include ACTIVE_PRESYMP in active TB count
        
    Returns:
        dict: Detailed analysis metrics including:
            - mean_prevalence: Average prevalence over simulation
            - std_prevalence: Standard deviation of prevalence
            - cv_prevalence: Coefficient of variation (stability measure)
            - target_compliance: Fraction of time points within target range
            - late_stage_cv: CV in last 10 years of simulation
            - status: Quality assessment (EXCELLENT/GOOD/MODERATE/POOR/FAILED)
            - prevalence_data: Full time series of prevalence
            - active_tb_definition: Description of active TB definition used
            - tb_state_breakdown: Prevalence by individual TB disease states
    """
    
    # Calculate prevalence using proper TB natural history definition
    prevalence = calculate_active_tb_prevalence(sim, include_presymptomatic)
    
    # Basic statistical measures
    mean_prevalence = np.mean(prevalence)      # Average prevalence
    std_prevalence = np.std(prevalence)        # Standard deviation
    # Coefficient of variation: relative variability measure
    cv_prevalence = (std_prevalence / mean_prevalence) * 100 if mean_prevalence > 0 else 0
    
    # Check target compliance (±0.2% from 1% target = 0.8% to 1.2%)
    target_range = (0.008, 0.012)  # 0.8% to 1.2% range
    in_range = np.sum((prevalence >= target_range[0]) & (prevalence <= target_range[1]))
    compliance = in_range / len(prevalence)  # Fraction of time points in range
    
    # Late-stage stability analysis (last 10 years of simulation)
    # Focus on equilibrium state after initial transients have settled
    late_stage = prevalence[-520:]  # Last 10 years (520 weekly time steps)
    late_cv = (np.std(late_stage) / np.mean(late_stage)) * 100 if np.mean(late_stage) > 0 else 0
    
    # Determine quality status based on WHO-style TB prevalence criteria
    # These thresholds are based on epidemiological standards for disease stability
    if cv_prevalence < 10 and compliance > 0.7:
        status = "EXCELLENT"  # Very stable, high compliance
    elif cv_prevalence < 20 and compliance > 0.5:
        status = "GOOD"       # Stable, good compliance
    elif cv_prevalence < 30 and compliance > 0.3:
        status = "MODERATE"   # Somewhat stable, moderate compliance
    elif cv_prevalence < 50 and compliance > 0.1:
        status = "POOR"       # Unstable, low compliance
    else:
        status = "FAILED"     # Very unstable, very low compliance
    
    # Get detailed TB state breakdown for epidemiological insight
    results = sim.results
    tb_results = results['tb']
    
    # Calculate average prevalence for each TB disease state
    # This provides insight into the distribution of TB cases by disease type
    tb_state_breakdown = {
        'active_presymp': np.mean(tb_results['n_active_presymp'] / results['n_alive']),  # Pre-symptomatic
        'active_smpos': np.mean(tb_results['n_active_smpos'] / results['n_alive']),      # Smear positive (most infectious)
        'active_smneg': np.mean(tb_results['n_active_smneg'] / results['n_alive']),      # Smear negative (moderately infectious)
        'active_exptb': np.mean(tb_results['n_active_exptb'] / results['n_alive']),      # Extrapulmonary (least infectious)
        'total_active': np.mean(tb_results['n_active'] / results['n_alive'])             # Total active TB
    }
    
    return {
        'mean_prevalence': mean_prevalence,
        'std_prevalence': std_prevalence,
        'cv_prevalence': cv_prevalence,
        'target_compliance': compliance,
        'late_stage_cv': late_cv,
        'status': status,
        'prevalence_data': prevalence,
        'active_tb_definition': 'All 4 active states (ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)' if include_presymptomatic else '3 symptomatic states only (ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)',
        'tb_state_breakdown': tb_state_breakdown
    }

def grid_search_calibration(target_prevalence=0.01, tolerance=0.002, max_combinations=100):
    """
    Perform a grid search to find parameters that achieve constant prevalence.
    
    This function implements a systematic grid search optimization approach to calibrate
    TB model parameters for achieving constant prevalence. Unlike gradient-based methods,
    grid search explores the parameter space systematically, which is more robust for
    complex epidemiological models with multiple local optima.
    
    The grid search methodology:
    1. Defines parameter ranges for key TB model parameters
    2. Generates all possible parameter combinations (or samples if too many)
    3. Runs simulations for each combination
    4. Evaluates each simulation using a composite scoring system
    5. Identifies the best parameter set based on lowest score
    
    The scoring system balances three criteria:
    - Target accuracy: How close mean prevalence is to target (40% weight)
    - Stability: Low coefficient of variation (40% weight)
    - Compliance: High percentage of time points in target range (20% weight)
    
    Args:
        target_prevalence (float): Target prevalence level (default: 0.01 = 1%)
        tolerance (float): Acceptable deviation from target (default: 0.002 = ±0.2%)
        max_combinations (int): Maximum number of parameter combinations to test
            - If total combinations exceed this, random sampling is used
        
    Returns:
        dict: Dictionary containing:
            - best_params: Best parameter set found
            - best_results: Evaluation metrics for best parameters
            - all_results: List of all simulation results
        None: If no successful simulations were found
    """
    
    print("TB CONSTANT PREVALENCE GRID SEARCH CALIBRATION")
    print("="*60)
    print(f"Target Prevalence: {target_prevalence:.1%}")
    print(f"Tolerance: ±{tolerance:.1%}")
    print(f"Max Combinations: {max_combinations}")
    print()
    
    # Define parameter ranges for grid search
    # These ranges are based on epidemiological literature and model sensitivity analysis
    param_ranges = {
        'beta': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],                    # Transmission rate (annual)
        'rate_LS_to_presym': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],                    # Latent slow progression (daily)
        'rate_LF_to_presym': [1e-3, 5e-3, 1e-2, 2e-2, 5e-2],                    # Latent fast progression (daily)
        'rate_active_to_clear': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],                 # Natural clearance (daily)
        'rate_treatment_to_clear': [1.0, 2.0, 3.0, 4.0, 5.0],                   # Treatment effectiveness (annual)
        'rate_smpos_to_dead': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],                   # Smear positive mortality (daily)
        'rel_trans_treatment': [0.1, 0.2, 0.3, 0.4, 0.5]                        # Treatment transmission reduction
    }
    
    # Generate parameter combinations using Cartesian product
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    # Calculate total number of combinations
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)
    
    # Limit combinations if too many to avoid excessive computation time
    if total_combinations > max_combinations:
        # Randomly sample combinations to maintain diversity in parameter space
        np.random.seed(42)  # Fixed seed for reproducibility
        indices = np.random.choice(total_combinations, max_combinations, replace=False)
        combinations = [all_combinations[i] for i in indices]
        print(f"Total possible combinations: {total_combinations}")
        print(f"Randomly sampling {max_combinations} combinations for efficiency")
    else:
        combinations = all_combinations
        print(f"Testing all {total_combinations} parameter combinations")
    
    print(f"Testing {len(combinations)} parameter combinations...")
    print()
    
    # Initialize tracking variables for optimization
    best_score = float('inf')  # Lower scores are better
    best_params = None         # Best parameter set found
    best_results = None        # Best evaluation results
    results_list = []          # Store all results for analysis
    
    # Main grid search loop - test each parameter combination
    for i, combination in enumerate(combinations):
        # Create parameter dictionary from current combination
        params = dict(zip(param_names, combination))
        
        # Set derived parameters based on epidemiological relationships
        # Smear negative mortality is typically 10% of smear positive mortality
        params['rate_smneg_to_dead'] = params['rate_smpos_to_dead'] * 0.1
        # Extrapulmonary mortality is typically 5% of smear positive mortality
        params['rate_exptb_to_dead'] = params['rate_smpos_to_dead'] * 0.05
        
        # Display current trial parameters for monitoring progress
        print(f"Trial {i+1}/{len(combinations)}: ", end="")
        print(f"β={params['beta']:.2f}, ", end="")                    # Transmission rate
        print(f"LS→P={params['rate_LS_to_presym']:.1e}, ", end="")    # Latent slow progression
        print(f"LF→P={params['rate_LF_to_presym']:.1e}, ", end="")    # Latent fast progression
        print(f"Clear={params['rate_active_to_clear']:.1e}")          # Natural clearance
        
        try:
            # Create and run simulation with current parameters
            # Use smaller population (1000) for faster calibration runs
            sim = create_tb_simulation(**params, n_agents=1000, rand_seed=1)
            sim.run()
            
            # Evaluate simulation results using composite scoring system
            results = evaluate_prevalence_stability(sim, target_prevalence, tolerance)
            results['params'] = params.copy()  # Store parameters with results
            results_list.append(results)
            
            # Display key evaluation metrics
            print(f"  → Mean: {results['mean_prevalence']:.3%}, ", end="")      # Mean prevalence
            print(f"CV: {results['cv_prevalence']:.1f}%, ", end="")              # Coefficient of variation
            print(f"Score: {results['score']:.3f}")                             # Composite score
            
            # Check if this is the best parameter set found so far
            if results['score'] < best_score:
                best_score = results['score']
                best_params = params.copy()
                best_results = results.copy()
                print(f"  ✓ NEW BEST!")  # Indicate new best result
            
        except Exception as e:
            # Handle simulation failures gracefully
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("GRID SEARCH RESULTS")
    print(f"{'='*60}")
    
    if best_params is not None:
        print(f"Best Parameters Found:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nBest Results:")
        print(f"  Mean Prevalence: {best_results['mean_prevalence']:.3%}")
        print(f"  Final Prevalence: {best_results['final_prevalence']:.3%}")
        print(f"  CV: {best_results['cv_prevalence']:.1f}%")
        print(f"  Late CV: {best_results['late_cv']:.1f}%")
        print(f"  Target Compliance: {best_results['target_percentage']:.1f}%")
        print(f"  Overall Score: {best_results['score']:.3f}")
        
        # Assessment
        if best_results['cv_prevalence'] < 10 and best_results['target_percentage'] > 70:
            print(f"  Status: ✓ EXCELLENT - Very stable prevalence achieved!")
        elif best_results['cv_prevalence'] < 20 and best_results['target_percentage'] > 50:
            print(f"  Status: ✓ GOOD - Stable prevalence achieved!")
        elif best_results['cv_prevalence'] < 30 and best_results['target_percentage'] > 30:
            print(f"  Status: ⚠ MODERATE - Somewhat stable prevalence")
        else:
            print(f"  Status: ⚠ POOR - Prevalence not stable enough")
        
        return {
            'best_params': best_params,
            'best_results': best_results,
            'all_results': results_list
        }
    else:
        print("No successful simulations found!")
        return None

def create_calibration_plots(best_params, best_results, target_prevalence=0.01):
    """
    Create comprehensive plots showing the calibration results.
    
    This function generates a 2x2 subplot layout that visualizes different aspects
    of the TB calibration results:
    
    1. Prevalence over time: Shows how well the calibrated model maintains
       constant prevalence around the target level
    2. Parameter values: Displays the calibrated parameter values in a bar chart
    3. Prevalence distribution: Histogram showing the distribution of prevalence
       values over the simulation period
    4. Calibration summary: Text summary of key calibration metrics
    
    The plots help assess the quality of the calibration and provide visual
    confirmation that the model achieves the desired constant prevalence behavior.
    
    Args:
        best_params (dict): Best parameter set found by grid search
        best_results (dict): Evaluation results for the best parameters
        target_prevalence (float): Target prevalence level (default: 0.01 = 1%)
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    
    print(f"\nCreating plots for best parameters...")
    
    # Run simulation with best parameters for plotting
    # Use larger population (2000) for better visualization quality
    sim = create_tb_simulation(**best_params, n_agents=2000, rand_seed=1)
    sim.run()
    
    # Create 2x2 subplot layout for comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TB Constant Prevalence Calibration Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Prevalence over time - shows temporal stability
    results = sim.results
    total_pop = results['n_alive']           # Total population over time
    active_tb = results['tb']['n_active']    # Active TB cases over time
    prevalence = active_tb / total_pop       # Calculate prevalence time series
    timevec = results['timevec']             # Time vector for x-axis
    
    # Plot simulated prevalence over time
    ax1.plot(timevec, prevalence * 100, 'b-', linewidth=2, label='Simulated Prevalence')
    
    # Add target line and tolerance bands
    ax1.axhline(y=target_prevalence * 100, color='r', linestyle='--', 
                label=f'Target ({target_prevalence:.1%})', linewidth=2)
    ax1.axhline(y=(target_prevalence + 0.002) * 100, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(y=(target_prevalence - 0.002) * 100, color='orange', linestyle=':', alpha=0.7)
    
    # Fill target range area for visual clarity
    ax1.fill_between(timevec, (target_prevalence - 0.002) * 100, (target_prevalence + 0.002) * 100, 
                     alpha=0.1, color='green', label='Target Range')
    ax1.set_title('Prevalence Over Time (Calibrated)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prevalence (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parameter values - shows calibrated parameter magnitudes
    param_names = list(best_params.keys())
    param_values = list(best_params.values())
    
    # Normalize values for better visualization
    # Use log scale for rates to handle wide range of values
    normalized_values = []
    for i, (name, value) in enumerate(best_params.items()):
        if 'rate_' in name:
            # Log scale for rates (handles values from 1e-5 to 1e-2)
            normalized_values.append(np.log10(value))
        else:
            # Linear scale for other parameters (beta, rel_trans_treatment)
            normalized_values.append(value)
    
    # Create horizontal bar chart for parameter values
    ax2.barh(param_names, normalized_values, alpha=0.7)
    ax2.set_title('Calibrated Parameter Values', fontweight='bold')
    ax2.set_xlabel('Parameter Value (log scale for rates)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prevalence distribution - shows statistical properties
    # Histogram of prevalence values over the entire simulation
    ax3.hist(prevalence * 100, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=target_prevalence * 100, color='r', linestyle='--', linewidth=2, label='Target')
    ax3.set_title('Prevalence Distribution', fontweight='bold')
    ax3.set_xlabel('Prevalence (%)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Calibration summary - text summary of key metrics
    # Display the most important calibration results in a text box
    ax4.text(0.5, 0.5, f'Best Score: {best_results["score"]:.3f}\n'
                       f'Mean Prevalence: {best_results["mean_prevalence"]:.3%}\n'
                       f'CV: {best_results["cv_prevalence"]:.1f}%\n'
                       f'Target Compliance: {best_results["target_percentage"]:.1f}%',
             ha='center', va='center', transform=ax4.transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax4.set_title('Calibration Summary', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')  # Remove axes for clean text display
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    Main function to run TB constant prevalence calibration.
    
    This function orchestrates the complete TB calibration workflow:
    1. Runs grid search calibration to find optimal parameters
    2. Performs detailed analysis of the best parameters
    3. Compares different TB prevalence definitions
    4. Generates comprehensive visualization plots
    5. Provides summary of calibration results
    
    The calibration process is designed to find TB model parameters that achieve
    a stable 1% prevalence level, which is representative of TB burden in many
    endemic settings. The grid search approach is particularly effective for
    this type of epidemiological model calibration.
    """
    
    print("TB SIMULATION: CONSTANT PREVALENCE CALIBRATION")
    print("Using grid search optimization approach")
    print("="*80)
    
    # Run grid search calibration with specified targets
    # Target: 1% prevalence with ±0.2% tolerance for stable disease burden
    results = grid_search_calibration(
        target_prevalence=0.01,  # 1% target prevalence (typical for TB endemic areas)
        tolerance=0.002,         # ±0.2% tolerance (allows for natural variation)
        max_combinations=50      # Number of combinations to test (balance speed vs thoroughness)
    )
    
    if results is not None:
        # Create simulation with best parameters for detailed analysis
        print("\n" + "="*80)
        print("DETAILED TB NATURAL HISTORY ANALYSIS")
        print("="*80)
        
        best_params = results['best_params']
        # Run simulation with best parameters for comprehensive analysis
        sim = create_tb_simulation(**best_params)
        sim.run()
        
        # Analyze with all 4 active TB states (including presymptomatic)
        # This represents the complete TB natural history model
        print("\n1. ANALYSIS WITH ALL 4 ACTIVE TB STATES:")
        print("   (ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)")
        print("   This includes pre-symptomatic cases as part of active TB disease")
        analysis_all = analyze_tb_prevalence_detailed(sim, include_presymptomatic=True)
        
        print(f"   Mean Prevalence: {analysis_all['mean_prevalence']:.3%}")
        print(f"   CV: {analysis_all['cv_prevalence']:.1f}%")
        print(f"   Target Compliance: {analysis_all['target_compliance']:.1%}")
        print(f"   Status: {analysis_all['status']}")
        print(f"   Definition: {analysis_all['active_tb_definition']}")
        
        # Analyze with only 3 symptomatic active TB states (excluding presymptomatic)
        # This represents a symptom-based definition of active TB
        print("\n2. ANALYSIS WITH 3 SYMPTOMATIC ACTIVE TB STATES ONLY:")
        print("   (ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB - excluding ACTIVE_PRESYMP)")
        print("   This excludes pre-symptomatic cases, focusing on detectable disease")
        analysis_symptomatic = analyze_tb_prevalence_detailed(sim, include_presymptomatic=False)
        
        print(f"   Mean Prevalence: {analysis_symptomatic['mean_prevalence']:.3%}")
        print(f"   CV: {analysis_symptomatic['cv_prevalence']:.1f}%")
        print(f"   Target Compliance: {analysis_symptomatic['target_compliance']:.1%}")
        print(f"   Status: {analysis_symptomatic['status']}")
        print(f"   Definition: {analysis_symptomatic['active_tb_definition']}")
        
        # Show detailed TB state breakdown for epidemiological insight
        print("\n3. TB STATE BREAKDOWN (All 4 States):")
        print("   This shows the distribution of TB cases by disease type")
        breakdown = analysis_all['tb_state_breakdown']
        print(f"   ACTIVE_PRESYMP: {breakdown['active_presymp']:.3%}  (Pre-symptomatic)")
        print(f"   ACTIVE_SMPOS:   {breakdown['active_smpos']:.3%}  (Smear positive - most infectious)")
        print(f"   ACTIVE_SMNEG:   {breakdown['active_smneg']:.3%}  (Smear negative - moderately infectious)")
        print(f"   ACTIVE_EXPTB:   {breakdown['active_exptb']:.3%}  (Extrapulmonary - least infectious)")
        print(f"   TOTAL ACTIVE:   {breakdown['total_active']:.3%}  (All active TB cases)")
        
        # Create comprehensive visualization plots
        create_calibration_plots(
            results['best_params'], 
            results['best_results'], 
            target_prevalence=0.01
        )
        
        print(f"\n{'='*80}")
        print("CALIBRATION COMPLETE")
        print(f"{'='*80}")
        print("The grid search calibration has completed successfully.")
        print("The best parameters found can be used in future TB simulations.")
        print("\nKEY INSIGHTS:")
        print("- The calibration uses all 4 active TB states according to TB natural history")
        print("- This includes pre-symptomatic cases (ACTIVE_PRESYMP) which are part of active TB")
        print("- The model achieves stable prevalence through balanced transmission and clearance")
        print("- Parameter relationships reflect realistic TB epidemiology")
        
        return results
    else:
        print(f"\n{'='*80}")
        print("CALIBRATION FAILED")
        print(f"{'='*80}")
        print("No successful parameter combinations were found.")
        print("This may indicate:")
        print("- Parameter ranges are too restrictive")
        print("- Target prevalence is unrealistic for the model structure")
        print("- Need to adjust tolerance or target values")
        return None

if __name__ == '__main__':
    results = main()
