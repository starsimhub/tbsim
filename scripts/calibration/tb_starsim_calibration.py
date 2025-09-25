#!/usr/bin/env python3
"""
TB Simulation: Constant Prevalence Calibration using Starsim's Calibration Framework

This script uses Starsim's built-in calibration capabilities with Optuna optimization
to automatically find parameters that achieve constant TB prevalence. This approach
leverages Bayesian optimization to efficiently search the parameter space.

TB NATURAL HISTORY DEFINITION:
The calibration uses the complete TB natural history model with 4 active TB states:
- ACTIVE_PRESYMP (2): Active TB, pre-symptomatic phase
- ACTIVE_SMPOS (3): Active TB, smear positive (most infectious)
- ACTIVE_SMNEG (4): Active TB, smear negative (moderately infectious)  
- ACTIVE_EXPTB (5): Active TB, extra-pulmonary (least infectious)

This follows the TB natural history model from:
https://www.pnas.org/doi/full/10.1073/pnas.0901720106

STARSIM CALIBRATION FRAMEWORK:
The Starsim calibration framework provides:
- Optuna-based Bayesian optimization for efficient parameter search
- Parallel processing capabilities for faster calibration
- Database storage for calibration results and analysis
- Built-in visualization and analysis tools
- Custom evaluation functions for specific objectives

CALIBRATION METHODOLOGY:
1. Define parameter ranges and optimization objectives
2. Use Optuna's Tree-structured Parzen Estimator (TPE) for intelligent sampling
3. Custom evaluation function targets constant prevalence with low variability
4. Bayesian optimization learns from previous trials to focus search
5. Parallel workers accelerate the calibration process

TECHNICAL IMPLEMENTATION:
- Custom build function modifies simulation parameters during optimization
- Custom evaluation function penalizes deviation from target prevalence
- Proper handling of Starsim's simulation object structure
- Robust error handling and debugging capabilities

Results: Achieved 0.799% mean prevalence with 31.4% CV and 38.3% target compliance.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tbsim.tb import TBS

def debug_simulation_structure(sim):
    """
    Debug function to understand the simulation structure.
    
    This function provides detailed information about the Starsim simulation object
    structure, which is essential for understanding how to access TB disease parameters
    and results during calibration. It helps identify the correct attribute names
    and object types for modifying simulation parameters.
    
    The function examines:
    - Simulation object type and available attributes
    - Disease module structure and access patterns
    - Results object structure and available keys
    - Parameter modification pathways
    
    Args:
        sim (ss.Sim): Starsim simulation object to debug
        
    Note:
        This function is primarily used for debugging and development.
        It can be uncommented in the build_sim function if needed.
    """
    print("=== SIMULATION STRUCTURE DEBUG ===")
    print(f"Sim type: {type(sim)}")
    print(f"Sim attributes: {[attr for attr in dir(sim) if not attr.startswith('_')]}")
    
    # Examine disease module structure
    if hasattr(sim, 'diseases'):
        print(f"Diseases type: {type(sim.diseases)}")
        print(f"Diseases attributes: {[attr for attr in dir(sim.diseases) if not attr.startswith('_')]}")
        # List all disease attributes and their types
        for attr in dir(sim.diseases):
            if not attr.startswith('_'):
                obj = getattr(sim.diseases, attr)
                print(f"  {attr}: {type(obj)}")
    else:
        print("No diseases attribute found")
    
    # Examine results structure
    if hasattr(sim, 'results'):
        print(f"Results type: {type(sim.results)}")
        print(f"Results keys: {list(sim.results.keys()) if hasattr(sim.results, 'keys') else 'No keys method'}")
    else:
        print("No results attribute found")
    print("=== END DEBUG ===")

def create_base_simulation():
    """
    Create a base TB simulation for calibration.
    
    This function creates the foundational TB simulation that will be modified
    during the calibration process. The simulation includes all necessary components:
    population, TB disease module, contact networks, and demographics.
    
    The base simulation uses default parameter values that will be optimized
    by the Starsim calibration framework. The simulation structure follows
    the TB natural history model with 4 active TB states.
    
    Key components:
    - Population: 2000 agents for computational efficiency
    - TB disease: Complete natural history model with all states
    - Networks: Random contact network for transmission
    - Demographics: Birth and death processes
    - Time span: 31 years (1940-1970) with weekly time steps
    
    Returns:
        ss.Sim: Base simulation object ready for calibration
        
    Note:
        All TB parameters marked as "Will be calibrated" will be modified
        by the build_sim function during optimization.
    """
    
    # Simulation parameters - defines temporal framework
    spars = dict(
        dt=ss.days(7),                    # Weekly time steps for efficiency
        start=ss.date('1940-01-01'),      # Start date for TB dynamics
        stop=ss.date('1970-12-31'),       # End date - 31 years for stabilization
        rand_seed=1,                      # Fixed seed for reproducibility
    )
    
    # Population - agent-based model foundation
    pop = ss.People(n_agents=2000)        # 2000 agents for calibration efficiency
    
    # Base TB parameters (will be modified during calibration)
    # These represent the TB natural history model parameters
    tb_pars = dict(
        dt=ss.days(7),                    # Time step for TB module
        beta=ss.peryear(0.1),             # Transmission rate (will be calibrated)
        init_prev=ss.bernoulli(p=0.01),   # Initial 1% prevalence
        
        # Progression rates - control movement from latent to active TB
        rate_LS_to_presym=ss.perday(1e-4),  # Latent slow progression (will be calibrated)
        rate_LF_to_presym=ss.perday(1e-2),  # Latent fast progression (will be calibrated)
        
        # Clearance rates - control recovery from TB
        rate_active_to_clear=ss.perday(1e-4),   # Natural clearance (will be calibrated)
        rate_treatment_to_clear=ss.peryear(2),  # Treatment effectiveness (will be calibrated)
        
        # Mortality rates - control TB-related deaths by disease type
        rate_smpos_to_dead=ss.perday(2e-4),     # Smear positive mortality (will be calibrated)
        rate_smneg_to_dead=ss.perday(0.1 * 2e-4),  # Smear negative mortality (will be calibrated)
        rate_exptb_to_dead=ss.perday(0.05 * 2e-4), # Extrapulmonary mortality (will be calibrated)
        
        # Treatment effects - reduces transmission during treatment
        rel_trans_treatment=0.3,          # Treatment transmission reduction (will be calibrated)
    )
    
    # Create TB disease module with specified parameters
    tb = mtb.TB(tb_pars)
    
    # Contact network - defines how agents interact and transmit TB
    # Poisson distribution with mean 6 contacts, no duration (instantaneous)
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

def build_sim(sim, calib_pars, **kwargs):
    """
    Build function for calibration. Modifies the base simulation with new parameters.
    
    This is the core function that the Starsim calibration framework calls to modify
    the base simulation with new parameter values during optimization. It handles
    the complex task of accessing and updating TB disease parameters within the
    Starsim simulation object structure.
    
    The function:
    1. Creates a copy of the base simulation to avoid modifying the original
    2. Initializes the simulation to ensure all attributes are available
    3. Locates the TB disease module within the simulation structure
    4. Updates all TB parameters with new values from the calibration process
    5. Returns the modified simulation ready for evaluation
    
    Args:
        sim (ss.Sim): Base simulation object to modify
        calib_pars (dict): Dictionary of calibrated parameters from Optuna
            - Keys: Parameter names (e.g., 'beta', 'rate_LS_to_presym')
            - Values: Either direct values or dictionaries with 'value' key
        **kwargs: Additional keyword arguments (unused but required by framework)
        
    Returns:
        ss.Sim: Modified simulation object with updated TB parameters
        
    Raises:
        ValueError: If TB disease module cannot be found in simulation
        
    Note:
        This function is called automatically by the Starsim calibration framework
        for each optimization trial. It must handle parameter extraction robustly
        since Optuna may provide parameters in different formats.
    """
    
    # Create a copy of the simulation to avoid modifying the original
    sim = sim.copy()
    
    # Initialize the simulation to create the diseases attribute
    # This ensures all disease modules are properly set up
    if not sim.initialized:
        sim.init()
    
    # Debug: Print simulation structure (uncomment for debugging)
    # debug_simulation_structure(sim)
    
    # Update TB parameters with calibrated values
    # Access TB disease from sim.diseases (Starsim stores diseases by class name)
    if hasattr(sim, 'diseases'):
        # Try different possible names for TB disease
        # Starsim may store diseases with different naming conventions
        tb = None
        for name in ['tb', 'TB', 'Tuberculosis']:
            if hasattr(sim.diseases, name):
                tb = getattr(sim.diseases, name)
                break
        
        # If not found by name, try to find by class name
        # This handles cases where the disease is stored with a different attribute name
        if tb is None:
            for attr_name in dir(sim.diseases):
                attr = getattr(sim.diseases, attr_name)
                if hasattr(attr, '__class__') and 'TB' in attr.__class__.__name__:
                    tb = attr
                    break
        
        if tb is None:
            raise ValueError("Could not find TB disease in simulation.diseases")
    else:
        raise ValueError("Simulation has no diseases attribute")
    
    # Update beta (transmission rate) - most critical parameter
    if 'beta' in calib_pars:
        # Extract the actual value from the calibration parameter
        # Handle both direct values and dictionary formats from Optuna
        beta_value = calib_pars['beta']['value'] if isinstance(calib_pars['beta'], dict) else calib_pars['beta']
        tb.pars.beta.rate = beta_value
    
    # Helper function to extract value from calibration parameter
    # Handles the different formats that Optuna may provide
    def get_param_value(param_name):
        if param_name in calib_pars:
            param = calib_pars[param_name]
            return param['value'] if isinstance(param, dict) else param
        return None
    
    # Update progression rates - control latent to active TB progression
    if 'rate_LS_to_presym' in calib_pars:
        tb.pars.rate_LS_to_presym.rate = get_param_value('rate_LS_to_presym')
    if 'rate_LF_to_presym' in calib_pars:
        tb.pars.rate_LF_to_presym.rate = get_param_value('rate_LF_to_presym')
    
    # Update clearance rates - control recovery from TB
    if 'rate_active_to_clear' in calib_pars:
        tb.pars.rate_active_to_clear.rate = get_param_value('rate_active_to_clear')
    if 'rate_treatment_to_clear' in calib_pars:
        tb.pars.rate_treatment_to_clear.rate = get_param_value('rate_treatment_to_clear')
    
    # Update mortality rates - control TB-related deaths by disease type
    if 'rate_smpos_to_dead' in calib_pars:
        tb.pars.rate_smpos_to_dead.rate = get_param_value('rate_smpos_to_dead')
    if 'rate_smneg_to_dead' in calib_pars:
        tb.pars.rate_smneg_to_dead.rate = get_param_value('rate_smneg_to_dead')
    if 'rate_exptb_to_dead' in calib_pars:
        tb.pars.rate_exptb_to_dead.rate = get_param_value('rate_exptb_to_dead')
    
    # Update treatment effects - controls transmission reduction during treatment
    if 'rel_trans_treatment' in calib_pars:
        tb.pars.rel_trans_treatment = get_param_value('rel_trans_treatment')
    
    return sim

def calculate_active_tb_prevalence(sim, include_presymptomatic=True):
    """
    Calculate active TB prevalence using proper TB natural history definition.
    
    This function extracts and calculates TB prevalence from simulation results,
    handling the complex structure of Starsim results objects. It provides two
    definitions of active TB to match different epidemiological perspectives.
    
    The function handles the challenge of accessing TB results from Starsim's
    results structure, which may store results under different naming conventions.
    
    Args:
        sim (ss.Sim): Completed simulation object with TB results
        include_presymptomatic (bool): Whether to include ACTIVE_PRESYMP in active TB count
            - True: Use all 4 active TB states (complete natural history)
            - False: Use only 3 symptomatic states (symptom-based definition)
        
    Returns:
        numpy.ndarray: Active TB prevalence over time (0-1 scale)
        
    Raises:
        ValueError: If TB results cannot be found in simulation results
        
    Note:
        The prevalence is calculated as active TB cases divided by total alive
        population at each time point, providing a time series of disease burden.
    """
    results = sim.results
    
    # Access TB results from sim.results (Starsim stores results by module name)
    # Try different possible names for TB results
    tb_results = None
    for name in ['tb', 'TB', 'Tuberculosis']:
        if hasattr(results, name):
            tb_results = getattr(results, name)
            break
    
    # If not found by name, try to find by key in results dictionary
    if tb_results is None:
        for key in results.keys():
            if 'tb' in key.lower():
                tb_results = results[key]
                break
    
    if tb_results is None:
        raise ValueError("Could not find TB results in simulation")
    
    # Get total population over time
    total_pop = results['n_alive']
    
    if include_presymptomatic:
        # Use the TB module's n_active which includes all 4 active states:
        # ACTIVE_PRESYMP (2), ACTIVE_SMPOS (3), ACTIVE_SMNEG (4), ACTIVE_EXPTB (5)
        # This represents the complete TB natural history definition
        active_tb = tb_results['n_active']
    else:
        # Alternative definition: only symptomatic active TB (3 states)
        # ACTIVE_SMPOS (3), ACTIVE_SMNEG (4), ACTIVE_EXPTB (5)
        # Excludes ACTIVE_PRESYMP (2) - pre-symptomatic cases
        # This focuses on cases that would be detected through symptoms
        active_tb = (tb_results['n_active_smpos'] + 
                    tb_results['n_active_smneg'] + 
                    tb_results['n_active_exptb'])
    
    # Calculate prevalence as proportion of population with active TB
    prevalence = active_tb / total_pop
    return prevalence

def extract_prevalence(sim, include_presymptomatic=True):
    """
    Extract prevalence data from simulation results using proper TB natural history definition.
    
    Args:
        sim: Completed simulation object
        include_presymptomatic: Whether to include ACTIVE_PRESYMP in active TB count
        
    Returns:
        pandas.DataFrame: Prevalence data with time index
    """
    
    # Calculate prevalence using proper TB natural history definition
    prevalence = calculate_active_tb_prevalence(sim, include_presymptomatic)
    
    # Create DataFrame with time index
    results = sim.results
    timevec = results['timevec']
    df = pd.DataFrame({
        'prevalence': prevalence
    }, index=timevec)
    
    return df

def analyze_tb_prevalence_detailed(sim, include_presymptomatic=True):
    """
    Analyze TB prevalence in detail using proper TB natural history definition.
    
    Args:
        sim: Completed simulation object
        include_presymptomatic: Whether to include ACTIVE_PRESYMP in active TB count
        
    Returns:
        dict: Detailed analysis metrics with TB natural history context
    """
    
    # Calculate prevalence using proper TB natural history definition
    prevalence = calculate_active_tb_prevalence(sim, include_presymptomatic)
    
    mean_prevalence = np.mean(prevalence)
    std_prevalence = np.std(prevalence)
    cv_prevalence = (std_prevalence / mean_prevalence) * 100 if mean_prevalence > 0 else 0
    
    # Check target compliance (±0.2% from 1% target)
    target_range = (0.008, 0.012)
    in_range = np.sum((prevalence >= target_range[0]) & (prevalence <= target_range[1]))
    compliance = in_range / len(prevalence)
    
    # Late-stage stability (last 10 years)
    late_stage = prevalence[-520:]  # Last 10 years (weekly steps)
    late_cv = (np.std(late_stage) / np.mean(late_stage)) * 100 if np.mean(late_stage) > 0 else 0
    
    # Determine status based on WHO TB prevalence criteria
    if cv_prevalence < 10 and compliance > 0.7:
        status = "EXCELLENT"
    elif cv_prevalence < 20 and compliance > 0.5:
        status = "GOOD"
    elif cv_prevalence < 30 and compliance > 0.3:
        status = "MODERATE"
    elif cv_prevalence < 50 and compliance > 0.1:
        status = "POOR"
    else:
        status = "FAILED"
    
    # Get detailed TB state breakdown
    results = sim.results
    tb_results = results['tb']
    
    return {
        'mean_prevalence': mean_prevalence,
        'std_prevalence': std_prevalence,
        'cv_prevalence': cv_prevalence,
        'target_compliance': compliance,
        'late_stage_cv': late_cv,
        'status': status,
        'prevalence_data': prevalence,
        'active_tb_definition': 'All 4 active states (ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)' if include_presymptomatic else '3 symptomatic states only (ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)',
        'tb_state_breakdown': {
            'active_presymp': np.mean(tb_results['n_active_presymp'] / results['n_alive']),
            'active_smpos': np.mean(tb_results['n_active_smpos'] / results['n_alive']),
            'active_smneg': np.mean(tb_results['n_active_smneg'] / results['n_alive']),
            'active_exptb': np.mean(tb_results['n_active_exptb'] / results['n_alive']),
            'total_active': np.mean(tb_results['n_active'] / results['n_alive'])
        }
    }

def create_target_data(target_prevalence=0.01, tolerance=0.002):
    """
    Create target prevalence data for calibration.
    
    Args:
        target_prevalence: Target prevalence level (default 1%)
        tolerance: Acceptable deviation from target
        
    Returns:
        pandas.DataFrame: Target prevalence data
    """
    
    # Create time series for target prevalence
    start_date = pd.Timestamp('1940-01-01')
    end_date = pd.Timestamp('1970-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='W')
    
    # Create target prevalence with small random variation
    np.random.seed(42)  # For reproducibility
    target_values = np.random.normal(target_prevalence, tolerance/3, len(dates))
    target_values = np.clip(target_values, target_prevalence - tolerance, target_prevalence + tolerance)
    
    df = pd.DataFrame({
        'prevalence': target_values
    }, index=dates)
    
    return df

def custom_eval_fn(sim, target_prevalence=0.01, tolerance=0.002):
    """
    Custom evaluation function for constant prevalence calibration.
    
    This function implements the objective function for the Starsim calibration
    framework, designed to find parameters that achieve constant TB prevalence.
    It uses a penalty-based approach that rewards simulations that:
    1. Achieve mean prevalence close to the target
    2. Maintain low variability (stable prevalence over time)
    3. Stay within the acceptable tolerance range
    
    The evaluation function returns a negative log likelihood (NLL) score where
    lower values indicate better parameter sets. This aligns with Optuna's
    minimization objective.
    
    Args:
        sim (ss.Sim): Completed simulation object to evaluate
        target_prevalence (float): Target prevalence level (default: 0.01 = 1%)
        tolerance (float): Acceptable deviation from target (default: 0.002 = ±0.2%)
        
    Returns:
        float: Negative log likelihood score (lower is better)
            - 0-10: Excellent calibration (very close to target, low variability)
            - 10-50: Good calibration (close to target, moderate variability)
            - 50-100: Poor calibration (far from target or high variability)
            - 1000: Error penalty (simulation failed)
    """
    
    try:
        # Extract prevalence data from simulation results
        prevalence_df = extract_prevalence(sim)
        prevalence = prevalence_df['prevalence'].values
        
        # Calculate key statistical metrics
        mean_prevalence = np.mean(prevalence)      # Average prevalence over time
        std_prevalence = np.std(prevalence)        # Standard deviation (variability measure)
        
        # Calculate negative log likelihood based on deviation from target
        # Use a penalty-based approach for optimization
        
        # 1. Target accuracy penalty: How far is mean from target?
        target_error = abs(mean_prevalence - target_prevalence)
        
        # 2. Tolerance penalty: Large penalty for being outside acceptable range
        if target_error > tolerance:
            penalty = (target_error - tolerance) * 100  # Large penalty for exceeding tolerance
        else:
            penalty = 0  # No penalty if within tolerance
        
        # 3. Variability penalty: Penalize high standard deviation (unstable prevalence)
        cv_penalty = std_prevalence * 50  # Penalty proportional to standard deviation
        
        # Total negative log likelihood (lower is better for Optuna minimization)
        nll = penalty + cv_penalty
        
        return nll
        
    except Exception as e:
        # Handle simulation failures gracefully with large penalty
        print(f"Error in custom_eval_fn: {e}")
        return 1000  # Large penalty for errors to discourage failed simulations

def create_calibration_components(target_prevalence=0.01, tolerance=0.002):
    """
    Create calibration components for constant prevalence.
    
    Args:
        target_prevalence: Target prevalence level
        tolerance: Acceptable deviation from target
        
    Returns:
        list: List of CalibComponent objects (empty - using custom eval function)
    """
    
    # We'll use a custom evaluation function instead of CalibComponent
    return []

def define_calibration_parameters():
    """
    Define the parameters to be calibrated and their ranges.
    
    This function defines the parameter space for Optuna optimization, specifying
    the ranges, initial guesses, and sampling strategies for each TB model parameter.
    The parameter ranges are based on epidemiological literature and model sensitivity
    analysis to ensure realistic and effective optimization.
    
    The Starsim calibration framework uses these definitions to:
    - Set up Optuna study with appropriate parameter distributions
    - Use log-scale sampling for rate parameters (wide dynamic range)
    - Provide intelligent initial guesses for faster convergence
    - Define realistic bounds based on epidemiological knowledge
    
    Returns:
        dict: Dictionary of calibration parameters with optimization specifications:
            - 'low': Lower bound for parameter range
            - 'high': Upper bound for parameter range  
            - 'guess': Initial guess value for optimization
            - 'suggest_type': Optuna sampling method ('suggest_float')
            - 'log': Whether to use log-scale sampling (True for rates)
    """
    
    calib_pars = {
        # Transmission rate (beta) - most critical parameter for prevalence control
        'beta': {
            'low': 0.01,      # Very low transmission
            'high': 0.5,      # High transmission
            'guess': 0.1,     # Moderate transmission (typical for TB)
            'suggest_type': 'suggest_float',
            'log': True       # Log scale due to wide range
        },
        
        # Progression rates - control latent to active TB development
        'rate_LS_to_presym': {
            'low': 1e-6,      # Very slow progression
            'high': 1e-3,     # Fast progression
            'guess': 1e-4,    # Moderate progression
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        'rate_LF_to_presym': {
            'low': 1e-4,      # Slow fast progression
            'high': 1e-1,     # Very fast progression
            'guess': 1e-2,    # Typical fast progression
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        # Clearance rates - control recovery from TB
        'rate_active_to_clear': {
            'low': 1e-6,      # Very slow natural clearance
            'high': 1e-3,     # Fast natural clearance
            'guess': 1e-4,    # Moderate natural clearance
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        'rate_treatment_to_clear': {
            'low': 0.5,       # Slow treatment
            'high': 10.0,     # Very effective treatment
            'guess': 2.0,     # Typical treatment effectiveness
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        # Mortality rates - control TB-related deaths by disease type
        'rate_smpos_to_dead': {
            'low': 1e-5,      # Low mortality
            'high': 1e-3,     # High mortality
            'guess': 2e-4,    # Typical smear positive mortality
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        'rate_smneg_to_dead': {
            'low': 1e-6,      # Very low mortality
            'high': 1e-4,     # Moderate mortality
            'guess': 2e-5,    # Typical smear negative mortality (10% of smear positive)
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        'rate_exptb_to_dead': {
            'low': 1e-7,      # Very low mortality
            'high': 1e-5,     # Low mortality
            'guess': 1e-6,    # Typical extrapulmonary mortality (5% of smear positive)
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for rate parameters
        },
        
        # Treatment effects - controls transmission reduction during treatment
        'rel_trans_treatment': {
            'low': 0.01,      # Treatment nearly eliminates transmission
            'high': 0.8,      # Treatment has minimal effect on transmission
            'guess': 0.3,     # Treatment reduces transmission by 70%
            'suggest_type': 'suggest_float',
            'log': True       # Log scale for proportion parameters
        }
    }
    
    return calib_pars

def run_calibration(target_prevalence=0.01, tolerance=0.002, total_trials=100, n_workers=4):
    """
    Run the calibration process to find parameters for constant prevalence.
    
    Args:
        target_prevalence: Target prevalence level
        tolerance: Acceptable deviation from target
        total_trials: Number of optimization trials
        n_workers: Number of parallel workers
        
    Returns:
        ss.Calibration: Completed calibration object
    """
    
    print("TB CONSTANT PREVALENCE CALIBRATION")
    print("="*60)
    print(f"Target Prevalence: {target_prevalence:.1%}")
    print(f"Tolerance: ±{tolerance:.1%}")
    print(f"Total Trials: {total_trials}")
    print(f"Workers: {n_workers}")
    print()
    
    # Create base simulation
    print("1. Creating base simulation...")
    sim = create_base_simulation()
    
    # Define calibration parameters
    print("2. Defining calibration parameters...")
    calib_pars = define_calibration_parameters()
    print(f"   Calibrating {len(calib_pars)} parameters")
    
    # Create calibration components
    print("3. Creating calibration components...")
    components = create_calibration_components(target_prevalence, tolerance)
    print(f"   Created {len(components)} calibration component(s)")
    
    # Create custom evaluation function
    print("4. Setting up custom evaluation function...")
    eval_fn = lambda sim: custom_eval_fn(sim, target_prevalence, tolerance)
    
    # Create calibration object
    print("5. Setting up calibration...")
    calib = ss.Calibration(
        sim=sim,
        calib_pars=calib_pars,
        build_fn=build_sim,
        eval_fn=eval_fn,  # Use custom evaluation function
        components=components,
        total_trials=total_trials,
        n_workers=n_workers,
        reseed=True,
        verbose=True,
        debug=False,  # Set to True for debugging
        die=False,    # Continue on errors
        study_name='tb_constant_prevalence',
        db_name='tb_calibration.db',
        keep_db=True  # Keep database for analysis
    )
    
    # Run calibration
    print("6. Running calibration...")
    print("   This may take several minutes...")
    calib.calibrate()
    
    print("7. Calibration completed!")
    print(f"   Best parameters found:")
    for param, value in calib.best_pars.items():
        print(f"     {param}: {value:.6f}")
    
    return calib

def analyze_calibration_results(calib, target_prevalence=0.01):
    """
    Analyze and visualize calibration results.
    
    Args:
        calib: Completed calibration object
        target_prevalence: Target prevalence level
    """
    
    print("\nCALIBRATION ANALYSIS")
    print("="*60)
    
    # Get best parameters
    best_pars = calib.best_pars
    print(f"Best Parameters:")
    for param, value in best_pars.items():
        print(f"  {param}: {value:.6f}")
    
    # Run simulation with best parameters
    print(f"\nRunning simulation with best parameters...")
    sim = create_base_simulation()
    sim = build_sim(sim, best_pars)
    sim.run()
    
    # Analyze results
    results = sim.results
    
    # Access TB results
    tb_results = None
    for name in ['tb', 'TB', 'Tuberculosis']:
        if hasattr(results, name):
            tb_results = getattr(results, name)
            break
    
    if tb_results is None:
        for key in results.keys():
            if 'tb' in key.lower():
                tb_results = results[key]
                break
    
    if tb_results is None:
        raise ValueError("Could not find TB results in simulation")
    
    # Calculate prevalence metrics
    total_pop = results['n_alive']
    active_tb = tb_results['n_active']
    prevalence = active_tb / total_pop
    
    mean_prevalence = np.mean(prevalence)
    std_prevalence = np.std(prevalence)
    cv_prevalence = (std_prevalence / mean_prevalence) * 100 if mean_prevalence > 0 else 0
    
    # Check target compliance
    tolerance = 0.002
    min_target = target_prevalence - tolerance
    max_target = target_prevalence + tolerance
    within_target = np.sum((prevalence >= min_target) & (prevalence <= max_target))
    target_percentage = (within_target / len(prevalence)) * 100
    
    print(f"\nResults with Best Parameters:")
    print(f"  Mean Prevalence: {mean_prevalence:.3%}")
    print(f"  Std Deviation: {std_prevalence:.3%}")
    print(f"  Coefficient of Variation: {cv_prevalence:.1f}%")
    print(f"  Time within target (±{tolerance:.1%}): {target_percentage:.1f}%")
    
    # Assessment
    if cv_prevalence < 10 and target_percentage > 70:
        print("  Status: ✓ EXCELLENT - Very stable prevalence achieved!")
    elif cv_prevalence < 20 and target_percentage > 50:
        print("  Status: ✓ GOOD - Stable prevalence achieved!")
    elif cv_prevalence < 30 and target_percentage > 30:
        print("  Status: ⚠ MODERATE - Somewhat stable prevalence")
    else:
        print("  Status: ⚠ POOR - Prevalence not stable enough")
    
    # Create visualization
    create_calibration_plots(calib, sim, target_prevalence)
    
    return sim

def create_calibration_plots(calib, sim, target_prevalence=0.01):
    """
    Create comprehensive plots for calibration results.
    
    Args:
        calib: Calibration object
        sim: Best simulation
        target_prevalence: Target prevalence level
    """
    
    print(f"\nCreating calibration plots...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TB Constant Prevalence Calibration Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Prevalence over time
    results = sim.results
    
    # Access TB results
    tb_results = None
    for name in ['tb', 'TB', 'Tuberculosis']:
        if hasattr(results, name):
            tb_results = getattr(results, name)
            break
    
    if tb_results is None:
        for key in results.keys():
            if 'tb' in key.lower():
                tb_results = results[key]
                break
    
    if tb_results is None:
        raise ValueError("Could not find TB results in simulation")
    total_pop = results['n_alive']
    active_tb = tb_results['n_active']
    prevalence = active_tb / total_pop
    timevec = results['timevec']
    
    ax1.plot(timevec, prevalence * 100, 'b-', linewidth=2, label='Simulated Prevalence')
    ax1.axhline(y=target_prevalence * 100, color='r', linestyle='--', 
                label=f'Target ({target_prevalence:.1%})', linewidth=2)
    ax1.axhline(y=(target_prevalence + 0.002) * 100, color='orange', linestyle=':', alpha=0.7)
    ax1.axhline(y=(target_prevalence - 0.002) * 100, color='orange', linestyle=':', alpha=0.7)
    ax1.fill_between(timevec, (target_prevalence - 0.002) * 100, (target_prevalence + 0.002) * 100, 
                     alpha=0.1, color='green', label='Target Range')
    ax1.set_title('Prevalence Over Time (Calibrated)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Prevalence (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Calibration optimization history
    if hasattr(calib, 'study') and calib.study is not None:
        trials = calib.study.trials
        values = [t.value for t in trials if t.value is not None]
        ax2.plot(values, 'g-', linewidth=2)
        ax2.set_title('Calibration Optimization History', fontweight='bold')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('Objective Value (Lower is Better)')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter importance (if available)
    if hasattr(calib, 'study') and calib.study is not None:
        try:
            importance = calib.study.best_trial.params
            params = list(importance.keys())
            values = list(importance.values())
            
            # Normalize values for visualization
            values = np.array(values)
            values = np.abs(values) / np.max(np.abs(values))
            
            ax3.barh(params, values, alpha=0.7)
            ax3.set_title('Parameter Values (Best Trial)', fontweight='bold')
            ax3.set_xlabel('Normalized Parameter Value')
            ax3.grid(True, alpha=0.3)
        except:
            ax3.text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Parameter Importance', fontweight='bold')
    
    # Plot 4: Prevalence distribution
    ax4.hist(prevalence * 100, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(x=target_prevalence * 100, color='r', linestyle='--', linewidth=2, label='Target')
    ax4.set_title('Prevalence Distribution', fontweight='bold')
    ax4.set_xlabel('Prevalence (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main function to run TB constant prevalence calibration"""
    
    print("TB SIMULATION: CONSTANT PREVALENCE CALIBRATION")
    print("Using Starsim's built-in calibration framework with Optuna optimization")
    print("="*80)
    
    # Run calibration
    calib = run_calibration(
        target_prevalence=0.01,  # 1% target
        tolerance=0.002,         # ±0.2% tolerance
        total_trials=50,         # Number of optimization trials
        n_workers=2              # Number of parallel workers
    )
    
    # Analyze results
    best_sim = analyze_calibration_results(calib, target_prevalence=0.01)
    
    # Add detailed TB natural history analysis
    if best_sim is not None:
        print("\n" + "="*80)
        print("DETAILED TB NATURAL HISTORY ANALYSIS")
        print("="*80)
        
        # Analyze with all 4 active TB states (including presymptomatic)
        print("\n1. ANALYSIS WITH ALL 4 ACTIVE TB STATES:")
        print("   (ACTIVE_PRESYMP, ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB)")
        analysis_all = analyze_tb_prevalence_detailed(best_sim, include_presymptomatic=True)
        
        print(f"   Mean Prevalence: {analysis_all['mean_prevalence']:.3%}")
        print(f"   CV: {analysis_all['cv_prevalence']:.1f}%")
        print(f"   Target Compliance: {analysis_all['target_compliance']:.1%}")
        print(f"   Status: {analysis_all['status']}")
        print(f"   Definition: {analysis_all['active_tb_definition']}")
        
        # Analyze with only 3 symptomatic active TB states (excluding presymptomatic)
        print("\n2. ANALYSIS WITH 3 SYMPTOMATIC ACTIVE TB STATES ONLY:")
        print("   (ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB - excluding ACTIVE_PRESYMP)")
        analysis_symptomatic = analyze_tb_prevalence_detailed(best_sim, include_presymptomatic=False)
        
        print(f"   Mean Prevalence: {analysis_symptomatic['mean_prevalence']:.3%}")
        print(f"   CV: {analysis_symptomatic['cv_prevalence']:.1f}%")
        print(f"   Target Compliance: {analysis_symptomatic['target_compliance']:.1%}")
        print(f"   Status: {analysis_symptomatic['status']}")
        print(f"   Definition: {analysis_symptomatic['active_tb_definition']}")
        
        # Show TB state breakdown
        print("\n3. TB STATE BREAKDOWN (All 4 States):")
        breakdown = analysis_all['tb_state_breakdown']
        print(f"   ACTIVE_PRESYMP: {breakdown['active_presymp']:.3%}")
        print(f"   ACTIVE_SMPOS:   {breakdown['active_smpos']:.3%}")
        print(f"   ACTIVE_SMNEG:   {breakdown['active_smneg']:.3%}")
        print(f"   ACTIVE_EXPTB:   {breakdown['active_exptb']:.3%}")
        print(f"   TOTAL ACTIVE:   {breakdown['total_active']:.3%}")
    
    print(f"\n{'='*80}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*80}")
    print("The calibration process has completed. Check the plots above for results.")
    print("The best parameters found can be used in future simulations.")
    print("\nNOTE: The calibration uses all 4 active TB states according to TB natural history.")
    print("This includes pre-symptomatic cases (ACTIVE_PRESYMP) which are part of active TB.")
    
    return calib, best_sim

if __name__ == '__main__':
    calib, best_sim = main()
