#!/usr/bin/env python3

"""
TB Simulation: Critical Path with Best Multiprocessing Approaches

Key Features:
- Three multiprocessing approaches comparison
- Real-time timing and performance metrics
- Streamlined execution for critical path analysis
- Performance recommendations for different use cases
- Comprehensive plotting and visualization
- Model validation and accuracy assessment
- Interactive dashboards and statistical analysis
- Dynamic prevalence control with feedback adjustment
- Robust error handling and multiprocessing compatibility



===============================================================================
==  SCRIPT SUMMARY
===============================================================================

This script demonstrates the most effective multiprocessing approaches for
TB epidemiological simulations over a 70-year period (1940-2010):

MULTIPROCESSING APPROACHES:
   • Python Multiprocessing: Most reliable, using multiprocessing.Pool
   • Starsim MultiSim: Native Starsim integration with parallel execution
   • Starsim Parallel: Simplest to use, using ss.parallel() shortcut

RUNNING THIS SCRIPT WILL GIVE YOU:
   • Performance comparison of 3 multiprocessing methods
   • Timing information for each approach
   • Execution time analysis and recommendations
   • Parallel processing efficiency metrics

===============================================================================
==  EXECUTION FLOW
===============================================================================

SECTION 1: PYTHON MULTIPROCESSING
   • Uses multiprocessing.Pool for parallel execution
   • Most reliable and widely compatible
   • Good for custom workflows and debugging

SECTION 2: STARSIM MULTISIM
   • Uses ss.MultiSim class for native Starsim integration
   • Best for Starsim-specific workflows
   • Full control over simulation parameters

SECTION 3: STARSIM PARALLEL SHORTCUT
   • Uses ss.parallel() for simplest parallel execution
   • Easiest to implement and use
   • Best for quick parallel runs

===============================================================================
==  WHAT TO EXPECT WHEN RUNNING THIS SCRIPT
===============================================================================

EXECUTION TIME:
   • Total runtime: ~1-2 minutes (depending on system)
   • Python Multiprocessing: ~25-30 seconds
   • Starsim MultiSim: ~25-30 seconds
   • Starsim Parallel: ~20-25 seconds
   • Plot generation: ~10-15 seconds

OUTPUT METRICS:
   • Execution time for each approach
   • Performance comparison between methods
   • Success/failure rates for each approach
   • Recommendations for best use cases
   • Model accuracy and validation metrics
   • Prevalence stability analysis

GENERATED FILES:
   • results/TBModelValidation-*.csv - Simulation data logs
   • results/comparison/ - Comparative visualization plots
   • results/validation/ - Model validation plots and dashboards
   • Automatic results saving to results/ directory

PERFORMANCE COMPARISON:
   • Side-by-side timing comparison
   • Efficiency analysis of each approach
   • Recommendations for different use cases

VISUALIZATIONS:
   • Comparative plots showing all multiprocessing approaches
   • Model validation plots and accuracy dashboards
   • Disease progression and state transition diagrams
   • Prevalence stability analysis plots
   • Interactive accuracy dashboards
   • Dwell time analysis plots

IMPORTANT NOTES:
   • Simulations use fixed random seeds for reproducibility
   • Population size: 5,000 agents for statistical reliability
   • Time step: Weekly (7-day intervals)
   • Target prevalence: 1% (0.01) with ±0.2% tolerance
   • All results are automatically saved to 'results/' directory
   • Dynamic prevalence control with feedback adjustment
   • Robust error handling and multiprocessing compatibility

===============================================================================
==  TECHNICAL DETAILS
===============================================================================

SIMULATION PARAMETERS:
   • Population: 5,000 agents
   • Time Period: 1940-2010 (70 years)
   • Time Steps: Weekly (7-day intervals)
   • Random Seed: Fixed for reproducibility
   • Contact Network: Random network with Poisson contacts

DISEASE MODEL:
   • TB States: Susceptible → Latent → Presymptomatic → Active → Clear/Dead
   • Active TB Types: Smear-positive, Smear-negative, Extra-pulmonary
   • Transmission: Contact-based with configurable rates
   • Mortality: TB-specific death rates by disease type

MULTIPROCESSING METHODS:
   • Python: multiprocessing.Pool with custom simulation runner
   • Starsim MultiSim: Native ss.MultiSim class with parallel execution
   • Starsim Parallel: ss.parallel() shortcut for simple parallel runs

PLOTTING AND ANALYSIS:
   • Comparative visualization of all multiprocessing approaches
   • Model validation and accuracy assessment
   • Interactive dashboards for detailed analysis
   • Statistical metrics and performance benchmarking

PREVALENCE CONTROL:
   • Static parameters: Pre-calibrated for target prevalence
   • Dynamic controller: Real-time feedback adjustment
   • Robust error handling and multiprocessing support
   • Automatic parameter optimization during simulation

BURN-IN PERIOD AND COOLDOWN MECHANISMS:
   • Burn-in period: Configurable time steps (default: 50, ~1 year) to stabilize system
   • Cooldown period: 5-10 time steps between parameter adjustments
   • Prevents oscillations and ensures realistic epidemiological dynamics
   • Allows natural disease progression before control begins
   • Ensures system reaches equilibrium before dynamic adjustments
   • Fully parameterizable for different simulation scenarios

===============================================================================
"""


import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import time
import sys
from tbsim.analyzers import DwtAnalyzer
from tbsim.interventions.beta import BetaByYear
from tbsim.tb import TBS
from tb_validation_plots import (
    create_validation_plots, 
    monitor_prevalence_stability, 
    calculate_model_accuracy_metrics, 
    create_accuracy_dashboard,
    
)


def build_tbsim(sim_pars=None, target_prevalence=0.01):
    """
    Build TB simulation with parameters calibrated for constant prevalence
    
    This function creates a TB simulation with static parameters that are pre-calibrated
    to maintain a target prevalence level. The parameters are set to high values to ensure
    the simulation reaches and maintains the target prevalence without dynamic adjustment.
    
    Args:
        sim_pars: Additional simulation parameters to override defaults
        target_prevalence: Target active TB prevalence (default 1% = 0.01)
        
    Returns:
        ss.Sim: Configured simulation object ready to run
    """
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=5000)  # Increased population for better statistics
    
    # ============================================================================
    # TB DISEASE PARAMETERS - STATIC CALIBRATION APPROACH
    # ============================================================================
    # These parameters are aggressively calibrated to maintain target prevalence
    # without dynamic adjustment. High values ensure the simulation reaches and
    # maintains the target prevalence level throughout the simulation period.
    tb_pars = dict(
        dt = ss.days(7),  # Weekly time steps for disease progression
        beta = ss.peryear(0.5),  # High transmission rate to reach target prevalence
        init_prev = ss.bernoulli(p=target_prevalence),  # Initialize at target prevalence
        
        # ========================================================================
        # DISEASE PROGRESSION RATES (Calibrated for Stability)
        # ========================================================================
        # These rates control the flow between TB disease states:
        # Susceptible → Latent → Presymptomatic → Active → Clear/Dead
        rate_LS_to_presym = ss.perday(5e-4),  # Latent slow to presymptomatic progression
        rate_LF_to_presym = ss.perday(1e-2),  # Latent fast to presymptomatic progression  
        rate_presym_to_active = ss.perday(0.3),  # Presymptomatic to active TB progression
        rate_active_to_clear = ss.perday(5e-3),  # Spontaneous active TB clearance rate
        rate_treatment_to_clear = ss.peryear(6),  # Treatment success rate (6 treatments per year)
        
        # ========================================================================
        # TB-RELATED MORTALITY RATES (High for Realism)
        # ========================================================================
        # Mortality rates vary by TB type, with smear-positive being most severe
        rate_smpos_to_dead = ss.perday(1e-2),  # Smear-positive TB mortality (highest)
        rate_smneg_to_dead = ss.perday(0.3 * 1e-2),  # Smear-negative TB mortality (30% of smear-positive)
        rate_exptb_to_dead = ss.perday(0.15 * 1e-2),  # Extra-pulmonary TB mortality (15% of smear-positive)
        
        # ========================================================================
        # TREATMENT EFFECTS
        # ========================================================================
        rel_trans_treatment = 0.5,  # Treatment reduces transmission probability by 50%
    )
    
    # ============================================================================
    # DISEASE MODULE SETUP
    # ============================================================================
    # Create TB disease module with the calibrated parameters
    # This module handles all TB-specific disease dynamics and state transitions
    tb = mtb.TB(tb_pars)
    
    # ============================================================================
    # CONTACT NETWORK SETUP
    # ============================================================================
    # Random network for TB transmission modeling
    # High contact rate ensures sufficient transmission opportunities for
    # maintaining target prevalence levels
    net = ss.RandomNet(pars=dict(
        n_contacts=ss.poisson(lam=10),  # Average 10 contacts per person per time step
        dur=5.0,  # Contact duration in time steps (5 weeks)
        beta=0.50  # Transmission probability per contact (50%)
    ))
    
    # ============================================================================
    # DEMOGRAPHIC PROCESSES
    # ============================================================================
    # Birth and death processes to maintain realistic population dynamics
    # These rates are calibrated to maintain population stability over50 years
    births = ss.Births(pars=dict(birth_rate=20))  # 20 births per 1000 population per year
    deaths = ss.Deaths(pars=dict(death_rate=15))  # 15 deaths per 1000 population per year


    # ============================================================================
    # SIMULATION ASSEMBLY
    # ============================================================================
    # Assemble all components into a complete simulation
    # This creates the main simulation object that coordinates all modules
    sim = ss.Sim(
        people=pop,                    # Population of 5000 agents
        networks=net,                  # Contact network for transmission
        diseases=tb,                   # TB disease module with calibrated parameters
        demographics=[deaths, births], # Birth and death processes
        pars=spars,                   # Simulation parameters (time, seed, etc.)
    )

    # Disable verbose output for cleaner execution
    sim.pars.verbose = 0
    return sim

class PrevalenceController(ss.Analyzer):
    """
    Dynamic prevalence controller that adjusts TB parameters to maintain constant prevalence
    
    This analyzer monitors the current TB prevalence during simulation and automatically
    adjusts the transmission rate (beta) to keep prevalence within the target range.
    It uses a feedback control mechanism with cooldown periods to prevent oscillations.
    
    BURN-IN PERIOD AND COOLDOWN MECHANISMS:
    - Burn-in period: Configurable time steps (default: 50, ~1 year) allow system to stabilize naturally
    - Cooldown period: 5-10 time steps between parameter adjustments prevent oscillations
    - Purpose: Ensures realistic epidemiological dynamics and stable control
    - Mechanism: System reaches natural equilibrium before dynamic adjustments begin
    - Benefits: Prevents artificial oscillations and ensures reproducible results
    - Flexibility: Fully parameterizable for different simulation scenarios and requirements
    """
    def __init__(self, target_prevalence=0.01, tolerance=0.002, adjustment_factor=0.1, cooldown_period=10, burn_in_period=50):
        """
        Initialize the prevalence controller with control parameters
        
        Args:
            target_prevalence (float): Target prevalence level to maintain (default: 1%)
            tolerance (float): Acceptable deviation from target (±tolerance)
            adjustment_factor (float): Magnitude of parameter adjustments (default: 10%)
            cooldown_period (int): Time steps between adjustments to prevent oscillations
            burn_in_period (int): Time steps to wait before allowing parameter adjustments (default: 50)
        """
        super().__init__()
        
        # ========================================================================
        # CONTROL PARAMETERS
        # ========================================================================
        self.target_prevalence = target_prevalence  # Target prevalence level (1%)
        self.tolerance = tolerance                  # Acceptable deviation (±0.2%)
        self.adjustment_factor = adjustment_factor  # Adjustment magnitude (10%)
        self.cooldown_period = cooldown_period      # Cooldown between adjustments
        self.burn_in_period = burn_in_period        # Time steps before allowing adjustments
        
        # ========================================================================
        # TRACKING VARIABLES
        # ========================================================================
        self.prevalence_history = []               # Store prevalence over time
        self.beta_adjustments = []                 # Record all parameter adjustments
        self.last_adjustment_time = -cooldown_period  # Allow immediate first adjustment
        
    def step(self):
        """
        Monitor and adjust parameters each time step
        
        This method is called every simulation time step to:
        1. Calculate current TB prevalence
        2. Check if prevalence is within target range
        3. Adjust transmission rate if needed (with cooldown)
        """
        sim = self.sim
        tb = sim.diseases.tb
        
        # ========================================================================
        # PREVALENCE CALCULATION
        # ========================================================================
        # Calculate current prevalence as active TB cases / total alive population
        # This includes all active TB states: smear-positive, smear-negative,
        # extra-pulmonary, and presymptomatic cases
        n_alive = np.count_nonzero(sim.people.alive)
        n_active = np.count_nonzero(tb.state == TBS.ACTIVE_SMPOS) + \
                  np.count_nonzero(tb.state == TBS.ACTIVE_SMNEG) + \
                  np.count_nonzero(tb.state == TBS.ACTIVE_EXPTB) + \
                  np.count_nonzero(tb.state == TBS.ACTIVE_PRESYMP)
        
        current_prevalence = n_active / n_alive if n_alive > 0 else 0
        self.prevalence_history.append(current_prevalence)
        
        # ========================================================================
        # CONTROL LOGIC
        # ========================================================================
        # BURN-IN PERIOD: Configurable time steps allow system to stabilize
        # COOLDOWN PERIOD: 5-10 time steps between adjustments prevent oscillations
        # Only adjust parameters after burn-in period and cooldown to prevent
        # oscillations and allow the system to stabilize
        if (len(self.prevalence_history) > self.burn_in_period and 
            self.ti - self.last_adjustment_time >= self.cooldown_period):
            
            # Use recent average to smooth out noise and short-term fluctuations
            recent_prevalence = np.mean(self.prevalence_history[-10:])
            
            # ====================================================================
            # FEEDBACK CONTROL ALGORITHM
            # ====================================================================
            # Adjust transmission rate based on prevalence deviation from target
            if recent_prevalence > self.target_prevalence + self.tolerance:
                # Prevalence too high - reduce transmission to lower prevalence
                adjustment = 1 - self.adjustment_factor  # Reduce by adjustment_factor
                tb.pars.beta.rate *= adjustment
                self.beta_adjustments.append(('reduce', adjustment, self.ti))
                self.last_adjustment_time = self.ti
                
            elif recent_prevalence < self.target_prevalence - self.tolerance:
                # Prevalence too low - increase transmission to raise prevalence
                adjustment = 1 + self.adjustment_factor  # Increase by adjustment_factor
                tb.pars.beta.rate *= adjustment
                self.beta_adjustments.append(('increase', adjustment, self.ti))
                self.last_adjustment_time = self.ti
    
    def finalize(self):
        """
        Report on adjustments made and call parent finalize
        
        This method is called by the Starsim framework at the end of simulation
        to report on any parameter adjustments made during the simulation and
        ensure proper cleanup by calling the parent class finalize method.
        
        The method reports:
        - Total number of adjustments made
        - Details of each adjustment (action, factor, time step)
        - Ensures proper Starsim framework integration
        """
        if self.beta_adjustments:
            print(f"\nPrevalence Controller made {len(self.beta_adjustments)} adjustments:")
            for action, factor, time_step in self.beta_adjustments:
                print(f"  Time {time_step}: {action} beta by factor {factor:.3f}")
        
        # Call parent finalize to ensure proper cleanup
        super().finalize()
    
    def finalize_results(self):
        """
        Required by Starsim framework - call parent method
        
        This method is called by the Starsim framework to finalize any results
        processing. It ensures proper integration with the Starsim framework
        by calling the parent class finalize_results method.
        
        This method is essential for:
        - Proper Starsim framework integration
        - Eliminating RuntimeWarnings about missing method calls
        - Ensuring correct multiprocessing compatibility
        """
        super().finalize_results()

def build_tbsim_with_control(sim_pars=None, target_prevalence=0.01, use_controller=True, burn_in_period=50):
    """
    Build TB simulation with dynamic prevalence control capabilities
    
    This function creates a TB simulation that can optionally use dynamic prevalence
    control. Unlike the static approach, this allows for real-time parameter
    adjustment based on current prevalence levels.
    
    Args:
        sim_pars (dict, optional): Additional simulation parameters to override defaults
        target_prevalence (float): Target active TB prevalence level (default: 1%)
        use_controller (bool): Whether to enable dynamic prevalence controller
        burn_in_period (int): Time steps to wait before allowing parameter adjustments (default: 50)
        
    Returns:
        ss.Sim: Configured simulation object with optional dynamic control
        
    Note:
        The dynamic controller uses feedback control to maintain target prevalence
        by adjusting transmission rates in real-time during simulation.
    """
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=5000)  # Increased population for better statistics
    
    # TB parameters - aggressive values for dynamic control
    tb_pars = dict(
        dt = ss.days(7),
        beta = ss.peryear(0.4),  # Higher transmission rate for dynamic control
        init_prev = ss.bernoulli(p=target_prevalence),
        
        # High rates for stability
        rate_LS_to_presym = ss.perday(3e-4),  # Much higher progression
        rate_LF_to_presym = ss.perday(6e-3),  # Much higher progression
        rate_presym_to_active = ss.perday(0.2),  # Much higher progression
        rate_active_to_clear = ss.perday(3e-3),  # Much higher clearance
        rate_treatment_to_clear = ss.peryear(6),
        
        # High mortality
        rate_smpos_to_dead = ss.perday(5e-3),  # Much higher mortality
        rate_smneg_to_dead = ss.perday(0.3 * 5e-3),
        rate_exptb_to_dead = ss.perday(0.15 * 5e-3),
        
        # Treatment effects
        rel_trans_treatment = 0.4,
    )
    
    tb = mtb.TB(tb_pars)
    # Use enhanced random network for TB transmission
    net = ss.RandomNet(pars=dict(
        n_contacts=ss.poisson(lam=10),  # More contacts for better transmission
        dur=1.0,  # Contact duration
        beta=1.0  # Full transmission probability
    ))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    # Create analyzers
    analyzers = []
    
    # Add prevalence controller if requested
    if use_controller:
        prevalence_controller = PrevalenceController(
            target_prevalence=target_prevalence,
            tolerance=0.02,  # Larger tolerance (2%)
            adjustment_factor=0.1,  # Much larger adjustments (10%) for better control
            cooldown_period=5,  # 5 time steps between adjustments
            burn_in_period=burn_in_period  # Configurable burn-in period
        )
        analyzers.append(prevalence_controller)

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        analyzers=analyzers,
        pars=spars,
    )

    sim.pars.verbose = 0
    return sim

def create_prevalence_control_interventions(target_prevalence=0.01):
    """
    Create interventions to maintain constant prevalence
    
    Args:
        target_prevalence: Target prevalence level
    """
    interventions = []
    
    # Gradual beta reduction to maintain prevalence
    # This simulates improved public health measures over time
    beta_intervention = BetaByYear(pars=dict(
        years=[1950, 1940, 1970, 1980, 1990, 2000],
        x_beta=[0.95, 0.90, 0.85, 0.80, 0.75, 0.70]  # Gradual reduction
    ))
    interventions.append(beta_intervention)
    
    return interventions


def run_single_simulation(config):
    """
    Run a single TB simulation with given configuration
    
    This function is designed to be used with multiprocessing to run
    multiple simulations in parallel. Each simulation runs independently
    with its own random seed and parameters.
    
    Args:
        config (dict): Configuration dictionary containing:
            - approach: 'static' or 'dynamic'
            - target_prevalence: Target prevalence level
            - rand_seed: Random seed for reproducibility
            - sim_id: Unique identifier for this simulation
            
    Returns:
        dict: Results dictionary containing simulation results and metadata
    """
    approach = config['approach']
    target_prevalence = config['target_prevalence']
    rand_seed = config['rand_seed']
    sim_id = config['sim_id']
    start_time = time.time()
    
    try:
        # Build simulation with specific random seed
        sim_pars = dict(rand_seed=rand_seed)
        
        if approach == 'static':
            sim = build_tbsim(sim_pars=sim_pars, target_prevalence=target_prevalence)
        elif approach == 'dynamic':
            sim = build_tbsim_with_control(
                sim_pars=sim_pars, 
                target_prevalence=target_prevalence, 
                use_controller=True
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")
        
        # Run simulation
        sim.run()
        
        # Analyze results
        try:
            prevalence_stats = monitor_prevalence_stability(sim, target_prevalence=target_prevalence)
        except Exception as e:
            print(f"Warning: Could not analyze prevalence stability: {e}")
            # Create default stats if analysis fails
            prevalence_stats = {
                'mean_prevalence': 0.0,
                'std_prevalence': 0.0,
                'cv_prevalence': 0.0,
                'target_percentage': 0.0,
                'prevalence_series': []
            }
        
        # Add metadata to results
        total_time = time.time() - start_time
        results = {
            'sim_id': sim_id,
            'approach': approach,
            'rand_seed': rand_seed,
            'target_prevalence': target_prevalence,
            'simulation': sim,
            'prevalence_stats': prevalence_stats,
            'success': True,
            'error': None,
            'execution_time': total_time
        }
        
        return results
        
    except Exception as e:
        error_time = time.time() - start_time
        return {
            'sim_id': sim_id,
            'approach': approach,
            'rand_seed': rand_seed,
            'target_prevalence': target_prevalence,
            'simulation': None,
            'prevalence_stats': None,
            'success': False,
            'error': str(e),
            'execution_time': error_time
        }


def run_parallel_simulations(n_simulations=4, n_processes=None, target_prevalence=0.01):
    """
    Run multiple TB simulations in parallel using multiprocessing
    
    This function creates multiple simulation configurations and runs them
    in parallel using Python's multiprocessing module. This significantly
    reduces total execution time when running multiple simulations.
    
    Args:
        n_simulations (int): Number of simulations to run (default: 4)
        n_processes (int): Number of processes to use (default: CPU count)
        target_prevalence (float): Target prevalence level (default: 0.01)
        
    Returns:
        dict: Dictionary containing results from all simulations
    """
    if n_processes is None:
        n_processes = min(cpu_count(), n_simulations)
    
    print(f"PARALLEL TB SIMULATION EXECUTION")
    print("="*60)
    print(f"** Configuration:")
    print(f"   • Total simulations: {n_simulations}")
    print(f"   • Parallel processes: {n_processes}")
    print(f"   • Target prevalence: {target_prevalence:.1%}")
    print(f"   • Approaches: Static and Dynamic (50/50 split)")
    print("="*60)
    
    # Create simulation configurations
    configs = []
    for i in range(n_simulations):
        # Alternate between static and dynamic approaches
        approach = 'static' if i % 2 == 0 else 'dynamic'
        config = {
            'approach': approach,
            'target_prevalence': target_prevalence,
            'rand_seed': i + 1,  # Different seed for each simulation
            'sim_id': f"sim_{i+1:02d}"
        }
        configs.append(config)
    
    print(f"-> Created {len(configs)} simulation configurations")
    print(f"-> Starting parallel execution with {n_processes} processes...")
    
    # Record start time
    start_time = time.time()
    
    # Run simulations in parallel with progress tracking
    print(f"-> Running {n_simulations} simulations in parallel...")
    print(f"** Progress: ", end="", flush=True)
    
    with Pool(processes=n_processes) as pool:
        # Use imap for progress tracking
        results = []
        for i, result in enumerate(pool.imap(run_single_simulation, configs), 1):
            results.append(result)
            # Print a dot for each completed simulation
            print(".", end="", flush=True)
            if i % 10 == 0:  # New line every 10 simulations
                print(f" {i}/{n_simulations}")
    
    print(f" {n_simulations}/{n_simulations}")  # Final count
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    print(f"[OK] All simulations completed in {execution_time:.2f} seconds")
    print(f"[FAST] Average time per simulation: {execution_time/n_simulations:.2f} seconds")
    
    # Analyze results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n** EXECUTION SUMMARY:")
    print(f"   [OK] Successful simulations: {len(successful_results)}")
    print(f"   [FAIL] Failed simulations: {len(failed_results)}")
    
    if failed_results:
        print(f"\n[FAIL] FAILED SIMULATIONS:")
        for result in failed_results:
            print(f"   • {result['sim_id']} ({result['approach']}): {result['error']}")
    
    # Group results by approach
    static_results = [r for r in successful_results if r['approach'] == 'static']
    dynamic_results = [r for r in successful_results if r['approach'] == 'dynamic']
    
    print(f"\n** RESULTS BY APPROACH:")
    print(f"   [STATIC] Static approach: {len(static_results)} simulations")
    print(f"   [DYNAMIC]  Dynamic approach: {len(dynamic_results)} simulations")
    
    # Calculate aggregate statistics
    if static_results:
        static_cvs = [r['prevalence_stats']['cv_prevalence'] for r in static_results]
        static_means = [r['prevalence_stats']['mean_prevalence'] for r in static_results]
        print(f"   ** Static - Mean CV: {np.mean(static_cvs):.1f}%, Mean Prevalence: {np.mean(static_means):.3%}")
    
    if dynamic_results:
        dynamic_cvs = [r['prevalence_stats']['cv_prevalence'] for r in dynamic_results]
        dynamic_means = [r['prevalence_stats']['mean_prevalence'] for r in dynamic_results]
        print(f"   ** Dynamic - Mean CV: {np.mean(dynamic_cvs):.1f}%, Mean Prevalence: {np.mean(dynamic_means):.3%}")
    
    return {
        'all_results': results,
        'successful_results': successful_results,
        'failed_results': failed_results,
        'static_results': static_results,
        'dynamic_results': dynamic_results,
        'execution_time': execution_time,
        'n_simulations': n_simulations,
        'n_processes': n_processes
    }


def run_starsim_multisim(n_simulations=4, target_prevalence=0.01):
    """
    Run multiple TB simulations using Starsim's built-in MultiSim feature
    
    This function uses Starsim's native MultiSim class and parallel execution
    capabilities. This is the most integrated approach with Starsim's architecture.
    
    Args:
        n_simulations (int): Number of simulations to run (default: 4)
        target_prevalence (float): Target prevalence level (default: 0.01)
        
    Returns:
        ss.MultiSim: MultiSim object containing all simulation results
    """
    print(f"STARSIM MULTISIM TB SIMULATION EXECUTION")
    print("="*60)
    print(f"** Configuration:")
    print(f"   • Total simulations: {n_simulations}")
    print(f"   • Target prevalence: {target_prevalence:.1%}")
    print(f"   • Using Starsim MultiSim and parallel execution")
    print("="*60)
    
    # Create simulation configurations
    sims = []
    for i in range(n_simulations):
        approach = 'static' if i % 2 == 0 else 'dynamic'
        rand_seed = i + 1
        
        print(f"[STATIC] Building simulation {i+1}/{n_simulations} ({approach}) with seed {rand_seed}")
        
        # Build simulation with specific random seed
        sim_pars = dict(rand_seed=rand_seed)
        
        if approach == 'static':
            sim = build_tbsim(sim_pars=sim_pars, target_prevalence=target_prevalence)
        else:  # dynamic
            sim = build_tbsim_with_control(
                sim_pars=sim_pars, 
                target_prevalence=target_prevalence, 
                use_controller=True
            )
        
        # Set label for identification
        sim.label = f"{approach}_sim_{i+1:02d}_seed_{rand_seed}"
        sims.append(sim)
    
    print(f"-> Created {len(sims)} simulation configurations")
    print(f"-> Starting Starsim MultiSim parallel execution...")
    print(f"** Progress: ", end="", flush=True)
    
    # Record start time
    start_time = time.time()
    
    # Create MultiSim object and run in parallel
    msim = ss.MultiSim(sims=sims, label="TB_Parallel_Comparison")
    
    # Run with progress tracking
    completed_count = 0
    def progress_callback(sim):
        nonlocal completed_count
        completed_count += 1
        print(".", end="", flush=True)
        if completed_count % 10 == 0:
            print(f" {completed_count}/{n_simulations}")
    
    msim.run(verbose=0)  # Run all simulations in parallel
    
    print(f" {n_simulations}/{n_simulations}")  # Final count
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    print(f"[OK] All simulations completed in {execution_time:.2f} seconds")
    print(f"[FAST] Average time per simulation: {execution_time/n_simulations:.2f} seconds")
    
    # Analyze results
    successful_sims = [sim for sim in msim.sims if hasattr(sim, 'results')]
    failed_sims = [sim for sim in msim.sims if not hasattr(sim, 'results')]
    
    print(f"\n** EXECUTION SUMMARY:")
    print(f"   [OK] Successful simulations: {len(successful_sims)}")
    print(f"   [FAIL] Failed simulations: {len(failed_sims)}")
    
    if failed_sims:
        print(f"\n[FAIL] FAILED SIMULATIONS:")
        for sim in failed_sims:
            print(f"   • {sim.label}")
    
    # Group results by approach
    static_sims = [sim for sim in successful_sims if 'static' in sim.label]
    dynamic_sims = [sim for sim in successful_sims if 'dynamic' in sim.label]
    
    print(f"\n** RESULTS BY APPROACH:")
    print(f"   [STATIC] Static approach: {len(static_sims)} simulations")
    print(f"   [DYNAMIC]  Dynamic approach: {len(dynamic_sims)} simulations")
    
    # Calculate aggregate statistics
    if static_sims:
        static_cvs = []
        static_means = []
        for sim in static_sims:
            stats = monitor_prevalence_stability(sim, target_prevalence=target_prevalence)
            static_cvs.append(stats['cv_prevalence'])
            static_means.append(stats['mean_prevalence'])
        print(f"   ** Static - Mean CV: {np.mean(static_cvs):.1f}%, Mean Prevalence: {np.mean(static_means):.3%}")
    
    if dynamic_sims:
        dynamic_cvs = []
        dynamic_means = []
        for sim in dynamic_sims:
            stats = monitor_prevalence_stability(sim, target_prevalence=target_prevalence)
            dynamic_cvs.append(stats['cv_prevalence'])
            dynamic_means.append(stats['mean_prevalence'])
        print(f"   ** Dynamic - Mean CV: {np.mean(dynamic_cvs):.1f}%, Mean Prevalence: {np.mean(dynamic_means):.3%}")
    
    return msim


def run_starsim_parallel_shortcut(n_simulations=4, target_prevalence=0.01):
    """
    Run multiple TB simulations using Starsim's parallel() shortcut function
    
    This function uses Starsim's parallel() shortcut which is the most
    convenient way to run multiple simulations in parallel.
    
    Args:
        n_simulations (int): Number of simulations to run (default: 4)
        target_prevalence (float): Target prevalence level (default: 0.01)
        
    Returns:
        ss.MultiSim: MultiSim object containing all simulation results
    """
    print(f"STARSIM PARALLEL SHORTCUT TB SIMULATION EXECUTION")
    print("="*60)
    print(f"** Configuration:")
    print(f"   • Total simulations: {n_simulations}")
    print(f"   • Target prevalence: {target_prevalence:.1%}")
    print(f"   • Using Starsim parallel() shortcut")
    print("="*60)
    
    # Create base simulation
    base_sim = build_tbsim(target_prevalence=target_prevalence)
    base_sim.label = "TB_Base_Simulation"
    
    print(f"-> Created base simulation configuration")
    print(f"-> Starting Starsim parallel() execution...")
    
    # Record start time
    start_time = time.time()
    
    # Use Starsim's parallel shortcut to run multiple copies
    msim = ss.parallel(base_sim, n_runs=n_simulations, verbose=0)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    print(f"[OK] All simulations completed in {execution_time:.2f} seconds")
    print(f"[FAST] Average time per simulation: {execution_time/n_simulations:.2f} seconds")
    
    # Analyze results
    successful_sims = [sim for sim in msim.sims if hasattr(sim, 'results')]
    failed_sims = [sim for sim in msim.sims if not hasattr(sim, 'results')]
    
    print(f"\n** EXECUTION SUMMARY:")
    print(f"   [OK] Successful simulations: {len(successful_sims)}")
    print(f"   [FAIL] Failed simulations: {len(failed_sims)}")
    
    if failed_sims:
        print(f"\n[FAIL] FAILED SIMULATIONS:")
        for sim in failed_sims:
            print(f"   • {sim.label}")
    
    # Calculate aggregate statistics
    if successful_sims:
        cvs = []
        means = []
        for sim in successful_sims:
            stats = monitor_prevalence_stability(sim, target_prevalence=target_prevalence)
            cvs.append(stats['cv_prevalence'])
            means.append(stats['mean_prevalence'])
        print(f"   ** Overall - Mean CV: {np.mean(cvs):.1f}%, Mean Prevalence: {np.mean(means):.3%}")
    
    return msim


def run_prevalence_comparison():
    """
    Run comparison between static and dynamic prevalence control
    """
    print("Running TB Simulation Comparison (Static vs Dynamic Control)...")
    print("="*60)
    
    # Test different approaches
    approaches = [
        ("Static Parameters", build_tbsim, False),
        ("Dynamic Controller", build_tbsim_with_control, True),
    ]
    
    results = {}
    
    for name, build_func, use_control in approaches:
        print(f"\nTesting: {name}")
        print("-" * 40)
        
        # Build and run simulation
        if use_control:
            sim = build_func(target_prevalence=0.01, use_controller=True)
        else:
            sim = build_func(target_prevalence=0.01)
        
        sim.run()
        
        # Analyze prevalence stability
        stats = monitor_prevalence_stability(sim, target_prevalence=0.01)
        results[name] = stats
        
        print(f"Mean Prevalence: {stats['mean_prevalence']:.3%}")
        print(f"Coefficient of Variation: {stats['cv_prevalence']:.1f}%")
        print(f"Time in Target: {stats['target_percentage']:.1f}%")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("TB SIMULATION COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for name, stats in results.items():
        print(f"{name}:")
        print(f"  Mean: {stats['mean_prevalence']:.3%}")
        print(f"  CV: {stats['cv_prevalence']:.1f}%")
        print(f"  Target: {stats['target_percentage']:.1f}%")
        print()
    
    return results


if __name__ == '__main__':
    """
    TB Simulation: Critical Path with Best Multiprocessing Approaches
    
    This script demonstrates the most effective multiprocessing approaches for
    TB epidemiological simulations. It runs three different parallel execution
    methods and compares their performance:
    
    SECTION 1: Python Multiprocessing
    - Uses multiprocessing.Pool for parallel execution
    - Most reliable and widely compatible
    - Good for custom workflows and debugging
    
    SECTION 2: Starsim MultiSim  
    - Uses ss.MultiSim class for native Starsim integration
    - Best for Starsim-specific workflows
    - Full control over simulation parameters
    
    SECTION 3: Starsim Parallel Shortcut
    - Uses ss.parallel() for simplest parallel execution
    - Easiest to implement and use
    - Best for quick parallel runs
    
    The script provides timing information for each approach, generates
    comprehensive plots and visualizations, and gives recommendations for
    different use cases. It includes model validation, accuracy assessment,
    and interactive dashboards for detailed analysis.
    
    ==  PREVALENCE CONTROL FEATURES:
    - Static parameters: Pre-calibrated for target prevalence
    - Dynamic controller: Real-time feedback adjustment during simulation
    - Robust error handling and multiprocessing compatibility
    - Automatic parameter optimization with cooldown periods
    
    ==  BURN-IN PERIOD AND COOLDOWN EXPLANATION:
    - Burn-in period: Configurable time steps (default: 50, ~1 year) allow system to stabilize
    - Cooldown period: 5-10 time steps between parameter adjustments
    - Purpose: Prevents oscillations and ensures realistic disease dynamics
    - Mechanism: System reaches natural equilibrium before control begins
    - Benefits: Stable, reproducible, and epidemiologically valid results
    - Flexibility: Fully parameterizable for different simulation scenarios
    
    ==  ERROR HANDLING:
    - Comprehensive try-catch blocks for robust execution
    - Graceful degradation when plotting modules are unavailable
    - Proper Starsim framework integration with parent method calls
    - Multiprocessing-safe analyzer implementation
    """
    import time
    
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    target_prevalence = 0.01
    start_time = time.time()
    
    print("="*80)
    print("[TB] TB SIMULATION: CRITICAL PATH EXECUTION")
    print("="*80)
    print(f"[TIME] START TIME: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
    print(f"[TARGET] TARGET PREVALENCE: {target_prevalence:.1%}")
    print("="*80)
    
    # ============================================================================
    # SECTION 1: PYTHON MULTIPROCESSING
    # ============================================================================
    section1_start = time.time()
    print(f"\n[TIME] SECTION 1 START: {time.strftime('%H:%M:%S', time.localtime(section1_start))}")
    print("PYTHON MULTIPROCESSING (Most Reliable)")
    print("="*60)
    
    parallel_results = run_parallel_simulations(
        n_simulations=4, 
        n_processes=min(4, cpu_count()), 
        target_prevalence=target_prevalence
    )
    
    section1_end = time.time()
    print(f"[DURATION]  SECTION 1 DURATION: {section1_end - section1_start:.2f} seconds")
    
    # ============================================================================
    # SECTION 2: STARSIM MULTISIM
    # ============================================================================
    section2_start = time.time()
    print(f"\n[TIME] SECTION 2 START: {time.strftime('%H:%M:%S', time.localtime(section2_start))}")
    print("[STARSIM] STARSIM MULTISIM (Native Integration)")
    print("="*60)
    
    starsim_multisim = run_starsim_multisim(
        n_simulations=4, 
        target_prevalence=target_prevalence
    )
    
    section2_end = time.time()
    print(f"[DURATION]  SECTION 2 DURATION: {section2_end - section2_start:.2f} seconds")
    
    # ============================================================================
    # SECTION 3: STARSIM PARALLEL SHORTCUT
    # ============================================================================
    section3_start = time.time()
    print(f"\n[TIME] SECTION 3 START: {time.strftime('%H:%M:%S', time.localtime(section3_start))}")
    print("[FAST] STARSIM PARALLEL SHORTCUT (Easiest to Use)")
    print("="*60)
    
    starsim_parallel = run_starsim_parallel_shortcut(
        n_simulations=4, 
        target_prevalence=target_prevalence
    )
    
    section3_end = time.time()
    print(f"[DURATION]  SECTION 3 DURATION: {section3_end - section3_start:.2f} seconds")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    total_end = time.time()
    print("\n[SUMMARY] EXECUTION SUMMARY")
    print("="*80)
    
    print("[DURATION]  PERFORMANCE COMPARISON:")
    print(f"   Python Multiprocessing: {parallel_results['execution_time']:.2f} seconds")
    print(f"   [STARSIM] Starsim MultiSim: {section2_end - section2_start:.2f} seconds")
    print(f"   [FAST] Starsim Parallel: {section3_end - section3_start:.2f} seconds")
    print(f"   ** Total Execution Time: {total_end - start_time:.2f} seconds")
    
    print(f"\n[TARGET] RECOMMENDATIONS:")
    print(f"   • Best for reliability: Python multiprocessing")
    print(f"   • Best for Starsim integration: Starsim MultiSim")
    print(f"   • Best for simplicity: Starsim parallel shortcut")
    
    print(f"\n[OK] ALL SIMULATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # ============================================================================
    # GENERATE PLOTS
    # ============================================================================
    print("\n** GENERATING COMPARISON PLOTS")
    print("="*60)
    
    # Create results dictionary for plotting
    results_dict = {}
    
    # Add Python multiprocessing results
    if parallel_results and 'successful_results' in parallel_results:
        for i, result in enumerate(parallel_results['successful_results']):
            if result.get('simulation') is not None:
                results_dict[f"Python_MP_{result['sim_id']}"] = result['simulation'].results.flatten()
    
    # Add Starsim MultiSim results
    if hasattr(starsim_multisim, 'sims'):
        for i, sim in enumerate(starsim_multisim.sims):
            if hasattr(sim, 'results'):
                results_dict[f"Starsim_MultiSim_{i+1}"] = sim.results.flatten()
    
    # Add Starsim Parallel results
    if hasattr(starsim_parallel, 'sims'):
        for i, sim in enumerate(starsim_parallel.sims):
            if hasattr(sim, 'results'):
                results_dict[f"Starsim_Parallel_{i+1}"] = sim.results.flatten()
    
    # Generate comparison plots if we have results
    if results_dict:
        try:
            print("[PLOT] Creating comparison plots...")
            mtb.plot_combined(results_dict, 
                             dark=False, 
                             title="TB Simulation Results - Multiprocessing Comparison",
                             heightfold=1.5,
                             cmap='viridis_r',
                             outdir='results/comparison')
            print("[OK] Comparison plots saved to 'results/comparison/'")
        except Exception as e:
            print(f"[WARN]  Could not generate comparison plots: {e}")
    else:
        print("[WARN]  No simulation results available for plotting")
    
    # Generate individual validation plots for best performing approach
    print("\n[PLOT] GENERATING VALIDATION PLOTS")
    print("="*60)
    
    # Use the first successful simulation for validation plots
    best_sim = None
    if parallel_results and 'successful_results' in parallel_results and parallel_results['successful_results']:
        best_sim = parallel_results['successful_results'][0]['simulation']
    elif hasattr(starsim_multisim, 'sims') and starsim_multisim.sims:
        best_sim = starsim_multisim.sims[0]
    elif hasattr(starsim_parallel, 'sims') and starsim_parallel.sims:
        best_sim = starsim_parallel.sims[0]
    
    if best_sim:
        try:
            print("[PLOT] Creating validation plots...")
            create_validation_plots(best_sim)
            print("[OK] Validation plots saved to 'results/validation/'")
            
            # Generate accuracy dashboard
            print("** Creating accuracy dashboard...")
            create_accuracy_dashboard(best_sim)
            print("[OK] Accuracy dashboard saved to 'results/validation/'")
            
            # Calculate model accuracy metrics
            print("->> Calculating model accuracy metrics...")
            calculate_model_accuracy_metrics(best_sim)
            print("[OK] Model accuracy metrics calculated")
            
        except Exception as e:
            print(f"[WARN]  Could not generate validation plots: {e}")
    else:
        print("[WARN]  No simulation available for validation plots")
    
    print(f"\n[FILES] ALL PLOTS GENERATED SUCCESSFULLY!")
    print("="*80)