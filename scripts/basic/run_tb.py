#!/usr/bin/env python3
"""
TB Simulation Comparison: Static vs Dynamic Prevalence Control

This module provides comprehensive TB simulation capabilities comparing different
approaches to maintaining constant prevalence levels. It implements two main
strategies:

1. **Static Calibrated Parameters**: Uses pre-calibrated, fixed parameters
   that are tuned to maintain target prevalence without real-time adjustment.

2. **Dynamic Prevalence Controller**: Implements a feedback control system
   that monitors prevalence during simulation and automatically adjusts
   transmission parameters to maintain target levels.

The module includes comprehensive validation, analysis, and visualization
capabilities to assess model performance and accuracy.

Key Features:
- Multi-approach simulation comparison
- Real-time prevalence monitoring and control
- Comprehensive validation and accuracy metrics
- Interactive visualizations and dashboards
- Statistical analysis of model performance

Author: TB Simulation Team
Version: 1.0
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
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
    # These rates are calibrated to maintain population stability over 70 years
    births = ss.Births(pars=dict(birth_rate=20))  # 20 births per 1000 population per year
    deaths = ss.Deaths(pars=dict(death_rate=15))  # 15 deaths per 1000 population per year

    # ============================================================================
    # ANALYZER SETUP
    # ============================================================================
    # Create dwell time analyzer for comprehensive model validation
    # This analyzer tracks disease progression times and state transitions
    # for accuracy assessment and model validation
    dwell_analyzer = DwtAnalyzer(scenario_name="TB_Model_Validation")

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
        analyzers=dwell_analyzer,      # Validation and analysis tools
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
    """
    def __init__(self, target_prevalence=0.01, tolerance=0.002, adjustment_factor=0.1, cooldown_period=10):
        """
        Initialize the prevalence controller with control parameters
        
        Args:
            target_prevalence (float): Target prevalence level to maintain (default: 1%)
            tolerance (float): Acceptable deviation from target (±tolerance)
            adjustment_factor (float): Magnitude of parameter adjustments (default: 10%)
            cooldown_period (int): Time steps between adjustments to prevent oscillations
        """
        super().__init__()
        
        # ========================================================================
        # CONTROL PARAMETERS
        # ========================================================================
        self.target_prevalence = target_prevalence  # Target prevalence level (1%)
        self.tolerance = tolerance                  # Acceptable deviation (±0.2%)
        self.adjustment_factor = adjustment_factor  # Adjustment magnitude (10%)
        self.cooldown_period = cooldown_period      # Cooldown between adjustments
        
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
        # Only adjust parameters after burn-in period and cooldown to prevent
        # oscillations and allow the system to stabilize
        if (len(self.prevalence_history) > 50 and 
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
        """Report on adjustments made"""
        if self.beta_adjustments:
            print(f"\nPrevalence Controller made {len(self.beta_adjustments)} adjustments:")
            for action, factor, time_step in self.beta_adjustments:
                print(f"  Time {time_step}: {action} beta by factor {factor:.3f}")
    
    def finalize_results(self):
        """Required by Starsim framework"""
        pass

def build_tbsim_with_control(sim_pars=None, target_prevalence=0.01, use_controller=True):
    """
    Build TB simulation with dynamic prevalence control capabilities
    
    This function creates a TB simulation that can optionally use dynamic prevalence
    control. Unlike the static approach, this allows for real-time parameter
    adjustment based on current prevalence levels.
    
    Args:
        sim_pars (dict, optional): Additional simulation parameters to override defaults
        target_prevalence (float): Target active TB prevalence level (default: 1%)
        use_controller (bool): Whether to enable dynamic prevalence controller
        
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
            cooldown_period=5  # 5 time steps between adjustments
        )
        analyzers.append(prevalence_controller)
    
    # Add dwell time analyzer
    dwell_analyzer = DwtAnalyzer(scenario_name="TB_Model_Validation")
    analyzers.append(dwell_analyzer)

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
        years=[1950, 1960, 1970, 1980, 1990, 2000],
        x_beta=[0.95, 0.90, 0.85, 0.80, 0.75, 0.70]  # Gradual reduction
    ))
    interventions.append(beta_intervention)
    
    return interventions

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

def run_all_approaches():
    """
    Run all simulation approaches sequentially and provide comprehensive analysis
    
    This function executes three different approaches to TB simulation:
    1. Static calibrated parameters - uses fixed, pre-calibrated parameters
    2. Dynamic prevalence controller - uses feedback control to maintain target prevalence
    3. Comprehensive comparison - compares the performance of both approaches
    
    Returns:
        tuple: (all_results, all_sims, best_approach) containing results and best performing method
    """
    # ============================================================================
    # INITIALIZATION
    # ============================================================================
    target_prevalence = 0.01  # Target prevalence: 1%
    all_results = {}          # Store results from all approaches
    all_sims = {}            # Store simulation objects for analysis
    
    print("TB SIMULATION: RUNNING ALL APPROACHES")
    print("="*60)
    
    # ============================================================================
    # APPROACH 1: STATIC CALIBRATED PARAMETERS
    # ============================================================================
    # This approach uses pre-calibrated, fixed parameters that are tuned to
    # maintain target prevalence without any real-time adjustment
    print("\n" + "="*60)
    print("APPROACH 1: STATIC CALIBRATED PARAMETERS")
    print("="*60)
    print("Running with static calibrated parameters...")
    
    # Build and run simulation with static parameters
    sim1 = build_tbsim(target_prevalence=target_prevalence)
    sim1.run()
    
    # Analyze prevalence stability for this approach
    prevalence_stats1 = monitor_prevalence_stability(sim1, target_prevalence=target_prevalence)
    all_results["Static Parameters"] = prevalence_stats1
    all_sims["Static Parameters"] = sim1
    
    print(f"✓ Static approach completed")
    print(f"  Mean Prevalence: {prevalence_stats1['mean_prevalence']:.3%}")
    print(f"  Coefficient of Variation: {prevalence_stats1['cv_prevalence']:.1f}%")
    print(f"  Time in Target: {prevalence_stats1['target_percentage']:.1f}%")
    
    # ============================================================================
    # APPROACH 2: DYNAMIC PREVALENCE CONTROLLER
    # ============================================================================
    # This approach uses real-time feedback control to maintain target prevalence
    # by automatically adjusting transmission parameters during simulation
    print("\n" + "="*60)
    print("APPROACH 2: DYNAMIC PREVALENCE CONTROLLER")
    print("="*60)
    print("Running with dynamic prevalence controller...")
    
    # Build and run simulation with dynamic control
    sim2 = build_tbsim_with_control(target_prevalence=target_prevalence, use_controller=True)
    sim2.run()
    
    # Analyze prevalence stability for this approach
    prevalence_stats2 = monitor_prevalence_stability(sim2, target_prevalence=target_prevalence)
    all_results["Dynamic Controller"] = prevalence_stats2
    all_sims["Dynamic Controller"] = sim2
    
    print(f"✓ Dynamic controller approach completed")
    print(f"  Mean Prevalence: {prevalence_stats2['mean_prevalence']:.3%}")
    print(f"  Coefficient of Variation: {prevalence_stats2['cv_prevalence']:.1f}%")
    print(f"  Time in Target: {prevalence_stats2['target_percentage']:.1f}%")
    
    # ============================================================================
    # COMPREHENSIVE COMPARISON
    # ============================================================================
    # Run detailed comparison between approaches and determine best performer
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    comparison_results = run_prevalence_comparison()
    
    # ============================================================================
    # BEST APPROACH DETERMINATION
    # ============================================================================
    # Determine best performing approach based on coefficient of variation
    # Lower CV indicates more stable prevalence (better performance)
    best_approach = None
    best_cv = float('inf')
    for name, stats in all_results.items():
        if stats['cv_prevalence'] < best_cv:
            best_cv = stats['cv_prevalence']
            best_approach = name
    
    print(f"\n🏆 BEST PERFORMING APPROACH: {best_approach}")
    print(f"   Coefficient of Variation: {best_cv:.1f}%")
    
    # Summary comparison
    print(f"\n" + "="*60)
    print("COMPREHENSIVE COMPARISON SUMMARY")
    print(f"="*60)
    
    for name, stats in all_results.items():
        status = "🏆 BEST" if name == best_approach else ""
        print(f"{name} {status}:")
        print(f"  Mean Prevalence: {stats['mean_prevalence']:.3%}")
        print(f"  Coefficient of Variation: {stats['cv_prevalence']:.1f}%")
        print(f"  Time in Target: {stats['target_percentage']:.1f}%")
        print()
    
    return all_results, all_sims, best_approach


def create_comprehensive_analysis(all_sims, best_approach):
    """
    Create comprehensive analysis and visualizations for all approaches
    
    This function generates detailed analysis using the best performing simulation:
    - Model accuracy metrics calculation
    - Accuracy dashboard creation
    - Validation plots generation
    - Comparative results visualization
    
    Args:
        all_sims: Dictionary of all simulation results
        best_approach: Name of the best performing approach
    """
    print(f"\n" + "="*60)
    print("CREATING COMPREHENSIVE ANALYSIS")
    print(f"="*60)
    
    # ============================================================================
    # DETAILED ANALYSIS USING BEST PERFORMING APPROACH
    # ============================================================================
    # Use the best performing simulation for comprehensive analysis and visualization
    best_sim = all_sims[best_approach]
    print(f"Using {best_approach} for detailed analysis...")
    
    # ============================================================================
    # STEP 1: QUANTITATIVE ACCURACY METRICS
    # ============================================================================
    # Calculate statistical metrics to assess model accuracy and performance
    print("\nCalculating model accuracy metrics...")
    calculate_model_accuracy_metrics(best_sim)
    
    # ============================================================================
    # STEP 2: INTERACTIVE ACCURACY DASHBOARD
    # ============================================================================
    # Create comprehensive dashboard with multiple visualization panels
    print("Creating accuracy dashboard...")
    create_accuracy_dashboard(best_sim)
    
    # ============================================================================
    # STEP 3: COMPREHENSIVE VALIDATION PLOTS
    # ============================================================================
    # Generate detailed validation plots including dwell time analysis,
    # state transitions, and model accuracy assessments
    print("Creating validation plots...")
    create_validation_plots(best_sim)
    
    # ============================================================================
    # STEP 4: COMPARATIVE VISUALIZATION
    # ============================================================================
    # Create comparative visualization showing all approaches side-by-side
    print("Creating comparative results plot...")
    results_dict = {}
    for name, sim in all_sims.items():
        results_dict[name] = sim.results.flatten()
    
    # Generate combined plot showing all approaches for direct comparison
    mtb.plot_combined(results_dict, 
                     dark=False, 
                     title="TB Simulation Results - All Approaches Comparison",
                     heightfold=1.5,
                     outdir='results/comparison')
    
    print("✓ All analysis and visualizations completed")


if __name__ == '__main__':
    """
    Main execution block for TB simulation comparison
    
    This script compares different approaches to maintaining constant TB prevalence:
    - Static parameters: Pre-calibrated fixed parameters
    - Dynamic control: Real-time parameter adjustment based on prevalence
    - Analysis: Comprehensive comparison and visualization of results
    """
    # Set target prevalence (1% = 0.01)
    target_prevalence = 0.01
    
    print("TB SIMULATION: RUNNING ALL APPROACHES")
    print("="*60)
    print("This will run two approaches and their comparison:")
    print("1. Static calibrated parameters")
    print("2. Dynamic prevalence controller") 
    print("3. Comprehensive comparison of both approaches")
    print("\nStarting execution...")
    
    # ============================================================================
    # EXECUTION PHASE
    # ============================================================================
    # Execute all simulation approaches and collect results
    all_results, all_sims, best_approach = run_all_approaches()
    
    # ============================================================================
    # ANALYSIS PHASE
    # ============================================================================
    # Generate comprehensive analysis and visualizations using best approach
    create_comprehensive_analysis(all_sims, best_approach)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    # Display final results and completion status
    print(f"\n" + "="*60)
    print("ALL APPROACHES COMPLETED SUCCESSFULLY!")
    print(f"="*60)
    print(f"Best performing approach: {best_approach}")
    print("Results and plots have been generated.")
    print("Check the 'results' directory for output files.")
    
    # Display plots if running interactively
    plt.show()