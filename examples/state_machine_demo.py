#!/usr/bin/env python3
"""
TB State Machine Demonstration

This script demonstrates the TB state machine implementation by:
1. Creating a simulation with the state machine-enabled TB model
2. Running the simulation and tracking state transitions
3. Comparing results with the original TB implementation
4. Visualizing state transitions and results
5. Demonstrating state machine features and capabilities

Usage:
    python state_machine_demo.py [--compare] [--visualize] [--duration DAYS]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import starsim as ss
import sys
import os

# Add the parent directory to the path to import tbsim
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tbsim import TB, TBWithStateMachine, TBS
from tbsim.state_machine import TBStateManager


def create_demo_simulation(use_state_machine=True, duration=365):
    """
    Create a demonstration simulation with TB.
    
    Args:
        use_state_machine: Whether to use the state machine implementation
        duration: Simulation duration in days
        
    Returns:
        Simulation object with TB module
    """
    # Create simulation
    sim = ss.Sim(
        start=0,
        end=duration,
        dt=1,  # Daily time steps
        n_agents=1000,
        networks='random',
        demographics='age',
        analyzers='default'
    )
    
    # Create TB module
    if use_state_machine:
        tb = TBWithStateMachine(
            pars={
                'beta': ss.peryear(0.1),  # 10% annual infection rate
                'init_prev': ss.bernoulli(0.05),  # 5% initial prevalence
                'rate_LS_to_presym': ss.perday(3e-5),  # Slow progression
                'rate_LF_to_presym': ss.perday(6e-3),  # Fast progression
                'rate_presym_to_active': ss.perday(3e-2),
                'rate_active_to_clear': ss.perday(2.4e-4),
                'rate_treatment_to_clear': ss.peryear(6),  # 2-month treatment
                'rel_trans_presymp': 0.1,
                'rel_trans_smpos': 1.0,
                'rel_trans_smneg': 0.3,
                'rel_trans_exptb': 0.05,
            }
        )
    else:
        tb = TB(
            pars={
                'beta': ss.peryear(0.1),
                'init_prev': ss.bernoulli(0.05),
                'rate_LS_to_presym': ss.perday(3e-5),
                'rate_LF_to_presym': ss.perday(6e-3),
                'rate_presym_to_active': ss.perday(3e-2),
                'rate_active_to_clear': ss.perday(2.4e-4),
                'rate_treatment_to_clear': ss.peryear(6),
                'rel_trans_presymp': 0.1,
                'rel_trans_smpos': 1.0,
                'rel_trans_smneg': 0.3,
                'rel_trans_exptb': 0.05,
            }
        )
    
    # Add TB to simulation
    sim.add_module(tb)
    
    return sim, tb


def run_simulation(sim, tb, verbose=True):
    """
    Run the simulation and collect results.
    
    Args:
        sim: Simulation object
        tb: TB module
        verbose: Whether to print progress
        
    Returns:
        Dictionary with simulation results
    """
    if verbose:
        print(f"Running simulation for {sim.end} days...")
        print(f"Initial population: {sim.n_agents}")
        print(f"Using state machine: {hasattr(tb, 'state_manager') and tb.state_manager is not None}")
    
    # Run simulation
    sim.run()
    
    # Collect results
    results = {
        'time': sim.results['timevec'],
        'prevalence': tb.results['prevalence'],
        'new_infections': tb.results['new_infections'],
        'new_active': tb.results['new_active'],
        'new_deaths': tb.results['new_deaths'],
        'n_active': tb.results['n_active'],
        'n_infectious': tb.results['n_infectious'],
    }
    
    # Add state-specific results if available
    if hasattr(tb.results, 'n_latent_slow'):
        results['n_latent_slow'] = tb.results['n_latent_slow']
        results['n_latent_fast'] = tb.results['n_latent_fast']
        results['n_active_presymp'] = tb.results['n_active_presymp']
        results['n_active_smpos'] = tb.results['n_active_smpos']
        results['n_active_smneg'] = tb.results['n_active_smneg']
        results['n_active_exptb'] = tb.results['n_active_exptb']
    
    if verbose:
        print(f"Simulation completed!")
        print(f"Final prevalence: {results['prevalence'][-1]:.3f}")
        print(f"Total new infections: {np.sum(results['new_infections'])}")
        print(f"Total new active cases: {np.sum(results['new_active'])}")
        print(f"Total deaths: {np.sum(results['new_deaths'])}")
    
    return results


def compare_implementations(duration=365):
    """
    Compare the state machine implementation with the original implementation.
    
    Args:
        duration: Simulation duration in days
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 60)
    print("COMPARING TB IMPLEMENTATIONS")
    print("=" * 60)
    
    # Run with state machine
    print("\n1. Running with State Machine Implementation...")
    sim_sm, tb_sm = create_demo_simulation(use_state_machine=True, duration=duration)
    results_sm = run_simulation(sim_sm, tb_sm, verbose=True)
    
    # Run with original implementation
    print("\n2. Running with Original Implementation...")
    sim_orig, tb_orig = create_demo_simulation(use_state_machine=False, duration=duration)
    results_orig = run_simulation(sim_orig, tb_orig, verbose=True)
    
    # Compare results
    print("\n3. Comparing Results...")
    comparison = {}
    
    for key in ['prevalence', 'new_infections', 'new_active', 'new_deaths']:
        if key in results_sm and key in results_orig:
            sm_final = results_sm[key][-1]
            orig_final = results_orig[key][-1]
            diff = sm_final - orig_final
            pct_diff = (diff / orig_final * 100) if orig_final != 0 else 0
            
            comparison[key] = {
                'state_machine': sm_final,
                'original': orig_final,
                'difference': diff,
                'percent_difference': pct_diff
            }
            
            print(f"{key:15}: SM={sm_final:8.2f}, Orig={orig_final:8.2f}, Diff={diff:+8.2f} ({pct_diff:+6.1f}%)")
    
    return {
        'state_machine': results_sm,
        'original': results_orig,
        'comparison': comparison,
        'tb_sm': tb_sm,
        'tb_orig': tb_orig
    }


def demonstrate_state_machine_features(tb):
    """
    Demonstrate various state machine features.
    
    Args:
        tb: TB module with state machine
    """
    print("\n" + "=" * 60)
    print("STATE MACHINE FEATURES DEMONSTRATION")
    print("=" * 60)
    
    if not hasattr(tb, 'state_manager') or tb.state_manager is None:
        print("State machine not available in this TB instance.")
        return
    
    # 1. State Statistics
    print("\n1. Current State Distribution:")
    stats = tb.get_state_statistics()
    for state, count in stats.items():
        print(f"   {state:20}: {count:4d}")
    
    # 2. Transition Matrix
    print("\n2. Available Transitions:")
    transitions = tb.get_transition_matrix()
    for (from_state, to_state), rate in transitions.items():
        print(f"   {from_state:15} -> {to_state:15}: {rate:.2e}")
    
    # 3. State Machine Validation
    print("\n3. State Machine Validation:")
    validation = tb.validate_state_machine()
    print(f"   Valid: {validation['valid']}")
    if validation['errors']:
        print(f"   Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"   Warnings: {validation['warnings']}")
    
    # 4. Export Configuration
    print("\n4. State Machine Configuration:")
    config = tb.export_state_machine_config()
    print(f"   States defined: {len(config['states'])}")
    print(f"   Transitions defined: {len(config['transitions'])}")
    print(f"   Parameters exported: {len(config['parameters'])}")
    
    # 5. State Machine Manager Details
    print("\n5. State Machine Manager Details:")
    sm = tb.state_manager.state_machine
    print(f"   Total states: {len(sm.states)}")
    print(f"   State IDs: {list(sm.states.keys())}")
    
    # Count transitions per state
    for state_id, state_obj in sm.states.items():
        print(f"   {state_obj.name:15}: {len(state_obj.transitions)} transitions")


def create_visualizations(results, tb_sm=None, save_plots=True):
    """
    Create visualizations of the simulation results.
    
    Args:
        results: Simulation results dictionary
        tb_sm: TB module with state machine (optional)
        save_plots: Whether to save plots to files
    """
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # 1. Basic Results Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TB Simulation Results', fontsize=16)
    
    time = results['time']
    
    # Prevalence
    axes[0, 0].plot(time, results['prevalence'])
    axes[0, 0].set_title('TB Prevalence')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Prevalence')
    axes[0, 0].grid(True)
    
    # New Infections
    axes[0, 1].plot(time, results['new_infections'])
    axes[0, 1].set_title('New Infections per Day')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('New Infections')
    axes[0, 1].grid(True)
    
    # Active Cases
    axes[1, 0].plot(time, results['n_active'])
    axes[1, 0].set_title('Active TB Cases')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Number of Cases')
    axes[1, 0].grid(True)
    
    # Deaths
    axes[1, 1].plot(time, results['new_deaths'])
    axes[1, 1].set_title('TB Deaths per Day')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Deaths')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('tb_simulation_results.png', dpi=300, bbox_inches='tight')
        print("Saved: tb_simulation_results.png")
    plt.show()
    
    # 2. State Distribution Plot (if state machine available)
    if tb_sm and hasattr(tb_sm, 'state_manager') and tb_sm.state_manager is not None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get state-specific results
        state_results = {}
        for state_name in ['n_latent_slow', 'n_latent_fast', 'n_active_presymp', 
                          'n_active_smpos', 'n_active_smneg', 'n_active_exptb']:
            if state_name in results:
                state_results[state_name] = results[state_name]
        
        if state_results:
            for state_name, data in state_results.items():
                ax.plot(time, data, label=state_name.replace('n_', '').replace('_', ' ').title())
            
            ax.set_title('TB State Distribution Over Time')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Number of Individuals')
            ax.legend()
            ax.grid(True)
            
            if save_plots:
                plt.savefig('tb_state_distribution.png', dpi=300, bbox_inches='tight')
                print("Saved: tb_state_distribution.png")
            plt.show()
        
        # 3. Transition Matrix Visualization
        try:
            fig = tb_sm.plot_state_transitions()
            if save_plots:
                fig.savefig('tb_transition_matrix.png', dpi=300, bbox_inches='tight')
                print("Saved: tb_transition_matrix.png")
        except Exception as e:
            print(f"Could not create transition matrix plot: {e}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description='TB State Machine Demonstration')
    parser.add_argument('--compare', action='store_true', 
                       help='Compare state machine with original implementation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--duration', type=int, default=365,
                       help='Simulation duration in days (default: 365)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to files')
    
    args = parser.parse_args()
    
    print("TB State Machine Demonstration")
    print("=" * 60)
    print(f"Simulation duration: {args.duration} days")
    print(f"Compare implementations: {args.compare}")
    print(f"Create visualizations: {args.visualize}")
    print(f"Save plots: {not args.no_save}")
    
    if args.compare:
        # Compare implementations
        comparison_results = compare_implementations(args.duration)
        
        if args.visualize:
            # Create comparison visualizations
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('TB Implementation Comparison', fontsize=16)
            
            time = comparison_results['state_machine']['time']
            
            # Prevalence comparison
            axes[0, 0].plot(time, comparison_results['state_machine']['prevalence'], 
                           label='State Machine', linewidth=2)
            axes[0, 0].plot(time, comparison_results['original']['prevalence'], 
                           label='Original', linewidth=2, linestyle='--')
            axes[0, 0].set_title('Prevalence Comparison')
            axes[0, 0].set_xlabel('Time (days)')
            axes[0, 0].set_ylabel('Prevalence')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # New infections comparison
            axes[0, 1].plot(time, comparison_results['state_machine']['new_infections'], 
                           label='State Machine', linewidth=2)
            axes[0, 1].plot(time, comparison_results['original']['new_infections'], 
                           label='Original', linewidth=2, linestyle='--')
            axes[0, 1].set_title('New Infections Comparison')
            axes[0, 1].set_xlabel('Time (days)')
            axes[0, 1].set_ylabel('New Infections')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Active cases comparison
            axes[1, 0].plot(time, comparison_results['state_machine']['n_active'], 
                           label='State Machine', linewidth=2)
            axes[1, 0].plot(time, comparison_results['original']['n_active'], 
                           label='Original', linewidth=2, linestyle='--')
            axes[1, 0].set_title('Active Cases Comparison')
            axes[1, 0].set_xlabel('Time (days)')
            axes[1, 0].set_ylabel('Active Cases')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Deaths comparison
            axes[1, 1].plot(time, comparison_results['state_machine']['new_deaths'], 
                           label='State Machine', linewidth=2)
            axes[1, 1].plot(time, comparison_results['original']['new_deaths'], 
                           label='Original', linewidth=2, linestyle='--')
            axes[1, 1].set_title('Deaths Comparison')
            axes[1, 1].set_xlabel('Time (days)')
            axes[1, 1].set_ylabel('Deaths')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            if not args.no_save:
                plt.savefig('tb_implementation_comparison.png', dpi=300, bbox_inches='tight')
                print("Saved: tb_implementation_comparison.png")
            plt.show()
        
        # Demonstrate state machine features
        demonstrate_state_machine_features(comparison_results['tb_sm'])
        
    else:
        # Single simulation with state machine
        print("\nRunning single simulation with state machine...")
        sim, tb = create_demo_simulation(use_state_machine=True, duration=args.duration)
        results = run_simulation(sim, tb, verbose=True)
        
        if args.visualize:
            create_visualizations(results, tb, save_plots=not args.no_save)
        
        # Demonstrate state machine features
        demonstrate_state_machine_features(tb)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
