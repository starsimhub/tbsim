#!/usr/bin/env python3
"""
TB State Machine Demonstration Script

This script demonstrates the new TB state machine implementation by:
1. Running simulations with both original and state machine implementations
2. Comparing results between the two approaches
3. Demonstrating state machine features and capabilities
4. Creating visualizations of state transitions and results

Based on the original run_tb.py script but enhanced with state machine functionality.
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

# Add the parent directory to the path to import the state machine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tbsim.tb_with_state_machine import TBWithStateMachine
from tbsim.state_machine import TBStateManager


def build_tbsim_original(sim_pars=None):
    """
    Build TB simulation using the original TB implementation.
    
    Args:
        sim_pars: Optional simulation parameters to override defaults
        
    Returns:
        Simulation object with original TB implementation
    """
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(
        dt = ss.days(7),
        beta = ss.peryear(0.025),
        init_prev = ss.bernoulli(0.01),
        rate_LS_to_presym = ss.perday(3e-5),
        rate_LF_to_presym = ss.perday(6e-3),
        rate_presym_to_active = ss.perday(3e-2),
        rate_active_to_clear = ss.perday(2.4e-4),
        rate_treatment_to_clear = ss.peryear(6),
        rel_trans_presymp = 0.1,
        rel_trans_smpos = 1.0,
        rel_trans_smneg = 0.3,
        rel_trans_exptb = 0.05,
    ))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    sim.pars.verbose = 0
    return sim


def build_tbsim_state_machine(sim_pars=None, use_state_machine=True):
    """
    Build TB simulation using the state machine implementation.
    
    Args:
        sim_pars: Optional simulation parameters to override defaults
        use_state_machine: Whether to use the state machine (True) or fallback to original (False)
        
    Returns:
        Simulation object with state machine TB implementation
    """
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=1000)
    tb = TBWithStateMachine(
        pars=dict(
            dt = ss.days(7),
            beta = ss.peryear(0.025),
            init_prev = ss.bernoulli(0.01),
            rate_LS_to_presym = ss.perday(3e-5),
            rate_LF_to_presym = ss.perday(6e-3),
            rate_presym_to_active = ss.perday(3e-2),
            rate_active_to_clear = ss.perday(2.4e-4),
            rate_treatment_to_clear = ss.peryear(6),
            rel_trans_presymp = 0.1,
            rel_trans_smpos = 1.0,
            rel_trans_smneg = 0.3,
            rel_trans_exptb = 0.05,
        ),
        use_state_machine=use_state_machine
    )
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    sim.pars.verbose = 0
    return sim


def run_comparison_simulation():
    """
    Run comparison between original and state machine implementations.
    
    Returns:
        Dictionary with results from both implementations
    """
    print("=" * 60)
    print("TB STATE MACHINE COMPARISON")
    print("=" * 60)
    
    # Run original implementation
    print("\n1. Running Original TB Implementation...")
    sim_orig = build_tbsim_original()
    sim_orig.run()
    
    # Run state machine implementation
    print("\n2. Running State Machine TB Implementation...")
    sim_sm = build_tbsim_state_machine(use_state_machine=True)
    sim_sm.run()
    
    # Collect results
    results_orig = sim_orig.results.flatten()
    results_sm = sim_sm.results.flatten()
    
    # Compare key metrics
    print("\n3. Comparing Results...")
    comparison_metrics = [
        'prevalence', 'new_infections', 'new_active', 'new_deaths',
        'n_active', 'n_infectious', 'cum_infections'
    ]
    
    for metric in comparison_metrics:
        if metric in results_orig and metric in results_sm:
            orig_final = results_orig[metric][-1] if hasattr(results_orig[metric], '__getitem__') else results_orig[metric]
            sm_final = results_sm[metric][-1] if hasattr(results_sm[metric], '__getitem__') else results_sm[metric]
            diff = sm_final - orig_final
            pct_diff = (diff / orig_final * 100) if orig_final != 0 else 0
            print(f"{metric:15}: Orig={orig_final:8.2f}, SM={sm_final:8.2f}, Diff={diff:+8.2f} ({pct_diff:+6.1f}%)")
    
    return {
        'original': results_orig,
        'state_machine': results_sm,
        'sim_orig': sim_orig,
        'sim_sm': sim_sm
    }


def demonstrate_state_machine_features(sim_sm):
    """
    Demonstrate various state machine features.
    
    Args:
        sim_sm: Simulation object with state machine TB
    """
    print("\n" + "=" * 60)
    print("STATE MACHINE FEATURES DEMONSTRATION")
    print("=" * 60)
    
    tb_sm = sim_sm.diseases.tbwithstatemachine
    
    if not hasattr(tb_sm, 'state_manager') or tb_sm.state_manager is None:
        print("State machine not available in this TB instance.")
        return
    
    # 1. State Statistics
    print("\n1. Current State Distribution:")
    stats = tb_sm.get_state_statistics()
    for state, count in stats.items():
        print(f"   {state:20}: {count:4d}")
    
    # 2. Transition Matrix
    print("\n2. Available Transitions:")
    transitions = tb_sm.get_transition_matrix()
    for (from_state, to_state), rate in transitions.items():
        print(f"   {from_state:15} -> {to_state:15}: {rate:.2e}")
    
    # 3. State Machine Validation
    print("\n3. State Machine Validation:")
    validation = tb_sm.validate_state_machine()
    print(f"   Valid: {validation['valid']}")
    if validation.get('errors'):
        print(f"   Errors: {validation['errors']}")
    if validation.get('warnings'):
        print(f"   Warnings: {validation['warnings']}")
    
    # 4. Export Configuration
    print("\n4. State Machine Configuration:")
    config = tb_sm.export_state_machine_config()
    print(f"   States defined: {len(config['states'])}")
    print(f"   Transitions defined: {len(config['transitions'])}")
    print(f"   Parameters exported: {len(config['parameters'])}")


def create_comparison_plots(results):
    """
    Create comparison plots between original and state machine implementations.
    
    Args:
        results: Dictionary with results from both implementations
    """
    print("\n" + "=" * 60)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("=" * 60)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('TB Implementation Comparison: Original vs State Machine', fontsize=16)
    
    # Get time vector
    time = results['original']['timevec']
    
    # Prevalence comparison
    axes[0, 0].plot(time, results['original']['prevalence'], 
                   label='Original', linewidth=2, alpha=0.8)
    axes[0, 0].plot(time, results['state_machine']['prevalence'], 
                   label='State Machine', linewidth=2, alpha=0.8)
    axes[0, 0].set_title('TB Prevalence Comparison')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Prevalence')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # New infections comparison
    axes[0, 1].plot(time, results['original']['new_infections'], 
                   label='Original', linewidth=2, alpha=0.8)
    axes[0, 1].plot(time, results['state_machine']['new_infections'], 
                   label='State Machine', linewidth=2, alpha=0.8)
    axes[0, 1].set_title('New Infections Comparison')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('New Infections')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Active cases comparison
    axes[1, 0].plot(time, results['original']['n_active'], 
                   label='Original', linewidth=2, alpha=0.8)
    axes[1, 0].plot(time, results['state_machine']['n_active'], 
                   label='State Machine', linewidth=2, alpha=0.8)
    axes[1, 0].set_title('Active TB Cases Comparison')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Active Cases')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Deaths comparison
    axes[1, 1].plot(time, results['original']['new_deaths'], 
                   label='Original', linewidth=2, alpha=0.8)
    axes[1, 1].plot(time, results['state_machine']['new_deaths'], 
                   label='State Machine', linewidth=2, alpha=0.8)
    axes[1, 1].set_title('TB Deaths Comparison')
    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('Deaths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tb_implementation_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: tb_implementation_comparison.png")
    plt.show()


def create_state_machine_visualizations(sim_sm):
    """
    Create state machine specific visualizations.
    
    Args:
        sim_sm: Simulation object with state machine TB
    """
    tb_sm = sim_sm.diseases.tbwithstatemachine
    
    if not hasattr(tb_sm, 'state_manager') or tb_sm.state_manager is None:
        print("State machine not available for visualization.")
        return
    
    # 1. State transition matrix
    try:
        fig = tb_sm.plot_state_transitions()
        fig.savefig('tb_state_transition_matrix.png', dpi=300, bbox_inches='tight')
        print("Saved: tb_state_transition_matrix.png")
        plt.show()
    except Exception as e:
        print(f"Could not create transition matrix plot: {e}")
    
    # 2. State distribution over time
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get state-specific results
        time = sim_sm.results['timevec']
        state_results = {}
        
        for state_name in ['n_latent_slow', 'n_latent_fast', 'n_active_presymp', 
                          'n_active_smpos', 'n_active_smneg', 'n_active_exptb']:
            if state_name in sim_sm.results:
                state_results[state_name] = sim_sm.results[state_name]
        
        if state_results:
            for state_name, data in state_results.items():
                ax.plot(time, data, label=state_name.replace('n_', '').replace('_', ' ').title())
            
            ax.set_title('TB State Distribution Over Time (State Machine)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Individuals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('tb_state_distribution.png', dpi=300, bbox_inches='tight')
            print("Saved: tb_state_distribution.png")
            plt.show()
    except Exception as e:
        print(f"Could not create state distribution plot: {e}")


def run_single_simulation(use_state_machine=True):
    """
    Run a single simulation with specified implementation.
    
    Args:
        use_state_machine: Whether to use state machine implementation
    """
    print("=" * 60)
    print(f"RUNNING SINGLE SIMULATION ({'State Machine' if use_state_machine else 'Original'})")
    print("=" * 60)
    
    if use_state_machine:
        sim = build_tbsim_state_machine(use_state_machine=True)
    else:
        sim = build_tbsim_original()
    
    sim.run()
    
    # Print basic results
    results = sim.results.flatten()
    print(f"\nSimulation completed!")
    
    # Handle prefixed result keys
    prefix = 'tbwithstatemachine_' if use_state_machine else 'tb_'
    if use_state_machine:
        prefix = 'tbwithstatemachine_'
    else:
        prefix = 'tb_'
    
    print(f"Final prevalence: {results[f'{prefix}prevalence'][-1]:.3f}")
    print(f"Total new infections: {np.sum(results[f'{prefix}new_infections'])}")
    print(f"Total new active cases: {np.sum(results[f'{prefix}new_active'])}")
    print(f"Total deaths: {np.sum(results[f'{prefix}new_deaths'])}")
    
    # Demonstrate state machine features if available
    if use_state_machine:
        demonstrate_state_machine_features(sim)
        create_state_machine_visualizations(sim)
    
    # Create standard plots
    results_dict = {'state_machine' if use_state_machine else 'original': results}
    mtb.plot_combined(results_dict, dark=False, filter=mtb.FILTERS.important_metrics)
    plt.show()
    
    return sim


def main():
    """Main function to run the TB state machine demonstration."""
    parser = argparse.ArgumentParser(description='TB State Machine Demonstration')
    parser.add_argument('--mode', choices=['single', 'compare', 'original', 'state_machine'], 
                       default='compare', help='Run mode')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save plots to files')
    
    args = parser.parse_args()
    
    print("TB State Machine Demonstration Script")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Create visualizations: {args.visualize}")
    print(f"Save plots: {not args.no_save}")
    
    if args.mode == 'compare':
        # Run comparison between implementations
        results = run_comparison_simulation()
        
        if args.visualize:
            create_comparison_plots(results)
        
        # Demonstrate state machine features
        demonstrate_state_machine_features(results['sim_sm'])
        
    elif args.mode == 'single':
        # Run single simulation with state machine
        sim = run_single_simulation(use_state_machine=True)
        
    elif args.mode == 'original':
        # Run single simulation with original implementation
        sim = run_single_simulation(use_state_machine=False)
        
    elif args.mode == 'state_machine':
        # Run single simulation with state machine
        sim = run_single_simulation(use_state_machine=True)
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
