#!/usr/bin/env python3
"""
Simple TB State Machine Example

This is a simplified version of the TB state machine demonstration script
that shows the basic usage of the new state machine implementation.

Based on the original run_tb.py script but using the state machine.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add the parent directory to the path to import the state machine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tbsim.tb_with_state_machine import TBWithStateMachine


def build_tbsim_state_machine():
    """Build TB simulation using the state machine implementation."""
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )

    pop = ss.People(n_agents=1000)
    tb = TBWithStateMachine(
        pars=dict(
            dt = ss.days(7),
            beta = ss.peryear(0.025),
            init_prev = ss.bernoulli(0.01),
        ),
        use_state_machine=True  # Enable state machine
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


def main():
    """Run a simple TB simulation with the state machine."""
    print("Simple TB State Machine Example")
    print("=" * 40)
    
    # Build simulation with state machine
    sim = build_tbsim_state_machine()
    
    # Run simulation
    print("Running simulation...")
    sim.run()
    
    # Print basic results
    results = sim.results.flatten()
    print(f"\nSimulation completed!")
    print(f"Final prevalence: {results['tbwithstatemachine_prevalence'][-1]:.3f}")
    print(f"Total new infections: {sum(results['tbwithstatemachine_new_infections'])}")
    print(f"Total new active cases: {sum(results['tbwithstatemachine_new_active'])}")
    print(f"Total deaths: {sum(results['tbwithstatemachine_new_deaths'])}")
    
    # Demonstrate state machine features
    tb = sim.diseases.tbwithstatemachine
    print(f"\nState Machine Features:")
    print(f"Using state machine: {hasattr(tb, 'state_manager') and tb.state_manager is not None}")
    
    if hasattr(tb, 'state_manager') and tb.state_manager is not None:
        # Get state statistics
        stats = tb.get_state_statistics()
        print(f"Current state distribution:")
        for state, count in stats.items():
            print(f"  {state}: {count}")
        
        # Validate state machine
        validation = tb.validate_state_machine()
        print(f"State machine valid: {validation['valid']}")
        
        # Get transition matrix
        transitions = tb.get_transition_matrix()
        print(f"Available transitions: {len(transitions)}")
    
    # Create comprehensive plots
    print(f"\nCreating comprehensive plots...")
    
    # Create time vector based on results length
    n_steps = len(results['tbwithstatemachine_prevalence'])
    time_vec = np.arange(n_steps) * 7  # 7 days per time step
    
    # Create a detailed plot showing all TB statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('TB State Machine Simulation Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Prevalence and Incidence
    ax1 = axes[0, 0]
    ax1.plot(time_vec, results['tbwithstatemachine_prevalence'], 'b-', linewidth=2, label='Total Prevalence')
    ax1.plot(time_vec, results['tbwithstatemachine_prevalence_active'], 'r-', linewidth=2, label='Active Prevalence')
    ax1.set_title('TB Prevalence Over Time')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Prevalence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: New Cases and Deaths
    ax2 = axes[0, 1]
    ax2.plot(time_vec, results['tbwithstatemachine_new_infections'], 'g-', linewidth=2, label='New Infections')
    ax2.plot(time_vec, results['tbwithstatemachine_new_active'], 'orange', linewidth=2, label='New Active Cases')
    ax2.plot(time_vec, results['tbwithstatemachine_new_deaths'], 'r-', linewidth=2, label='New Deaths')
    ax2.set_title('New TB Cases and Deaths Over Time')
    ax2.set_xlabel('Time (days)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: State Distribution
    ax3 = axes[1, 0]
    ax3.plot(time_vec, results['tbwithstatemachine_n_latent_slow'], 'b-', linewidth=2, label='Latent Slow')
    ax3.plot(time_vec, results['tbwithstatemachine_n_latent_fast'], 'c-', linewidth=2, label='Latent Fast')
    ax3.plot(time_vec, results['tbwithstatemachine_n_active_presymp'], 'y-', linewidth=2, label='Pre-symptomatic')
    ax3.plot(time_vec, results['tbwithstatemachine_n_active_smpos'], 'r-', linewidth=2, label='Smear Positive')
    ax3.plot(time_vec, results['tbwithstatemachine_n_active_smneg'], 'orange', linewidth=2, label='Smear Negative')
    ax3.plot(time_vec, results['tbwithstatemachine_n_active_exptb'], 'purple', linewidth=2, label='Extra-pulmonary')
    ax3.set_title('TB State Distribution Over Time')
    ax3.set_xlabel('Time (days)')
    ax3.set_ylabel('Number of Individuals')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Infectious and Notifications
    ax4 = axes[1, 1]
    ax4.plot(time_vec, results['tbwithstatemachine_n_infectious'], 'r-', linewidth=2, label='Infectious')
    ax4.plot(time_vec, results['tbwithstatemachine_new_notifications_15+'], 'g-', linewidth=2, label='New Notifications (15+)')
    ax4.plot(time_vec, results['tbwithstatemachine_n_detectable_15+'], 'b-', linewidth=2, label='Detectable Cases (15+)')
    ax4.set_title('Infectious and Detection Metrics')
    ax4.set_xlabel('Time (days)')
    ax4.set_ylabel('Count')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Also create the standard TB plot
    print(f"\nCreating standard TB plot...")
    results_dict = {'state_machine': results}
    mtb.plot_combined(results_dict, dark=False)
    plt.show()
    
    print("Done!")


if __name__ == '__main__':
    main()
