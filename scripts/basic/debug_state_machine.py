#!/usr/bin/env python3
"""
Debug script to understand what's happening with the state machine implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tbsim as mtb
import starsim as ss
import numpy as np
from tbsim.tb_with_state_machine import TBWithStateMachine

def debug_state_machine():
    """Debug the state machine implementation."""
    print("=== TB State Machine Debug ===")
    
    # Create a simple simulation
    pop = ss.People(n_agents=100)
    tb = TBWithStateMachine(
        pars=dict(
            dt = ss.days(7),
            beta = ss.peryear(0.1),  # Higher transmission for testing
            init_prev = ss.bernoulli(0.1),  # 10% initial prevalence
        ),
        use_state_machine=True
    )
    
    sim = ss.Sim(
        people=pop,
        diseases=tb,
        pars=dict(
            dt = ss.days(7),
            start = ss.date('1940-01-01'),
            stop = ss.date('1940-12-31'),  # Short simulation
            rand_seed = 1,
            verbose = 0
        )
    )
    
    print(f"Initial state distribution:")
    stats = tb.get_state_statistics()
    for state, count in stats.items():
        print(f"  {state}: {count}")
    
    print(f"\nState machine enabled: {tb.use_state_machine}")
    print(f"State manager exists: {tb.state_manager is not None}")
    
    if tb.state_manager:
        print(f"Available transitions: {len(tb.get_transition_matrix())}")
        for (from_state, to_state), rate in tb.get_transition_matrix().items():
            print(f"  {from_state} -> {to_state}: {rate}")
    
    # Run a few time steps
    print(f"\nRunning simulation...")
    sim.run()
    
    print(f"\nFinal state distribution:")
    stats = tb.get_state_statistics()
    for state, count in stats.items():
        print(f"  {state}: {count}")
    
    # Check if any transitions actually happened
    results = sim.results.flatten()
    print(f"\nResults:")
    print(f"  New infections: {sum(results['tbwithstatemachine_new_infections'])}")
    print(f"  New active cases: {sum(results['tbwithstatemachine_new_active'])}")
    print(f"  New deaths: {sum(results['tbwithstatemachine_new_deaths'])}")
    
    # Check the actual state transitions that happened
    print(f"\nActual state transitions in the simulation:")
    for ti in range(len(results['tbwithstatemachine_new_active'])):
        if results['tbwithstatemachine_new_active'][ti] > 0:
            print(f"  Time {ti}: {results['tbwithstatemachine_new_active'][ti]} new active cases")
        if results['tbwithstatemachine_new_deaths'][ti] > 0:
            print(f"  Time {ti}: {results['tbwithstatemachine_new_deaths'][ti]} new deaths")

if __name__ == '__main__':
    debug_state_machine()
