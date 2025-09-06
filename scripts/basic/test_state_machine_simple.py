#!/usr/bin/env python3
"""
Simple test to understand what's happening with the state machine.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tbsim as mtb
import starsim as ss
import numpy as np
from tbsim.tb_with_state_machine import TBWithStateMachine

def test_state_machine():
    """Test the state machine implementation."""
    print("=== Testing State Machine Implementation ===")
    
    # Create a simple simulation
    pop = ss.People(n_agents=500)
    tb = TBWithStateMachine(
        pars=dict(
            dt = ss.days(7),
            beta = ss.peryear(0.1),  # Higher transmission for testing
            init_prev = ss.bernoulli(0.1),  # 10% initial prevalence
        ),
        use_state_machine=True
    )
    
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))
    
    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=dict(
            dt = ss.days(7),
            start = ss.date('1940-01-01'),
            stop = ss.date('2010-12-31'),  # Longer simulation like working script
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
    
    # Run a few time steps manually to see what happens
    print(f"\nRunning simulation step by step...")
    
    # Run the simulation
    print(f"\nRunning simulation...")
    sim.run()
    
    # Check results
    results = sim.results.flatten()
    print(f"\nFinal results:")
    print(f"  New infections: {sum(results['tbwithstatemachine_new_infections'])}")
    print(f"  New active cases: {sum(results['tbwithstatemachine_new_active'])}")
    print(f"  New deaths: {sum(results['tbwithstatemachine_new_deaths'])}")
    
    # Check state distribution
    stats = tb.get_state_statistics()
    print(f"\nFinal state distribution:")
    for state, count in stats.items():
        if count > 0:
            print(f"  {state}: {count}")
    
    # Check if any transitions happened during the simulation
    print(f"\nTransitions during simulation:")
    for i in range(len(results['tbwithstatemachine_new_active'])):
        new_active = results['tbwithstatemachine_new_active'][i]
        new_deaths = results['tbwithstatemachine_new_deaths'][i]
        if new_active > 0 or new_deaths > 0:
            print(f"  Time {i}: {new_active} new active, {new_deaths} new deaths")

if __name__ == '__main__':
    test_state_machine()
