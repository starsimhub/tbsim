#!/usr/bin/env python3
"""
Debug the infection process to understand what's happening.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tbsim as mtb
import starsim as ss
import numpy as np
from tbsim.tb_with_state_machine import TBWithStateMachine

def debug_infection_process():
    """Debug the infection process."""
    print("=== Debugging Infection Process ===")
    
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
            stop = ss.date('1940-12-31'),  # Short simulation
            rand_seed = 1,
            verbose = 0
        )
    )
    
    # Run the simulation and check results
    print(f"Running simulation...")
    sim.run()
    
    # Check results
    results = sim.results.flatten()
    print(f"\nFinal results:")
    print(f"  New infections: {sum(results['tbwithstatemachine_new_infections'])}")
    print(f"  New active cases: {sum(results['tbwithstatemachine_new_active'])}")
    print(f"  New deaths: {sum(results['tbwithstatemachine_new_deaths'])}")
    
    # Check final state distribution
    print(f"\nFinal state distribution:")
    for state_id in mtb.TBS.all():
        count = np.count_nonzero(tb.state == state_id)
        if count > 0:
            print(f"  State {state_id}: {count}")
    
    # Check if any transitions happened during the simulation
    print(f"\nTransitions during simulation:")
    for i in range(len(results['tbwithstatemachine_new_active'])):
        new_active = results['tbwithstatemachine_new_active'][i]
        new_deaths = results['tbwithstatemachine_new_deaths'][i]
        if new_active > 0 or new_deaths > 0:
            print(f"  Time {i}: {new_active} new active, {new_deaths} new deaths")
    
    # Check initial prevalence parameter
    init_prev = tb.pars.init_prev
    print(f"\nInitial prevalence parameter: {init_prev}")
    print(f"Expected initial infections: {init_prev.p * 500}")

if __name__ == '__main__':
    debug_infection_process()
