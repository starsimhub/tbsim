#!/usr/bin/env python3
"""
Test the original TB implementation for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tbsim as mtb
import starsim as ss
import numpy as np

def test_original_tb():
    """Test the original TB implementation."""
    print("=== Testing Original TB Implementation ===")
    
    # Create a simple simulation
    pop = ss.People(n_agents=500)
    tb = mtb.TB(
        pars=dict(
            dt = ss.days(7),
            beta = ss.peryear(0.1),  # Higher transmission for testing
            init_prev = ss.bernoulli(0.1),  # 10% initial prevalence
        )
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
            stop = ss.date('2010-12-31'),  # Longer simulation
            rand_seed = 1,
            verbose = 0
        )
    )
    
    print(f"Running simulation...")
    sim.run()
    
    # Check results
    results = sim.results.flatten()
    print(f"\nFinal results:")
    print(f"  New infections: {sum(results['tb_new_infections'])}")
    print(f"  New active cases: {sum(results['tb_new_active'])}")
    print(f"  New deaths: {sum(results['tb_new_deaths'])}")
    
    # Check if any transitions happened during the simulation
    print(f"\nTransitions during simulation:")
    for i in range(len(results['tb_new_active'])):
        new_active = results['tb_new_active'][i]
        new_deaths = results['tb_new_deaths'][i]
        if new_active > 0 or new_deaths > 0:
            print(f"  Time {i}: {new_active} new active, {new_deaths} new deaths")

if __name__ == '__main__':
    test_original_tb()
