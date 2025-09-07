#!/usr/bin/env python3
"""
Basic simulation test to see if the simulation is working
"""

import tbsim as mtb
import starsim as ss
import numpy as np
import sys
import os

# Add the parent directory to the path to import the state machine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tbsim.tb import TB


def main():
    print("Basic Simulation Test")
    print("=" * 40)
    
    # Create a simple simulation
    pop = ss.People(n_agents=50)
    networks = ss.RandomNet()
    
    # Create TB module
    tb = TB(
        init_prev=0.2,  # 20% initial prevalence
        beta=ss.peryear(2.0),  # High transmission rate
    )
    
    # Create simulation
    sim = ss.Sim(
        people=pop,
        networks=networks,
        diseases=tb,
        start='1940-01-01',
        stop='1942-01-01',  # 2 years
        verbose=True  # Enable verbose output
    )
    
    print(f"Simulation created:")
    print(f"  Sim attributes: {[attr for attr in dir(sim) if not attr.startswith('_')]}")
    print(f"  Now: {sim.now}")
    print(f"  Ti: {sim.ti}")
    
    print(f"\nRunning simulation...")
    sim.run()
    
    print(f"\nSimulation completed!")
    print(f"  Results: {sim.results}")
    
    # Check if the disease module was called
    print(f"\nDisease module check:")
    print(f"  TB module: {tb}")
    print(f"  TB state: {tb.state}")
    print(f"  TB results: {tb.results}")


if __name__ == '__main__':
    main()
