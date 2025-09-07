#!/usr/bin/env python3
"""
Test to see if the original TB class step method is called
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
    print("Original TB Step Test")
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
        verbose=False
    )
    
    print(f"Running simulation...")
    
    # Add debug output to see if step method is called
    original_step = tb.step
    step_count = 0
    
    def debug_step():
        nonlocal step_count
        step_count += 1
        print(f"  Step {step_count} called")
        return original_step()
    
    tb.step = debug_step
    
    sim.run()
    
    print(f"  Total steps called: {step_count}")


if __name__ == '__main__':
    main()
