"""
Simple TB Simulation Example
============================

This script demonstrates a basic TB simulation using Starsim v3.0.1.
It shows the minimal setup required to run a TB simulation.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a simple TB simulation
    sim = ss.Sim(
        # Population
        people=ss.People(n_agents=500),
        
        # Modules - all in a single list for v3.0.1
        modules=[
            # Network for disease transmission
            ss.RandomNet(dict(n_contacts=ss.poisson(lam=3), dur=0)),
            
            # TB disease module
            mtb.TB(dict(
                init_prev=ss.bernoulli(0.02),  # 2% initial prevalence
                beta=ss.peryear(0.01),         # Transmission rate
            )),
            
            # Demographics
            ss.Births(pars=dict(birth_rate=15)),
            ss.Deaths(pars=dict(death_rate=10)),
        ],
        
        # Simulation parameters
        pars=dict(
            dt=ss.days(7),                    # Weekly time steps
            start=ss.date('2020-01-01'),      # Start date
            stop=ss.date('2025-12-31'),       # End date (5 years)
            verbose=0,                        # Reduce output
        )
    )
    
    # Run the simulation
    print("Running TB simulation...")
    sim.run()
    
    # Display basic results
    print(f"\nSimulation completed!")
    print(f"Final population: {sim.people.alive.sum()}")
    
    # Find the TB module in the module list
    tb_module = None
    for module in sim.module_list:
        if isinstance(module, mtb.TB):
            tb_module = module
            break
    
    if tb_module:
        print(f"TB cases: {(tb_module.state > 0).sum()}")
    else:
        print("TB module not found")
    
    # Plot results
    print("\nGenerating plots...")
    results = sim.results.flatten()
    results = {'simple_tb': results}
    
    # Create a simple plot
    mtb.plot_combined(
        results, 
        dark=False, 
        # n_cols=2, 
        # filter=mtb.FILTERS.important_metrics
    )
    
    # plt.tight_layout()
    # plt.show()
    
    return sim

if __name__ == '__main__':
    sim = main()
