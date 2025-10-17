"""
Basic TB Simulation Script

This script demonstrates the simplest possible TB simulation setup using TBSim.
It creates a minimal simulation with only the TB disease module and default parameters,
then runs the simulation and plots the results.

Purpose:
--------
This script serves as the entry point for new users to understand TB modeling basics.
It shows:
- How to create a basic TB simulation
- How to run the simulation
- How to access and visualize results

Components:
-----------
- TB disease module with default parameters
- Weekly time steps (7 days)
- 70-year simulation period (1940-2010)
- Default population size and structure

Usage:
------
    python scripts/basic/run_tb.py

Output:
-------
Displays a combined plot showing all TB metrics over time using default visualization.
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt


def build_tbsim(sim_pars=None):
    """
    Build a basic TB simulation with default parameters.
    
    Creates the simplest possible TB simulation with only the TB disease
    module. Uses default parameters for population size, transmission rates,
    and disease progression.
    
    Parameters
    ----------
    sim_pars : dict, optional
        Additional simulation parameters to override defaults.
        Not currently used but reserved for future extensions.
    
    Returns
    -------
    ss.Sim
        Configured TB simulation ready to run
        
    Examples
    --------
    Create and run basic TB simulation:
    
        >>> sim = build_tbsim()
        >>> sim.run()
        >>> sim.plot()
    
    Notes
    -----
    The simulation uses:
    - 70-year period (1940-2010) to show long-term TB dynamics
    - Weekly time steps for computational efficiency
    - Default TB parameters (see mtb.TB documentation)
    - No demographics, interventions, or custom networks
    """

    sim = ss.Sim(diseases=mtb.TB(),pars=dict(dt = ss.days(7), start = ss.date('1940-01-01'), stop = ss.date('2010')))
    return sim

if __name__ == '__main__':
    """
    Main execution block for basic TB simulation.
    
    This block:
    1. Creates a basic TB simulation with default parameters
    2. Runs the simulation for 70 years (1940-2010)
    3. Prints simulation parameters for reference
    4. Flattens results for plotting
    5. Creates a combined plot of all TB metrics
    6. Displays the plot
    
    The plot shows all TB disease states and transitions over time,
    providing a complete picture of TB dynamics under default parameters.
    """
    # Create an empty Sim object (not used, kept for compatibility)
    sim = ss.Sim()
    
    # Build the actual TB simulation with disease module
    sim = build_tbsim()
    
    # Run the simulation
    sim.run()
    
    # Print simulation parameters for reference
    print(sim.pars)
    
    # Flatten results into a dictionary for plotting
    results = {'TB DEFAULTS  ': sim.results.flatten()}
    
    # Create combined plot showing all TB metrics
    mtb.plot_combined(results, title='TB MODEL WITH DEFAULT PARAMETERS', dark=False, heightfold=1.5)
    
    # Display the plot
    plt.show()