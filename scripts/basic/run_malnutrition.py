"""
Malnutrition Simulation Script

This script demonstrates how to set up and run a simulation with only the malnutrition
comorbidity module. It shows the natural dynamics of malnutrition in a population
over time without TB disease.

Purpose:
--------
This script is useful for:
- Understanding malnutrition dynamics independently from TB
- Testing the malnutrition module in isolation
- Developing baseline malnutrition scenarios
- Calibrating malnutrition parameters

Components:
-----------
- Malnutrition disease module
- Random contact network
- Demographics (births and deaths)
- 31-year simulation period (1990-2020)

Usage:
------
    python scripts/basic/run_malnutrition.py

Output:
-------
Displays plots of malnutrition metrics over time including prevalence,
incidence, and demographic dynamics.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def make_malnutrition():
    """
    Create a malnutrition-only simulation with default parameters.
    
    This function sets up a complete simulation focused on malnutrition dynamics
    without TB disease. It includes population, network, demographics, and the
    malnutrition disease module.
    
    The simulation models:
    - Malnutrition prevalence and transitions
    - Demographic changes (births and deaths)
    - Population dynamics over 31 years
    
    Returns
    -------
    ss.Sim
        Configured malnutrition simulation ready to run
        
    Examples
    --------
    Create and run malnutrition simulation:
    
        >>> sim = make_malnutrition()
        >>> sim.run()
        >>> sim.plot()
    
    Notes
    -----
    The simulation uses:
    - 200 agents (small population for fast execution)
    - Weekly time steps
    - 31-year period (1990-2020)
    - Balanced birth and death rates (both 5 per 1000)
    - Random network with average 5 contacts per person
    """
    # --------- Disease ----------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)
    
    # --------- People ----------
    n_agents = 200
    pop = ss.People(n_agents=n_agents)
    
    # -------- simulation -------
    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1990-01-01'),
        stop=ss.date('2020-12-31'),  # we dont use dur, as duration gets calculated internally.
    )
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))
    sim = ss.Sim(people=pop, 
                 diseases=nut, 
                 demographics=[deaths, births],
                 networks=net,
                 pars=sim_pars)
    return sim


if __name__ == '__main__':
    """
    Main execution block for malnutrition simulation.
    
    This block:
    1. Creates a malnutrition simulation
    2. Runs the simulation for 31 years (1990-2020)
    3. Flattens results for plotting
    4. Creates a combined plot with dark theme
    5. Displays plots showing malnutrition dynamics
    
    The plot shows malnutrition prevalence, incidence, and demographic
    changes over time in a 3-column grid layout with dark theme.
    """
    # Make Malnutrition simulation
    sim_n = make_malnutrition()
    
    # Run the simulation
    sim_n.run()
    
    # Flatten results into dictionary for plotting
    results = {'malnutrition': sim_n.results.flatten()}
    
    # Create combined plot with dark theme and 3-column layout
    mtb.plot_combined(results, n_cols=3, dark=True)
    
    