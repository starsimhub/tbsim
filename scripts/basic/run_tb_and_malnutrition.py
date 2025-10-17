"""
TB and Malnutrition Co-Infection Simulation

This script demonstrates how to model TB and malnutrition as co-occurring comorbidities
using the TB-Nutrition connector. It shows how the two diseases interact and affect
each other's dynamics.

Purpose:
--------
This script demonstrates:
- Setting up TB-malnutrition co-infection models
- Using the TB-Nutrition connector for disease interactions
- Modeling socioeconomic status (SES) effects on disease
- Comparing different simulation configurations
- Using two different builder functions for flexibility

Key Features:
-------------
- TB-Nutrition connector models bidirectional disease interactions
- Malnutrition increases TB susceptibility and progression
- TB can worsen nutritional status
- SES (socioeconomic status) as an extra population state
- Demographics included (births and deaths via Pregnancy)

Components:
-----------
- TB disease module (1% annual transmission, 25% initial prevalence)
- Malnutrition module with default parameters
- TB-Nutrition connector for disease interactions
- Random contact network (Poisson with λ=5)
- Demographics: Pregnancy (15 per 1000) and Deaths (10 per 1000)
- Extra state: SES (30% probability of low SES)

Usage:
------
    python scripts/basic/run_tb_and_malnutrition.py

Output:
-------
Displays two simulation runs:
1. Using make_tb_nut() - verbose configuration
2. Using make_tb_nut_02() - compact configuration with custom parameters
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

def make_tb_nut():
    """
    Create a TB-malnutrition co-infection simulation (verbose configuration).
    
    This function creates a complete simulation modeling TB and malnutrition
    as interacting comorbidities. It uses a verbose configuration style with
    clearly separated components.
    
    The simulation includes:
    - TB disease dynamics
    - Malnutrition dynamics
    - TB-Nutrition connector for bidirectional interactions
    - Demographics (births and deaths)
    - Socioeconomic status (SES) as population attribute
    
    Returns
    -------
    ss.Sim
        Configured TB-malnutrition simulation ready to run
        
    Examples
    --------
    Create and run simulation:
    
        >>> sim = make_tb_nut()
        >>> sim.run()
        >>> sim.plot()
    
    Notes
    -----
    The simulation runs from 1990-2010 (21 years) with:
    - 10,000 agents
    - Weekly time steps
    - 30% of population with low SES (Bernoulli p=0.3)
    - Balanced demographics (15 births, 10 deaths per 1000)
    - Verbose progress reporting (every 10% completion)
    """
    # --------- People ----------
    n_agents = 10000
    extra_states = [
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = ss.peryear(0.01), 
        init_prev = 0.25,
        )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # -------- Network ---------
    # Network parameters
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
        )
    # Initialize a random network
    net = ss.RandomNet(net_pars)

    # Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # Connector
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt=ss.days(7),
        start = ss.date('1990-01-01'),
        stop = ss.date('2010-12-31'),
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    sim.pars.verbose = 0.1 # Print status every 10% of simulation

    return sim


def make_tb_nut_02(agents=1000, start=1980, stop=2020, dt=ss.days(7)):
    """
    Create a TB-malnutrition co-infection simulation (compact configuration).
    
    This function provides a more compact way to create the same TB-malnutrition
    simulation as make_tb_nut(), with parameters exposed for easy customization.
    It demonstrates a cleaner coding style using inline dictionary definitions.
    
    Parameters
    ----------
    agents : int, default=1000
        Number of agents in the population
    start : int, default=1980
        Simulation start year
    stop : int, default=2020
        Simulation stop year
    dt : ss.TimePar, default=ss.days(7)
        Simulation time step (default: weekly)
    
    Returns
    -------
    ss.Sim
        Configured TB-malnutrition simulation ready to run
        
    Examples
    --------
    Create with default parameters:
    
        >>> sim = make_tb_nut_02()
    
    Create with custom population and time range:
    
        >>> sim = make_tb_nut_02(agents=5000, start=2000, stop=2030)
    
    Notes
    -----
    This function uses the same components as make_tb_nut() but with:
    - Compact dictionary-based parameter specification
    - Parameterized population size and time range
    - More flexible configuration options
    - Same TB-Nutrition connector and disease interactions
    """
    pop = ss.People(n_agents=agents, extra_states=[ss.FloatArr('SES', default=ss.bernoulli(p=0.3))])
    tb = mtb.TB({'beta': ss.peryear(0.01), 'init_prev': 0.25})
    nut = mtb.Malnutrition({})
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    dems = [ss.Pregnancy(pars={'fertility_rate': 15}), ss.Deaths(pars={'death_rate': 10})]
    cn = mtb.TB_Nutrition_Connector({})
    sim_pars = {'dt': dt, 'start': ss.date(f'{start}-01-01'), 'stop': ss.date(f'{stop}-12-31')}
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    sim.pars.verbose = 0.1
    return sim

if __name__ == '__main__':  
    """
    Main execution block demonstrating both TB-malnutrition builder functions.
    
    This block:
    1. Creates and runs simulation using make_tb_nut() (verbose style)
    2. Displays results
    3. Creates and runs simulation using make_tb_nut_02() (compact style) with custom agents
    4. Uses built-in plot() method to visualize results
    5. Displays plots
    
    This demonstrates that both builder functions produce equivalent simulations
    but with different levels of parameter customization and code style.
    """
    # Run simulation using verbose configuration style
    sim_tbn = make_tb_nut()
    sim_tbn.run()
    plt.show()

    # Run simulation using compact configuration style with custom population
    sim_tbn = make_tb_nut_02(agents=1500)
    sim_tbn.run()
    sim_tbn.plot()  # Use built-in plotting method
    plt.show()