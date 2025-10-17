"""
HIV-Only Simulation Module

This script demonstrates HIV simulation in isolation, without TB disease. It is useful
for testing HIV dynamics, interventions, and demographics independently before adding
the complexity of TB-HIV co-infection.

Purpose:
--------
- Test HIV disease module in isolation
- Explore HIV intervention effects (ART coverage, prevalence control)
- Test demographic effects (births/deaths) on HIV dynamics
- Validate HIV natural history and intervention mechanics
- Provide baseline HIV-only scenarios for comparison with TB-HIV models

Scenarios:
----------
The script runs multiple scenarios comparing:
- HIV with/without interventions
- HIV with/without demographics (births and deaths)

Components:
-----------
- HIV disease module (30% initial prevalence, 50% on ART)
- Random network (average 5 contacts per person)
- Optional HivInterventions for prevalence/ART control
- Optional demographics (births and deaths)

Usage:
------
    python scripts/hiv/run_hiv.py

Output:
-------
- Multi-panel plot comparing scenarios
- Results show HIV prevalence, ART coverage, deaths over time
- 56-year simulation period (1980-2035)

Notes:
------
This is a foundational script for understanding HIV dynamics before
adding TB co-infection complexity. The scenarios help identify how
interventions and demographics affect HIV outcomes independently.
"""

import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
from shared_functions import make_hiv_interventions, make_demographics, plot_results


def sim_setup( n_agents=10_000,
            Intvs=True,
            Demgs=False,
            verbose_log=False,
        ) -> ss.Sim:
    """
    Create an HIV-only simulation with configurable interventions and demographics.
    
    This function builds a complete HIV simulation without TB disease. It allows
    testing HIV natural history, interventions, and demographic effects in isolation.
    
    Parameters
    ----------
    n_agents : int, default=10_000
        Number of agents in the simulation population
    Intvs : bool, default=True
        Whether to include HIV interventions (ART and prevalence control)
    Demgs : bool, default=False
        Whether to include demographics (births and deaths)
    verbose_log : bool, default=False
        Whether to show detailed simulation progress messages
    
    Returns
    -------
    ss.Sim
        Configured HIV simulation ready to run
        
    Simulation Configuration:
    -------------------------
    - Period: 1980-2035 (56 years)
    - Time step: 7 days (weekly)
    - Initial HIV prevalence: 30%
    - Initial ART coverage: 50%
    - Random network: Poisson(5) contacts
    - Interventions: HivInterventions (if enabled)
    - Demographics: Balanced births/deaths (if enabled)
    
    Examples
    --------
    Basic HIV simulation:
    
        >>> sim = sim_setup()
        >>> sim.run()
    
    HIV without interventions:
    
        >>> sim = sim_setup(Intvs=False)
        >>> sim.run()
    
    HIV with demographics:
    
        >>> sim = sim_setup(Demgs=True)
        >>> sim.run()
    
    Notes
    -----
    The HIV interventions maintain prevalence and ART coverage at target levels,
    which is useful for studying steady-state HIV dynamics. Without interventions,
    HIV prevalence follows natural epidemic curves.
    """

    sim_pars = dict(
        dt=ss.days(7),
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        verbose=verbose_log
    )

    people = ss.People(n_agents=n_agents)

    hiv = mtb.HIV(pars=dict(
        init_prev=ss.bernoulli(p=0.30),
        init_onart=ss.bernoulli(p=0.50),
        dt=ss.days(7),
    ))
    
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    sim = ss.Sim(
        people=people,
        diseases=hiv,
        pars=sim_pars,
        networks=network,
        interventions=make_hiv_interventions(Intvs),    
        demographics=make_demographics(Demgs),      
    )

    return sim

# HIV Basic Run
if __name__ == '__main__':
    args = []
    # args.append(dict(Intvs=False,Demgs=False))
    # args.append(dict(Intvs=False,Demgs=True))
    args.append(dict(Intvs=True,Demgs=True))
    args.append(dict(Intvs=True,Demgs=False))
    results = {}
    for i, arg in enumerate(args):
        print(f"Running scenario: {arg}")
        sim = sim_setup(**arg).run()  
        results[str(arg)] = sim.results.flatten()
        
    plot_results(results, dark=False)