"""
TB-HIV Co-infection Simulation with Interventions

This script demonstrates TB and HIV co-infection modeling with optional HIV interventions
and the TB-HIV connector. It shows how HIV status affects TB progression and how HIV
interventions can indirectly benefit TB outcomes.

Purpose:
--------
- Model TB-HIV co-infection with disease interactions
- Test HIV interventions' effects on TB outcomes
- Demonstrate TB-HIV connector functionality
- Compare scenarios with/without HIV interventions
- Show how ART coverage affects TB natural history

Key Concepts:
-------------
- **TB-HIV Connector**: Models how HIV status affects TB progression rates
- **HIV Interventions**: Maintain HIV prevalence and ART coverage at target levels
- **ART Effect**: Reduces TB activation risk in people living with HIV
- **Bidirectional Interaction**: HIV increases TB risk, TB outcomes affected by HIV control

Components:
-----------
- TB disease module (25% initial prevalence)
- HIV disease module (0% initial prevalence, controlled by interventions)
- TB-HIV Connector for disease interactions
- HIV Interventions for prevalence/ART control
- Random network (Poisson(2) contacts)

Usage:
------
    python scripts/hiv/run_tbhiv.py

Output:
-------
- Multi-panel plot comparing scenarios with/without HIV interventions
- Shows TB and HIV metrics side-by-side
- Demonstrates intervention effects on both diseases
- 56-year simulation period (1980-2035)

Notes:
------
This script is useful for understanding:
- How HIV interventions affect TB burden
- The importance of ART in TB control
- Disease interaction mechanics in the model
- Intervention cost-benefit analysis
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import shared_functions as sf


def build_tbhiv_sim(Intvs=True, tb=True, includehiv = True, Demgs= True, simpars = None) -> ss.Sim:
    """
    Build a TB-HIV co-infection simulation with optional interventions.
    
    Creates a complete TB-HIV simulation with configurable components. The simulation
    models disease interactions through the TB-HIV connector and allows testing
    various intervention strategies.
    
    Parameters
    ----------
    Intvs : bool, default=True
        Whether to include HIV interventions (prevalence/ART control)
    tb : bool, default=True
        Whether to include TB disease (useful for HIV-only tests)
    includehiv : bool, default=True
        Whether to include HIV disease (useful for TB-only tests)
    Demgs : bool, default=True
        Whether to include demographics (births/deaths)
        Currently not implemented in the function body
    simpars : dict, optional
        Override simulation parameters (dt, start, stop, rand_seed)
    
    Returns
    -------
    ss.Sim
        Configured TB-HIV simulation ready to run
        
    Simulation Configuration:
    -------------------------
    - Period: 1980-2035 (56 years)
    - Population: 1,000 agents
    - Time step: 7 days (weekly)
    - TB: From common_functions.make_tb()
    - HIV: 0% initial prevalence (controlled by intervention)
    - Network: Random with Poisson(2) contacts
    - Connector: TB-HIV interactions
    - Interventions: HivInterventions (if enabled)
    
    Examples
    --------
    Basic TB-HIV simulation with interventions:
    
        >>> sim = build_tbhiv_sim()
        >>> sim.run()
    
    TB-HIV without interventions:
    
        >>> sim = build_tbhiv_sim(Intvs=False)
        >>> sim.run()
    
    HIV-only simulation:
    
        >>> sim = build_tbhiv_sim(tb=False)
        >>> sim.run()
    
    Notes
    -----
    The TB-HIV connector models how HIV status affects TB:
    - Increases TB activation rate from latent TB
    - Modifies TB progression rates
    - Affects TB treatment outcomes
    - ART reduces (but doesn't eliminate) these effects
    """

    # --- Simulation Parameters ---
    default_simpars = dict(
        dt=ss.days(7),
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
    )
    people = ss.People(n_agents=1000)

    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
        init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
    )
    hiv = mtb.HIV(pars=hiv_pars)

    # --- Network ---
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    # --- Assemble Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=[sf.make_tb(include=tb), hiv],
        interventions=sf.make_hiv_interventions(include=Intvs),
        # demographics=demographics,
        networks=network,
        connectors=sf.make_tb_hiv_connector(include=tb),
        pars=default_simpars,
        verbose=0,
    )
    return sim


if __name__ == '__main__':
    args = []
    args.append(dict(Intvs=False,Demgs=False))
    args.append(dict(Intvs=True,Demgs=False))
    
    results = {}
    for i, arg in enumerate(args):
        print(f"Running scenario: {arg}")
        sim = build_tbhiv_sim(**arg).run()  
        results[str(arg)] = sim.results.flatten()
        
    sf.plot_results(results)