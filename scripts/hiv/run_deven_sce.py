"""
TB-HIV Scenario Comparisons with Connector and Intervention Testing

This script demonstrates TB-HIV co-infection modeling with flexible configuration
of HIV interventions and the TB-HIV connector. It compares scenarios ranging from
no HIV to controlled HIV prevalence, showing how different HIV states affect TB.

Purpose:
--------
- Compare TB-only vs. TB-HIV co-infection scenarios
- Test TB-HIV connector effects on TB dynamics
- Evaluate HIV intervention effects on TB burden
- Demonstrate different HIV control strategies
- Show importance of TB-HIV interactions in modeling

Scenarios:
----------
1. **No HIV**: TB-only baseline
   - 0% HIV prevalence
   - No TB-HIV connector
   - Pure TB natural history
   
2. **Initial HIV prevalence = 30%**: High HIV, no control
   - 30% HIV prevalence at baseline
   - 77% on ART initially
   - TB-HIV connector enabled
   - Natural dynamics after initialization
   
3. **Controlled HIV Prevalence 30%**: Active HIV control
   - 30% HIV prevalence target maintained
   - 30% ART coverage maintained
   - TB-HIV connector enabled
   - Active intervention from 1981-2030

Key Concepts:
-------------
- **TB-HIV Connector**: Modifies TB progression rates based on HIV/ART status
  - acute_multiplier: 1.7× for acute HIV
  - latent_multiplier: 2.5× for chronic HIV
  - aids_multiplier: 2.9× for AIDS stage
- **HIV Interventions**: Maintain target prevalence and ART coverage
- **Scenario Comparison**: Shows net effect of HIV on TB burden

Components:
-----------
- TB disease module (from shared_functions)
- HIV disease module (configurable prevalence and ART)
- TB-HIV Connector with stage-specific multipliers
- HIV Interventions for prevalence/ART control
- Random network (Poisson(2) contacts)
- Very small population (100 agents) for fast testing

Usage:
------
    python scripts/hiv/run_deven_sce.py

Output:
-------
- Multi-panel plot comparing all scenarios
- Dark theme with viridis colormap
- Shows TB and HIV metrics over 51 years (1980-2030)
- Demonstrates connector and intervention effects

Notes:
------
This script is useful for:
- Understanding TB-HIV connector mechanics
- Validating intervention implementations
- Comparing HIV control strategies
- Teaching TB-HIV interaction concepts
- Quick testing of TB-HIV models (small population)
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import shared_functions as sf
import shared_functions as utils

def build_tbhiv_sim(include_intv=False, include_cnn=False, hiv_pars=None, hiv_intv_pars=None, Demgs=False) -> ss.Sim:
    """
    Construct a TB-HIV simulation with optional interventions and connector.
    
    Creates a flexible TB-HIV model where HIV disease, TB-HIV connector, and HIV
    interventions can be independently enabled or disabled. This allows testing
    each component's effect in isolation or combination.
    
    Parameters
    ----------
    include_intv : bool, default=False
        Whether to include HIV interventions (prevalence/ART control)
    include_cnn : bool, default=False
        Whether to include TB-HIV connector (disease interactions)
    hiv_pars : dict, optional
        Override HIV disease parameters. If None, uses default:
        - init_prev: Set by hiv_pars
        - init_onart: Set by hiv_pars
    hiv_intv_pars : dict, optional
        HIV intervention parameters (requires include_intv=True):
        - prevalence: Target HIV prevalence
        - percent_on_ART: Target ART coverage
        - start, stop: Intervention period
        - dt: Time step for intervention
    Demgs : bool, default=False
        Whether to include demographics (births/deaths)
        Currently not implemented in function body
    
    Returns
    -------
    ss.Sim
        Configured TB-HIV simulation ready to run
        
    Simulation Configuration:
    -------------------------
    - Period: 1980-2030 (51 years)
    - Population: 100 agents (very small for testing)
    - Time step: 7 days (weekly)
    - TB: From shared_functions.make_tb()
    - HIV: Configurable via hiv_pars
    - Network: Random with Poisson(2) contacts
    - Connector: Optional, with stage-specific multipliers
    - Interventions: Optional HIV control
    
    TB-HIV Connector Multipliers:
    ------------------------------
    - acute_multiplier: 1.7 (TB activation rate during acute HIV)
    - latent_multiplier: 2.5 (TB activation during chronic HIV)
    - aids_multiplier: 2.9 (TB activation during AIDS)
    
    Examples
    --------
    TB-only simulation (no HIV):
    
        >>> hiv_pars = dict(init_prev=ss.bernoulli(p=0.00),
        ...                 init_onart=ss.bernoulli(p=0.00))
        >>> sim = build_tbhiv_sim(hiv_pars=hiv_pars)
        >>> sim.run()
    
    TB-HIV with connector but no intervention:
    
        >>> hiv_pars = dict(init_prev=ss.bernoulli(p=0.30),
        ...                 init_onart=ss.bernoulli(p=0.77))
        >>> sim = build_tbhiv_sim(include_cnn=True, hiv_pars=hiv_pars)
        >>> sim.run()
    
    TB-HIV with intervention and connector:
    
        >>> intv_pars = dict(prevalence=0.30, percent_on_ART=0.30,
        ...                  start=ss.date('1981-05-01'),
        ...                  stop=ss.date('2030-12-31'),
        ...                  dt=ss.days(7))
        >>> sim = build_tbhiv_sim(include_intv=True, include_cnn=True,
        ...                       hiv_intv_pars=intv_pars)
        >>> sim.run()
    
    Notes
    -----
    The very small population (100 agents) makes this script fast for testing
    but results will be noisy. For production use, increase n_agents to 1000+.
    
    The connector parameters (acute_multiplier, etc.) are based on epidemiological
    evidence that HIV increases TB activation risk, with effect varying by HIV stage.
    """
    
    sim_pars = dict(
        dt=ss.days(7),   # Simulation's Time unit and time-step size.
        start=ss.date('1980-01-01'),  
        stop=ss.date('2030-12-31'), # Simulation's start and stop dates
        verbose=0,          # Verbosity level   
    )

    people = ss.People(n_agents=100)
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    tb = sf.make_tb()
    hiv = sf.make_hiv(hiv_pars=hiv_pars)
    
    # Please note, this multiplier is used to adjust the rate of progression 
    # from latent to presynptomatic TB (TB state 'rr_activation'):
    cnn_pars = dict(
                acute_multiplier     = 1.7,
                latent_multiplier    = 2.5,
                aids_multiplier      = 2.9,
                )
    connector = utils.make_tb_hiv_connector(pars=cnn_pars) if include_cnn else None
    interventions = utils.make_hiv_interventions(include=include_intv, pars=hiv_intv_pars) if include_intv else None
    
    return ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=network,
        interventions=interventions,
        connectors=connector,
        pars=sim_pars,
    )


def get_scenarios():
    """
    Define TB-HIV simulation scenarios with different HIV configurations.
    
    Returns a dictionary of scenarios that progressively add HIV disease and
    interventions to show their incremental effects on TB dynamics.
    
    Returns
    -------
    dict
        Dictionary mapping scenario names to build_tbhiv_sim() parameters:
        {scenario_name: dict(include_intv=..., include_cnn=..., hiv_pars=...), ...}
    
    Scenarios Defined:
    ------------------
    1. **No HIV**: TB-only baseline
       - include_intv=False: No interventions
       - include_cnn=False: No TB-HIV connector
       - hiv_pars: 0% prevalence, 0% on ART
       - Purpose: Pure TB dynamics without HIV confounding
    
    2. **Initial HIV prevalence = 30%**: Uncontrolled TB-HIV
       - include_intv=False: No interventions
       - include_cnn=True: TB-HIV connector active
       - hiv_pars: 30% initial prevalence, 77% on ART
       - Purpose: TB-HIV natural dynamics with high baseline ART
    
    3. **Controlled HIV Prevalence 30%**: Controlled TB-HIV
       - include_intv=True: Active HIV control
       - include_cnn=True: TB-HIV connector active
       - hiv_intv_pars: Maintain 30% prevalence, 30% ART from 1981-2030
       - Purpose: TB-HIV with active intervention management
    
    Examples
    --------
    Get scenarios and run individually:
    
        >>> scenarios = get_scenarios()
        >>> for name, kwargs in scenarios.items():
        ...     sim = build_tbhiv_sim(**kwargs)
        ...     sim.run()
    
    Run specific scenario:
    
        >>> scenarios = get_scenarios()
        >>> sim = build_tbhiv_sim(**scenarios['No HIV'])
        >>> sim.run()
    
    Notes
    -----
    The scenarios are ordered by increasing complexity:
    1. TB only (simplest)
    2. TB-HIV without intervention
    3. TB-HIV with intervention (most complex)
    
    This ordering helps identify which component (HIV disease vs. intervention)
    contributes most to observed changes in TB dynamics.
    """
    return {
        "No HIV": dict(
            include_intv=False,
            include_cnn=False,
            hiv_pars=dict(
                init_prev=ss.bernoulli(p=0.00),
                init_onart=ss.bernoulli(p=0.00)
        )),
        "Initial HIV prevalence = 30%": dict( 
            include_intv=False, 
            include_cnn=True,
            hiv_pars=dict(
                init_prev=ss.bernoulli(p=0.30),
                init_onart=ss.bernoulli(p=0.77)
        )),
        "Controlled HIV Prevalence 30%": dict(
            include_intv=True,
            include_cnn=True,
            hiv_intv_pars=dict(
                prevalence=0.30,
                percent_on_ART=0.30,
                start=ss.date('1981-05-01'),
                stop=ss.date('2030-12-31'),
                dt=ss.days(7),
                )
        ),
    }


def main():
    """
    Run all TB-HIV scenarios and plot comparative results.
    
    Executes all predefined scenarios sequentially, collects results, and generates
    a comprehensive comparison plot with dark theme and viridis colormap.
    
    Workflow:
    ---------
    1. Get scenario definitions from get_scenarios()
    2. Run each scenario and collect flattened results
    3. Plot results with dark theme for better visibility
    4. Display plot to user
    
    Output:
    -------
    - Console: Progress messages for each scenario
    - Plot: Multi-panel figure with all scenarios overlaid
    - Plot style: Dark theme with viridis colormap
    
    Notes
    -----
    The dark theme (dark=True) and viridis colormap provide good visibility
    for comparing multiple time series. The plot shows how HIV presence and
    control affect TB dynamics over the 51-year simulation period.
    """
    scenarios = get_scenarios()
    results = {}

    for name, kwargs in scenarios.items():
        print(f"\nRunning scenario: {name}")
        sim = build_tbhiv_sim(**kwargs)
        sim.run()
        results[name] = sim.results.flatten()

    # utils.plot_results(results)
    # utils.plot_results_1(results)
    utils.plot_results(results, dark=True, cmap='viridis' )

if __name__ == '__main__':
    main()