HIV Comorbidity Module
=====================

This module implements HIV disease modeling and its interactions with tuberculosis, including disease progression, ART treatment, and TB-HIV connectors.

HIV Disease Model
----------------

.. automodule:: tbsim.comorbidities.hiv.hiv
   :members:
   :undoc-members:
   :show-inheritance:

HIV State Enumeration
--------------------

.. automodule:: tbsim.comorbidities.hiv.hiv.HIVState
   :members:
   :undoc-members:
   :show-inheritance:

HIV Interventions
----------------

.. automodule:: tbsim.comorbidities.hiv.intervention
   :members:
   :undoc-members:
   :show-inheritance:

TB-HIV Connector
----------------

.. automodule:: tbsim.comorbidities.hiv.tb_hiv_cnn
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The HIV comorbidity module provides comprehensive modeling of HIV infection and its bidirectional interactions with tuberculosis:

**HIV Disease States**
   - **ATRISK (0)**: HIV-negative but at risk
   - **ACUTE (1)**: Recently infected with HIV
   - **LATENT (2)**: Chronic HIV infection
   - **AIDS (3)**: Advanced stage of HIV infection

**Disease Progression**
   - Stochastic progression through HIV states
   - ART treatment effects on progression rates
   - Age and risk factor considerations
   - Natural history modeling

**TB-HIV Interactions**
   - HIV modifies TB progression and mortality risks
   - State-specific risk multipliers (ACUTE: 1.22x, LATENT: 1.90x, AIDS: 2.60x)
   - ART effects on TB-HIV interactions
   - Bidirectional disease dynamics

Key Features
-----------

**HIV Disease Modeling**
   - Initial infection and ART status assignment
   - Disease progression through ACUTE → LATENT → AIDS
   - ART treatment effects on progression rates
   - Population-level prevalence and incidence tracking

**Intervention Framework**
   - Prevalence adjustment to target levels
   - ART coverage management
   - Age-specific targeting
   - Dynamic intervention timing

**TB-HIV Integration**
   - Risk ratio calculations for TB progression
   - State-dependent risk modifications
   - ART status consideration in risk calculations
   - Real-time parameter adjustment during simulation

Usage Examples
-------------

Basic HIV simulation:

.. code-block:: python

   from tbsim.comorbidities.hiv.hiv import HIV
   import starsim as ss
   
   sim = ss.Sim()
   hiv = HIV()
   sim.add_module(hiv)
   sim.run()

With custom parameters:

.. code-block:: python

   from tbsim.comorbidities.hiv.hiv import HIV
   
   hiv = HIV(pars={
       'init_prev': 0.15,           # 15% initial HIV prevalence
       'init_onart': 0.60,          # 60% of infected on ART initially
       'acute_to_latent': 0.1,      # Progression rate from acute to latent
       'latent_to_aids': 0.05,      # Progression rate from latent to AIDS
       'art_progression_factor': 0.1 # ART reduces progression by 90%
   })
   
   sim.add_module(hiv)
   sim.run()

HIV interventions:

.. code-block:: python

   from tbsim.comorbidities.hiv.intervention import HivInterventions
   
   # Create HIV intervention
   hiv_intervention = HivInterventions(pars={
       'mode': 'both',              # Apply both prevalence and ART adjustments
       'prevalence': 0.20,          # Target 20% HIV prevalence
       'percent_on_ART': 0.775,     # Target 77.5% ART coverage
       'start': ss.date('2000-01-01'),
       'stop': ss.date('2035-12-31')
   })
   
   sim.add_intervention(hiv_intervention)
   sim.run()

TB-HIV integration:

.. code-block:: python

   from tbsim.comorbidities.hiv.tb_hiv_cnn import TB_HIV_Connector
   from tbsim import TB, HIV
   
   # Add both diseases
   sim.add_module(TB())
   sim.add_module(HIV())
   
   # Add connector for interactions
   connector = TB_HIV_Connector(pars={
       'acute_multiplier': 1.5,     # Custom risk multiplier for acute HIV
       'latent_multiplier': 2.0,    # Custom risk multiplier for latent HIV
       'aids_multiplier': 3.0       # Custom risk multiplier for AIDS
   })
   
   sim.add_connector(connector)
   sim.run()

Accessing results:

.. code-block:: python

   # HIV-specific results
   hiv_results = hiv.results
   hiv_prevalence = hiv_results.hiv_prevalence
   art_coverage = hiv_results.on_art
   state_distribution = {
       'atrisk': hiv_results.atrisk,
       'acute': hiv_results.acute,
       'latent': hiv_results.latent,
       'aids': hiv_results.aids
   }
   
   # Current states
   current_hiv_states = hiv.state
   on_art_status = hiv.on_ART

Mathematical Framework
---------------------

**HIV Progression**
   - Exponential progression rates between states
   - ART modification: effective_rate = base_rate × art_factor
   - State-specific progression probabilities

**TB-HIV Risk Ratios**
   - RR_activation = base_rate × HIV_multiplier
   - Multipliers: ACUTE (1.22), LATENT (1.90), AIDS (2.60)
   - Real-time adjustment during simulation

**Intervention Effects**
   - Prevalence targeting with age constraints
   - ART coverage management with eligibility criteria
   - Dynamic parameter adjustment based on targets

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings.
