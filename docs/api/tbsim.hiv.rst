HIV Comorbidity Module
=====================

This module implements HIV disease modeling and its interactions with tuberculosis, including disease progression, ART treatment, and TB-HIV connectors.

HIV Disease Model
----------------

.. automodule:: tbsim.comorbidities.hiv
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

   from tbsim import HIV
   import starsim as ss

   sim = ss.Sim(diseases=HIV())
   sim.run()

HIV interventions:

.. code-block:: python

   from tbsim import HivInterventions

   # Create HIV intervention
   hiv_intervention = HivInterventions(pars={
       'mode': 'both',
       'prevalence': 0.20,
       'percent_on_ART': 0.775,
   })

TB-HIV integration:

.. code-block:: python

   from tbsim import TB, HIV, TB_HIV_Connector

   connector = TB_HIV_Connector(pars={
       'acute_multiplier': 1.5,
       'latent_multiplier': 2.0,
       'aids_multiplier': 3.0
   })

   sim = ss.Sim(
       diseases=[TB(), HIV()],
       connectors=connector
   )
   sim.run()
