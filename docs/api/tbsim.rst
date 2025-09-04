Core TBsim Package
=================

This page documents the core package and main modules of the TBsim framework.

Main Package
-----------

.. automodule:: tbsim
   :members:
   :undoc-members:
   :show-inheritance:

Package Overview
---------------

TBsim is a comprehensive tuberculosis modeling framework built on the Starsim platform. It provides:

- **Core TB Disease Model**: Complete tuberculosis simulation with state transitions
- **Comorbidity Modeling**: HIV and malnutrition integration with TB dynamics
- **Intervention Framework**: DOTS, BCG, diagnostics, and treatment protocols
- **Network Structures**: Household and social network modeling
- **Analysis Tools**: Dwell time analysis, visualization, and post-processing
- **Data Utilities**: Demographics, plotting, and probability calculations

Key Components
-------------

**TB Disease Model** (`tbsim.tb`)
   Main tuberculosis simulation class that handles disease dynamics, transmission, and state transitions.

**TBS State Enumeration** (`tbsim.tb.TBS`)
   Comprehensive state definitions for TB progression including latent, active, and treatment states.

**Network Structures** (`tbsim.networks`)
   Social network implementations for modeling transmission patterns and household structures.

**Comorbidity Connectors**
   - `tbsim.comorbidities.hiv.tb_hiv_cnn.TB_HIV_Connector`: TB-HIV interaction modeling
   - `tbsim.comorbidities.malnutrition.tb_malnut_cnn.TB_Nutrition_Connector`: TB-malnutrition interactions

**Analysis Framework** (`tbsim.analyzers`)
   Comprehensive tools for analyzing simulation results including dwell time analysis and visualization.

**Intervention Modules** (`tbsim.interventions.*`)
   Complete intervention framework for TB control including DOTS implementation, enhanced diagnostics, and treatment protocols.

Usage Examples
-------------

Basic TB simulation:

.. code-block:: python

   from tbsim import TB
   import starsim as ss
   
   sim = ss.Sim()
   tb = TB()
   sim.add_module(tb)
   sim.run()

With comorbidities:

.. code-block:: python

   from tbsim import TB, HIV, Malnutrition
   from tbsim.comorbidities.hiv.tb_hiv_cnn import TB_HIV_Connector
   from tbsim.comorbidities.malnutrition.tb_malnut_cnn import TB_Nutrition_Connector
   
   sim = ss.Sim()
   tb = TB()
   hiv = HIV()
   malnutrition = Malnutrition()
   
   sim.add_module(tb)
   sim.add_module(hiv)
   sim.add_module(malnutrition)
   
   # Add connectors for disease interactions
   sim.add_connector(TB_HIV_Connector())
   sim.add_connector(TB_Nutrition_Connector())
   
   sim.run()

With interventions:

.. code-block:: python

   from tbsim.interventions.enhanced_tb_treatment import create_dots_treatment
   from tbsim.interventions.bcg import BCGProtection
   
   # Add DOTS treatment
   dots = create_dots_treatment()
   sim.add_intervention(dots)
   
   # Add BCG vaccination
   bcg = BCGProtection()
   sim.add_intervention(bcg)
   
   sim.run()

Analysis and visualization:

.. code-block:: python

   from tbsim.analyzers import DwtAnalyzer
   
   # Add analyzer to simulation
   analyzer = DwtAnalyzer(scenario_name="Baseline")
   sim.add_analyzer(analyzer)
   
   sim.run()
   
   # Access results and create visualizations
   analyzer.sankey_agents()
   analyzer.histogram_with_kde()
   analyzer.graph_state_transitions_enhanced() 