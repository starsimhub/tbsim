Malnutrition Comorbidity Module
==============================

This module implements malnutrition disease modeling and its interactions with tuberculosis, including anthropometric measurements, nutritional interventions, and TB-malnutrition connectors.

Malnutrition Disease Model
-------------------------

.. automodule:: tbsim.comorbidities.malnutrition
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The malnutrition comorbidity module provides comprehensive modeling of nutritional status and its bidirectional interactions with tuberculosis:

**Anthropometric Modeling**
   - **Weight and Height**: LMS (Lambda-Mu-Sigma) method for growth reference curves
   - **Percentile Tracking**: Individual weight and height percentiles (0.0-1.0)
   - **Micronutrient Status**: Z-score based micronutrient tracking
   - **Age and Sex Specific**: Reference data from WHO growth standards

**Nutritional Interventions**
   - **Macronutrient Supplementation**: Affects weight percentile evolution
   - **Micronutrient Supplementation**: Influences susceptibility to TB
   - **RATIONS Trial Integration**: Framework for nutritional intervention studies
   - **Dynamic Effects**: Time-dependent intervention impacts

**TB-Malnutrition Interactions**
   - Malnutrition modifies TB activation and clearance rates
   - BMI-based risk functions using Lönnroth et al. relationships
   - Supplementation effects on TB dynamics
   - Bidirectional disease interactions

Usage Examples
-------------

Basic malnutrition simulation:

.. code-block:: python

   from tbsim import Malnutrition
   import starsim as ss

   sim = ss.Sim(diseases=Malnutrition())
   sim.run()

TB-malnutrition integration:

.. code-block:: python

   from tbsim import TB, Malnutrition, TB_Nutrition_Connector

   connector = TB_Nutrition_Connector(pars={
       'rr_activation_func': TB_Nutrition_Connector.lonnroth_bmi_rr,
       'rr_clearance_func': TB_Nutrition_Connector.supplementation_rr,
       'relsus_func': TB_Nutrition_Connector.compute_relsus
   })

   sim = ss.Sim(
       diseases=[TB(), Malnutrition()],
       connectors=connector
   )
   sim.run()
