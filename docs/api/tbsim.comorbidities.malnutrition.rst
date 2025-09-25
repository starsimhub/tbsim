Malnutrition Comorbidity Module
==============================

This module implements malnutrition disease modeling and its interactions with tuberculosis, including anthropometric measurements, nutritional interventions, and TB-malnutrition connectors.

Malnutrition Disease Model
-------------------------

.. automodule:: tbsim.comorbidities.malnutrition.malnutrition
   :members:
   :undoc-members:
   :show-inheritance:

TB-Malnutrition Connector
-------------------------

.. automodule:: tbsim.comorbidities.malnutrition.tb_malnut_cnn
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

Key Features
-----------

**LMS Growth Modeling**
   - Cole's LMS method for anthropometric measurements
   - Age and sex-specific reference parameters
   - Percentile to measurement conversion
   - Growth curve interpolation

**Nutritional Status Tracking**
   - Weight percentile evolution with random walk processes
   - Height percentile (assumed constant)
   - Micronutrient z-score tracking
   - Intervention effect modeling

**Intervention Framework**
   - Macronutrient supplementation effects on weight
   - Micronutrient supplementation effects on susceptibility
   - Time-dependent intervention impacts
   - Individual-level intervention assignment

**TB Integration**
   - Risk ratio calculations for TB progression
   - BMI-based risk functions
   - Supplementation protective effects
   - Real-time parameter adjustment

Usage Examples
-------------

Basic malnutrition simulation:

.. code-block:: python

   from tbsim.comorbidities.malnutrition.malnutrition import Malnutrition
   import starsim as ss
   
   sim = ss.Sim(diseases=Malnutrition())
   sim.run()

With custom parameters:

.. code-block:: python

   from tbsim.comorbidities.malnutrition.malnutrition import Malnutrition
   
   malnutrition = Malnutrition(pars={
       'init_prev': 0.05,     # 5% initial malnutrition prevalence
       'beta': 0.8            # Custom transmission rate
   })
   
   sim = ss.Sim(diseases=malnutrition)
   sim.run()

TB-malnutrition integration:

.. code-block:: python

   from tbsim.comorbidities.malnutrition.tb_malnut_cnn import TB_Nutrition_Connector
   from tbsim import TB, Malnutrition
   
   # Add both diseases and connector for interactions
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

Accessing anthropometric data:

.. code-block:: python

   # Get current measurements
   weight_kg = malnutrition.weight()           # Weight in kilograms
   height_cm = malnutrition.height()           # Height in centimeters
   
   # Get percentiles
   weight_percentiles = malnutrition.weight_percentile
   height_percentiles = malnutrition.height_percentile
   micronutrient_status = malnutrition.micro
   
   # Get intervention status
   receiving_macro = malnutrition.receiving_macro
   receiving_micro = malnutrition.receiving_micro

Mathematical Framework
---------------------

**LMS Transformation**
   - X = M × (L×S×Z + 1)^(1/L) for L ≠ 0
   - X = M × exp(S×Z) for L = 0
   - Where X = measurement, M = median, L = skewness, S = coefficient of variation
   - Z = Φ^(-1)(percentile) is the inverse normal CDF

**Weight Evolution**
   - Random walk process: ΔW_i(t) ~ N(μ_i(t), σ_i(t)²)
   - Drift function: μ_i(t) = 1.0 × t for supplemented individuals
   - Scale function: σ_i(t) = 0.01 × t for all individuals
   - Percentile clipping: W_i(t+1) = clip(W_i(t+1), 0.025, 0.975)

**TB Risk Modification**
   - BMI calculation: BMI = 10,000 × weight(kg) / height(cm)²
   - Lönnroth relationship: log(incidence) = -0.05×(BMI-15) + 2
   - Sigmoid transformation: RR = scale / (1 + 10^(-slope × (x-x0)))
   - Supplementation effects: RR = 1.0 for non-supplemented, rate_ratio for supplemented

**Susceptibility Modification**
   - Relative susceptibility based on micronutrient status
   - Threshold-based logic: rel_sus = 1.0 if micro ≥ 0.2, 2.0 if micro < 0.2
   - Z-score threshold represents approximately 42nd percentile

Data Requirements
----------------

**Anthropometry Reference Data**
   - File: `tbsim/data/anthropometry.csv`
   - Columns: Sex, Age, Weight_L, Weight_M, Weight_S, Height_L, Height_M, Height_S
   - Age in months, LMS parameters for weight and height
   - WHO growth standards or custom reference data

**Intervention Parameters**
   - Macronutrient supplementation timing and duration
   - Micronutrient supplementation protocols
   - RATIONS trial specific parameters
   - Age and eligibility criteria

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings.
