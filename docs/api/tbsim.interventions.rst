TB Interventions
===============

This module provides various intervention strategies for tuberculosis control and prevention.

Main Interventions Module
------------------------

.. automodule:: tbsim.interventions
   :members:
   :undoc-members:
   :show-inheritance:

Available Intervention Types
--------------------------

**BCG Vaccination** (`tbsim.interventions.bcg`)
   Bacillus Calmette-Guérin vaccination for TB prevention

**Beta Interventions** (`tbsim.interventions.beta`)
   Time-varying intervention parameters

**Treatment Protocols** (`tbsim.interventions.interventions`)
   Various TB treatment regimens and protocols

**Preventive Therapy** (`tbsim.interventions.tpt`)
   Isoniazid preventive therapy (IPT) implementation

**Enhanced Treatment** (`tbsim.interventions.enhanced_tb_treatment`)
   Advanced treatment with configurable drug types and protocols

**Basic Treatment** (`tbsim.interventions.tb_treatment`)
   Basic TB treatment with success/failure logic

**Diagnostic Testing** (`tbsim.interventions.tb_diagnostic`)
   TB diagnostic testing with configurable accuracy

**Enhanced Diagnostics** (`tbsim.interventions.enhanced_tb_diagnostic`)
   Advanced diagnostics with stratified parameters and multiple methods

**Health-Seeking Behavior** (`tbsim.interventions.healthseeking`)
   Modeling of care-seeking behavior for TB cases

**TB Health-Seeking** (`tbsim.interventions.tb_health_seeking`)
   TB-specific health-seeking with rate-based probabilities

**Drug Types and Parameters** (`tbsim.interventions.tb_drug_types`)
   Comprehensive TB drug regimen definitions and parameters

Subpackages
----------

.. toctree::
   :maxdepth: 4

   tbsim.interventions.bcg
   tbsim.interventions.beta
   tbsim.interventions.interventions
   tbsim.interventions.tpt
   tbsim.interventions.enhanced_tb_treatment
   tbsim.interventions.tb_treatment
   tbsim.interventions.tb_diagnostic
   tbsim.interventions.enhanced_tb_diagnostic
   tbsim.interventions.healthseeking
   tbsim.interventions.tb_health_seeking
   tbsim.interventions.tb_drug_types

Usage Examples
-------------

Adding BCG vaccination:

.. code-block:: python

   from tbsim.interventions.bcg import BCG
   from tbsim import TB
   
   tb = TB()
   bcg = BCG()
   
   sim = ss.Sim(
       diseases=tb,
       interventions=bcg
   )
   sim.run()

Implementing preventive therapy:

.. code-block:: python

   from tbsim.interventions.tpt import TPT
   
   tpt = TPT()
   sim = ss.Sim(interventions=tpt)

Key Features
-----------

- **Modular Design**: Mix and match different intervention types
- **Configurable Parameters**: Adjust intervention effectiveness and coverage
- **Time-Varying Implementation**: Model interventions that change over time
- **Integration**: Seamlessly integrate with TB and comorbidity models
- **Evaluation**: Built-in tools for assessing intervention impact

For detailed information about specific intervention types, see the individual subpackage documentation above. 