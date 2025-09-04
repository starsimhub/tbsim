Basic TB Treatment
=================

This module provides basic TB treatment functionality for TBsim simulations.

Main TB Treatment Module
------------------------

.. automodule:: tbsim.interventions.tb_treatment
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**TBTreatment**
   Basic TB treatment intervention with success/failure logic

Key Features
-----------

- **Treatment Initiation**: Start treatment for diagnosed TB cases
- **Success/Failure Logic**: Realistic treatment outcome modeling
- **Care-seeking Behavior**: Post-treatment health-seeking patterns
- **Flag Management**: Reset diagnosis and testing flags after failure
- **Result Tracking**: Comprehensive treatment outcome monitoring

Usage Examples
-------------

Basic TB treatment:

.. code-block:: python

   from tbsim.interventions.tb_treatment import TBTreatment
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add basic treatment
   treatment = TBTreatment()
   sim.add_module(treatment)
   
   sim.run()

Custom treatment parameters:

.. code-block:: python

   treatment = TBTreatment(
       treatment_success_rate=0.9,  # 90% success rate
       reseek_multiplier=3.0,       # 3x care-seeking after failure
       reset_flags=True             # Reset diagnosis flags
   )

Key Methods
-----------

**Treatment Management**
   - `step()`: Execute treatment logic each time step
   - `init_results()`: Initialize treatment result tracking
   - `update_results()`: Update results during simulation

**Treatment Logic**
   - Automatically identifies diagnosed TB cases
   - Applies treatment success/failure probabilities
   - Manages post-treatment care-seeking behavior
   - Tracks treatment outcomes and statistics

Treatment Outcomes
-----------------

The module handles:
- **Treatment Success**: Complete TB clearance and recovery
- **Treatment Failure**: Failed treatment with renewed care-seeking
- **Care-seeking Multipliers**: Increased health-seeking after failure
- **Flag Management**: Reset of diagnosis and testing status

For advanced treatment capabilities, see the :doc:`tbsim.interventions.enhanced_tb_treatment` module.
