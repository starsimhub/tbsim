BCG Vaccination Interventions
============================

This module provides BCG (Bacillus Calmette-Guérin) vaccination interventions for TBsim simulations.

Main BCG Module
---------------

.. automodule:: tbsim.interventions.bcg
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**BCGProtection**
   Main class for implementing BCG vaccination interventions

Key Features
-----------

- **Vaccination Modeling**: Realistic BCG vaccination implementation
- **Protection Levels**: Variable protection against TB infection
- **Age-based Targeting**: Vaccination strategies by age group
- **Effectiveness Tracking**: Monitor vaccination impact over time
- **Integration**: Seamless integration with TB and comorbidity models

Usage Examples
-------------

Basic BCG intervention:

.. code-block:: python

   from tbsim.interventions.bcg import BCGProtection
   from tbsim import TB
   
   # Add TB module and BCG vaccination
   tb = TB()
   bcg = BCGProtection()
   
   sim = ss.Sim(
       diseases=tb,
       interventions=bcg
   )
   sim.run()

BCG with custom parameters:

.. code-block:: python

   bcg = BCGProtection(
       protection_level=0.8,  # 80% protection
       target_age_groups=['infant', 'child'],
       vaccination_rate=0.9   # 90% coverage
   )

Key Methods
-----------

**Vaccination Management**
   - `check_eligibility()`: Determine who can receive vaccination
   - `is_protected()`: Check if individual has BCG protection
   - `step()`: Execute vaccination logic each time step

**Results Tracking**
   - `init_results()`: Initialize vaccination result tracking
   - `update_results()`: Update results during simulation
   - `get_summary_stats()`: Get vaccination summary statistics

**Analysis**
   - `calculate_tb_impact()`: Assess vaccination impact on TB outcomes
   - `debug_population()`: Debug vaccination status

BCG Effectiveness
----------------

BCG vaccination provides:
- **Infection Protection**: Reduced risk of TB infection
- **Disease Protection**: Lower progression to active disease
- **Severity Reduction**: Less severe disease if infection occurs
- **Childhood Protection**: Particularly effective in children

For detailed analysis of BCG impact, use the :doc:`tbsim.analyzers` module. 