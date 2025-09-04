Preventive Therapy Interventions
==============================

This module provides Isoniazid Preventive Therapy (IPT) interventions for TBsim simulations.

Main TPT Module
---------------

.. automodule:: tbsim.interventions.tpt
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**TPT**
   Main class for implementing Isoniazid Preventive Therapy

Key Features
-----------

- **Preventive Therapy**: IPT implementation for latent TB infection
- **Risk-based Targeting**: Therapy for high-risk individuals
- **Treatment Protocols**: Standard IPT regimens and durations
- **Effectiveness Tracking**: Monitor therapy impact over time
- **Integration**: Seamless integration with TB and comorbidity models

Usage Examples
-------------

Basic TPT intervention:

.. code-block:: python

   from tbsim.interventions.tpt import TPT
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add preventive therapy
   tpt = TPT()
   sim.add_module(tpt)
   
   sim.run()

TPT with custom parameters:

.. code-block:: python

   tpt = TPT(
       treatment_duration=6,  # 6 months of therapy
       target_groups=['hiv_positive', 'household_contacts'],
       effectiveness=0.9       # 90% effectiveness
   )

Key Methods
-----------

**Therapy Management**
   - `identify_candidates()`: Find eligible individuals for therapy
   - `start_therapy()`: Begin preventive therapy
   - `monitor_adherence()`: Track therapy adherence
   - `step()`: Execute therapy logic each time step

**Results Tracking**
   - `init_results()`: Initialize therapy result tracking
   - `update_results()`: Update results during simulation
   - `get_summary_stats()`: Get therapy summary statistics

**Analysis**
   - `calculate_effectiveness()`: Assess therapy impact
   - `identify_barriers()`: Find factors affecting therapy success

TPT Effectiveness
----------------

Isoniazid Preventive Therapy provides:
- **Infection Prevention**: Reduced risk of TB infection
- **Disease Prevention**: Lower progression to active disease
- **Risk Reduction**: Particularly effective in high-risk groups
- **Cost-effectiveness**: High impact at relatively low cost

Target Populations
-----------------

TPT is typically targeted at:
- **HIV-positive individuals**: High risk of TB progression
- **Household contacts**: Recent exposure to active TB
- **Immunocompromised**: Weakened immune systems
- **High-risk groups**: Based on demographic and health factors

For detailed analysis of TPT impact, use the :doc:`tbsim.analyzers` module. 