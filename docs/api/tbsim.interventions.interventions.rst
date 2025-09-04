Core Intervention Functionality
==============================

This module provides the core intervention infrastructure and base classes for TBsim interventions.

Main Interventions Module
------------------------

.. automodule:: tbsim.interventions.interventions
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**BaseIntervention**
   Base class for all TBsim interventions

**InterventionManager**
   Central manager for coordinating multiple interventions

Key Features
-----------

- **Base Classes**: Common functionality for all interventions
- **Intervention Management**: Coordinate multiple intervention types
- **Standardized Interface**: Consistent API across interventions
- **Integration Support**: Easy integration with TB and comorbidity models
- **Extensibility**: Framework for custom intervention development

Usage Examples
-------------

Creating custom interventions:

.. code-block:: python

   from tbsim.interventions.interventions import BaseIntervention
   
   class CustomIntervention(BaseIntervention):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)
           # Custom initialization
       
       def step(self):
           # Custom intervention logic
           pass

Using intervention manager:

.. code_block:: python

   from tbsim.interventions.interventions import InterventionManager
   
   # Create manager for multiple interventions
   manager = InterventionManager()
   
   # Add interventions
   manager.add_intervention(bcg)
   manager.add_intervention(tpt)
   manager.add_intervention(treatment)

Key Methods
-----------

**BaseIntervention**
   - `__init__()`: Initialize intervention parameters
   - `step()`: Execute intervention logic each time step
   - `init_results()`: Initialize result tracking
   - `update_results()`: Update results during simulation

**InterventionManager**
   - `add_intervention()`: Register new intervention
   - `remove_intervention()`: Remove intervention
   - `get_interventions()`: List all active interventions
   - `step_all()`: Execute all interventions

Intervention Lifecycle
---------------------

1. **Initialization**: Set up parameters and state
2. **Registration**: Add to simulation framework
3. **Execution**: Run intervention logic each time step
4. **Tracking**: Monitor intervention effects and outcomes
5. **Analysis**: Evaluate intervention impact and effectiveness

For specific intervention types, see the individual intervention modules above. 