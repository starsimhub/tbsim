Beta Interventions
=================

This module provides time-varying intervention parameters for TBsim simulations.

Main Beta Module
----------------

.. automodule:: tbsim.interventions.beta
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**BetaByYear**
   Class for implementing time-varying transmission parameters

Key Features
-----------

- **Time-varying Parameters**: Dynamic intervention parameters over time
- **Year-based Changes**: Annual updates to transmission rates
- **Flexible Configuration**: Customizable parameter schedules
- **Integration**: Seamless integration with TB transmission models
- **Historical Modeling**: Support for historical intervention patterns

Usage Examples
-------------

Basic beta intervention:

.. code-block:: python

   from tbsim.interventions.beta import BetaByYear
   from tbsim import TB
   
   # Add TB module and time-varying transmission parameters
   tb = TB()
   beta = BetaByYear()
   
   sim = ss.Sim(
       diseases=tb,
       interventions=beta
   )
   sim.run()

Custom beta schedule:

.. code-block:: python

   beta = BetaByYear(
       baseline_rate=0.1,
       year_changes={
           2020: 0.08,  # 20% reduction in 2020
           2025: 0.06,  # Further reduction in 2025
           2030: 0.04   # Target rate by 2030
       }
   )

Key Methods
-----------

**Parameter Management**
   - `get_beta_for_year()`: Get transmission rate for specific year
   - `update_parameters()`: Update intervention parameters
   - `step()`: Execute parameter updates each time step

**Configuration**
   - `set_baseline_rate()`: Set initial transmission rate
   - `add_year_change()`: Add year-specific parameter changes
   - `get_current_rate()`: Get current transmission rate

Time-varying Parameters
----------------------

Beta interventions allow modeling of:
- **Historical Changes**: Past intervention implementations
- **Future Plans**: Planned intervention rollouts
- **Seasonal Effects**: Time-based parameter variations
- **Policy Changes**: Impact of new policies over time

This enables realistic modeling of intervention effectiveness and temporal dynamics. 