TB Health-Seeking Interventions
==============================

This module provides TB-specific health-seeking behavior modeling for TBsim simulations.

Main TB Health-Seeking Module
-----------------------------

.. automodule:: tbsim.interventions.tb_health_seeking
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**HealthSeekingBehavior**
   TB-specific health-seeking intervention with rate-based care-seeking probabilities

Key Features
-----------

- **Rate-based Probabilities**: Care-seeking rates with time unit specification
- **TB State Targeting**: Specific targeting of active TB cases
- **Timing Control**: Optional start/stop time windows
- **Single-use Option**: Expire intervention after successful care-seeking
- **Treatment Integration**: Automatic treatment initiation for care-seekers

Usage Examples
-------------

Basic TB health-seeking:

.. code-block:: python

   from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
   from tbsim import TB
   import starsim as ss
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add health-seeking behavior
   health_seeking = HealthSeekingBehavior()
   sim.add_module(health_seeking)
   
   sim.run()

Custom health-seeking parameters:

.. code-block:: python

   health_seeking = HealthSeekingBehavior(
       initial_care_seeking_rate=ss.perday(0.2),  # 20% daily rate
       single_use=False,                          # Continue triggering
       start=2020,                                # Start in 2020
       stop=2030                                  # Stop in 2030
   )

Key Methods
-----------

**Behavior Management**
   - `step()`: Execute health-seeking behavior each time step
   - `p_care_seeking()`: Calculate care-seeking probabilities
   - `init_results()`: Initialize behavior result tracking
   - `update_results()`: Update results during simulation

**Care-seeking Logic**
   - Identifies individuals with active TB (smear-positive, smear-negative, EPTB)
   - Applies rate-based care-seeking probabilities
   - Triggers automatic treatment initiation
   - Manages intervention expiration
   - Tracks care-seeking outcomes

Health-Seeking Parameters
------------------------

**initial_care_seeking_rate**: Base care-seeking rate with time units (default: 0.1 per day)
**start**: Optional start time for intervention
**stop**: Optional stop time for intervention
**single_use**: Whether to expire after successful care-seeking (default: True)

Care-seeking Outcomes
--------------------

The module tracks:
- **New Care-seekers**: Individuals who sought care in current timestep
- **Total Care-seekers**: Cumulative care-seeking behavior
- **Eligible Individuals**: Active TB cases who haven't sought care
- **Treatment Initiation**: Automatic treatment start for care-seekers

Integration with TB Module
-------------------------

**Automatic Treatment**: Care-seeking automatically triggers treatment
**State Management**: Updates sought_care flags and treatment status
**Result Tracking**: Comprehensive care-seeking behavior monitoring
**Timing Control**: Flexible intervention timing and duration

For basic health-seeking behavior, see the :doc:`tbsim.interventions.healthseeking` module.
