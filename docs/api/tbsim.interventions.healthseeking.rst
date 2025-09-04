Health-Seeking Behavior Interventions
====================================

This module provides health-seeking behavior modeling for TBsim simulations.

Main Health-Seeking Module
--------------------------

.. automodule:: tbsim.interventions.healthseeking
   :members:
   :undoc-members:
   :show-inheritance:

Available Classes
----------------

**HealthSeekingBehavior**
   Intervention that triggers care-seeking behavior for individuals with active TB

Key Features
-----------

- **Care-seeking Triggers**: Automatic health-seeking for active TB cases
- **Configurable Probabilities**: Adjustable care-seeking rates
- **Timing Control**: Optional start/stop time windows
- **Single-use Option**: Expire intervention after successful care-seeking
- **Result Tracking**: Monitor care-seeking behavior over time

Usage Examples
-------------

Basic health-seeking behavior:

.. code-block:: python

   from tbsim.interventions.healthseeking import HealthSeekingBehavior
   from tbsim import TB
   
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
       prob=0.2,              # 20% daily probability of seeking care
       single_use=False,      # Continue triggering care-seeking
       start=2020,            # Start in 2020
       stop=2030              # Stop in 2030
   )

Key Methods
-----------

**Behavior Management**
   - `step()`: Execute health-seeking behavior each time step
   - `init_results()`: Initialize behavior result tracking
   - `update_results()`: Update results during simulation

**Care-seeking Logic**
   - Identifies individuals with active TB
   - Applies care-seeking probability
   - Triggers treatment initiation
   - Manages intervention expiration
   - Tracks care-seeking outcomes

Behavior Parameters
------------------

**prob**: Daily probability of seeking care (default: 0.1)
**single_use**: Whether to expire after successful care-seeking (default: True)
**start**: Optional start time for intervention
**stop**: Optional stop time for intervention

Health-Seeking Outcomes
----------------------

The module tracks:
- **Eligible Individuals**: Active TB cases who haven't sought care
- **Care-seeking Events**: Successful health-seeking behavior
- **Treatment Initiation**: Automatic treatment start for care-seekers
- **Behavior Patterns**: Care-seeking trends over time

Integration with Other Modules
-----------------------------

**TB Module**: Automatically triggers treatment for care-seekers
**Diagnostic Module**: Care-seeking leads to testing opportunities
**Treatment Module**: Care-seeking initiates treatment pathways

For comprehensive health-seeking integration, see the :doc:`tbsim.interventions.tb_health_seeking` module.
