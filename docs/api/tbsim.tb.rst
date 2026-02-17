TB Disease Model
===============

This module implements the core tuberculosis disease model for TBsim simulations, including disease dynamics, state transitions, and transmission modeling.

Main TB Module
--------------

.. automodule:: tbsim.tb
   :members:
   :undoc-members:
   :show-inheritance:

TB State Enumeration
--------------------

.. autoclass:: tbsim.tb.TBS
   :members:
   :undoc-members:
   :show-inheritance:

TB Disease Class
----------------

.. autoclass:: tbsim.tb.TB
   :members:
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

The TB disease model implements a comprehensive tuberculosis simulation with the following key features:

**State Structure**
   - **Susceptible States**: NONE (-1) - No TB infection
   - **Latent States**: LATENT_SLOW (0), LATENT_FAST (1) - TB infection without symptoms
   - **Active States**: ACTIVE_PRESYMP (2), ACTIVE_SMPOS (3), ACTIVE_SMNEG (4), ACTIVE_EXPTB (5)
   - **Outcome States**: DEAD (8) - TB-related death, PROTECTED (100) - BCG protection

**Disease Progression**
   - Stochastic state transitions with time-dependent probabilities
   - Latent to active progression with different rates for slow/fast progressors
   - Active disease progression through pre-symptomatic, symptomatic, and treatment phases
   - Natural clearance and recovery mechanisms

**Transmission Dynamics**
   - Person-to-person transmission based on infectious states
   - Network-based transmission through social contacts
   - Age and risk factor dependent transmission rates

**Treatment Integration**
   - Treatment initiation and management
   - Treatment effectiveness and adherence modeling
   - Treatment outcome tracking

Key Methods
-----------

**State Transitions**
   - `p_latent_to_presym()`: Latent to pre-symptomatic progression
   - `p_presym_to_clear()`: Natural clearance from pre-symptomatic state
   - `p_presym_to_active()`: Progression to active disease
   - `p_active_to_clear()`: Recovery from active disease
   - `p_active_to_death()`: TB-related mortality

**Treatment Management**
   - `start_treatment()`: Initiate treatment for eligible individuals
   - `step_die()`: Handle TB-related deaths

**Results Tracking**
   - Prevalence and incidence monitoring
   - State-specific population counts
   - Treatment coverage and outcomes
   - Mortality and recovery rates

Usage Examples
-------------

Basic TB simulation:

.. code-block:: python

   from tbsim import TB
   import starsim as ss
   
   sim = ss.Sim(diseases=TB())
   sim.run()

With custom parameters:

.. code-block:: python

   from tbsim import TB
   
   # Customize TB parameters
   tb = TB(pars={
       'beta': ss.peryear(0.5),           # Transmission rate
       'init_prev': ss.bernoulli(0.01),   # Initial prevalence
   })
   
   sim = ss.Sim(diseases=tb)
   sim.run()

Accessing results:

.. code-block:: python

   # Get current state counts
   current_states = tb.state
   infected_count = len(tb.infected)
   active_count = len(tb.active)
   
   # Access results
   results = tb.results
   prevalence = results.prevalence
   incidence = results.incidence
   mortality = results.mortality

State Management
---------------

**State Transitions**
   The model implements a comprehensive state transition system where agents move between different TB states based on:

   - **Natural Progression**: Time-dependent transition probabilities
   - **Treatment Effects**: Modified progression rates under treatment
   - **Network Effects**: Transmission through social contacts
   - **Risk Factors**: Age, comorbidities, and other individual characteristics

**Infection Dynamics**
   - New infections occur through contact with infectious individuals
   - Latent infections can progress to active disease or clear naturally
   - Active disease can lead to treatment, recovery, or death
   - Treatment modifies progression rates and transmission potential

**Population Dynamics**
   - Birth and death processes affect TB dynamics
   - Age-specific risk factors influence disease progression
   - Network structure determines transmission patterns
   - Intervention effects modify disease parameters

For detailed information about specific methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings. 