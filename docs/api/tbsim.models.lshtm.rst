LSHTM-Style TB Model
====================

This module provides individual-based TB compartmental models with states and
transitions inspired by LSHTM (London School of Hygiene & Tropical Medicine)
formulations. Two variants are available: **TB_LSHTM** (latent → active
progression) and **TB_LSHTM_Acute** (adds an acute infectious state immediately
after infection).

Module Reference
----------------

.. automodule:: tbsim.models.tb_lshtm
   :members: TBSL, TB_LSHTM, TB_LSHTM_Acute
   :undoc-members:
   :show-inheritance:

Model Overview
--------------

**State flow (TB_LSHTM)**

- Susceptible → [transmission] → INFECTION (latent) → CLEARED | UNCONFIRMED | ASYMPTOMATIC
- From UNCONFIRMED: RECOVERED | ASYMPTOMATIC
- From ASYMPTOMATIC: UNCONFIRMED | SYMPTOMATIC
- From SYMPTOMATIC: ASYMPTOMATIC | TREATMENT | DEAD
- From TREATMENT: SYMPTOMATIC | TREATED
- CLEARED, RECOVERED, TREATED are susceptible to reinfection (modifiers pi, rho)

**TB_LSHTM_Acute** extends the base by inserting an ACUTE state between
infection and the usual INFECTION (latent) state. New infections enter ACUTE
first, then transition to INFECTION; acute cases are infectious with relative
transmissibility ``alpha``.

Key Parameters
--------------

- **Transmission**: ``beta``; reinfection risks ``pi`` (recovered), ``rho`` (treated); asymptomatic transmissibility ``kappa``
- **From latent (INFECTION)**: ``infcle``, ``infunc``, ``infasy`` (clear, unconfirmed, asymptomatic)
- **From UNCONFIRMED**: ``uncrec``, ``uncasy``
- **From ASYMPTOMATIC**: ``asyunc``, ``asysym``
- **From SYMPTOMATIC**: ``symasy``, ``theta`` (start treatment), ``mutb`` (TB death)
- **From TREATMENT**: ``phi`` (failure), ``delta`` (completion)

Key Methods
----------

- **set_prognoses(uids, sources)**: Set prognoses for newly infected agents (latent → scheduled transition)
- **transition(uids, to)**: Sample competing exponential transitions; returns next state and time
- **step()**: Advance TB state for agents whose scheduled transition time has been reached
- **start_treatment(uids)**: Move eligible agents onto treatment (or clear latent infection)
- **step_die(uids)**: Update TB state when agents die (e.g. background mortality)

Usage Examples
-------------

Basic LSHTM TB simulation:

.. code-block:: python

   from tbsim import TB_LSHTM
   import starsim as ss

   sim = ss.Sim(diseases=TB_LSHTM())
   sim.run()

With custom parameters:

.. code-block:: python

   from tbsim import TB_LSHTM

   tb = TB_LSHTM(pars={
       'beta': ss.peryear(0.3),
       'init_prev': ss.bernoulli(0.02),
       'kappa': 0.82,
   })
   sim = ss.Sim(diseases=tb)
   sim.run()

Using the acute variant (ACUTE state after infection):

.. code-block:: python

   from tbsim import TB_LSHTM_Acute

   tb = TB_LSHTM_Acute(pars={'alpha': 0.5})  # relative transmissibility when acute
   sim = ss.Sim(diseases=tb)
   sim.run()

Accessing state and results:

.. code-block:: python

   from tbsim import TBSL

   # State counts
   n_symptomatic = (tb.state == TBSL.SYMPTOMATIC).sum()
   # Results
   results = tb.results
   prevalence = results.prevalence_active
   incidence = results.incidence_kpy
