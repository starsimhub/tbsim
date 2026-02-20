Examples
========

This section provides practical examples of TBsim usage across different scenarios and applications.

Basic TB Simulation
-------------------

The simplest way to run a TB simulation:

.. code-block:: python

   from tbsim import TB, TBS
   import starsim as ss
   
   # Create a basic TB simulation
   sim = ss.Sim(diseases=TB())
   
   # Run the simulation
   sim.run()
   
   # Plot results
   sim.plot()

TB with Interventions
--------------------

Adding BCG vaccination and treatment interventions:

.. code-block:: python

   from tbsim.interventions.bcg import BCGProtection
   from tbsim.interventions.tpt import TPTInitiation
   from tbsim import TB
   
   # Add TB module and interventions
   tb = TB()
   bcg = BCGProtection()
   tpt = TPTInitiation()
   
   sim = ss.Sim(
       diseases=tb,
       interventions=[bcg, tpt]
   )
   sim.run()

TB-HIV Comorbidity
------------------

Modeling TB and HIV together:

.. code-block:: python

   from tbsim.comorbidities.hiv.hiv import HIV
   from tbsim import TB
   
   # Add both modules
   tb = TB()
   hiv = HIV()
   
   sim = ss.Sim(diseases=[tb, hiv])
   sim.run()

Household Networks
-----------------

Using household-based social networks:

.. code-block:: python

   from tbsim.networks import HouseholdNet
   from tbsim import TB
   
   # Create household network and TB
   households = HouseholdNet()
   tb = TB()
   
   sim = ss.Sim(
       networks=households,
       diseases=tb
   )
   sim.run()

Advanced Analysis
----------------

Using the built-in analyzers:

.. code-block:: python

   from tbsim.analyzers import DwtAnalyzer, DwtPlotter
   
   # Analyze simulation results
   analyzer = DwtAnalyzer()
   results = analyzer.analyze(sim)
   
   # Create plots
   plotter = DwtPlotter()
   plotter.plot(results)

Parameter Sweeps
----------------

Running multiple parameter combinations:

.. code-block:: python

   import numpy as np
   
   # Define parameter ranges
   transmission_rates = np.linspace(0.1, 0.5, 5)
   
   results = []
   for rate in transmission_rates:
       sim = ss.Sim(diseases=TB(pars={'beta': ss.peryear(rate)}))
       sim.run()
       results.append(sim.results)

Script Examples
--------------

The ``tbsim_examples/`` directory contains ready-to-run examples:

- **Basic TB**: ``run_tb.py`` - Simple TB simulation
- **LSHTM Model**: ``run_tb_lshtm.py`` - Spectrum of TB disease natural history
- **Malnutrition**: ``run_malnutrition.py`` - TB and malnutrition comorbidity
- **TB-HIV**: ``run_tbhiv.py`` - TB-HIV coinfection model
- **Interventions**: ``run_tb_interventions.py`` - BCG, TPT, and beta scenarios

For more detailed tutorials and step-by-step guides, see the :doc:`tutorials` section.
