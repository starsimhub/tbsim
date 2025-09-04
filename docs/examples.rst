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
   sim = ss.Sim()
   tb = TB()
   sim.add_module(tb)
   
   # Run the simulation
   sim.run()
   
   # Plot results
   sim.plot()

TB with Interventions
--------------------

Adding BCG vaccination and treatment interventions:

.. code-block:: python

   from tbsim.interventions import BCG, TPT
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add TB module
   tb = TB()
   sim.add_module(tb)
   
   # Add interventions
   bcg = BCG()
   tpt = TPT()
   sim.add_module(bcg)
   sim.add_module(tpt)
   
   sim.run()

TB-HIV Comorbidity
------------------

Modeling TB and HIV together:

.. code-block:: python

   from tbsim.comorbidities.hiv import HIV
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Add both modules
   tb = TB()
   hiv = HIV()
   sim.add_module(tb)
   sim.add_module(hiv)
   
   sim.run()

Household Networks
-----------------

Using household-based social networks:

.. code-block:: python

   from tbsim.networks import HouseholdNet
   from tbsim import TB
   
   sim = ss.Sim()
   
   # Create household network
   households = HouseholdNet()
   sim.add_module(households)
   
   # Add TB with network transmission
   tb = TB()
   sim.add_module(tb)
   
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
       sim = ss.Sim()
       tb = TB(transmission_rate=rate)
       sim.add_module(tb)
       sim.run()
       results.append(sim.results)

Script Examples
--------------

The `scripts/` directory contains many ready-to-run examples:

- **Basic TB**: `run_tb.py` - Simple TB simulation
- **Interventions**: `run_tb_interventions.py` - TB with various interventions
- **Comorbidities**: `run_tb_and_malnutrition.py` - TB and malnutrition
- **HIV Integration**: `run_tbhiv_scens.py` - TB-HIV scenarios
- **Calibration**: `tb_calibration_south_africa.py` - Parameter fitting

For more detailed tutorials and step-by-step guides, see the :doc:`tutorials` section.
