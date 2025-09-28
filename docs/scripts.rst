Scripts
=======

This section provides documentation for all Python scripts in the TBsim project. These scripts demonstrate various use cases, run simulations, and provide examples of how to use the TBsim framework.

Scripts are organized by functionality and purpose:

Basic Scripts
-------------

Basic simulation scripts that demonstrate core TBsim functionality.

.. toctree::
   :maxdepth: 2

   scripts/basic

Burn-in Scripts
---------------

Scripts for running burn-in simulations to establish equilibrium states before main analysis.

.. toctree::
   :maxdepth: 2

   scripts/burn_in

Calibration Scripts
-------------------

Scripts for calibrating model parameters against real-world data.

.. toctree::
   :maxdepth: 2

   scripts/calibration

HIV Scripts
-----------

Scripts for running TB-HIV co-infection simulations and scenarios.

.. toctree::
   :maxdepth: 2

   scripts/hiv

Intervention Scripts
--------------------

Scripts for testing and running various TB intervention scenarios.

.. toctree::
   :maxdepth: 2

   scripts/interventions

Optimization Scripts
--------------------

Scripts for parameter optimization and sensitivity analysis.

.. toctree::
   :maxdepth: 2

   scripts/optimization

How-to Scripts
--------------

Tutorial and example scripts demonstrating specific features and workflows.

.. toctree::
   :maxdepth: 2

   scripts/howto

Plotting Scripts
----------------

Scripts for generating visualizations and plots from simulation results.

.. toctree::
   :maxdepth: 2

   scripts/plots

Root Level Scripts
------------------

Main simulation scripts located in the scripts root directory.

.. toctree::
   :maxdepth: 2

   scripts/root

Script Overview
---------------

**Basic Scripts** (`scripts/basic/`)
   Core simulation examples including TB-only simulations, malnutrition modeling, and basic scenario runs.

**Burn-in Scripts** (`scripts/burn_in/`)
   Specialized scripts for running extended burn-in periods to establish model equilibrium, particularly for South African demographic scenarios.

**Calibration Scripts** (`scripts/calibration/`)
   Parameter calibration tools for fitting model outputs to observed epidemiological data.

**HIV Scripts** (`scripts/hiv/`)
   TB-HIV co-infection modeling scripts with various scenario configurations and analysis tools.

**Intervention Scripts** (`scripts/interventions/`)
   Comprehensive intervention testing including diagnostics, treatment protocols, and health-seeking behavior modifications.

**Optimization Scripts** (`scripts/optimization/`)
   Parameter optimization and sensitivity analysis tools for model refinement.

**How-to Scripts** (`scripts/howto/`)
   Educational examples and tutorials for learning TBsim features and best practices.

**Plotting Scripts** (`scripts/plots/`)
   Visualization tools for generating publication-quality figures from simulation results.

**Root Level Scripts** (`scripts/`)
   Main simulation runners and comprehensive scenario scripts for production use.

All scripts include comprehensive docstrings and are designed to be both educational examples and production-ready simulation tools. Each script can be run independently and includes detailed parameter documentation.
