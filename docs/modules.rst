API Reference
============

This section provides comprehensive API documentation for all TBsim modules and components, automatically generated from Python docstrings.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   api/tbsim
   api/tbsim.tb
   api/tbsim.networks

Analysis and Visualization
--------------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.analyzers
   api/tbsim.plots

Interventions
-------------

.. toctree::
   :maxdepth: 2

   api/tbsim.interventions
   api/tbsim.interventions.bcg
   api/tbsim.interventions.beta
   api/tbsim.interventions.enhanced_tb_diagnostic
   api/tbsim.interventions.enhanced_tb_treatment
   api/tbsim.interventions.healthseeking
   api/tbsim.interventions.interventions
   api/tbsim.interventions.tb_diagnostic
   api/tbsim.interventions.tb_drug_types
   api/tbsim.interventions.tb_health_seeking
   api/tbsim.interventions.tb_treatment
   api/tbsim.interventions.tpt

Comorbidities
-------------

.. toctree::
   :maxdepth: 2

   api/tbsim.hiv
   api/tbsim.malnutrition

Configuration and Support
-------------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.version

Data and Utilities
------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.data

Module Overview
---------------

**Core TB Model** (`tbsim.tb`)
   Main tuberculosis simulation module with disease dynamics, transmission, and state transitions. Implements the TBS state enumeration and TB disease class.

**Networks** (`tbsim.networks`)
   Social network structures for modeling transmission patterns, including household networks and RATIONS trial specific implementations.

**Analyzers** (`tbsim.analyzers`)
   Comprehensive data analysis tools including dwell time analysis (DWT), visualization, and post-processing capabilities for simulation results.

**Interventions** (`tbsim.interventions.*`)
   Various intervention modules for TB control and prevention, including DOTS implementation, BCG vaccination, enhanced diagnostics, and treatment protocols.

**Comorbidities** (`tbsim.comorbidities.hiv`, `tbsim.comorbidities.malnutrition`)
   Modeling of HIV, malnutrition, and other co-occurring conditions with bidirectional interactions with TB dynamics.

**Plots** (`tbsim.plots`)
   Plotting and visualization tools for simulation results.

**Data** (`tbsim.data`)
   Anthropometric reference data for malnutrition modeling.

For detailed information about each module, click on the links above or use the search functionality. All documentation is automatically generated from Python docstrings to ensure accuracy and completeness. 