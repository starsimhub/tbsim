API Reference
============

This section provides comprehensive API documentation for all TBsim modules and components, automatically generated from Python docstrings.

Core Modules
------------

.. toctree::
   :maxdepth: 2

   api/tbsim
   api/tbsim.tb
   api/tbsim.tb_lshtm
   api/tbsim.networks

Analysis and Visualization
--------------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.analyzers

Utilities
---------

.. toctree::
   :maxdepth: 2

   api/tbsim.utils

Interventions
-------------

.. toctree::
   :maxdepth: 2

   api/tbsim.interventions

Comorbidities
-------------

.. toctree::
   :maxdepth: 2

   api/tbsim.comorbidities
   api/tbsim.comorbidities.hiv
   api/tbsim.comorbidities.malnutrition

Configuration and Support
-------------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.config
   api/tbsim.misc
   api/tbsim.misc.tbterms
   api/tbsim.version
   api/tbsim.wrappers

Data and Utilities
------------------

.. toctree::
   :maxdepth: 2

   api/tbsim.data

Module Overview
---------------

**Natural History TB Model** (`tbsim.tb`)
   Main tuberculosis simulation module with disease dynamics, transmission, and state transitions. Implements the TBS state enumeration and TB disease class.

**LSHTM-Style TB Model** (`tbsim.tb_lshtm`)
   Individual-based TB compartmental models with LSHTM-inspired states and transitions: :class:`~tbsim.tb_lshtm.TB_LSHTM` (latent → active) and :class:`~tbsim.tb_lshtm.TB_LSHTM_Acute` (adds acute infectious state). State labels in :class:`~tbsim.tb_lshtm.TBSL`.

**Networks** (`tbsim.networks`)
   Social network structures for modeling transmission patterns, including household networks and RATIONS trial specific implementations.

**Analyzers** (`tbsim.analyzers`)
   Comprehensive data analysis tools including dwell time analysis (DWT), visualization, and post-processing capabilities for simulation results.

**Interventions** (`tbsim.interventions.*`)
   Various intervention modules for TB control and prevention, including DOTS implementation, BCG vaccination, enhanced diagnostics, and treatment protocols.

**Comorbidities** (`tbsim.comorbidities.*`)
   Modeling of HIV, malnutrition, and other co-occurring conditions with bidirectional interactions with TB dynamics.

**Utilities** (`tbsim.utils.*`)
   Helper functions for demographics, plotting, and data processing.

**Configuration** (`tbsim.config`)
   Parameter management, simulation configuration, and result directory creation utilities.

**Data** (`tbsim.data`)
   Data extraction, processing utilities, and anthropometric reference data for malnutrition modeling.

**Miscellaneous** (`tbsim.misc.*`)
   TB terminology definitions, version information, and additional utility functions.

For detailed information about each module, click on the links above or use the search functionality. All documentation is automatically generated from Python docstrings to ensure accuracy and completeness. 