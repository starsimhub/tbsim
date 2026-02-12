TBsim Utilities
==============

This module provides various utility functions and tools for TBsim simulations, including demographics and plotting.

Main Utilities Module
--------------------

.. automodule:: tbsim.utils
   :members:
   :undoc-members:
   :show-inheritance:

Available Utility Modules
------------------------

**Demographics** (`tbsim.utils.demographics`)
   Population demographics and age structure utilities

**Plotting** (`tbsim.utils.plots`)
   Visualization and plotting tools for simulation results

**General Utilities** (`tbsim.utils`)
   Common utility functions and helpers

Subpackages
-----------

.. toctree::
   :maxdepth: 2

   tbsim.utils.demographics
   tbsim.utils.plots

Key Features
-----------

- **Demographic Modeling**: Age-structured population utilities
- **Visualization Tools**: Comprehensive plotting capabilities
- **Data Processing**: Utilities for handling simulation data
- **Export Functions**: Tools for saving and sharing results

Usage Examples
-------------

Demographics utilities:

.. code-block:: python

   from tbsim.utils.demographics import create_age_structure
   
   age_structure = create_age_structure(pop_size=10000)

Plotting utilities:

.. code-block:: python

   from tbsim.utils.plots import plot_tb_incidence
   
   plot_tb_incidence(sim_results)

Utility Functions
----------------

**Demographics**
   - Age distribution generation
   - Population structure modeling
   - Demographic parameter calculations

**Plotting**
   - Time series visualization
   - Comparative plots
- Custom chart creation
- Export to various formats

For detailed information about specific utility functions, see the individual submodule documentation above. 