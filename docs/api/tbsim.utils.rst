TBsim Utilities
==============

This module provides various utility functions and tools for TBsim simulations, including demographics, plotting, and probability calculations.

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

**Probabilities** (`tbsim.utils.probabilities`)
   Probability calculations and statistical utilities

**General Utilities** (`tbsim.utils`)
   Common utility functions and helpers

Key Features
-----------

- **Demographic Modeling**: Age-structured population utilities
- **Visualization Tools**: Comprehensive plotting capabilities
- **Statistical Functions**: Probability and statistical calculations
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

Probability calculations:

.. code-block:: python

   from tbsim.utils.probabilities import calculate_transmission_prob
   
   prob = calculate_transmission_prob(contact_rate, infectivity)

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

**Probabilities**
   - Transmission probability calculations
   - Risk factor modeling
   - Statistical distributions
   - Uncertainty quantification

For detailed information about specific utility functions, see the individual submodule documentation above. 