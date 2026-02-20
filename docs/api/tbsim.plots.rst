Plotting Utilities
=================

This module provides comprehensive plotting and visualization tools for TBsim simulation results.

Main Plots Module
-----------------

.. automodule:: tbsim.plots
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
-----------

- **Result Visualization**: Plot simulation outputs and trends
- **Comparative Analysis**: Compare multiple simulation scenarios
- **Custom Plotting**: Flexible plotting with customizable parameters
- **Export Capabilities**: Save plots in various formats
- **Interactive Plots**: Dynamic visualizations for exploration

Usage Examples
-------------

Basic result plotting:

.. code-block:: python

   from tbsim.plots import plot_results
   
   # Plot basic simulation results
   plot_results(sim_results)
   
   # Plot with custom parameters
   plot_results(
       sim_results,
       title='TB Simulation Results',
       save_path='results.png'
   )

Combined result plotting:

.. code-block:: python

   from tbsim.plots import plot_combined
   
   # Plot multiple simulation results
   plot_combined(
       [results1, results2, results3],
       labels=['Baseline', 'Intervention A', 'Intervention B']
   )

Custom plotting:

.. code-block:: python

   from tbsim.plots import create_custom_plot
   
   # Create custom visualization
   fig = create_custom_plot(
       data=sim_results,
       plot_type='incidence_trends',
       style='seaborn'
   )

Available Plot Types
-------------------

**Time Series Plots**
   - Incidence trends over time
   - Prevalence changes
   - Treatment outcomes

**Comparative Plots**
   - Multiple scenario comparison
   - Intervention effectiveness
   - Parameter sensitivity

**Distribution Plots**
   - Age distribution of cases
   - Geographic distribution
   - Risk factor distributions

**Export Options**
   - PNG, PDF, SVG formats
   - High-resolution outputs
   - Publication-ready figures

For advanced plotting and analysis, see the :doc:`tbsim.analyzers` module. 