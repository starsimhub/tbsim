TBsim Analyzers
==============

This module provides comprehensive analysis tools for TBsim simulation results, including dwell time analysis (DWT), visualization, and post-processing capabilities.

Main Analyzers Module
---------------------

.. automodule:: tbsim.analyzers
   :members:
   :undoc-members:
   :show-inheritance:

Available Analysis Tools
-----------------------

**DwtAnalyzer**
   Records and analyzes dwell times during simulation execution. Tracks how long agents spend in different states and provides comprehensive analysis capabilities for understanding state transition patterns.

**DwtPlotter**
   Comprehensive visualization tools for analyzing dwell time data from tuberculosis simulations. Supports both interactive (Plotly) and static (Matplotlib) visualizations with various chart types.

**DwtPostProcessor**
   Extends DwtPlotter to provide aggregation and analysis capabilities for multiple simulation results. Can combine results from multiple CSV files and perform comparative analysis across different scenarios.

Key Features
-----------

- **Real-time Dwell Time Tracking**: Automatic state change detection and dwell time recording during simulation
- **Multiple Visualization Types**: Sankey diagrams, network graphs, histograms, interactive charts, and Kaplan-Meier survival curves
- **Comprehensive Statistical Analysis**: Mean, mode, and count statistics for state transitions
- **Data Aggregation**: Batch processing of multiple simulation results
- **Interactive Visualizations**: Plotly-based interactive charts with hover information
- **Professional Styling**: Publication-ready visualizations with consistent formatting

Visualization Capabilities
-------------------------

**Sankey Diagrams**
   - State transition flows with agent counts and dwell times
   - Age-stratified analysis with multiple subplots
   - Interactive hover information and color coding

**Network Graphs**
   - Directed graph visualization of state transitions
   - Edge thickness proportional to transition frequency
   - Statistical annotations (mean, mode, agent count)
   - Multiple layout algorithms (spring, circular, spectral, etc.)

**Histograms and Distributions**
   - Dwell time distributions with kernel density estimation
   - State-specific analysis with automatic bin sizing
   - Cumulative distribution functions for custom transitions

**Interactive Charts**
   - Bar charts grouped by dwell time categories
   - Stacked visualizations for state transitions
   - Reinfection analysis by age groups and state transitions

**Survival Analysis**
   - Kaplan-Meier survival curves for dwell times
   - Time-to-event analysis for state transitions
   - Confidence intervals and statistical validation

Usage Examples
-------------

Basic dwell time analysis:

.. code-block:: python

   import starsim as ss
   from tbsim import TB
   from tbsim.analyzers import DwtAnalyzer
   
   # Create simulation with analyzer
   sim = ss.Sim(diseases=[TB()])
   sim.add_analyzer(DwtAnalyzer(scenario_name="Baseline"))
   sim.run()
   
   # Access analyzer results
   analyzer = sim.analyzers[0]
   analyzer.plot_dwell_time_validation()
   analyzer.sankey_agents()

Post-processing multiple runs:

.. code-block:: python

   from tbsim.analyzers import DwtPostProcessor
   
   # Aggregate multiple simulation results
   postproc = DwtPostProcessor(directory='results', prefix='Baseline')
   postproc.sankey_agents()
   postproc.histogram_with_kde()

Direct data analysis:

.. code-block:: python

   from tbsim.analyzers import DwtPlotter
   
   # Analyze existing data file
   plotter = DwtPlotter(file_path='results/Baseline-20240101120000.csv')
   plotter.sankey_agents()
   plotter.graph_state_transitions_curved()

Advanced visualizations:

.. code-block:: python

   # Enhanced network graph with professional styling
   plotter.graph_state_transitions_enhanced(
       subtitle="Enhanced TB State Transitions",
       colormap='plasma',
       figsize=(18, 14),
       node_size_scale=1200,
       edge_width_scale=10
   )
   
   # Age-stratified Sankey diagrams
   plotter.sankey_agents_by_age_subplots(
       bins=[0, 18, 65, 100],  # Child, Adult, Elderly
       scenario="Age-stratified Analysis"
   )

Analysis Capabilities
--------------------

**State Transition Analysis**: Comprehensive tracking of all state changes with timing information

**Dwell Time Distributions**: Statistical analysis of time spent in each TB state

**Network Visualization**: Graph-based representation of transition patterns and frequencies

**Comparative Analysis**: Multi-scenario comparison and batch processing capabilities

**Data Export**: CSV and metadata export for further analysis in external tools

**Validation Tools**: Distribution validation against expected theoretical patterns

For detailed information about specific analysis methods and parameters, see the individual class documentation above. All methods include comprehensive mathematical models and implementation details in their docstrings. 