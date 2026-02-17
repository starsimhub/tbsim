# Comprehensive DwtAnalyzer Plots Example

This script demonstrates all available plotting methods in the `DwtAnalyzer` class from the TB simulation module. The analyzer provides comprehensive tools for analyzing and visualizing dwell time data from tuberculosis simulation runs.

## Overview

The `DwtAnalyzer` class offers three main categories of functionality:

1. **DwtAnalyzer**: Records dwell times during simulation execution
2. **DwtPlotter**: Creates various visualizations of dwell time data  
3. **DwtPostProcessor**: Aggregates and processes multiple simulation results

## Available Plot Types

### 1. Sankey Diagrams
Sankey diagrams show the flow of agents between different states with the width of the flow representing the number of agents.

- **`sankey_agents(subtitle="")`**: Basic Sankey diagram for all agents
- **`sankey_dwelltimes(subtitle='')`**: Sankey diagram with dwell time information
- **`sankey_agents_by_age_subplots(bins, scenario, includecycles)`**: Sankey diagrams stratified by age groups
- **`sankey_agents_even_age_ranges(number_of_plots, scenario)`**: Sankey diagrams with evenly distributed age ranges

### 2. Network Graphs
Network graphs visualize state transitions as directed graphs with statistical annotations.

- **`graph_state_transitions(states, subtitle, layout, colormap, onlymodel)`**: Basic state transition network
- **`graph_state_transitions_curved(states, subtitle, layout, curved_ratio, colormap, onlymodel, graphseed)`**: Curved network graph with edge thickness proportional to agent count

### 3. Histograms and Distributions
Distribution analysis tools for understanding dwell time patterns.

- **`histogram_with_kde(subtitle="")`**: Histogram with kernel density estimation for dwell time distributions
- **`plot_dwell_time_validation()`**: Static histogram validation plot for dwell time distributions
- **`plot_dwell_time_validation_interactive()`**: Interactive histogram validation plot

### 4. Interactive Bar Charts
Interactive Plotly-based visualizations for detailed analysis.

- **`barchar_all_state_transitions_interactive(dwell_time_bins, filter_states)`**: Interactive bar chart of all state transitions grouped by dwell time categories
- **`reinfections_age_bins_bars_interactive(target_states, barmode, scenario)`**: Interactive reinfection analysis by age groups
- **`reinfections_percents_bars_interactive(target_states, scenario)`**: Interactive reinfection percentages across population
- **`reinfections_bystates_bars_interactive(target_states, scenario, barmode)`**: Interactive reinfection analysis by state transitions

### 5. Stacked Bar Charts
Stacked visualizations showing cumulative time and transition patterns.

- **`stacked_bars_states_per_agent_static()`**: Static stacked bar chart showing cumulative dwell time per agent
- **`stackedbars_dwelltime_state_interactive(bin_size, num_bins)`**: Interactive stacked bar charts of dwell times by state
- **`stackedbars_subplots_state_transitions(bin_size, num_bins)`**: Subplot stacked bar charts for state transitions by dwell time

### 6. Custom Transition Analysis
Specialized analysis for specific state transition patterns.

- **`subplot_custom_transitions(transitions_dict)`**: Plot cumulative distribution of dwell times for custom state transitions

### 7. Survival Analysis
Survival analysis tools for understanding state persistence.

- **`plot_kaplan_meier(dwell_time_col, event_observed_col)`**: Kaplan-Meier survival curve for dwell time analysis

## Usage

### Running the Comprehensive Example

```bash
cd scripts
python comprehensive_analyzer_plots_example.py
```

### Basic Usage Pattern

```python
import tbsim as mtb
import starsim as ss

# 1. Create simulation with analyzer
sim = ss.Sim(diseases=mtb.TB(), analyzers=DwtAnalyzer(scenario_name="My Analysis"), pars=dict(dt = ss.days(7), start = ss.date('1940'), stop = ss.date('2010')))
sim.run()

# 2. Generate plots
analyzer.sankey_agents(subtitle="State Transitions")
analyzer.histogram_with_kde(subtitle="Dwell Time Analysis")
analyzer.graph_state_transitions_curved(subtitle="Network Analysis")

# 3. Access generated data file
file_path = analyzer.file_path
print(f"Data saved to: {file_path}")
```

### Using DwtPlotter with Existing Data

```python
from tbsim.analyzers import DwtPlotter

# Analyze existing data file
plotter = DwtPlotter(file_path='results/my_simulation.csv')
plotter.sankey_agents()
plotter.histogram_with_kde()
```

### Using DwtPostProcessor for Multiple Results

```python
from tbsim.analyzers import DwtPostProcessor

# Aggregate multiple simulation results
postproc = DwtPostProcessor(directory='results', prefix='Baseline')
postproc.sankey_agents(subtitle="Aggregated Results")
postproc.histogram_with_kde(subtitle="Aggregated Distributions")
```

## Key Features

### Real-time Data Collection
- Tracks dwell times during simulation execution
- Records state transitions with timing information
- Captures agent demographics and reinfection patterns

### Multiple Visualization Types
- **Static plots**: Matplotlib-based for publication quality
- **Interactive plots**: Plotly-based for exploration
- **Network visualizations**: NetworkX-based for complex relationships
- **Statistical analysis**: Survival curves and distribution fitting

### Data Export and Reuse
- Automatically saves data to CSV files
- Supports post-processing of multiple simulation runs
- Enables batch analysis across different scenarios

### Customization Options
- Configurable color schemes and layouts
- Adjustable bin sizes and time ranges
- Filterable state selections
- Custom transition definitions

## Output Files

The analyzer generates:
1. **CSV data file**: Contains all dwell time and transition data
2. **Interactive plots**: Displayed in browser or notebook
3. **Static plots**: Saved as matplotlib figures
4. **Network graphs**: Visualized state transition networks

## Dependencies

Required packages:
- `tbsim`: TB simulation framework
- `starsim`: Agent-based simulation framework
- `plotly`: Interactive visualizations
- `matplotlib`: Static plotting
- `networkx`: Network graph visualizations
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scipy`: Statistical functions
- `lifelines`: Survival analysis

## Example Output

The comprehensive example will generate:
- 8+ different types of Sankey diagrams
- 2 network graph visualizations
- 3 histogram/distribution plots
- 4 interactive bar charts
- 3 stacked bar chart variations
- 1 custom transition analysis
- 1 survival curve
- Additional plots using the DwtPlotter directly

Each plot type provides different insights into the TB simulation dynamics, from high-level state transition patterns to detailed agent-level analysis. 