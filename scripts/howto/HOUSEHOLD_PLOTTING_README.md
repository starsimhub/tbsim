# Household Plotting Functionality

This document describes the household plotting functionality added to the `tbsim` package, specifically designed to visualize the `HouseholdNetGeneric` network structure.

## Overview

The household plotting functionality provides visual representations of household networks used in TB simulations. It includes two main plotting functions:

1. **`plot_household_structure`**: Basic visualization of household networks
2. **`plot_household_network_analysis`**: Comprehensive analysis with multiple subplots

## Features

- **Visual Network Representation**: Shows households as clusters of connected nodes
- **Complete Graph Visualization**: Displays all connections within households
- **Household Size Distribution**: Histogram of household sizes
- **Network Connectivity Analysis**: Distribution of connections per household
- **Agent Age Distribution**: Age demographics (when available)
- **Flexible Theming**: Support for both dark and light themes
- **Automatic Saving**: Figures saved with timestamps
- **Integration**: Seamless integration with `HouseholdNetGeneric` class

## Usage

### Basic Usage

```python
from tbsim.utils.plots import plot_household_structure, plot_household_network_analysis

# Create household structure
households = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]

# Basic household plot
plot_household_structure(
    households=households,
    people=people_object,
    title="My Household Network",
    show_household_ids=True,
    show_agent_ids=False,
    max_households_to_show=20,
    dark=True,
    savefig=True,
    outdir='results/household_plots'
)

# Comprehensive analysis
plot_household_network_analysis(
    households=households,
    people=people_object,
    figsize=(15, 10),
    dark=True,
    savefig=True,
    outdir='results/household_plots'
)
```

### Integration with Simulation

The household plotting is integrated into the simulation creation process in `scripts/run_tb_bcg_tpt.py`:

```python
# Build simulation with household plotting
sim = build_sim(
    scenario=scenario,
    show_household_plot=True,
    household_plot_type='analysis'  # 'basic', 'analysis', or 'both'
)
```

### Plot Types

1. **`'basic'`**: Simple household structure visualization
2. **`'analysis'`**: Comprehensive analysis with multiple subplots
3. **`'both'`**: Generate both basic and analysis plots

## Function Parameters

### `plot_household_structure`

- `households`: List of household lists (agent UIDs)
- `people`: Starsim People object (optional)
- `figsize`: Figure size tuple (default: (12, 8))
- `dark`: Use dark theme (default: True)
- `savefig`: Save figure (default: True)
- `outdir`: Output directory (default: 'results')
- `title`: Plot title
- `show_household_ids`: Show household IDs (default: True)
- `show_agent_ids`: Show individual agent IDs (default: False)
- `max_households_to_show`: Maximum households to display (default: 50)

### `plot_household_network_analysis`

- `households`: List of household lists (agent UIDs)
- `people`: Starsim People object (optional)
- `figsize`: Figure size tuple (default: (15, 10))
- `dark`: Use dark theme (default: True)
- `savefig`: Save figure (default: True)
- `outdir`: Output directory (default: 'results')

## Output

The plotting functions generate:

1. **Visual Display**: Interactive matplotlib plots
2. **Saved Files**: PNG files with timestamps in the specified directory
3. **Statistics**: Summary statistics displayed on the plots

## Example Output

The plots show:
- Household clusters with complete graphs
- Household size distribution
- Network connectivity analysis
- Agent age distribution (when available)
- Summary statistics (total agents, households, connections)

## Integration with HouseholdNetGeneric

The plotting functionality is specifically designed to work with the optimized `HouseholdNetGeneric` class:

```python
# Create household structure
households = create_sample_households(500)

# Use in simulation
networks = [
    mtb.HouseholdNetGeneric(hhs=households, pars={'add_newborns': True})
]

# Generate plots during simulation creation
sim = build_sim(
    show_household_plot=True,
    household_plot_type='analysis'
)
```

## Error Handling

The plotting functions include robust error handling for:
- Missing or inaccessible age data
- Empty household structures
- Invalid agent IDs
- Missing people objects

## Performance

- Optimized for large household networks
- Automatic limiting of displayed households for performance
- Efficient memory usage
- Fast rendering with matplotlib

## Files

- `tbsim/utils/plots.py`: Main plotting functions
- `scripts/run_tb_bcg_tpt.py`: Integration with simulation
- `example_household_plots.py`: Usage examples
- `HOUSEHOLD_PLOTTING_README.md`: This documentation

## Dependencies

- matplotlib
- numpy
- starsim
- tbsim
- pandas (for age data)

## Future Enhancements

Potential improvements:
- Interactive network visualization
- Export to different formats
- Custom color schemes
- Animation of network evolution
- Integration with other network types 