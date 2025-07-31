# tbsim

The main tbsim module containing core simulation functionality.

## Overview

The `tbsim` module provides the main entry point for tuberculosis simulation functionality. It includes:

- Core simulation classes
- Utility functions
- Configuration management
- Data processing tools

## Main Classes

### TB Simulation

The primary class for running tuberculosis simulations.

```python
import tbsim

# Create a TB simulation
sim = tbsim.tb.TB()

# Configure parameters
sim.set_parameters({
    'population_size': 10000,
    'initial_infected': 100,
    'transmission_rate': 0.1,
    'recovery_rate': 0.05
})

# Run simulation
results = sim.run()
```

### Analyzers

Tools for analyzing simulation results.

```python
from tbsim.analyzers import Analyzer

# Create analyzer
analyzer = Analyzer(results)

# Generate plots
analyzer.plot_results()
```

## Key Functions

### Configuration

```python
import tbsim.config

# Load configuration
config = tbsim.config.load_config('config.yaml')

# Set parameters
tbsim.config.set_parameter('transmission_rate', 0.15)
```

### Utilities

```python
import tbsim.utils

# Demographic utilities
demographics = tbsim.utils.demographics.load_population_data()

# Plotting utilities
tbsim.utils.plots.plot_incidence(results)
```

## Module Structure

```
tbsim/
├── __init__.py          # Main module initialization
├── tb.py               # TB simulation classes
├── analyzers.py        # Analysis tools
├── config.py           # Configuration management
├── networks.py         # Network-based simulations
├── version.py          # Version information
├── wrappers.py         # Wrapper functions
├── comorbidities/      # Comorbidity modeling
├── interventions/      # Intervention implementations
├── utils/             # Utility functions
└── misc/              # Miscellaneous tools
```

## Getting Started

```python
import tbsim

# Basic usage
sim = tbsim.tb.TB()
results = sim.run()

# With interventions
from tbsim.interventions import BCG
bcg = BCG(coverage=0.8, efficacy=0.7)
sim.add_intervention(bcg)
results = sim.run()
```

For more detailed information, see the individual module documentation pages. 