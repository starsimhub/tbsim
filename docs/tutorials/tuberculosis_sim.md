# Tuberculosis Simulation Tutorial

This tutorial demonstrates how to run tuberculosis simulations using the tbsim library.

## Overview

The tuberculosis simulation module provides comprehensive tools for modeling TB transmission dynamics, including:

- Population-based transmission modeling
- Individual-based simulation
- Network-based transmission
- Intervention analysis
- Result visualization

## Basic Simulation

```python
import tbsim.tb as tb

# Create a TB simulation
sim = tb.TB()

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

## Advanced Features

### Network-based Simulation

```python
# Create network-based simulation
network_sim = tb.NetworkTB()

# Add network structure
network_sim.add_network_structure('random', p=0.01)

# Run with network
network_results = network_sim.run()
```

### Intervention Analysis

```python
from tbsim.interventions import BCG

# Add BCG vaccination
bcg = BCG(coverage=0.8, efficacy=0.7)
sim.add_intervention(bcg)

# Compare with and without intervention
baseline_results = sim.run()
intervention_results = sim.run()
```

## Analysis and Visualization

```python
from tbsim.analyzers import TBAnalyzer

# Create analyzer
analyzer = TBAnalyzer(results)

# Generate plots
analyzer.plot_incidence()
analyzer.plot_prevalence()
analyzer.plot_intervention_impact()
```

## Next Steps

- Explore the [TB Interventions tutorial](tb_interventions_tutorial.md) for more intervention examples
- Check the [API documentation](../api/tbsim.tb.md) for detailed parameter options
- Try the [TB-HIV Comorbidity tutorial](tbhiv_comorbidity.md) for modeling comorbidities 