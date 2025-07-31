# Example Tutorial

This tutorial provides a basic introduction to using the tbsim library.

## Getting Started

First, import the necessary modules:

```python
import tbsim
import tbsim.tb as tb
```

## Basic TB Simulation

Create a simple TB simulation:

```python
# Initialize TB simulation
sim = tb.TB()

# Run the simulation
results = sim.run()

# Analyze results
print(f"Simulation completed with {len(results)} time steps")
```

## Working with Interventions

Add interventions to your simulation:

```python
from tbsim.interventions import Intervention

# Create intervention
intervention = Intervention()

# Add to simulation
sim.add_intervention(intervention)

# Run with intervention
results = sim.run()
```

## Analyzing Results

Use the built-in analyzers:

```python
from tbsim.analyzers import Analyzer

# Create analyzer
analyzer = Analyzer(results)

# Generate plots
analyzer.plot_results()
```

## Next Steps

- Check out the [Tuberculosis Simulation tutorial](tuberculosis_sim.md) for more advanced examples
- Explore the [API documentation](../api/overview.md) for detailed function references
- Try the [TB Interventions tutorial](tb_interventions_tutorial.md) for intervention-specific examples 