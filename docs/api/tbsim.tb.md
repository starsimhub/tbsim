# tbsim.tb

Tuberculosis simulation module with TB transmission dynamics.

## Overview

The `tbsim.tb` module provides comprehensive tools for simulating tuberculosis transmission dynamics, including:

- Population-based transmission modeling
- Individual-based simulation
- Network-based transmission
- Intervention analysis
- Result visualization

## Main Classes

### TB

The primary TB simulation class.

```python
import tbsim.tb as tb

# Create a TB simulation
sim = tb.TB()

# Configure parameters
sim.set_parameters({
    'population_size': 10000,
    'initial_infected': 100,
    'transmission_rate': 0.1,
    'recovery_rate': 0.05,
    'mortality_rate': 0.02
})

# Run simulation
results = sim.run()
```

### NetworkTB

Network-based TB simulation.

```python
from tbsim.tb import NetworkTB

# Create network-based simulation
network_sim = NetworkTB()

# Add network structure
network_sim.add_network_structure('random', p=0.01)

# Run with network
network_results = network_sim.run()
```

## Key Methods

### Parameter Configuration

```python
# Set simulation parameters
sim.set_parameters({
    'population_size': 10000,
    'initial_infected': 100,
    'transmission_rate': 0.1,
    'recovery_rate': 0.05,
    'mortality_rate': 0.02,
    'latent_period': 30,
    'infectious_period': 180
})

# Get current parameters
params = sim.get_parameters()
```

### Running Simulations

```python
# Basic simulation
results = sim.run()

# Simulation with custom duration
results = sim.run(duration=365)

# Simulation with specific time steps
results = sim.run(time_steps=1000)
```

### Adding Interventions

```python
from tbsim.interventions import BCG, TPT

# Add BCG vaccination
bcg = BCG(coverage=0.8, efficacy=0.7)
sim.add_intervention(bcg)

# Add TB Preventive Therapy
tpt = TPT(coverage=0.6, efficacy=0.9)
sim.add_intervention(tpt)

# Run with interventions
results = sim.run()
```

### Adding Comorbidities

```python
from tbsim.comorbidities.hiv import HIV

# Add HIV comorbidity
hiv = HIV(prevalence=0.1, tb_risk_multiplier=3.0)
sim.add_comorbidity(hiv)

# Run with comorbidity
results = sim.run()
```

## Result Analysis

```python
# Access simulation results
incidence = results['incidence']
prevalence = results['prevalence']
mortality = results['mortality']

# Time series data
time_points = results['time']
susceptible = results['susceptible']
infected = results['infected']
recovered = results['recovered']

# Network data (if using NetworkTB)
network_results = results['network']
```

## Configuration Options

### Transmission Parameters

- `transmission_rate`: Probability of transmission per contact
- `contact_rate`: Average number of contacts per person per time unit
- `infectiousness`: Relative infectiousness of different TB states

### Population Parameters

- `population_size`: Total population size
- `initial_infected`: Number of initially infected individuals
- `age_distribution`: Age distribution of the population
- `risk_factors`: Distribution of risk factors

### Disease Parameters

- `latent_period`: Average duration of latent TB infection
- `infectious_period`: Average duration of infectious TB
- `recovery_rate`: Rate of spontaneous recovery
- `mortality_rate`: TB-related mortality rate

## Example Usage

```python
import tbsim.tb as tb
from tbsim.interventions import BCG
from tbsim.comorbidities.hiv import HIV

# Create simulation
sim = tb.TB()

# Configure parameters
sim.set_parameters({
    'population_size': 50000,
    'initial_infected': 500,
    'transmission_rate': 0.08,
    'recovery_rate': 0.04,
    'mortality_rate': 0.015
})

# Add interventions
bcg = BCG(coverage=0.75, efficacy=0.65)
sim.add_intervention(bcg)

# Add comorbidities
hiv = HIV(prevalence=0.08, tb_risk_multiplier=2.8)
sim.add_comorbidity(hiv)

# Run simulation
results = sim.run(duration=730)  # 2 years

# Analyze results
print(f"Final incidence: {results['incidence'][-1]}")
print(f"Final prevalence: {results['prevalence'][-1]}")
```

For more advanced features and detailed parameter options, see the tutorials and other module documentation. 