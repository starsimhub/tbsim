# Examples

This section provides practical examples of TBsim usage across different scenarios and applications.

## Basic TB Simulation

The simplest way to run a TB simulation:

```python
from tbsim import TB, TBS
import starsim as ss

# Create a basic TB simulation
sim = ss.Sim(diseases=TB())

# Run the simulation
sim.run()

# Plot results
sim.plot()
```

## TB with Interventions

Adding BCG vaccination and treatment interventions:

```python
from tbsim.interventions.bcg import BCGVx
from tbsim.interventions.tpt import TPTTx
from tbsim import TB

# Add TB module and interventions
tb = TB()
bcg = BCGVx()
tpt = TPTTx()

sim = ss.Sim(
    diseases=tb,
    interventions=[bcg, tpt]
)
sim.run()
```

## TB-HIV Comorbidity

Modeling TB and HIV together:

```python
from tbsim.comorbidities.hiv import HIV
from tbsim import TB

# Add both modules
tb = TB()
hiv = HIV()

sim = ss.Sim(diseases=[tb, hiv])
sim.run()
```

## Household Networks

Using household-based social networks:

```python
from tbsim.networks import HouseholdNet
from tbsim import TB

# Create household network and TB
households = HouseholdNet()
tb = TB()

sim = ss.Sim(
    networks=households,
    diseases=tb
)
sim.run()
```

## Advanced Analysis

Using the built-in analyzers:

```python
from tbsim import TB
from tbsim.analyzers import DwellTime
import starsim as ss

# Run simulation with dwell time analyzer
sim = ss.Sim(diseases=[TB()], analyzers=DwellTime(scenario_name="Baseline"))
sim.run()

# Create plots from the analyzer
sim.analyzers[0].plot('sankey')
sim.analyzers[0].plot('histogram')
sim.analyzers[0].plot('kaplan_meier')
```

## Parameter Sweeps

Running multiple parameter combinations:

```python
import numpy as np

# Define parameter ranges
transmission_rates = np.linspace(0.1, 0.5, 5)

results = []
for rate in transmission_rates:
    sim = ss.Sim(diseases=TB(pars={'beta': ss.peryear(rate)}))
    sim.run()
    results.append(sim.results)
```

## Script Examples

The `tbsim_examples/` directory contains ready-to-run examples:

- **Basic TB**: `run_tb.py` - Simple TB simulation
- **LSHTM Model**: `run_tb_lshtm.py` - Spectrum of TB disease natural history
- **Malnutrition**: `run_malnutrition.py` - TB and malnutrition comorbidity
- **TB-HIV**: `run_tbhiv.py` - TB-HIV coinfection model
- **Interventions**: `run_tb_interventions.py` - BCG, TPT, and beta scenarios

For more detailed tutorials and step-by-step guides, see the [tutorials](tutorials.md) section.
