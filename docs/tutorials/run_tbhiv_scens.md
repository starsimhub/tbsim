# TB-HIV Scenarios Tutorial

This tutorial demonstrates how to run TB-HIV comorbidity scenarios using tbsim.

## Overview

TB-HIV comorbidity modeling is crucial for understanding the complex interactions between tuberculosis and HIV infection. This tutorial covers:

- Setting up TB-HIV co-infection models
- Running multiple scenarios
- Analyzing comorbidity effects
- Comparing intervention strategies

## Basic TB-HIV Simulation

```python
import tbsim.tb as tb
from tbsim.comorbidities.hiv import HIV

# Create TB simulation
tb_sim = tb.TB()

# Add HIV comorbidity
hiv = HIV(
    prevalence=0.1,    # 10% HIV prevalence
    tb_risk_multiplier=3.0,  # HIV increases TB risk 3x
    treatment_effectiveness=0.8
)

tb_sim.add_comorbidity(hiv)

# Run simulation
results = tb_sim.run()
```

## Multiple Scenarios

```python
# Define scenarios
scenarios = {
    'baseline': {'hiv_prevalence': 0.05, 'interventions': []},
    'high_hiv': {'hiv_prevalence': 0.15, 'interventions': []},
    'with_art': {'hiv_prevalence': 0.1, 'interventions': ['art']},
    'with_tpt': {'hiv_prevalence': 0.1, 'interventions': ['tpt']}
}

# Run all scenarios
results = {}
for name, params in scenarios.items():
    sim = tb.TB()
    hiv = HIV(prevalence=params['hiv_prevalence'])
    sim.add_comorbidity(hiv)
    
    # Add interventions
    for intervention in params['interventions']:
        if intervention == 'art':
            sim.add_intervention(ARTIntervention())
        elif intervention == 'tpt':
            sim.add_intervention(TPTIntervention())
    
    results[name] = sim.run()
```

## Analysis

```python
from tbsim.analyzers import TBHIVAnalyzer

# Create TB-HIV specific analyzer
analyzer = TBHIVAnalyzer(results)

# Generate comorbidity plots
analyzer.plot_tb_hiv_interaction()
analyzer.plot_scenario_comparison()
analyzer.plot_intervention_effectiveness()

# Statistical analysis
analyzer.calculate_risk_ratios()
analyzer.analyze_mortality_rates()
```

## Advanced Features

### Network-based TB-HIV

```python
from tbsim.networks import NetworkTB

# Create network-based simulation
network_sim = NetworkTB()

# Add HIV comorbidity to network
hiv_network = HIVNetwork(prevalence=0.1)
network_sim.add_comorbidity(hiv_network)

# Run network simulation
network_results = network_sim.run()
```

### Time-varying Parameters

```python
# HIV prevalence changes over time
hiv = HIV(
    prevalence_function=lambda t: 0.05 + 0.01 * (t / 365),  # Increases over time
    tb_risk_function=lambda t: 2.0 + 0.5 * (t / 365)        # Risk increases over time
)

sim.add_comorbidity(hiv)
```

## Next Steps

- Explore the [TB-HIV Comorbidity tutorial](tbhiv_comorbidity.md) for detailed comorbidity modeling
- Check the [Comprehensive Analyzer Plots tutorial](comprehensive_analyzer_plots_example.md) for advanced analysis
- Review the [API documentation](../api/tbsim.comorbidities.hiv.md) for detailed HIV modeling options 