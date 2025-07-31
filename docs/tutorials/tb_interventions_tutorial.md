# TB Interventions Tutorial

This tutorial covers how to implement and analyze various tuberculosis interventions using tbsim.

## Available Interventions

tbsim supports several types of TB interventions:

- **BCG Vaccination**: Bacillus Calmette-Guérin vaccine
- **Tuberculosis Preventive Therapy (TPT)**: Preventive treatment for latent TB
- **Beta-lactam Antibiotics**: Treatment with beta-lactam antibiotics
- **Custom Interventions**: User-defined intervention strategies

## BCG Vaccination

```python
from tbsim.interventions.bcg import BCG

# Create BCG intervention
bcg = BCG(
    coverage=0.8,      # 80% coverage
    efficacy=0.7,      # 70% efficacy
    start_time=0,      # Start at simulation beginning
    duration=365       # Last for 1 year
)

# Add to simulation
sim.add_intervention(bcg)
```

## Tuberculosis Preventive Therapy (TPT)

```python
from tbsim.interventions.tpt import TPT

# Create TPT intervention
tpt = TPT(
    coverage=0.6,      # 60% coverage
    efficacy=0.9,      # 90% efficacy
    duration=6,        # 6-month treatment
    target_population='latent_tb'
)

# Add to simulation
sim.add_intervention(tpt)
```

## Beta-lactam Antibiotics

```python
from tbsim.interventions.beta import BetaIntervention

# Create beta-lactam intervention
beta = BetaIntervention(
    coverage=0.5,      # 50% coverage
    efficacy=0.8,      # 80% efficacy
    resistance_rate=0.1 # 10% resistance rate
)

# Add to simulation
sim.add_intervention(beta)
```

## Intervention Analysis

```python
from tbsim.analyzers import InterventionAnalyzer

# Create intervention analyzer
analyzer = InterventionAnalyzer(results)

# Compare interventions
analyzer.compare_interventions(['baseline', 'bcg', 'tpt'])

# Plot intervention impact
analyzer.plot_intervention_impact()
analyzer.plot_cost_effectiveness()
```

## Custom Interventions

```python
from tbsim.interventions.interventions import Intervention

class CustomIntervention(Intervention):
    def __init__(self, custom_param):
        super().__init__()
        self.custom_param = custom_param
    
    def apply(self, population, time):
        # Custom intervention logic
        pass

# Use custom intervention
custom = CustomIntervention(custom_param=0.5)
sim.add_intervention(custom)
```

## Next Steps

- Explore the [Comprehensive Analyzer Plots tutorial](comprehensive_analyzer_plots_example.md) for advanced analysis
- Check the [API documentation](../api/tbsim.interventions.md) for detailed intervention options
- Try the [TB-HIV Scenarios tutorial](run_tbhiv_scens.md) for complex scenarios 