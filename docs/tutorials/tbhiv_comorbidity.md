# TB-HIV Comorbidity Tutorial

This tutorial provides detailed guidance on modeling TB-HIV comorbidity using tbsim.

## Understanding TB-HIV Comorbidity

TB-HIV comorbidity represents a significant public health challenge where:

- HIV infection increases the risk of TB infection and progression
- TB can accelerate HIV disease progression
- Treatment interactions between TB and HIV medications
- Complex transmission dynamics in co-infected populations

## Setting Up TB-HIV Models

```python
import tbsim.tb as tb
from tbsim.comorbidities.hiv import HIV, HIVIntervention

# Create base TB simulation
tb_sim = tb.TB()

# Configure HIV comorbidity
hiv_comorbidity = HIV(
    prevalence=0.08,           # 8% HIV prevalence
    tb_risk_multiplier=3.2,    # HIV increases TB risk 3.2x
    progression_rate=0.15,     # Faster TB progression in HIV+
    mortality_multiplier=2.5,  # Higher mortality in co-infected
    treatment_effectiveness=0.75  # Reduced treatment effectiveness
)

# Add to simulation
tb_sim.add_comorbidity(hiv_comorbidity)
```

## HIV Treatment Interventions

```python
# Antiretroviral Therapy (ART)
art_intervention = HIVIntervention(
    intervention_type='art',
    coverage=0.7,              # 70% coverage
    effectiveness=0.85,        # 85% effectiveness
    start_time=365,            # Start after 1 year
    duration=730               # Continue for 2 years
)

# TB Preventive Therapy for HIV+
tpt_hiv = TPTIntervention(
    target_population='hiv_positive',
    coverage=0.6,
    effectiveness=0.9,
    duration=6
)

# Add interventions
tb_sim.add_intervention(art_intervention)
tb_sim.add_intervention(tpt_hiv)
```

## Advanced Comorbidity Modeling

### CD4-based Risk Stratification

```python
# HIV with CD4-based TB risk
hiv_cd4 = HIV(
    prevalence=0.1,
    cd4_based_risk=True,
    cd4_risk_parameters={
        'cd4_500': 1.5,    # Risk multiplier for CD4 > 500
        'cd4_350_500': 2.5, # Risk multiplier for CD4 350-500
        'cd4_200_350': 4.0, # Risk multiplier for CD4 200-350
        'cd4_200': 8.0      # Risk multiplier for CD4 < 200
    }
)
```

### Treatment Interactions

```python
# Model drug interactions
hiv_with_interactions = HIV(
    prevalence=0.1,
    model_drug_interactions=True,
    interaction_parameters={
        'rifampicin_art': 0.8,  # Rifampicin reduces ART effectiveness
        'art_tb_treatment': 0.9 # ART improves TB treatment response
    }
)
```

## Analysis and Visualization

```python
from tbsim.analyzers import TBHIVComorbidityAnalyzer

# Create comorbidity-specific analyzer
analyzer = TBHIVComorbidityAnalyzer(results)

# Generate comorbidity plots
analyzer.plot_comorbidity_incidence()
analyzer.plot_hiv_tb_interaction()
analyzer.plot_treatment_effectiveness()
analyzer.plot_mortality_by_comorbidity()

# Statistical analysis
analyzer.calculate_attributable_risk()
analyzer.analyze_treatment_interactions()
analyzer.estimate_population_impact()
```

## Scenario Analysis

```python
# Compare different HIV prevalence scenarios
scenarios = {
    'low_hiv': {'prevalence': 0.02, 'risk_multiplier': 2.0},
    'medium_hiv': {'prevalence': 0.08, 'risk_multiplier': 3.2},
    'high_hiv': {'prevalence': 0.15, 'risk_multiplier': 4.5}
}

results = {}
for name, params in scenarios.items():
    sim = tb.TB()
    hiv = HIV(**params)
    sim.add_comorbidity(hiv)
    results[name] = sim.run()

# Analyze scenarios
analyzer = TBHIVComorbidityAnalyzer(results)
analyzer.plot_scenario_comparison()
```

## Next Steps

- Explore the [Comprehensive Analyzer Plots tutorial](comprehensive_analyzer_plots_example.md) for advanced visualization
- Check the [TB-HIV Scenarios tutorial](run_tbhiv_scens.md) for scenario-based analysis
- Review the [API documentation](../api/tbsim.comorbidities.hiv.md) for detailed HIV modeling options 