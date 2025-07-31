# tbsim.analyzers

Analysis tools for processing and visualizing simulation results.

## Overview

The `tbsim.analyzers` module provides comprehensive tools for analyzing TB simulation results, including:

- Statistical analysis of simulation outputs
- Visualization and plotting capabilities
- Comparative analysis between scenarios
- Export and reporting functionality

## Main Classes

### Analyzer

The primary analysis class for TB simulation results.

```python
from tbsim.analyzers import Analyzer

# Create analyzer with simulation results
analyzer = Analyzer(results)

# Generate basic plots
analyzer.plot_results()
```

### ComprehensiveAnalyzer

Advanced analyzer with comprehensive plotting capabilities.

```python
from tbsim.analyzers import ComprehensiveAnalyzer

# Create comprehensive analyzer
analyzer = ComprehensiveAnalyzer(results)

# Generate dashboard
analyzer.plot_comprehensive_dashboard()
```

## Key Methods

### Basic Analysis

```python
# Create analyzer
analyzer = Analyzer(results)

# Basic plotting
analyzer.plot_incidence()
analyzer.plot_prevalence()
analyzer.plot_mortality()

# Statistical analysis
stats = analyzer.calculate_statistics()
summary = analyzer.get_summary()
```

### Advanced Analysis

```python
# Comprehensive analysis
analyzer = ComprehensiveAnalyzer(results)

# Multi-panel dashboards
analyzer.plot_custom_dashboard({
    'layout': '2x3',
    'plots': ['incidence_trend', 'prevalence_by_age', 'intervention_impact']
})

# Interactive plots
analyzer.plot_interactive_incidence()
analyzer.save_interactive_plots('interactive_plots.html')
```

### Comparative Analysis

```python
# Compare multiple scenarios
scenario_results = {
    'baseline': baseline_results,
    'intervention_a': intervention_a_results,
    'intervention_b': intervention_b_results
}

comparative_analyzer = ComprehensiveAnalyzer(scenario_results)
comparative_analyzer.plot_scenario_comparison()
comparative_analyzer.plot_relative_effectiveness()
```

## Plotting Functions

### Time Series Plots

```python
# Incidence trends
analyzer.plot_incidence_trend()
analyzer.plot_prevalence_trend()
analyzer.plot_mortality_trend()

# Seasonal patterns
analyzer.plot_seasonal_patterns()
analyzer.plot_trend_analysis()
```

### Demographic Analysis

```python
# Age-specific analysis
analyzer.plot_incidence_by_age()
analyzer.plot_prevalence_by_age()
analyzer.plot_mortality_by_age()

# Risk factor analysis
analyzer.plot_risk_factor_analysis()
analyzer.plot_population_pyramid()
```

### Intervention Analysis

```python
# Intervention impact
analyzer.plot_intervention_impact()
analyzer.plot_cost_effectiveness()
analyzer.plot_intervention_comparison()

# Coverage analysis
analyzer.plot_coverage_analysis()
analyzer.plot_efficacy_analysis()
```

## Statistical Analysis

```python
# Basic statistics
stats = analyzer.calculate_statistics()
print(f"Mean incidence: {stats['mean_incidence']}")
print(f"Peak incidence: {stats['peak_incidence']}")
print(f"Total cases: {stats['total_cases']}")

# Confidence intervals
analyzer.plot_confidence_intervals()
analyzer.plot_bootstrap_analysis()

# Sensitivity analysis
analyzer.plot_sensitivity_analysis()
analyzer.plot_parameter_uncertainty()
```

## Export and Reporting

```python
# Export plots
analyzer.export_plots(
    format='png',
    dpi=300,
    directory='plots/',
    filename_prefix='tb_analysis'
)

# Generate reports
analyzer.generate_report(
    title='TB Simulation Analysis Report',
    author='Your Name',
    include_plots=True,
    include_statistics=True,
    output_format='html'
)
```

## Example Usage

```python
from tbsim.analyzers import ComprehensiveAnalyzer
import tbsim.tb as tb

# Run simulation
sim = tb.TB()
sim.set_parameters({
    'population_size': 10000,
    'initial_infected': 100,
    'transmission_rate': 0.1
})
results = sim.run()

# Create analyzer
analyzer = ComprehensiveAnalyzer(results)

# Generate comprehensive analysis
analyzer.plot_comprehensive_dashboard()

# Export results
analyzer.export_plots(format='pdf', directory='analysis_results/')

# Generate report
analyzer.generate_report(
    title='TB Simulation Analysis',
    output_format='html'
)
```

## Configuration Options

### Plot Styling

```python
# Custom styling
custom_style = {
    'figure.figsize': (12, 8),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'font.size': 12
}

analyzer.set_plot_style(custom_style)
```

### Color Schemes

```python
# Custom colors
color_scheme = {
    'baseline': '#1f77b4',
    'intervention': '#ff7f0e',
    'high_risk': '#d62728',
    'low_risk': '#2ca02c'
}

analyzer.set_color_scheme(color_scheme)
```

For more detailed examples and advanced features, see the [Comprehensive Analyzer Plots tutorial](../tutorials/comprehensive_analyzer_plots_example.md). 