# Comprehensive Analyzer Plots Tutorial

This tutorial demonstrates advanced plotting and analysis capabilities using tbsim's comprehensive analyzer tools.

## Overview

The comprehensive analyzer provides extensive visualization and analysis tools for TB simulation results, including:

- Multi-panel dashboard plots
- Interactive visualizations
- Statistical analysis plots
- Custom plot configurations
- Export capabilities

## Basic Comprehensive Analysis

```python
from tbsim.analyzers import ComprehensiveAnalyzer

# Create comprehensive analyzer
analyzer = ComprehensiveAnalyzer(results)

# Generate basic comprehensive plots
analyzer.plot_comprehensive_dashboard()
```

## Advanced Plotting Features

### Multi-panel Dashboards

```python
# Create custom dashboard layout
dashboard_config = {
    'layout': '2x3',  # 2 rows, 3 columns
    'plots': [
        'incidence_trend',
        'prevalence_by_age',
        'intervention_impact',
        'mortality_rates',
        'transmission_network',
        'cost_effectiveness'
    ],
    'figsize': (15, 10)
}

analyzer.plot_custom_dashboard(dashboard_config)
```

### Interactive Plots

```python
# Generate interactive plots
analyzer.plot_interactive_incidence()
analyzer.plot_interactive_prevalence()
analyzer.plot_interactive_interventions()

# Save interactive plots
analyzer.save_interactive_plots('interactive_plots.html')
```

### Statistical Analysis Plots

```python
# Statistical analysis plots
analyzer.plot_confidence_intervals()
analyzer.plot_bootstrap_analysis()
analyzer.plot_sensitivity_analysis()
analyzer.plot_parameter_uncertainty()
```

## Custom Plot Configurations

### Styling and Themes

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
analyzer.plot_styled_results()
```

### Color Schemes

```python
# Custom color schemes
color_scheme = {
    'baseline': '#1f77b4',
    'intervention': '#ff7f0e',
    'high_risk': '#d62728',
    'low_risk': '#2ca02c'
}

analyzer.set_color_scheme(color_scheme)
analyzer.plot_with_custom_colors()
```

## Export and Sharing

### High-Resolution Exports

```python
# Export high-resolution plots
analyzer.export_plots(
    format='png',
    dpi=300,
    directory='plots/',
    filename_prefix='tb_analysis'
)

# Export as PDF for publications
analyzer.export_plots(
    format='pdf',
    directory='publication_plots/'
)
```

### Report Generation

```python
# Generate comprehensive report
analyzer.generate_report(
    title='TB Simulation Analysis Report',
    author='Your Name',
    include_plots=True,
    include_statistics=True,
    output_format='html'
)
```

## Advanced Analysis Features

### Time-Series Analysis

```python
# Time-series specific plots
analyzer.plot_seasonal_patterns()
analyzer.plot_trend_analysis()
analyzer.plot_forecasting()
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

### Network Analysis

```python
# Network-specific plots
analyzer.plot_transmission_network()
analyzer.plot_network_evolution()
analyzer.plot_centrality_analysis()
```

## Integration with Other Tools

### Jupyter Integration

```python
# Jupyter-specific features
analyzer.enable_jupyter_mode()
analyzer.plot_jupyter_dashboard()
```

### External Tool Integration

```python
# Export for external analysis
analyzer.export_to_r('r_analysis.R')
analyzer.export_to_stata('stata_analysis.dta')
analyzer.export_to_excel('excel_analysis.xlsx')
```

## Next Steps

- Explore the [API documentation](../api/tbsim.analyzers.md) for detailed analyzer options
- Check the [TB Interventions tutorial](tb_interventions_tutorial.md) for intervention-specific analysis
- Review the [TB-HIV Comorbidity tutorial](tbhiv_comorbidity.md) for comorbidity analysis examples 