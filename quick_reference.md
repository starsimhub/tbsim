# Age-Stratified Plotting: Quick Reference

## Basic Usage

### Traditional Approach (Existing)
```python
# Flatten results manually
flat_results = {'Scenario': sim.results.flatten()}

# Plot
pl.plot_results(
    flat_results=flat_results,
    keywords=['prevalence', 'incidence']
)
```

### New Age-Stratified Approach
```python
# Direct age stratification
pl.plot_results(
    results=sim.results,
    sim=sim,
    age_bins=[0, 5, 15, 30, 50, 200],
    keywords=['prevalence', 'incidence']
)
```

## Common Age Bin Definitions

### Standard Age Groups
```python
age_bins = [0, 5, 15, 30, 50, 200]  # Children, Teens, Young Adults, Middle-aged, Elderly
```

### Pediatric Focus
```python
age_bins = [0, 1, 5, 12, 18, 200]  # Infants, Toddlers, Children, Teens, Adults
```

### Working Age Focus
```python
age_bins = [0, 18, 25, 35, 45, 55, 65, 200]  # Young adults, prime working age, older workers
```

### Elderly Focus
```python
age_bins = [0, 65, 75, 85, 200]  # Elderly age groups
```

## Supported Metrics

The age stratification supports these metric types:
- **Prevalence**: `prevalence_active`, `prevalence`
- **Incidence**: `incidence_kpy`, `new_infections`
- **Deaths**: `deaths_ppy`, `new_deaths`
- **Latent TB**: `n_latent_slow`, `n_latent_fast`

## Key Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `results` | object | Yes* | Simulation results object with `.flatten()` method |
| `sim` | object | Yes* | Simulation object with people and disease data |
| `age_bins` | list | Yes* | Age bin boundaries (e.g., [0, 5, 15, 30, 50, 200]) |
| `flat_results` | dict | Yes* | Pre-flattened results (existing functionality) |

*Either `flat_results` OR (`results` + `sim` + `age_bins`) must be provided.

## Output

The function creates age-stratified scenarios:
- Age bin `[0, 5, 15, 30, 50, 200]` creates scenarios: "0-5", "5-15", "15-30", "30-50", "50-200"
- Each scenario contains age-specific versions of the metrics
- All existing plotting features work (keywords, styling, etc.)

## Error Handling

The function provides clear error messages for:
- Missing required parameters
- Invalid age bin definitions
- Conflicting parameter combinations

## Real-World Examples

### Pediatric TB Analysis
```python
pl.plot_results(
    results=sim.results,
    sim=sim,
    age_bins=[0, 1, 5, 12, 18, 200],
    keywords=['prevalence_active', 'incidence_kpy'],
    n_cols=2
)
```

### Multi-Scenario Comparison
```python
# Run multiple simulations
scenarios = {}
for scenario_name, params in scenario_params.items():
    sim = build_sim(params)
    sim.run()
    scenarios[scenario_name] = {'sim': sim, 'results': sim.results}

# Generate age-stratified results for each scenario
combined_results = {}
for scenario_name, data in scenarios.items():
    stratified = pl._generate_age_stratified_results(
        data['sim'], data['results'].flatten(), [0, 15, 50, 200]
    )
    for age_bin, metrics in stratified.items():
        combined_results[f"{scenario_name}_{age_bin}"] = metrics

# Plot combined results
pl.plot_results(flat_results=combined_results, keywords=['prevalence_active'])
```

## Benefits

1. **Age-Specific Insights**: Reveals patterns hidden in overall population data
2. **Flexible Age Groups**: Custom age bin definitions for different analysis needs
3. **Backward Compatible**: Works with existing code and workflows
4. **Easy Integration**: Minimal changes to existing scripts
5. **Comprehensive**: Supports all major TB metrics 