# Beta Calibration Framework for TB Model

## Overview

This framework provides comprehensive tools for calibrating and analyzing the beta parameter (transmission rate) in the TB model, incorporating literature evidence, systematic parameter sweeping, behavior comparison, and implementation review.

## Key Features

### 🔬 **Literature-Informed Beta Values**
- **Household transmission**: β_HH: 0.3-0.7 (high exposure, repeated contacts)
- **Community transmission**: β_community: 0.005-0.02 (low exposure, diffuse contacts)
- **High burden settings**: 0.01-0.05/day ≈ 0.3-1.5/month (intense transmission)
- **Calibrated settings**: Often calibrated to match incidence ~250-350/100,000

### 📊 **Comprehensive Parameter Sweep**
- Context-specific beta ranges based on literature
- Multi-trial simulations for statistical robustness
- Literature comparison and validation
- Comprehensive visualization and reporting

### 🔍 **Behavior Comparison Analysis**
- Systematic comparison across transmission contexts
- Analysis of transmission dynamics and prevalence patterns
- Sensitivity analysis and elasticity calculations
- Context-specific recommendations

### 🛠️ **Implementation Review**
- Detailed analysis of beta usage in TB module
- Review of transmission logic and force of infection calculations
- Comparison with literature expectations
- Implementation recommendations

## Framework Components

### 1. Beta Calibration Sweep (`beta_calibration_sweep.py`)

**Purpose**: Systematic calibration of beta parameter across different transmission contexts.

**Key Features**:
- Literature-based beta ranges for different contexts
- Multi-metric calibration scoring
- Comprehensive visualization
- Literature comparison
- Export of results

**Usage**:
```python
from scripts.calibration.beta_calibration_sweep import BetaCalibrationSweep
from tbsim.utils import create_south_africa_data

# Create calibration data
calibration_data = create_south_africa_data()

# Create sweep object
sweep = BetaCalibrationSweep(calibration_data, context='community')

# Run beta sweep
results_df = sweep.run_beta_sweep(
    n_people=10000,
    years=10,
    n_trials=3,
    target_year=2018
)

# Analyze results
analysis = sweep.analyze_results(results_df)

# Generate plots and reports
sweep.plot_results(results_df, analysis)
report = sweep.generate_report(results_df, analysis)
```

**Available Contexts**:
- `household`: High exposure, repeated contacts (β: 0.3-0.7)
- `community`: Low exposure, diffuse contacts (β: 0.005-0.02)
- `high_burden`: Intense transmission (β: 0.3-1.5)
- `calibrated`: Calibrated for incidence targets (β: 0.015-0.040)

### 2. Beta Behavior Comparison (`beta_behavior_comparison.py`)

**Purpose**: Compare model behavior across different beta values and transmission contexts.

**Key Features**:
- Systematic comparison of beta values across contexts
- Analysis of transmission dynamics, prevalence patterns, and case rates
- Visualization of model behavior differences
- Literature-informed beta value selection
- Export of comparative analysis results

**Usage**:
```python
from scripts.calibration.beta_behavior_comparison import BetaBehaviorComparison
from tbsim.utils import create_south_africa_data

# Create calibration data
calibration_data = create_south_africa_data()

# Create comparison object
comparison = BetaBehaviorComparison(calibration_data)

# Run behavior comparison
results_df = comparison.run_behavior_comparison(
    comparison_set='context_specific',
    n_people=10000,
    years=10,
    n_trials=2
)

# Analyze behavior patterns
analysis = comparison.analyze_behavior_patterns(results_df)

# Generate plots and reports
comparison.plot_behavior_comparison(results_df, analysis)
report = comparison.generate_behavior_report(results_df, analysis)
```

**Available Comparison Sets**:
- `literature_based`: Beta values from epidemiological literature
- `context_specific`: Representative beta values for different transmission contexts
- `sensitivity_analysis`: Beta values for sensitivity analysis around literature values

### 3. Beta Logic Review (`beta_logic_review.py`)

**Purpose**: Review and analyze beta-related logic in the TB module and Starsim code.

**Key Features**:
- Analysis of beta usage in TB module
- Review of transmission logic and force of infection calculations
- Comparison with literature expectations
- Identification of beta-related parameters and their relationships
- Documentation of beta implementation details

**Usage**:
```python
from scripts.calibration.beta_logic_review import BetaLogicReview

# Create review object
reviewer = BetaLogicReview()

# Generate comprehensive review
report = reviewer.generate_review_report()

# Create visualization
reviewer.plot_beta_implementation_summary()
```

## Literature Evidence Integration

### Beta Values from Literature

| Context | Approximate β | Notes |
|---------|---------------|-------|
| **Styblo rule (historical)** | ~10-12 secondary infections/year per smear+ case | Often used to back-calculate β |
| **HIV-negative, high burden (e.g. India)** | 0.01-0.05 / day | ~3.6-18 / year |
| **Household transmission studies** | β_HH: 0.3-0.7 | High because of repeated exposure |
| **Community settings** | β_community: 0.005-0.02 | Lower, diffuse contacts |
| **Starsim calibrations** | Often calibrated to match incidence ~250-350/100,000 | Calibrated β varies but usually not explicitly set |

### Context-Dependent Variation

Beta varies significantly by:
- **Age group**: Different transmission rates across age groups
- **Household vs. community transmission**: Higher rates in households due to repeated exposure
- **Smear status of index case**: Different transmissibility by TB state
- **HIV status or malnutrition level**: Modified susceptibility and transmissibility

## Implementation Details

### Beta Parameter in TB Module

**Definition**:
```python
beta = ss.rate_prob(0.025, unit='month')  # Default value
```

**Usage**:
- **Primary**: Force of infection calculation (λ = β × I × C)
- **Modified by**: Relative transmissibility parameters for different TB states
- **Individual variation**: `reltrans_het` parameter for person-level heterogeneity
- **Susceptibility**: `rel_sus_latentslow` for latent slow TB

**Related Parameters**:
- `rel_trans_presymp`: Relative transmissibility of pre-symptomatic cases
- `rel_trans_smpos`: Relative transmissibility of smear-positive cases
- `rel_trans_smneg`: Relative transmissibility of smear-negative cases
- `rel_trans_exptb`: Relative transmissibility of extra-pulmonary TB
- `rel_trans_treatment`: Relative transmissibility during treatment
- `reltrans_het`: Individual-level heterogeneity in infectiousness
- `rel_sus_latentslow`: Relative susceptibility of latent slow TB

### Parameter Validation

```python
# Validate rates
for k, v in self.pars.items():
    if k[:5] == 'rate_':
        assert isinstance(v, ss.TimePar), 'Rate parameters for TB must be TimePars'
```

## Recommended Beta Values by Context

### For Different Transmission Contexts

```python
# Household transmission (high exposure)
beta_household = ss.rate_prob(0.5, unit='month')  # β_HH: 0.3-0.7

# Community transmission (moderate exposure)  
beta_community = ss.rate_prob(0.025, unit='month')  # β_community: 0.005-0.02

# High burden settings (e.g., India)
beta_high_burden = ss.rate_prob(0.7, unit='month')  # 0.01-0.05/day ≈ 0.3-1.5/month

# Calibrated for incidence ~250-350/100,000
beta_calibrated = ss.rate_prob(0.025, unit='month')  # Current default
```

### Context-Specific Defaults

```python
beta_contexts = {
    'household': ss.rate_prob(0.5, unit='month'),
    'community': ss.rate_prob(0.025, unit='month'), 
    'high_burden': ss.rate_prob(0.7, unit='month'),
    'calibrated': ss.rate_prob(0.025, unit='month')
}
```

## Calibration Workflow

### 1. Literature Review
- Identify appropriate beta ranges for target context
- Review epidemiological studies and transmission settings
- Determine calibration targets and validation metrics

### 2. Parameter Sweep
- Define beta range based on literature evidence
- Run systematic parameter sweep with multiple trials
- Calculate calibration scores and validation metrics

### 3. Behavior Analysis
- Compare model behavior across beta values
- Analyze transmission dynamics and prevalence patterns
- Validate against literature expectations

### 4. Implementation Review
- Review beta usage in code implementation
- Verify parameter relationships and modifications
- Ensure consistency with literature expectations

### 5. Documentation and Reporting
- Generate comprehensive calibration reports
- Create visualizations for different contexts
- Document recommendations and best practices

## Output Files

### Calibration Results
- `beta_calibration_sweep_{context}_{timestamp}.csv`: Sweep results
- `beta_calibration_plots_{context}_{timestamp}.png`: Visualization plots
- `beta_calibration_report_{context}_{timestamp}.json`: Detailed report

### Behavior Comparison
- `beta_behavior_comparison_{comparison_set}_{timestamp}.csv`: Comparison results
- `beta_behavior_comparison_{timestamp}.png`: Behavior comparison plots
- `beta_behavior_report_{timestamp}.json`: Behavior analysis report

### Logic Review
- `beta_logic_review_report_{timestamp}.json`: Implementation review report
- `beta_implementation_summary_{timestamp}.png`: Implementation summary plot

## Key Findings

### Beta Implementation
1. **Correct parameter type**: Using `ss.rate_prob` with monthly units
2. **Literature alignment**: Default value (0.025) aligns with community transmission
3. **Parameter relationships**: Multiple modifiers for different TB states
4. **Individual heterogeneity**: Support for person-level variation
5. **Validation**: Proper type checking for rate parameters

### Calibration Insights
1. **Context matters**: Beta values vary significantly by transmission context
2. **Literature validation**: Implementation covers literature-expected ranges
3. **Sensitivity**: Model outputs are sensitive to beta changes
4. **Calibration targets**: Different contexts require different calibration approaches

### Recommendations
1. **Continue using `ss.rate_prob`**: Implementation is correct
2. **Add range validation**: Consider explicit range constraints
3. **Document relationships**: Clarify beta relationships with other parameters
4. **Context-specific defaults**: Use appropriate defaults for different settings
5. **Sensitivity analysis**: Include beta sensitivity in calibration framework

## Usage Examples

### Quick Start
```python
# Run community transmission calibration
python scripts/calibration/beta_calibration_sweep.py

# Compare behavior across contexts
python scripts/calibration/beta_behavior_comparison.py

# Review implementation details
python scripts/calibration/beta_logic_review.py
```

### Custom Calibration
```python
# Create custom calibration data
from tbsim.utils import CalibrationData, CalibrationTarget

custom_data = CalibrationData(
    case_notifications=your_case_data,
    age_prevalence=your_age_data,
    targets={
        'overall_prevalence': CalibrationTarget('overall_prevalence', 0.01),
        'case_rate': CalibrationTarget('case_rate', 150)
    },
    country='Your Country',
    description='Custom calibration data'
)

# Run calibration
sweep = BetaCalibrationSweep(custom_data, context='high_burden')
results = sweep.run_beta_sweep()
```

## Dependencies

- `tbsim`: TB simulation framework
- `starsim`: Base simulation framework
- `numpy`: Numerical computations
- `pandas`: Data manipulation
- `matplotlib`: Visualization
- `seaborn`: Statistical plotting
- `scipy`: Scientific computing

## Contributing

When contributing to the beta calibration framework:

1. **Follow literature evidence**: Base beta ranges on epidemiological studies
2. **Document context**: Clearly specify transmission context for beta values
3. **Validate implementation**: Ensure beta usage aligns with literature expectations
4. **Test thoroughly**: Run comprehensive tests across different contexts
5. **Update documentation**: Keep documentation current with implementation changes

## References

1. **Styblo rule**: Historical TB transmission rule
2. **Household transmission studies**: High exposure transmission rates
3. **Community transmission studies**: Low exposure transmission rates
4. **High burden settings**: Intense transmission environments
5. **Starsim documentation**: Framework implementation details

---

**Author**: TB Simulation Team  
**Version**: 1.0.0  
**Last Updated**: 2024 