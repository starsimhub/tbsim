# TB Model Calibration Scripts Guide

## Quick Start Sequence

```bash
cd scripts/calibration

# 1. Test everything works
python test_simple_calibration.py

# 2. See a demo
python run_sa_calibration_demo.py

# 3. Basic calibration
python tb_calibration_south_africa.py

# 4. Systematic beta sweep
cd beta && python beta_calibration_sweep.py

# 5. Behavior analysis
cd beta && python beta_behavior_comparison.py
```

## Overview

This guide provides a comprehensive overview of the TB model calibration scripts, their purpose, usage order, and expected outputs. The calibration framework is designed to systematically tune model parameters to match real-world epidemiological data.

## Script Organization

The calibration scripts are organized in a logical workflow from basic testing to comprehensive analysis:

```
scripts/calibration/
├── test_simple_calibration.py          # 1. Basic functionality testing
├── run_sa_calibration_demo.py          # 2. Single simulation demonstration
├── tb_calibration_south_africa.py      # 3. Basic South Africa calibration
├── tb_calibration_generalized.py       # 4. Generalized framework
├── beta/                               # 5. Beta-specific analysis
│   ├── beta_calibration_sweep.py       # Systematic beta parameter sweep
│   ├── beta_behavior_comparison.py     # Beta behavior analysis
│   └── beta_logic_review.py            # Beta implementation review
└── README_GENERALIZED_FRAMEWORK.md     # Framework documentation
```

## Script Summary

| Script | Purpose | Output |
|--------|---------|---------|
| `test_simple_calibration.py` | Test functionality | Console results |
| `run_sa_calibration_demo.py` | Single simulation demo | Plots + JSON report |
| `tb_calibration_south_africa.py` | Basic SA calibration | CSV data + plots |
| `tb_calibration_generalized.py` | Flexible framework | Multi-country results |
| `beta/beta_calibration_sweep.py` | Systematic beta testing | Sweep results + plots |
| `beta/beta_behavior_comparison.py` | Cross-beta analysis | Behavior plots + reports |
| `beta/beta_logic_review.py` | Code implementation review | Review reports |

## Usage Order and Workflow

### 1. **test_simple_calibration.py** - Basic Testing
**Purpose**: Verify that all calibration components work correctly

**When to run**: First, to ensure your environment is set up properly

**Command**:
```bash
cd scripts/calibration
python test_simple_calibration.py
```

**What you get**:
- Test results for data creation, calibration scoring, and simulation
- Validation that all required functions are working
- Quick feedback on any missing dependencies or configuration issues

**Output**: Console messages indicating test success/failure

---

### 2. **run_sa_calibration_demo.py** - Single Simulation Demo
**Purpose**: Demonstrate the calibration process with a single simulation

**When to run**: After testing, to see how calibration works in practice

**Command**:
```bash
python run_sa_calibration_demo.py
```

**What you get**:
- Single simulation with realistic South Africa parameters
- Comparison plots showing model vs. data
- Calibration report with metrics
- Parameter sensitivity analysis

**Outputs**:
- `sa_calibration_demo_YYYYMMDD_HHMM.png` - Comparison plots
- `sa_calibration_report_YYYYMMDD_HHMM.json` - Detailed report
- Console summary of calibration metrics

---

### 3. **tb_calibration_south_africa.py** - Basic Calibration
**Purpose**: Run a basic calibration for South Africa with default parameters

**When to run**: To get a baseline calibration result

**Command**:
```bash
python tb_calibration_south_africa.py
```

**What you get**:
- Single calibration simulation
- Basic plots comparing model to South Africa data
- Calibration report

**Outputs**:
- `sa_case_notifications_YYYYMMDD_HHMM.csv` - Case notification data
- `sa_age_prevalence_YYYYMMDD_HHMM.csv` - Age prevalence data
- Calibration plots and report

---

### 4. **tb_calibration_generalized.py** - Generalized Framework
**Purpose**: Use the new generalized calibration framework

**When to run**: For more flexible calibration with different countries/parameters

**Command**:
```bash
python tb_calibration_generalized.py
```

**What you get**:
- Generalized calibration for any country
- Configurable disease and intervention parameters
- Comprehensive plotting and analysis

**Outputs**:
- Generalized calibration results
- Multi-panel plots
- Structured calibration reports

---

### 5. **beta_calibration_sweep.py** - Systematic Beta Sweep
**Purpose**: Systematically test different beta values across transmission contexts

**When to run**: To find optimal beta values for different transmission settings

**Command**:
```bash
cd beta
python beta_calibration_sweep.py
```

**What you get**:
- Systematic sweep of beta values (0.005 to 1.5)
- Context-specific analysis (household, community, high burden, calibrated)
- Literature comparison
- Multi-metric calibration scoring

**Outputs**:
- `beta_calibration_sweep_YYYYMMDD_HHMM.csv` - Sweep results
- `beta_calibration_sweep_YYYYMMDD_HHMM.png` - Sweep plots
- `beta_calibration_report_YYYYMMDD_HHMM.json` - Detailed analysis

---

### 6. **beta_behavior_comparison.py** - Beta Behavior Analysis
**Purpose**: Compare model behavior across different beta values and contexts

**When to run**: To understand how beta affects model dynamics

**Command**:
```bash
cd beta
python beta_behavior_comparison.py
```

**What you get**:
- Behavior comparison across literature-based beta values
- Transmission context analysis
- Sensitivity analysis
- Comprehensive visualization

**Outputs**:
- `beta_behavior_comparison_*_YYYYMMDD_HHMM.csv` - Behavior results
- `beta_behavior_comparison_YYYYMMDD_HHMM.png` - Behavior plots
- `beta_behavior_report_YYYYMMDD_HHMM.json` - Behavior analysis

---

### 7. **beta_logic_review.py** - Implementation Review
**Purpose**: Review and analyze beta implementation in the codebase

**When to run**: To understand how beta is implemented and used

**Command**:
```bash
cd beta
python beta_logic_review.py
```

**What you get**:
- Analysis of beta usage in TB module
- Review of transmission logic
- Comparison with literature expectations
- Implementation documentation

**Outputs**:
- `beta_logic_review_YYYYMMDD_HHMM.json` - Review report
- `beta_logic_review_YYYYMMDD_HHMM.png` - Implementation summary plots

## Expected Outputs Summary

### Data Files (.csv)
- **Case notifications**: Model vs. data comparison
- **Age-stratified prevalence**: Age group analysis
- **Parameter sweep results**: Systematic parameter testing
- **Behavior comparison**: Cross-beta analysis

### Visualization Files (.png)
- **Multi-panel plots**: Comprehensive calibration visualizations
- **Violin plots**: Parameter sensitivity analysis
- **Comparison plots**: Model vs. data overlays
- **Behavior plots**: Cross-parameter dynamics

### Report Files (.json)
- **Structured reports**: Detailed calibration metrics
- **Analysis summaries**: Key findings and recommendations
- **Literature comparisons**: Model vs. literature expectations

## Key Metrics You'll Get

### Calibration Metrics
- **Composite calibration score**: Overall fit quality
- **Case notification MAPE**: Model vs. data fit
- **Age prevalence MAPE**: Age-stratified fit
- **Overall prevalence error**: Target prevalence match

### Parameter Analysis
- **Optimal beta values**: Best-fit transmission rates
- **Parameter sensitivity**: How parameters affect outcomes
- **Context-specific ranges**: Literature-informed parameter bounds
- **Uncertainty quantification**: Parameter confidence intervals

### Model Behavior
- **Transmission dynamics**: How beta affects disease spread
- **Age patterns**: Age-specific transmission patterns
- **Temporal trends**: Time-varying disease dynamics
- **Intervention effects**: How interventions modify transmission

## Common Parameters

```python
# Beta ranges by context
household: [0.3, 0.7]
community: [0.005, 0.02] 
high_burden: [0.3, 1.5]
calibrated: [0.015, 0.04]

# Target prevalence (South Africa)
overall_prevalence: 0.852% (2018 survey)
```

## Performance Tips

- Start with: `n_agents=1000, years=10, n_trials=2`
- Scale up: `n_agents=10000, years=50, n_trials=5`
- For sweeps: Focus on specific beta ranges

## Customization Options

### Parameter Ranges
- Modify beta ranges in `beta_calibration_sweep.py`
- Adjust simulation parameters in configuration classes
- Customize target data in calibration data objects

### Countries/Regions
- Use `tb_calibration_generalized.py` for different countries
- Modify demographic data sources
- Adjust target values for different regions

### Analysis Depth
- Change number of trials per parameter combination
- Modify simulation duration and population size
- Adjust plotting and reporting detail levels

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure `tbsim` is installed and in Python path
2. **Missing data**: Check that demographic data files are available
3. **Simulation failures**: Reduce population size or simulation duration
4. **Memory issues**: Use smaller parameter sweeps or fewer trials

### Performance Tips
- Start with smaller simulations (n_agents=1000, years=10)
- Use fewer trials initially (n_trials=2)
- Focus on specific parameter ranges of interest
- Use the generalized framework for faster iteration

## Next Steps

After running the calibration scripts:

1. **Review results**: Examine calibration scores and plots
2. **Identify optimal parameters**: Note best-fit beta values
3. **Validate findings**: Compare with literature expectations
4. **Iterate**: Refine parameter ranges based on results
5. **Document**: Record findings for future reference

This systematic approach ensures robust calibration and provides comprehensive insights into TB model behavior across different transmission contexts.

## Related Documentation

- **README_GENERALIZED_FRAMEWORK.md**: Framework architecture and advanced usage
- **beta/README.md**: Beta-specific parameter information and literature context 