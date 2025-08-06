# TBsim Calibration Features and Plotting

## Overview

TBsim provides comprehensive **calibration features** and **plotting capabilities** for tuberculosis modeling. These are core features of the TBsim framework, not just utilities, providing a **centralized, generalized, and reusable** approach to TB model calibration that can be used with different countries, data sources, and calibration scenarios.

## TBsim Calibration Features

### Core Calibration System

TBsim includes a complete calibration system in `tbsim/calibration/functions.py`:

- **`compute_age_stratified_prevalence()`** - Age-group prevalence calculations
- **`compute_case_notifications()`** - Case notification metrics  
- **`calculate_calibration_score()`** - Composite calibration scoring
- **`create_calibration_report()`** - Detailed calibration reports
- **`create_south_africa_data()`** - South Africa calibration data
- **`create_country_data()`** - Generic country data creator
- **`load_calibration_data_from_file()`** - Load data from JSON
- **`save_calibration_data_to_file()`** - Save data to JSON

### Simulation Framework

TBsim provides a complete simulation framework in `tbsim/calibration/simulation.py`:

- **`run_calibration_simulation_suite()`** - Main simulation runner
- **`run_calibration_simulation()`** - Backward compatibility function
- **`create_demographics()`** - Country-specific demographics
- **`create_tb_disease()`** - TB disease configuration
- **`create_hiv_disease()`** - HIV disease configuration
- **`create_interventions()`** - Intervention setup
- **`find_demographic_data()`** - Locate demographic files

## TBsim Plotting System

### CalibrationPlotter Class

TBsim includes a comprehensive plotting system in `tbsim/plotting/plots.py` that provides:

- **`plot_calibration_comparison()`** - Model vs data comparison plots
- **`plot_sweep_results()`** - Comprehensive parameter sweep analysis (12 subplots)
- **`plot_violin_plots()`** - Focused violin plots for parameter sensitivity

```python
from tbsim.plotting import CalibrationPlotter

plotter = CalibrationPlotter(style='default', figsize=(12, 8))

# Create comparison plots
fig = plotter.plot_calibration_comparison(sim, calibration_data, timestamp)

# Create sweep analysis
fig = plotter.plot_sweep_results(results_df, timestamp)

# Create violin plots
fig = plotter.plot_violin_plots(results_df, timestamp)
```

### Visualization Features

#### **Violin Plots**
- **Parameter-specific distributions** showing score variability
- **Mean trend lines** to identify optimal parameter ranges
- **Interactive visualization** of parameter sensitivity

#### **Comprehensive Sweep Analysis**
- **12-panel layout** including:
  - Score distributions and scatter plots
  - Violin plots for each parameter
  - 3D scatter plots and correlation heatmaps
  - Box plots for categorical analysis

#### **Calibration Comparison Plots**
- **Model vs data** comparisons
- **Age-stratified prevalence** analysis
- **Overall prevalence** trends
- **Care cascade** visualization

## TBsim Core Data Structures

### CalibrationData Class
```python
@dataclass
class CalibrationData:
    case_notifications: pd.DataFrame
    age_prevalence: pd.DataFrame
    targets: Dict[str, CalibrationTarget]
    country: str
    description: str
```

### CalibrationTarget Class
```python
@dataclass
class CalibrationTarget:
    name: str
    value: float
    year: Optional[int] = None
    age_group: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None
```

## TBsim Configuration System

### DiseaseConfig
```python
@dataclass
class DiseaseConfig:
    beta: float = 0.020
    rel_sus_latentslow: float = 0.15
    tb_mortality: float = 3e-4
    init_prev: float = 0.10
    # ... other parameters
```

### SimulationConfig
```python
@dataclass
class SimulationConfig:
    start_year: int = 1850
    years: int = 200
    n_agents: int = 1000
    seed: int = 0
    verbose: int = 0
    dt: int = 30
    unit: str = 'day'
```

### InterventionConfig
```python
@dataclass
class InterventionConfig:
    include_hiv: bool = True
    include_health_seeking: bool = True
    include_diagnostic: bool = True
    include_treatment: bool = True
    # ... other settings
```

## Using TBsim's Core Features

### Basic Usage

```python
from tbsim.calibration import run_calibration_simulation_suite
from tbsim.plotting import CalibrationPlotter

# Run calibration for South Africa
sim = run_calibration_simulation_suite("South Africa")

# Run calibration for Vietnam (with fallback to SA demographics)
sim = run_calibration_simulation_suite("Vietnam")
```

### Parameter Sweep

```python
from tbsim.calibration import run_calibration_simulation_suite, SimulationConfig, DiseaseConfig

# Run parameter sweep manually
configs = []
for beta in [0.01, 0.02, 0.03]:
    disease_config = DiseaseConfig(beta=beta)
    sim_config = SimulationConfig(n_agents=1000, years=50)
    sim = run_calibration_simulation_suite("South Africa", disease_config=disease_config, sim_config=sim_config)
    configs.append(sim)
```

### Custom Configuration

```python
from tbsim.calibration import (
    DiseaseConfig, InterventionConfig, SimulationConfig,
    run_calibration_simulation_suite
)

# Custom disease configuration
disease_config = DiseaseConfig(
    beta=0.025,
    rel_sus_latentslow=0.20,
    tb_mortality=4e-4
)

# Custom simulation configuration
sim_config = SimulationConfig(
    n_agents=2000,
    years=100,
    seed=42
)

# Run simulation
sim = run_calibration_simulation_suite(
    country_name="South Africa",
    disease_config=disease_config,
    sim_config=sim_config
)
```

## Multi-Country Support

TBsim supports calibration for multiple countries:

### Adding a New Country

1. **Create demographic data files**:
   ```
   tbsim/data/CountryName_CBR.csv
   tbsim/data/CountryName_ASMR.csv
   ```

2. **Create calibration data function**:
   ```python
   def create_country_name_data():
       case_notification_data = {
           'year': [2000, 2005, 2010, 2015, 2020],
           'rate_per_100k': [your_data_here],
           'total_cases': [your_data_here],
           'source': ['WHO Global TB Report'] * 5
       }
       
       age_prevalence_data = {
           'age_group': ['15-24', '25-34', '35-44', '45-54', '55-64', '65+'],
           'prevalence_per_100k': [your_data_here],
           'prevalence_percent': [your_data_here],
           'sample_size': [your_data_here],
           'source': ['Country Survey 2018'] * 6
       }
       
       targets = {
           'overall_prevalence_2018': CalibrationTarget(
               name='overall_prevalence_2018',
               value=your_target_value,
               year=2018,
               description='Overall TB prevalence from 2018 survey',
               source='Country TB Prevalence Survey 2018'
           ),
           # ... other targets
       }
       
       return CalibrationData(
           case_notifications=pd.DataFrame(case_notification_data),
           age_prevalence=pd.DataFrame(age_prevalence_data),
           targets=targets,
           country='Country Name',
           description='TB model calibration data for Country Name'
       )
   ```

3. **Use TBsim's calibration features**:
   ```python
   sim, data, report, plotter = run_calibration_analysis("Country Name")
   ```

## Example Workflows

### **Workflow 1: Single Country Calibration**
```python
from scripts.calibration.tb_calibration_generalized import run_calibration_analysis

# Run complete calibration analysis using TBsim features
sim, data, report, plotter = run_calibration_analysis("South Africa")
```

### **Workflow 2: Parameter Sweep**
```python
from scripts.calibration.tb_calibration_generalized import run_parameter_sweep

# Run parameter sweep using TBsim's sweep capabilities
sweep_summary, results_df, best_sim, calibration_data = run_parameter_sweep(
    "South Africa", 
    max_simulations=100
)
```

### **Workflow 3: Custom Configuration**
```python
from tbsim.calibration import (
    DiseaseConfig, SimulationConfig, run_calibration_simulation_suite,
    CalibrationPlotter, create_south_africa_data
)

# Custom configurations using TBsim's config classes
disease_config = DiseaseConfig(beta=0.025, rel_sus_latentslow=0.20)
sim_config = SimulationConfig(n_agents=2000, years=150)

# Run simulation using TBsim's calibration suite
sim = run_calibration_simulation_suite(
    country_name="South Africa",
    disease_config=disease_config,
    sim_config=sim_config
)

# Create plots using TBsim's plotting features
calibration_data = create_south_africa_data()
plotter = CalibrationPlotter()
fig = plotter.plot_calibration_comparison(sim, calibration_data, "custom")
```

## Customization Options

### **Plotting Styles**
```python
plotter = CalibrationPlotter(
    style='seaborn-v0_8',  # Different matplotlib style
    figsize=(16, 10)       # Custom figure size
)
```

### **Calibration Weights**
```python
weights = {
    'case_notifications': 0.5,  # 50% weight
    'age_prevalence': 0.3,      # 30% weight
    'overall_prevalence': 0.2   # 20% weight
}

score = calculate_calibration_score(sim, calibration_data, weights=weights)
```

### **Age Groups**
```python
custom_age_groups = [(0, 14), (15, 29), (30, 44), (45, 59), (60, 200)]
prevalence = compute_age_stratified_prevalence(
    sim, 
    age_groups=custom_age_groups
)
```

## Output Files

TBsim's core features generate comprehensive output files:

- **`calibration_comparison_[Country]_[timestamp].pdf`** - Model vs data comparison
- **`calibration_report_[Country]_[timestamp].json`** - Detailed calibration metrics
- **`calibration_sweep_results_[Country]_[timestamp].pdf`** - Comprehensive sweep analysis
- **`calibration_violin_plots_[Country]_[timestamp].pdf`** - Focused violin plots
- **`calibration_sweep_results_[Country]_[timestamp].csv`** - Sweep results data
- **`calibration_sweep_summary_[Country]_[timestamp].json`** - Sweep summary
- **`case_notifications_[Country]_[timestamp].csv`** - Case notification data
- **`age_prevalence_[Country]_[timestamp].csv`** - Age prevalence data

## Error Handling

TBsim's core features include robust error handling:

- **Missing demographic data** - Falls back to South Africa data
- **Missing intervention results** - Uses fallback calculations
- **File not found errors** - Graceful degradation with warnings
- **Import errors** - Fallback to direct module creation

## Performance Optimization

TBsim's core features are optimized for performance:

- **Configurable simulation size** - Adjust n_agents and years for speed vs accuracy
- **Parallel processing ready** - Framework designed for easy parallelization
- **Memory efficient** - Structured data management
- **Caching friendly** - Results can be easily cached and reused

## File Structure

```
tbsim/calibration/
├── __init__.py              # Calibration module exports
├── functions.py             # Calibration functions and data structures
└── simulation.py            # Simulation framework and configuration

tbsim/plotting/
├── __init__.py              # Plotting module exports
└── plots.py                 # Plotting system (includes CalibrationPlotter)

tbsim/simulation/
├── __init__.py              # Simulation module exports
└── factory.py               # Factory functions for simulation components

tbsim/calibration/           # Calibration functions and simulation
tbsim/plotting/              # Plotting and visualization
tbsim/simulation/            # Simulation factory functions
├── demographics.py          # Demographic utilities
└── probabilities.py         # Probability utilities

scripts/calibration/
├── tb_calibration_generalized.py    # Uses TBsim calibration features
├── tb_calibration_south_africa.py   # Original (still functional)
├── tb_calibration_sweep.py          # Original sweep (still functional)
├── test_simple_calibration.py       # Basic functionality testing
├── run_sa_calibration_demo.py       # Single simulation demonstration
├── beta/                            # Beta-specific analysis
│   ├── beta_calibration_sweep.py    # Systematic beta parameter sweep
│   ├── beta_behavior_comparison.py  # Beta behavior analysis
│   └── beta_logic_review.py         # Beta implementation review
└── README_TBSIM_CALIB_FEATURES.md   # This file
```

## Future Enhancements

TBsim's core features are designed for future expansion:

- **Parallel parameter sweeps** - Multi-core processing
- **Machine learning integration** - Automated parameter optimization
- **Real-time visualization** - Interactive plotting during sweeps
- **Database integration** - Persistent result storage
- **API endpoints** - Web-based calibration interface

## Contributing to TBsim

To extend TBsim's core features:

1. **Add new countries** - Create demographic data and calibration targets
2. **Add new metrics** - Extend CalibrationTarget and scoring functions
3. **Add new plots** - Extend CalibrationPlotter class
4. **Add new interventions** - Extend InterventionConfig and creation functions

## Support

For questions about TBsim's core features:

1. Check the existing scripts for examples
2. Review the docstrings in the utility functions
3. Examine the generated output files for debugging
4. Use the backward compatibility functions as reference

---

**TBsim's core features provide a robust, flexible, and reusable foundation for TB model calibration across different countries and scenarios, with comprehensive plotting capabilities for analysis and visualization.** 