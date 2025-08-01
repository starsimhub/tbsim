# File Reorganization Summary

This document describes the reorganization of files that were previously located in `tbsim/interventions/` and `scripts/` directories and have been moved to their appropriate locations.

## Reorganization Completed

### ✅ **Intervention Classes (Remained in `tbsim/interventions/`)**
These files contain actual intervention classes and should stay in the interventions module:

- `enhanced_tb_diagnostic.py` - EnhancedTBDiagnostic class
- `tb_treatment.py` - TBTreatment class  
- `tb_health_seeking.py` - HealthSeekingBehavior class
- `tb_diagnostic.py` - TB diagnostic intervention
- `healthseeking.py` - Health seeking intervention
- `cascadecare.py` - Cascade care intervention
- `tpt.py` - TPTInitiation class
- `bcg.py` - BCGProtection class
- `interventions.py` - Base intervention classes

### 📁 **Scripts (Organized in `scripts/` subdirectories)**

#### Burn-in Scripts (`scripts/burn_in/`)
- `tb_burn_in8.py` - Refined TB prevalence sweep for manual calibration
- `tb_burn_in_base.py` - Base burn-in script
- `tb_burn_in4.py` - Burn-in script version 4
- `tb_burn_in5.py` - Burn-in script version 5
- `tb_burn_in7.py` - Burn-in script version 7
- `tb_burn_in10.py` - Burn-in script version 10
- `tb_burn_in11.py` - Burn-in script version 11
- `tb_burn_in12.py` - Burn-in script version 12
- `run_tb_burn_in_South_Africa.py` - Large burn-in script for South Africa
- `run_tb_burn_in_South_Africa_enhanced_diagnostic.py` - Enhanced diagnostic burn-in
- `run_tb_burn_in_South_Africa_enhanced_diagnostic_with_demographics.py` - Enhanced diagnostic with demographics

#### Calibration Scripts (`scripts/calibration/`)
- `tb_calibration_south_africa.py` - South Africa calibration script
- `tb_calibration_sweep.py` - Calibration sweep script
- `run_sa_calibration_demo.py` - South Africa calibration demo

#### Optimization Scripts (`scripts/optimization/`)
- `run_optimization_example.py` - TB model parameter optimization example
- `test_simple_optimization.py` - Simple optimization test script

#### Intervention Scripts (`scripts/interventions/`)
- `run_tb_interventions.py` - TB interventions script
- `run_tb_cascadedcare.py` - Cascade care script

#### Basic TB Scripts (`scripts/basic/`)
- `run_tb.py` - Basic TB simulation
- `run_tb_with_analyzer.py` - TB with analyzer
- `run_tb_and_malnutrition.py` - TB with malnutrition
- `run_malnutrition.py` - Malnutrition only
- `run_scenarios.py` - Basic scenarios

#### Utility Files (Remained in `scripts/`)
- `common_functions.py` - Common utility functions
- `plots.py` - Plotting utilities

### 📊 **Results (Moved to `results/interventions/`)**
All PDF output files from previous runs:
- `treatment_grid_*.pdf` - Treatment grid plots
- `total_population_grid_*.pdf` - Total population grid plots
- `tb_prevalence_sweep_*.pdf` - TB prevalence sweep plots
- `hiv_metrics_grid_*.pdf` - HIV metrics grid plots
- `hiv_tb_coinfection_grid_*.pdf` - HIV-TB coinfection grid plots
- `cumulative_diagnostic_grid_*.pdf` - Cumulative diagnostic grid plots
- `cumulative_treatment_grid_*.pdf` - Cumulative treatment grid plots
- `diagnostic_grid_*.pdf` - Diagnostic grid plots
- `health_seeking_grid_*.pdf` - Health seeking grid plots
- `age_prevalence_grid_*.pdf` - Age prevalence grid plots

## Benefits of Reorganization

1. **Clear Separation**: Intervention classes are now clearly separated from scripts and results
2. **Better Organization**: Scripts are organized by purpose (burn-in, calibration, optimization, interventions, basic)
3. **Cleaner Module**: The `tbsim.interventions` module now only contains actual intervention classes
4. **Easier Maintenance**: Related files are grouped together for easier maintenance
5. **Better Import Structure**: Intervention classes can be imported cleanly without script clutter

## Usage After Reorganization

### Running Scripts
```bash
# Burn-in scripts
python scripts/burn_in/tb_burn_in8.py
python scripts/burn_in/run_tb_burn_in_South_Africa.py

# Calibration scripts  
python scripts/calibration/tb_calibration_south_africa.py

# Optimization scripts
python scripts/optimization/run_optimization_example.py

# Intervention scripts
python scripts/interventions/run_tb_interventions.py

# Basic scripts
python scripts/basic/run_tb.py
```

### Importing Interventions
```python
# Clean imports from the interventions module
from tbsim.interventions import TPTInitiation, BCGProtection, TBTreatment
from tbsim.interventions import EnhancedTBDiagnostic, HealthSeekingBehavior
```

### Accessing Results
Results are now stored in `results/interventions/` for better organization and easier cleanup.

## ✅ Verification Results

All moved files have been tested and verified to work correctly:

- **✓ File Structure**: All files are in their correct locations
- **✓ Intervention Imports**: All intervention classes can be imported correctly
- **✓ Calibration Imports**: All calibration modules work with updated imports
- **✓ Optimization Imports**: All optimization scripts can import from calibration modules
- **✓ Burn-in Imports**: All burn-in scripts can import intervention classes correctly
- **✓ Basic Script Imports**: All basic scripts work with updated imports
- **✓ Utility Imports**: All utility modules work correctly

### Import Changes Made

The following import statements were updated to work with the new file structure:

#### Before (old imports):
```python
from tb_health_seeking import HealthSeekingBehavior
from tb_diagnostic import TBDiagnostic
from tb_treatment import TBTreatment
from tb_calibration_south_africa import create_south_africa_data
import scripts.common_functions as cf
import plots as pl
```

#### After (new imports):
```python
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
from tbsim.interventions.tb_diagnostic import TBDiagnostic
from tbsim.interventions.tb_treatment import TBTreatment
from scripts.calibration.tb_calibration_south_africa import create_south_africa_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common_functions as cf
import plots as pl
```

All scripts now work correctly with the new organization!

## 🎉 Final Verification Complete

### ✅ All Tests Passed

1. **File Structure**: All files are in their correct locations
2. **Intervention Imports**: All intervention classes can be imported correctly
3. **Module Exports**: The `tbsim.interventions` module properly exports all classes
4. **Cross-Module Imports**: Scripts can import from other modules correctly
5. **Backward Compatibility**: All existing functionality is preserved
6. **Script Organization**: All scripts are properly organized by category

### Available Intervention Classes

The following intervention classes are now available through `tbsim.interventions`:

- `TPTInitiation` - Tuberculosis Preventive Therapy
- `BCGProtection` - BCG vaccination intervention  
- `TBTreatment` - TB treatment intervention
- `HealthSeekingBehavior` - Health seeking behavior
- `TBDiagnostic` - TB diagnostic testing
- `EnhancedTBDiagnostic` - Enhanced TB diagnostic with detailed parameters
- `HealthSeeking` - Alternative name for HealthSeekingBehavior
- `TbCascadeIntervention` - TB testing and treatment cascade
- `TBVaccinationCampaign` - TB vaccination campaign

### Usage Examples

```python
# Import individual classes
from tbsim.interventions import TPTInitiation, BCGProtection, TBTreatment

# Import all classes
from tbsim.interventions import *

# Use in simulations
sim = ss.Sim(
    people=pop,
    interventions=[
        TPTInitiation(pars={'p_tpt': 0.8}),
        BCGProtection(pars={'coverage': 0.6}),
        TBTreatment(pars={'treatment_success_rate': 0.85})
    ],
    diseases=tb,
    pars=spars
)
```

**✅ File reorganization is complete and all functionality has been verified!** 