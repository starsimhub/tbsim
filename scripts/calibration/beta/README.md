# Beta Analysis Scripts

This folder contains scripts specifically focused on beta parameter analysis and calibration for the TB model.

## Scripts

### beta_calibration_sweep.py
Systematic calibration sweep of beta values across different transmission contexts.

**Usage**: `python beta_calibration_sweep.py`

*For detailed workflow, see ../CALIBRATION_SCRIPTS_GUIDE.md*

### beta_behavior_comparison.py
Comprehensive comparison of model behavior across different beta values and transmission contexts.

**Usage**: `python beta_behavior_comparison.py`

*For detailed workflow, see ../CALIBRATION_SCRIPTS_GUIDE.md*

### beta_logic_review.py
Review and analysis of beta implementation in the codebase.

**Usage**: `python beta_logic_review.py`

*For detailed workflow, see ../CALIBRATION_SCRIPTS_GUIDE.md*

## Outputs

All scripts generate:
- CSV files with results
- PNG plots for visualization
- JSON reports with detailed analysis

*For detailed output descriptions, see ../CALIBRATION_SCRIPTS_GUIDE.md*

## Literature Context

Beta values are based on epidemiological literature:
- **Household**: High exposure, repeated contacts
- **Community**: Low exposure, diffuse contacts
- **High burden**: Settings with high transmission rates
- **Calibrated**: Values tuned to match incidence data 