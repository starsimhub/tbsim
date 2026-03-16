# What's new

All notable changes to the codebase are documented in this file.

## Version 0.7.0 (2026-03-16)
- Added `tbsim.Sim`, a convenience wrapper around `ss.Sim` that auto-routes flat parameters between the sim and the TB module, provides TB-specific defaults (demographics, networks, disease), and supports a `tb_model` argument for selecting among TB model variants. Includes a `demo()` function for quick setup.
- Refactored all diagnostic and treatment interventions into a product/delivery architecture:
    - Added `Dx` diagnostic product class with DataFrame-based test definitions supporting state, age, and HIV stratification
    - Added built-in diagnostic products: `Xpert`, `OralSwab`, `FujiLAM`, `CAD`
    - Added `DxDelivery` intervention for delivering diagnostics with eligibility filtering, coverage, and false-negative retry logic
    - Added `Tx` treatment product class with drug-type-based efficacy
    - Added built-in treatment products: `DOTS`, `DOTSImproved`, `FirstLine`, `SecondLine`
    - Added `TxDelivery` intervention for delivering treatments with success/failure tracking and retry logic
    - Simplified `drug_types.py` to a single dictionary of drug parameters (previously ~600 lines of class hierarchy)
    - Removed old monolithic `tb_diagnostic.py`, `tb_drug_types.py`, and `tb_treatment.py`
- Removed `TB_EMOD`; only `TB_LSHTM` and `TB_LSHTM_Acute` are supported now
- Added `HouseholdStats` analyzer for tracking household size distributions, age-mixing matrices, and contact patterns over time when using `ss.HouseholdNet`. Includes visualization methods for household statistics, age-mixing heatmaps, and normalized contact matrices.
- Renamed `tb_health_seeking.py` to `health_seeking.py` and simplified the `HealthSeekingBehavior` class
- Moved `immigration.py` to an archive folder
- Added admin files (`code_of_conduct.md`, `contributing.md`, `CHANGELOG.md`)
- Updated `starsim` dependency to v3.2.1 for `HouseholdNet` support