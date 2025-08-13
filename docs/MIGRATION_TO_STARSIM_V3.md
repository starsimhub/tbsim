# Migration Guide: TBSim from Starsim v2 to v3

This document outlines the key changes needed to migrate TBSim from Starsim v2 to v3, based on the official Starsim v3.0.0 changelog.

## Overview

Starsim v3 includes significant changes to time handling, rate functions, and module structure. This guide covers the most important changes affecting TBSim.

## Key Changes

### 1. Time Handling

**Before (v2):**
```python
sim = ss.Sim(
    unit='day',
    dt=7,
    start=sc.date('1940-01-01'),
    stop=sc.date('2010-12-31')
)
```

**After (v3):**
```python
sim = ss.Sim(
    dt=ss.days(7),
    start=sc.date('1940-01-01'),
    stop=sc.date('2010-12-31')
)
```

**Changes:**
- Remove `unit` parameter from `ss.Sim()`
- Use `ss.days()`, `ss.years()`, etc. for time units
- `dt` now takes time parameters instead of numbers

### 2. Rate Functions

**Before (v2):**
```python
beta = ss.rate_prob(0.0025, unit='year')
beta = ss.beta(0.1)
```

**After (v3):**
```python
beta = ss.peryear(0.0025)  # Preferred
beta = ss.prob(0.1)        # Literal equivalent of ss.beta()
```

**Changes:**
- `ss.rate_prob()` → `ss.per()` (preferred) or `ss.prob()`
- `ss.beta()` → `ss.prob()`
- `ss.rate()` → `ss.freq()` (preferred) or `ss.per()`

### 3. State Definitions

**Before (v2):**
```python
ss.State('on_treatment', default=False)
ss.State('ever_infected', default=False)
```

**After (v3):**
```python
ss.BoolState('on_treatment', default=False)
ss.BoolState('ever_infected', default=False)
```

**Changes:**
- `ss.State()` → `ss.BoolState()`

### 4. Module Constructor Changes

**Before (v2):**
```python
tb = mtb.TB(dict(
    unit='day',
    dt=7,
    beta=ss.rate_prob(0.0025, unit='year')
))
```

**After (v3):**
```python
tb = mtb.TB(dict(
    dt=ss.days(7),
    beta=ss.peryear(0.0025)
))
```

## Migration Status

### Completed Files

The following files have been migrated to Starsim v3:

- `scripts/basic/run_tb.py` - Basic TB simulation
- `tbsim/tb.py` - Core TB module
- `tbsim/interventions/interventions.py` - Intervention states
- `scripts/common_functions.py` - Common utility functions
- `scripts/burn_in/tb_burn_in_base.py` - Burn-in base script

### Files Needing Migration

The following files still need migration:

#### Scripts Directory
- `scripts/basic/run_tb_and_malnutrition.py`
- `scripts/basic/run_tb_with_analyzer.py`
- `scripts/basic/run_scenarios.py`
- `scripts/basic/run_malnutrition.py`
- `scripts/interventions/run_tb_interventions.py`
- `scripts/interventions/run_tb_interventions_backup.py`
- `scripts/interventions/run_tb_cascadedcare.py`
- `scripts/run_tb_baseline_and_beta.py`
- `scripts/run_tb_bcg_beta.py`
- `scripts/run_tb_bcg_tpt.py`
- `scripts/run_tb_interventions.py`
- All files in `scripts/burn_in/`
- All files in `scripts/calibration/`
- All files in `scripts/hiv/`
- All files in `scripts/howto/`

#### TBSim Core Modules
- `tbsim/interventions/beta.py`
- `tbsim/interventions/tb_health_seeking.py`
- `tbsim/interventions/tpt.py`
- `tbsim/interventions/cascadecare.py`
- `tbsim/interventions/tb_diagnostic.py`
- `tbsim/interventions/tb_treatment.py`
- `tbsim/interventions/healthseeking.py`
- `tbsim/interventions/enhanced_tb_diagnostic.py`
- `tbsim/comorbidities/hiv/hiv.py`
- `tbsim/comorbidities/hiv/tb_hiv_cnn.py`
- `tbsim/comorbidities/malnutrition/malnutrition.py`
- `tbsim/comorbidities/malnutrition/tb_malnut_cnn.py`
- `tbsim/analyzers.py`
- `tbsim/networks.py`
- `tbsim/utils/plots.py`
- `tbsim/wrappers.py`

#### Test Files
- All files in `tests/`

#### Documentation
- All tutorial files in `docs/tutorials/`

## Migration Script

A migration script has been created at `scripts/migrate_to_starsim_v3.py` to help automate the conversion process.

**Usage:**
```bash
# Dry run to see what would be changed
python scripts/migrate_to_starsim_v3.py --dry-run

# Migrate specific file
python scripts/migrate_to_starsim_v3.py --file scripts/basic/run_tb.py

# Migrate all files
python scripts/migrate_to_starsim_v3.py
```

## Manual Migration Steps

For files that need manual attention:

1. **Update Sim constructors:**
   - Remove `unit` parameter
   - Convert `dt` to use time units (e.g., `dt=7` → `dt=ss.days(7)`)

2. **Update rate functions:**
   - `ss.rate_prob(x, unit='year')` → `ss.peryear(x)`
   - `ss.rate_prob(x, unit='month')` → `ss.permonth(x)`
   - `ss.rate_prob(x, unit='day')` → `ss.perday(x)`
   - `ss.beta(x)` → `ss.prob(x)`

3. **Update state definitions:**
   - `ss.State()` → `ss.BoolState()`

4. **Update module parameters:**
   - Remove `unit` parameters from module constructors
   - Update time-based parameters to use new time units

## Testing

After migration, test the following:

1. **Basic functionality:**
   ```bash
   python scripts/basic/run_tb.py
   ```

2. **Run test suite:**
   ```bash
   cd tests
   python -m pytest test_*.py
   ```

3. **Check for import errors:**
   ```bash
   python -c "import tbsim; print('Import successful')"
   ```

## Known Issues

1. **Time parameter conversions:** Some time parameters may need manual adjustment for correct behavior
2. **Rate function semantics:** `ss.per()` vs `ss.prob()` choice depends on intended meaning
3. **Module initialization:** Some modules may need updates to their `__init__` methods

## Resources

- [Starsim v3.0.0 Changelog](https://github.com/starsim/starsim/releases/tag/v3.0.0)
- [Starsim Migration Guide](https://github.com/starsim/starsim/tree/main/docs/migration_v2v3)
- [Starsim Documentation](https://starsim.readthedocs.io/)

## Notes

- The migration script provides automated conversion but may not catch all edge cases
- Manual review is recommended for complex files
- Test thoroughly after migration to ensure correct behavior
- Some patterns may need context-specific adjustments 