# Starsim v3 Migration Summary

## Overview

Based on the Starsim v3.0.0 changelog, I've analyzed the TBSim codebase and identified the key migration requirements. This document summarizes the work completed and provides guidance for completing the migration.

## Key Findings from Changelog Analysis

### Major Changes Affecting TBSim

1. **Time Handling Revolution**
   - `unit` parameter removed from `ss.Sim()` constructor
   - `dt` now takes time parameters (e.g., `ss.days(7)`) instead of numbers
   - Time is now based on precise datetime stamps (pd.Timestamp)

2. **Rate Function Changes**
   - `ss.rate_prob()` → `ss.per()` (preferred) or `ss.prob()`
   - `ss.beta()` → `ss.prob()`
   - `ss.rate()` → `ss.freq()` (preferred) or `ss.per()`

3. **State System Updates**
   - `ss.State()` → `ss.BoolState()`
   - New state management system with better type safety

4. **Module Constructor Changes**
   - `unit` parameter removed from module constructors
   - Time parameters now use explicit time units

## Migration Work Completed

### Files Successfully Migrated

1. **Core TB Module** (`tbsim/tb.py`)
   - Updated `ss.rate_prob()` → `ss.permonth()`
   - Updated `ss.State()` → `ss.BoolState()`

2. **Intervention States** (`tbsim/interventions/interventions.py`)
   - Updated all `ss.State()` → `ss.BoolState()`

3. **Basic Script** (`scripts/basic/run_tb.py`)
   - Removed `unit` parameter from `ss.Sim()`
   - Updated `dt=7` → `dt=ss.days(7)`
   - Updated `ss.rate_prob()` → `ss.peryear()`

4. **Common Functions** (`scripts/common_functions.py`)
   - Updated `ss.beta()` → `ss.prob()`
   - Removed `unit` parameters

5. **Burn-in Base** (`scripts/burn_in/tb_burn_in_base.py`)
   - Updated Sim constructor parameters
   - Updated rate functions

6. **Baseline Script** (`scripts/run_tb_baseline_and_beta.py`)
   - Updated time parameters and rate functions

### Migration Tools Created

1. **Migration Script** (`scripts/migrate_to_starsim_v3.py`)
   - Automated conversion of common patterns
   - Supports dry-run mode for preview
   - Handles file-specific or bulk migration

2. **Migration Guide** (`docs/MIGRATION_TO_STARSIM_V3.md`)
   - Comprehensive documentation of changes
   - Before/after examples
   - Step-by-step migration instructions

## Migration Patterns Identified

### High-Frequency Patterns (Found in 50+ files)

1. **Sim Constructor Changes**
   ```python
   # Before
   sim = ss.Sim(unit='day', dt=7, ...)
   
   # After  
   sim = ss.Sim(dt=ss.days(7), ...)
   ```

2. **Rate Function Updates**
   ```python
   # Before
   beta = ss.rate_prob(0.0025, unit='year')
   beta = ss.beta(0.1)
   
   # After
   beta = ss.peryear(0.0025)
   beta = ss.prob(0.1)
   ```

3. **State Definition Changes**
   ```python
   # Before
   ss.State('on_treatment', default=False)
   
   # After
   ss.BoolState('on_treatment', default=False)
   ```

### Medium-Frequency Patterns (Found in 10-50 files)

1. **Module Parameter Updates**
   ```python
   # Before
   tb = mtb.TB(dict(unit='day', dt=7, ...))
   
   # After
   tb = mtb.TB(dict(dt=ss.days(7), ...))
   ```

2. **Time Parameter Conversions**
   ```python
   # Before
   dt=30, unit='day'
   
   # After
   dt=ss.days(30)
   ```

## Files Requiring Migration

### High Priority (Core Functionality)
- `tbsim/interventions/beta.py`
- `tbsim/interventions/tb_health_seeking.py`
- `tbsim/interventions/tpt.py`
- `tbsim/analyzers.py`
- `scripts/interventions/run_tb_interventions.py`

### Medium Priority (Scripts)
- All files in `scripts/burn_in/`
- All files in `scripts/calibration/`
- All files in `scripts/hiv/`

### Low Priority (Documentation/Tests)
- All files in `tests/`
- All files in `docs/tutorials/`

## Migration Strategy

### Phase 1: Core Modules (Complete)
- ✅ Core TB module
- ✅ Intervention states
- ✅ Basic scripts

### Phase 2: Intervention Modules (Next)
- Beta intervention
- Health seeking behavior
- TPT (Treatment as Prevention)
- Enhanced diagnostics

### Phase 3: Scripts and Utilities
- Burn-in scripts
- Calibration scripts
- HIV integration scripts

### Phase 4: Testing and Documentation
- Test suite updates
- Tutorial updates
- Documentation fixes

## Testing Recommendations

1. **Basic Functionality Test**
   ```bash
   python scripts/basic/run_tb.py
   ```

2. **Import Test**
   ```bash
   python -c "import tbsim; print('Import successful')"
   ```

3. **Test Suite**
   ```bash
   cd tests && python -m pytest test_*.py
   ```

## Known Issues and Considerations

1. **Time Parameter Semantics**
   - Some time parameters may need manual adjustment
   - Context matters for `ss.per()` vs `ss.prob()` choice

2. **Module Initialization**
   - Some modules may need `__init__` method updates
   - Parameter validation may need updates

3. **Backward Compatibility**
   - Old parameter sets may not work without updates
   - Some API changes are breaking

## Next Steps

1. **Complete Core Module Migration**
   - Migrate remaining intervention modules
   - Update analyzers and utilities

2. **Script Migration**
   - Use migration script for bulk conversion
   - Manual review of complex scripts

3. **Testing and Validation**
   - Comprehensive testing of migrated code
   - Validation of simulation results

4. **Documentation Updates**
   - Update tutorials and examples
   - Update API documentation

## Resources

- **Migration Script**: `scripts/migrate_to_starsim_v3.py`
- **Migration Guide**: `docs/MIGRATION_TO_STARSIM_V3.md`
- **Starsim v3 Changelog**: [Official Release Notes](https://github.com/starsim/starsim/releases/tag/v3.0.0)
- **Starsim Migration Guide**: [Official Migration Documentation](https://github.com/starsim/starsim/tree/main/docs/migration_v2v3)

## Notes

- The migration script provides automated conversion but requires manual review
- Some patterns need context-specific handling
- Test thoroughly after each phase of migration
- Consider creating a test suite to validate migration correctness 