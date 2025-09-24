# SFT (System Functional Testing) Validation Suite

This directory contains validation scripts for testing the effectiveness and functionality of TB simulation interventions, particularly the BCG intervention.

## Overview

The SFT validation suite ensures that interventions are working correctly and making measurable differences in TB disease modeling indicators compared to baseline scenarios.

## Validation Tests

### BCG Intervention Effectiveness Validation

**File:** `validation_bcg_effectiveness.py`

**Purpose:** Validates that the BCG intervention makes a measurable difference in TB disease modeling indicators compared to baseline.

**What it tests:**
- BCG vaccination coverage and effectiveness
- Individual-level risk modifier changes
- Population-level TB disease indicators
- Clinical significance of BCG protection

**Key Metrics Validated:**
- **Activation Risk Reduction:** ~42% average reduction in TB activation risk
- **Clearance Rate Improvement:** ~39% average improvement in bacterial clearance
- **Death Risk Reduction:** ~90% average reduction in TB mortality risk
- **Vaccine Effectiveness:** >90% in target population (0-5 years)
- **Population Coverage:** ~28% of total population vaccinated

**Expected Results:**
- *BCG intervention successfully applied
- *High vaccine effectiveness (>90%)
- *Measurable population-level protection effects
- *Significant reduction in TB progression risk
- *Improved bacterial clearance capacity
- *Substantial reduction in TB mortality risk

## Running the Validation Tests

### Run Individual Test
```bash
python tests/SFT/validation_bcg_effectiveness.py
```

### Run All SFT Tests
```bash
python tests/SFT/run_validation.py
```

## Test Results Interpretation

### *PASSED
- BCG intervention successfully applied to target population
- Measurable individual-level protection effects
- Population-level impact on TB disease indicators
- Biologically plausible protection mechanisms

### X FAILED
- BCG intervention not applied (no vaccinations)
- No measurable impact on TB disease indicators
- Technical errors in intervention implementation

## Key Validation Criteria

1. **Vaccination Success:** >0 individuals vaccinated in target age group
2. **Vaccine Effectiveness:** >80% of vaccinated individuals show protection
3. **Risk Modifier Changes:** Measurable changes in TB risk modifiers
4. **Population Coverage:** Reasonable coverage based on age targeting
5. **Clinical Impact:** Clear evidence of TB protection benefits

## Technical Notes

- Tests use standardized population with realistic age distribution
- Baseline comparison ensures intervention impact is measurable
- Individual-level analysis validates biological mechanisms
- Population-level analysis validates epidemiological impact
- Error handling ensures robust testing across different scenarios

## Dependencies

- `tbsim` - TB simulation framework
- `starsim` - Simulation framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations

## Troubleshooting

### Common Issues

1. **No vaccinations applied:**
   - Check age range targeting (default: 0-5 years)
   - Verify intervention timing (start/stop dates)
   - Ensure population has individuals in target age group

2. **Low vaccine effectiveness:**
   - Check efficacy parameter (default: 0.9)
   - Verify vaccine response modeling
   - Review individual protection assignment

3. **No measurable impact:**
   - Verify risk modifier application
   - Check protection duration settings
   - Ensure intervention is properly linked to TB model

### Debug Mode

For detailed debugging, modify the validation script to include:
- Individual-level modifier values
- Protection status tracking
- Step-by-step intervention application
- Detailed error reporting

## Contributing

When adding new validation tests:

1. Follow the naming convention: `validation_[intervention_name].py`
2. Include comprehensive documentation
3. Test both individual and population-level impacts
4. Provide clear pass/fail criteria
5. Update this README with test description

## Contact

For questions about SFT validation tests, contact the TB simulation team.
