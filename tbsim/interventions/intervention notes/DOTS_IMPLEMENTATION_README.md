# TBsim DOTS Implementation

This document describes the enhanced DOTS (Directly Observed Treatment, Short-course) implementation in TBsim that matches EMOD-Generic's approach.

## Overview

The new implementation provides detailed drug type modeling similar to EMOD-Generic, with specific drug regimens, parameters, and effects. This replaces the generic treatment approach with a more sophisticated drug type system.

## Key Features

### 1. Drug Type Enumeration (`TBDrugType`)

Matches EMOD-Generic's `TBDrugType` enum with the same values:

```python
class TBDrugType(IntEnum):
    DOTS = 1                    # Directly Observed Treatment, Short-course
    DOTS_IMPROVED = 2           # Improved DOTS
    EMPIRIC_TREATMENT = 3       # Empirical treatment
    FIRST_LINE_COMBO = 4        # First-line combination therapy
    SECOND_LINE_COMBO = 5       # Second-line combination therapy
    THIRD_LINE_COMBO = 6        # Third-line combination therapy
    LATENT_TREATMENT = 7        # Treatment for latent TB
```

### 2. Drug Parameters (`TBDrugParameters`)

Each drug type has specific parameters similar to EMOD-Generic's `TBDrugTypeParameters`:

- `inactivation_rate`: Rate of drug inactivation
- `cure_rate`: Rate of cure/clearance
- `resistance_rate`: Rate of resistance development
- `relapse_rate`: Rate of relapse after treatment
- `mortality_rate`: Rate of mortality reduction
- `duration`: Treatment duration in days
- `adherence_rate`: Expected adherence rate
- `cost_per_course`: Cost per treatment course

### 3. Enhanced Treatment Class (`EnhancedTBTreatment`)

Replaces the generic `TBTreatment` with drug-specific treatment logic:

```python
from tbsim.interventions import create_dots_treatment

# Create DOTS treatment
dots_treatment = create_dots_treatment()

# Create improved DOTS treatment
dots_improved = create_dots_improved_treatment()

# Create first-line combination treatment
first_line = create_first_line_treatment()
```

## Usage Examples

### Basic DOTS Treatment

```python
import tbsim
from tbsim.interventions import create_dots_treatment

# Create simulation with DOTS treatment
sim = ss.Sim(
    people=ss.People(n_agents=1000, extra_states=tbsim.get_extrastates()),
    diseases=tbsim.TB({'init_prev': ss.bernoulli(0.25)}),
    interventions=[
        tbsim.HealthSeekingBehavior(pars={'initial_care_seeking_rate': ss.perday(0.25)}),
        tbsim.TBDiagnostic(pars={'coverage': 0.8, 'sensitivity': 0.85, 'specificity': 0.95}),
        create_dots_treatment(),  # Uses DOTS drug type
    ],
    networks=ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0}),
    pars=dict(start=2000, stop=2010, dt=ss.days(1)/12),
)
```

### Comparing Drug Types

```python
from tbsim.interventions import TBDrugType, get_drug_parameters

# Compare different drug types
dots_params = get_drug_parameters(TBDrugType.DOTS)
dots_improved_params = get_drug_parameters(TBDrugType.DOTS_IMPROVED)
first_line_params = get_drug_parameters(TBDrugType.FIRST_LINE_COMBO)

print(f"DOTS cure rate: {dots_params.cure_rate:.3f}")
print(f"DOTS Improved cure rate: {dots_improved_params.cure_rate:.3f}")
print(f"First Line cure rate: {first_line_params.cure_rate:.3f}")
```

### Custom Drug Parameters

```python
from tbsim.interventions import EnhancedTBTreatment, TBDrugType

# Create custom DOTS treatment with modified parameters
custom_dots = EnhancedTBTreatment(pars={
    'drug_type': TBDrugType.DOTS,
    'treatment_success_rate': 0.90,  # Override default
})
```

## Drug Type Parameters

### DOTS (Standard)
- Cure Rate: 85%
- Inactivation Rate: 10%
- Resistance Rate: 2%
- Relapse Rate: 5%
- Mortality Reduction: 80%
- Duration: 180 days
- Cost: $100 per course

### DOTS Improved
- Cure Rate: 90%
- Inactivation Rate: 8%
- Resistance Rate: 1.5%
- Relapse Rate: 3%
- Mortality Reduction: 85%
- Duration: 180 days
- Cost: $150 per course

### First Line Combination
- Cure Rate: 95%
- Inactivation Rate: 5%
- Resistance Rate: 1%
- Relapse Rate: 2%
- Mortality Reduction: 90%
- Duration: 120 days
- Cost: $200 per course

### Second Line Combination
- Cure Rate: 75%
- Inactivation Rate: 12%
- Resistance Rate: 3%
- Relapse Rate: 8%
- Mortality Reduction: 70%
- Duration: 240 days
- Cost: $500 per course

### Third Line Combination
- Cure Rate: 60%
- Inactivation Rate: 20%
- Resistance Rate: 8%
- Relapse Rate: 15%
- Mortality Reduction: 50%
- Duration: 360 days
- Cost: $1000 per course

### Latent Treatment
- Cure Rate: 90%
- Inactivation Rate: 2%
- Resistance Rate: 0.5%
- Relapse Rate: 1%
- Mortality Reduction: 95%
- Duration: 90 days
- Cost: $50 per course

## Results Tracking

The enhanced treatment system tracks drug-specific results:

```python
# Access results
results = sim.results['enhancedtbtreatment']

# Basic metrics
total_treated = np.sum(results['n_treated'].values)
total_success = np.sum(results['n_treatment_success'].values)
total_failure = np.sum(results['n_treatment_failure'].values)

# Drug type information
drug_type_used = results['drug_type_used'].values[0]
```

## Compatibility with EMOD-Generic

The new implementation maintains compatibility with EMOD-Generic's approach:

1. **Same Drug Type Values**: Enum values match EMOD-Generic exactly
2. **Similar Parameter Structure**: Drug parameters follow the same structure
3. **Compatible Treatment Logic**: Treatment effects are modeled similarly
4. **Equivalent Results**: Output format matches EMOD-Generic expectations

## Testing

Run the test script to verify the implementation:

```bash
python tbsim/test_dots_implementation.py
```

This will:
- Test drug parameters system
- Run simulations with different drug types
- Compare treatment outcomes
- Generate visualization plots

## Migration from Old System

To migrate from the old generic treatment system:

```python
# Old approach
from tbsim.interventions import TBTreatment
old_treatment = TBTreatment(pars={'treatment_success_rate': 0.85})

# New approach
from tbsim.interventions import create_dots_treatment
new_treatment = create_dots_treatment()  # Uses DOTS with 85% success rate
```

## Future Enhancements

Planned enhancements include:

1. **MDR-TB Integration**: Drug resistance evolution modeling
2. **Treatment Adherence**: Individual adherence tracking
3. **Drug Interactions**: Multi-drug regimen effects
4. **Cost Analysis**: Economic evaluation capabilities
5. **Treatment Sequencing**: Multiple treatment line modeling

## Files Modified

- `tbsim/interventions/tb_drug_types.py` - New drug type system
- `tbsim/interventions/enhanced_tb_treatment.py` - Enhanced treatment class
- `tbsim/interventions/__init__.py` - Updated exports
- `tbsim/test_dots_implementation.py` - Test script
- `tbsim/DOTS_IMPLEMENTATION_README.md` - This documentation

## Summary

The new DOTS implementation brings TBsim's treatment modeling to parity with EMOD-Generic, providing:

- ✅ Detailed drug type enumeration
- ✅ Drug-specific parameters
- ✅ Enhanced treatment logic
- ✅ Comprehensive results tracking
- ✅ EMOD-Generic compatibility
- ✅ Easy migration path

This implementation ensures that TBsim can now model DOTS and other TB drug regimens with the same level of detail and sophistication as EMOD-Generic.
