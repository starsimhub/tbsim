# TB.py Docstring Improvements Summary

## Overview
Enhanced all method docstrings in `tbsim/tb.py` to provide more specific and detailed information about the functionality of each method. The improvements focus on mathematical details, state requirements, usage examples, and integration with the Starsim framework.

## Methods Updated

### 1. Probability Calculation Methods

#### `_init_transition_probabilities()`
- **Added**: Explanation of placeholder purpose and future extensibility
- **Added**: Context about dynamic probability calculation approach
- **Added**: Note about consistency with original design pattern

#### `p_latent_to_presym(sim, uids)`
- **Added**: Mathematical details with specific rate values (3e-5 and 6e-3 per day)
- **Added**: Individual risk modifier explanation (rr_activation)
- **Added**: State requirements (LATENT_SLOW or LATENT_FAST)
- **Added**: Usage example with transition mask application
- **Added**: Detailed parameter types and return value description

#### `p_presym_to_clear(sim, uids)`
- **Added**: Treatment dependency explanation (zero for untreated)
- **Added**: Treatment rate details (6 per year, 2-month duration)
- **Added**: State requirements (ACTIVE_PRESYMP)
- **Added**: Usage example showing treatment-only clearance
- **Added**: Mathematical formula explanation

#### `p_presym_to_active(sim, uids)`
- **Added**: Uniform progression explanation
- **Added**: Active state assignment details
- **Added**: Rate value (3e-2 per day)
- **Added**: Usage example with state assignment
- **Added**: No individual risk modifiers note

#### `p_active_to_clear(sim, uids)`
- **Added**: Dual rate system (natural vs. treatment)
- **Added**: Specific rate values and treatment effect
- **Added**: Post-clearance state explanation (return to susceptible)
- **Added**: Individual risk modifier details (rr_clearance)
- **Added**: Usage example with state transition

#### `p_active_to_death(sim, uids)`
- **Added**: State-specific death rates with exact values
- **Added**: Treatment effect (rr_death = 0 for treated)
- **Added**: Death process explanation
- **Added**: Individual risk modifier details (rr_death)
- **Added**: Usage example with death marking

### 2. Core Disease Methods

#### `infectious` (property)
- **Added**: Detailed transmission rate values for each state
- **Added**: State-specific transmissibility explanation
- **Added**: Treatment effect on transmission
- **Added**: Usage example with infectious count
- **Added**: Framework integration details

#### `set_prognoses(uids, sources)`
- **Added**: Progression type assignment details (10% fast, 90% slow)
- **Added**: State initialization explanation
- **Added**: Individual risk factor descriptions
- **Added**: Active state pre-assignment details
- **Added**: Reinfection handling explanation
- **Added**: Framework integration context

#### `step()`
- **Added**: Detailed state transition sequence
- **Added**: Age-specific tracking explanation
- **Added**: Treatment effects summary
- **Added**: Transmission rate update process
- **Added**: Result tracking details
- **Added**: Framework integration context

#### `start_treatment(uids)`
- **Added**: Detailed treatment effects (mortality, transmission, clearance)
- **Added**: Eligibility requirements for active states
- **Added**: Notification tracking explanation
- **Added**: Treatment duration details (2 months average)
- **Added**: Usage example with return value

#### `step_die(uids)`
- **Added**: Death source explanation (TB vs. other causes)
- **Added**: State cleanup details
- **Added**: Framework integration context
- **Added**: Important notes about death handling
- **Added**: Usage example

### 3. Result Tracking Methods

#### `init_results()`
- **Added**: Detailed result variable categories
- **Added**: Age-specific results explanation (15+ years)
- **Added**: Incidence and cumulative measures
- **Added**: Derived epidemiological indicators
- **Added**: Detection and treatment metrics
- **Added**: Framework integration context

#### `update_results()`
- **Added**: State count update details
- **Added**: Detection metrics calculation
- **Added**: Derived indicator formulas
- **Added**: Age-specific calculation details
- **Added**: Population scaling explanation
- **Added**: Time step integration details

#### `finalize_results()`
- **Added**: Cumulative calculation details
- **Added**: Purpose and usage explanation
- **Added**: Framework integration context
- **Added**: Result availability information

#### `plot()`
- **Added**: Plot contents and features
- **Added**: Excluded variables explanation
- **Added**: Usage examples
- **Added**: Limitations and customization notes
- **Added**: Return value details

## Key Improvements

### 1. Mathematical Details
- Added specific rate values and formulas
- Explained probability calculations
- Detailed individual risk modifiers
- Treatment effect specifications

### 2. State Requirements
- Clear state validation requirements
- State transition explanations
- Pre/post condition details

### 3. Usage Examples
- Code examples for key methods
- Parameter usage demonstrations
- Return value handling

### 4. Framework Integration
- Starsim framework context
- Automatic calling mechanisms
- Integration with base classes

### 5. Epidemiological Context
- TB-specific terminology
- Age-specific considerations
- Treatment and detection metrics

## Benefits

1. **Better Understanding**: Developers can now understand the exact functionality of each method
2. **Mathematical Clarity**: Specific rate values and formulas are documented
3. **Usage Guidance**: Examples show how to use methods correctly
4. **Framework Integration**: Clear explanation of Starsim integration
5. **Epidemiological Context**: TB-specific details and terminology
6. **Maintenance**: Easier to maintain and modify code with detailed documentation

## Documentation Standards

All docstrings now follow a consistent format:
- **Brief description** of method purpose
- **Detailed explanation** of functionality
- **Mathematical details** where applicable
- **State requirements** and validation
- **Parameter descriptions** with types
- **Return value descriptions**
- **Usage examples** for key methods
- **Framework integration** context
- **Raises/Exceptions** documentation

This comprehensive documentation makes the TB disease model much more accessible and maintainable for developers working with the TBsim codebase.
