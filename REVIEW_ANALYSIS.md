# TBsim Code Review: Starsim Integration and TB.py Changes

## Executive Summary

This review analyzes the integration between TBsim and the Starsim framework, focusing on recent changes made to `tbsim/tb.py`. The changes represent a significant refactoring from using Starsim's built-in probability filtering system to a more explicit probability calculation approach.

## 1. Starsim Framework Analysis

### 1.1 Core Architecture
- **Base Classes**: Starsim uses a hierarchical disease model:
  - `ss.Disease`: Base class for all diseases (non-communicable)
  - `ss.Infection`: Inherits from Disease, adds transmission logic
  - `ss.SIR`: Example implementation of Infection

### 1.2 Key Components
- **States**: Defined using `ss.define_states()` with various state types (BoolState, FloatArr, etc.)
- **Parameters**: Defined using `ss.define_pars()` with time-based parameters (ss.perday, ss.peryear)
- **Transmission**: Handled automatically by `ss.Infection.infect()` method
- **Probability System**: Uses `ss.bernoulli`, `ss.choice`, and other distribution classes

### 1.3 Probability and Filtering System
The original Starsim approach uses:
```python
# Old approach - using filter() method
new_presymp_uids = self.p_latent_to_presym.filter(latent_uids)
```

The `filter()` method:
- Returns UIDs that meet the probability criteria
- Internally calls `rvs()` to generate random values
- Handles the random number generation automatically
- Returns only the "successful" UIDs

## 2. TB.py Changes Analysis

### 2.1 What Changed
The code was refactored from using Starsim's `filter()` method to explicit probability calculation:

**Before (Old Approach):**
```python
new_presymp_uids = self.p_latent_to_presym.filter(latent_uids)
```

**After (New Approach):**
```python
if len(latent_uids):
    probs = self.p_latent_to_presym(self.sim, latent_uids)
    transition_mask = np.random.random(len(probs)) < probs
    new_presymp_uids = latent_uids[transition_mask]
else:
    new_presymp_uids = np.array([], dtype=int)
```

### 2.2 Changes Made
1. **Latent to Pre-symptomatic transition** (lines 488-496)
2. **Pre-symptomatic to Clear transition** (lines 510-512)
3. **Pre-symptomatic to Active transition** (lines 515-517)
4. **Active to Clear transition** (lines 525-531)
5. **Active to Death transition** (lines 545-551)

### 2.3 Key Improvements
1. **Explicit Probability Calculation**: The new approach explicitly calculates probabilities using the custom probability methods
2. **Better Error Handling**: Added checks for empty UID arrays
3. **More Control**: Direct control over random number generation
4. **Consistency**: All transitions now use the same pattern

## 3. Technical Analysis

### 3.1 Probability Methods
The TB class implements custom probability calculation methods:
- `p_latent_to_presym()`: Calculates transition probabilities from latent to pre-symptomatic
- `p_presym_to_clear()`: Calculates clearance probabilities for pre-symptomatic cases
- `p_presym_to_active()`: Calculates progression to active disease
- `p_active_to_clear()`: Calculates clearance for active cases
- `p_active_to_death()`: Calculates death probabilities

### 3.2 Rate Conversion
The methods convert Starsim rate parameters to probabilities:
```python
rate = ss.per(ratevals_arr, unit=unit)
prob = rate.to_prob()
```

### 3.3 Individual Risk Factors
The new approach properly incorporates individual risk modifiers:
- `rr_activation`: Multiplier for latent-to-active transition
- `rr_clearance`: Multiplier for active-to-clearance transition  
- `rr_death`: Multiplier for active-to-death transition

## 4. Compatibility Assessment

### 4.1 Starsim Compatibility
✅ **Fully Compatible**: The changes maintain full compatibility with Starsim:
- Still inherits from `ss.Infection`
- Uses Starsim's state management system
- Maintains Starsim's parameter system
- Compatible with Starsim's transmission logic

### 4.2 Framework Integration
✅ **Well Integrated**: The TB class properly integrates with Starsim:
- Uses `super().__init__()` and `super().step()`
- Implements required methods (`set_prognoses`, `step_die`)
- Follows Starsim patterns for state definition
- Uses Starsim's result tracking system

## 5. Code Quality Assessment

### 5.1 Strengths
1. **Explicit Logic**: The probability calculations are now explicit and easier to understand
2. **Robust Error Handling**: Added proper checks for empty arrays
3. **Consistent Pattern**: All transitions follow the same pattern
4. **Better Documentation**: Methods are well-documented with clear docstrings
5. **Individual Heterogeneity**: Properly handles individual risk factors

### 5.2 Areas for Improvement
1. **Code Duplication**: The transition pattern is repeated multiple times
2. **Magic Numbers**: Some hardcoded values could be parameters
3. **Error Messages**: Could be more descriptive in some cases

## 6. Recommendations

### 6.1 Immediate Actions
1. **Test Thoroughly**: Ensure all transitions work correctly with the new approach
2. **Validate Results**: Compare simulation outputs before/after changes
3. **Update Documentation**: Ensure all probability methods are documented

### 6.2 Future Improvements
1. **Refactor Common Pattern**: Create a helper method for the transition pattern:
   ```python
   def _apply_transition(self, uids, prob_method, sim):
       if len(uids) == 0:
           return np.array([], dtype=int)
       probs = prob_method(sim, uids)
       mask = np.random.random(len(probs)) < probs
       return uids[mask]
   ```

2. **Add Validation**: Add more robust validation for probability calculations
3. **Performance Optimization**: Consider vectorizing operations where possible

## 7. Conclusion

The changes to `tbsim/tb.py` represent a well-executed refactoring that:
- Maintains full compatibility with the Starsim framework
- Improves code clarity and maintainability
- Provides better control over probability calculations
- Handles edge cases more robustly

The refactoring is a positive improvement that makes the TB model more explicit and easier to understand while maintaining all the benefits of the Starsim framework. The changes should be tested thoroughly but appear to be a solid improvement to the codebase.

## 8. Testing Recommendations

1. **Unit Tests**: Test each probability method individually
2. **Integration Tests**: Test the full TB simulation workflow
3. **Regression Tests**: Compare results with previous versions
4. **Edge Case Tests**: Test with empty populations, extreme parameters
5. **Performance Tests**: Ensure no significant performance degradation
