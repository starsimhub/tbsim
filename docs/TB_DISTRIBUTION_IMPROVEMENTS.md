# TB Class Distribution Improvements

This document outlines the improvements made to the TB class using StarSim's advanced distribution API features.

## Overview

The TB class has been enhanced to leverage StarSim's `Dist` and `Dists` classes for better random number generation management, state tracking, and distribution scaling capabilities.

## Key Improvements

### 1. Better Distribution Management with `Dists` Class

**Before:**
```python
self.p_latent_to_presym = ss.bernoulli(p=self.p_latent_to_presym)
self.p_presym_to_clear = ss.bernoulli(p=self.p_presym_to_clear)
# ... individual distributions scattered throughout
```

**After:**
```python
# Create transition probability distributions with proper naming and state management
self.transition_dists = ss.Dists()
self.transition_dists.p_latent_to_presym = ss.bernoulli(name='p_latent_to_presym')
self.transition_dists.p_presym_to_clear = ss.bernoulli(name='p_presym_to_clear')
self.transition_dists.p_presym_to_active = ss.bernoulli(name='p_presym_to_active')
self.transition_dists.p_active_to_clear = ss.bernoulli(name='p_active_to_clear')
self.transition_dists.p_active_to_death = ss.bernoulli(name='p_active_to_death')
```

**Benefits:**
- Centralized distribution management
- Automatic state tracking and synchronization
- Easier debugging and monitoring
- Better memory management

### 2. Named Distributions for Better Tracking

**Before:**
```python
init_prev = ss.bernoulli(0.01)
beta = ss.peryear(0.025)
```

**After:**
```python
init_prev = ss.bernoulli(0.01, name='init_prev')
beta = ss.peryear(0.025, name='beta')
```

**Benefits:**
- Better debugging and error messages
- Easier identification in logs and results
- Improved code readability

### 3. Individual Heterogeneity with Normal Distributions

**Before:**
```python
reltrans_het = ss.constant(v=1.0)
```

**After:**
```python
reltrans_het = ss.normal(loc=1.0, scale=0.2, name='reltrans_het')
```

**Benefits:**
- More realistic individual variation
- Better representation of biological heterogeneity
- Improved model accuracy

### 4. Proper Distribution Initialization

**New Method:**
```python
def init(self, sim):
    """Initialize the TB module with proper distribution management"""
    super().init(sim)
    
    # Initialize all distributions in the transition_dists collection
    self.transition_dists.init(sim=sim, module=self)
    
    # Initialize individual heterogeneity distribution
    self.pars.reltrans_het.init(sim=sim, module=self)
    
    return
```

**Benefits:**
- Proper random number generator setup
- Consistent state management
- Better reproducibility

### 5. Dynamic Distribution Parameter Updates

**Before:**
```python
# Convert to Starsim rate object and apply to_prob()
rate = ss.per(ratevals_arr, unit=unit)
prob = rate.to_prob()
return prob
```

**After:**
```python
# Update the distribution parameters dynamically
self.transition_dists.p_latent_to_presym.set(p=ratevals_arr)

# Return the probability distribution for filtering
return self.transition_dists.p_latent_to_presym
```

**Benefits:**
- More efficient parameter updates
- Better state consistency
- Cleaner code structure

### 6. Advanced Distribution Features

#### Time-Varying Parameters
```python
def create_time_varying_distributions(self):
    """Demonstrate advanced distribution features with time-varying parameters"""
    # Example of time-varying transmission rates
    self.time_varying_beta = ss.bernoulli(
        p=lambda self, sim, uids: np.interp(
            sim.year, 
            [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010], 
            [0.02, 0.025, 0.03, 0.035, 0.04, 0.035, 0.03, 0.025]
        ),
        name='time_varying_beta'
    )
```

#### Age-Dependent Parameters
```python
# Example of age-dependent transition rates
self.age_dependent_activation = ss.bernoulli(
    p=lambda self, sim, uids: np.where(
        self.sim.people.age[uids] < 15,
        0.5 * self.pars.rate_LS_to_presym.rate,  # Lower rate for children
        self.pars.rate_LS_to_presym.rate  # Normal rate for adults
    ),
    name='age_dependent_activation'
)
```

#### Multi-Random Distributions
```python
# Example of multi-random distribution for pairwise transmission
self.pairwise_transmission = ss.multi_random(
    names=['source', 'target'],
    name='pairwise_transmission'
)
```

### 7. Distribution State Management

```python
def manage_distribution_states(self):
    """Demonstrate advanced distribution state management"""
    # Get current state of all distributions
    current_states = {}
    for name, dist in self.transition_dists.items():
        current_states[name] = dist.get_state()
    
    # Example of jumping distributions to a specific time
    if hasattr(self, 'sim') and self.sim is not None:
        self.transition_dists.jump(to=self.sim.ti)
    
    return current_states
```

### 8. Distribution Scaling Capabilities

```python
def demonstrate_distribution_scaling(self):
    """Demonstrate distribution scaling capabilities"""
    # Example of scaling a distribution by time
    scaled_rate = self.pars.rate_LS_to_presym * 2  # Double the rate
    scaled_prob = scaled_rate.to_prob()  # Convert to probability
    
    # Example of checking scale types
    scale_type = ss.distributions.scale_types.check_predraw(self.pars.rate_LS_to_presym)
    
    return {
        'original_rate': self.pars.rate_LS_to_presym,
        'scaled_rate': scaled_rate,
        'scaled_prob': scaled_prob,
        'scale_type': scale_type
    }
```

## Usage Examples

### Basic Usage
```python
import tbsim as mtb
import starsim as ss

# Create TB module with improved distributions
tb = mtb.TB(dict(
    beta = ss.peryear(0.025, name='transmission_rate'),
    init_prev = ss.bernoulli(0.01, name='initial_prevalence'),
    reltrans_het = ss.normal(loc=1.0, scale=0.2, name='transmission_heterogeneity'),
))
```

### Advanced Features
```python
# Access distribution states
states = tb.manage_distribution_states()

# Demonstrate scaling
scaling_info = tb.demonstrate_distribution_scaling()

# Create time-varying distributions
tb.create_time_varying_distributions()
```

## Benefits Summary

1. **Better Reproducibility**: Proper random number generator management
2. **Improved Performance**: More efficient distribution updates
3. **Enhanced Debugging**: Named distributions and state tracking
4. **Greater Flexibility**: Support for time-varying and age-dependent parameters
5. **Better Memory Management**: Centralized distribution handling
6. **Advanced Features**: Multi-random distributions and scaling capabilities
7. **Cleaner Code**: More organized and maintainable structure

## Migration Guide

To migrate existing code to use the improved TB class:

1. **Update Distribution Creation**: Add names to all distributions
2. **Use Transition Dists**: Replace individual distributions with `transition_dists` collection
3. **Initialize Properly**: Ensure `init()` method is called
4. **Update Parameter Updates**: Use `.set()` method for dynamic updates
5. **Leverage Advanced Features**: Add time-varying and age-dependent parameters as needed

## Testing

Run the improved simulation to see the benefits:

```bash
python scripts/basic/run_tb_improved.py
```

This will demonstrate all the new features and show the improved distribution management in action.
