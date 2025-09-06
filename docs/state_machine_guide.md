# TB State Machine Implementation Guide

## Overview

The TB State Machine is a comprehensive implementation that addresses the requirements for treating states as objects, encapsulation, transition management, and centralized control. This guide provides detailed documentation on how to use and extend the state machine system.

## Architecture

### Core Components

1. **TBState (Abstract Base Class)**: Defines the interface for all TB states
2. **Concrete State Classes**: Implement specific state behavior (LatentSlowState, ActiveSmposState, etc.)
3. **TBTransition**: Represents transitions between states with rates and conditions
4. **TBStateMachine**: Core state machine that manages transitions
5. **TBStateManager**: High-level interface for integrating with TB module
6. **TBWithStateMachine**: Enhanced TB class that uses the state machine

### Key Features

- **Encapsulated States**: Each state is an object with its own behavior and properties
- **Transition Management**: Centralized handling of what can be triggered from each state
- **Rate Control**: States know their transition rates and when to progress
- **Order Management**: Transitions are processed in priority order
- **Manager Integration**: Centralized state machine manager/handler

## State Definitions

### TB States

```python
class TBS(IntEnum):
    NONE = -1           # No TB infection (susceptible)
    LATENT_SLOW = 0     # Latent TB with slow progression
    LATENT_FAST = 1     # Latent TB with fast progression  
    ACTIVE_PRESYMP = 2  # Active TB in pre-symptomatic phase
    ACTIVE_SMPOS = 3    # Active TB, smear positive (most infectious)
    ACTIVE_SMNEG = 4    # Active TB, smear negative (moderately infectious)
    ACTIVE_EXPTB = 5    # Active TB, extra-pulmonary (least infectious)
    DEAD = 8            # Death from TB
    PROTECTED = 100     # Protected from TB (e.g., from BCG vaccination)
```

### State Classes

Each state class implements the `TBState` interface:

```python
class LatentSlowState(TBState):
    def __init__(self):
        super().__init__(TBS.LATENT_SLOW, "Latent Slow", "Latent TB with slow progression")
    
    def is_infectious(self, uids, module):
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids, module):
        return np.zeros(len(uids))
    
    def on_entry(self, uids, sim, module):
        module.susceptible[uids] = True  # Slow progressors remain susceptible
        module.latent_tb_state[uids] = TBS.LATENT_SLOW
```

## Transition System

### Transition Definition

Transitions are defined with rates, conditions, and priorities:

```python
transition = TBTransition(
    target_state=target_state,
    rate=ss.perday(3e-5),  # Transition rate
    condition=condition_function,  # Optional condition
    probability_func=probability_function,  # Optional custom probability
    priority=1  # Execution order (lower numbers first)
)
```

### Transition Processing

The state machine processes transitions in priority order:

1. **Priority 1**: Latent → Pre-symptomatic, Pre-symptomatic → Clear, Active → Clear
2. **Priority 2**: Pre-symptomatic → Active, Active → Death

### Rate Management

Transition rates are managed through the `ss.TimePar` system:

```python
# Define rates with units
rate_LS_to_presym = ss.perday(3e-5)  # 3e-5 per day
rate_treatment_to_clear = ss.peryear(6)  # 6 per year

# Convert to probabilities
prob = rate.to_prob()  # Automatic conversion based on time step
```

## Usage Examples

### Basic Usage

```python
from tbsim import TBWithStateMachine
import starsim as ss

# Create simulation with state machine-enabled TB
sim = ss.Sim()
tb = TBWithStateMachine(use_state_machine=True)
sim.add_module(tb)
sim.run()

# Get state statistics
stats = tb.get_state_statistics()
print(f"Current state distribution: {stats}")

# Get transition matrix
transitions = tb.get_transition_matrix()
print(f"Available transitions: {transitions}")
```

### Advanced Configuration

```python
# Custom parameters
tb = TBWithStateMachine(pars={
    'beta': 0.5,
    'rate_LS_to_presym': ss.perday(1e-4),  # Faster progression
    'rel_trans_smpos': 1.5,  # Higher transmission
})

# Validate state machine
validation = tb.validate_state_machine()
if not validation['valid']:
    print(f"Validation errors: {validation['errors']}")

# Export configuration
config = tb.export_state_machine_config()
```

### Visualization

```python
# Plot state transition matrix
fig = tb.plot_state_transitions()
fig.savefig('tb_transitions.png')

# Plot standard TB results
fig = tb.plot()
fig.show()
```

## Integration with Existing Code

### Drop-in Replacement

The `TBWithStateMachine` class is designed as a drop-in replacement for the original `TB` class:

```python
# Original code
from tbsim import TB
tb = TB()

# Enhanced code with state machine
from tbsim import TBWithStateMachine
tb = TBWithStateMachine(use_state_machine=True)

# Both work identically with Starsim
sim = ss.Sim()
sim.add_module(tb)
sim.run()
```

### Backward Compatibility

The state machine can be disabled for backward compatibility:

```python
# Use original implementation
tb = TBWithStateMachine(use_state_machine=False)
```

## Extending the State Machine

### Adding New States

1. **Define the state class**:

```python
class NewState(TBState):
    def __init__(self):
        super().__init__(TBS.NEW_STATE, "New State", "Description")
    
    def is_infectious(self, uids, module):
        return np.ones(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids, module):
        return np.full(len(uids), 0.5)
    
    def on_entry(self, uids, sim, module):
        # State-specific initialization
        pass
```

2. **Add to state machine**:

```python
def _initialize_states(self):
    # ... existing states ...
    self.states[TBS.NEW_STATE] = NewState()
```

3. **Define transitions**:

```python
def _setup_transitions(self):
    # ... existing transitions ...
    
    # Add new transition
    new_state = self.states[TBS.NEW_STATE]
    new_state.add_transition(TBTransition(
        target_state=self.states[TBS.ACTIVE_SMPOS],
        rate=ss.perday(1e-3),
        priority=1
    ))
```

### Custom Transition Conditions

```python
def custom_condition(uids, sim, module):
    """Custom condition for transition."""
    return module.some_condition[uids]

transition = TBTransition(
    target_state=target_state,
    rate=ss.perday(1e-4),
    condition=custom_condition,
    priority=1
)
```

### Custom Probability Functions

```python
def custom_probability(uids, sim, module):
    """Custom probability calculation."""
    base_rate = 1e-4
    modifiers = module.risk_factors[uids]
    return 1 - np.exp(-base_rate * modifiers * sim.t.dt)

transition = TBTransition(
    target_state=target_state,
    rate=ss.perday(1e-4),  # Used as fallback
    probability_func=custom_probability,
    priority=1
)
```

## Testing and Validation

### State Machine Validation

```python
# Validate state machine configuration
validation = tb.validate_state_machine()

if validation['valid']:
    print("State machine is valid")
else:
    print(f"Errors: {validation['errors']}")
    print(f"Warnings: {validation['warnings']}")
```

### Unit Testing

```python
import unittest

class TestTBStateMachine(unittest.TestCase):
    def setUp(self):
        self.tb = TBWithStateMachine()
        self.sim = ss.Sim()
        self.sim.add_module(self.tb)
        self.sim.initialize()
    
    def test_state_transitions(self):
        # Test specific state transitions
        pass
    
    def test_transmission_rates(self):
        # Test transmission rate calculations
        pass
    
    def test_validation(self):
        # Test state machine validation
        validation = self.tb.validate_state_machine()
        self.assertTrue(validation['valid'])
```

## Performance Considerations

### State Machine vs Original

The state machine implementation has minimal performance overhead:

- **Memory**: Slight increase due to state objects
- **CPU**: Negligible overhead for transition processing
- **Maintainability**: Significant improvement in code organization

### Optimization Tips

1. **Batch Processing**: Transitions are processed in batches for efficiency
2. **Conditional Evaluation**: Conditions are only evaluated when needed
3. **Priority Ordering**: Transitions are sorted by priority once during initialization

## Troubleshooting

### Common Issues

1. **State Machine Not Initialized**:
   ```python
   # Ensure state machine is enabled
   tb = TBWithStateMachine(use_state_machine=True)
   ```

2. **Invalid States**:
   ```python
   # Check for invalid states
   validation = tb.validate_state_machine()
   if not validation['valid']:
       print(validation['errors'])
   ```

3. **Transition Not Working**:
   ```python
   # Check transition matrix
   transitions = tb.get_transition_matrix()
   print(transitions)
   ```

### Debug Mode

Enable debug logging to trace state transitions:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# State transitions will be logged
tb = TBWithStateMachine(use_state_machine=True)
```

## Future Enhancements

### Planned Features

1. **Dynamic State Addition**: Runtime addition of new states
2. **State History Tracking**: Track state transition history
3. **Parallel Processing**: Multi-threaded transition processing
4. **State Machine Visualization**: Interactive state diagram
5. **Custom State Behaviors**: Plugin system for custom state logic

### Contributing

To contribute to the state machine implementation:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Submit pull requests with clear descriptions

## Conclusion

The TB State Machine provides a robust, maintainable, and extensible framework for managing TB disease progression. It successfully addresses all the requirements:

- ✅ **States as Objects**: Each state is encapsulated with its own behavior
- ✅ **Encapsulation**: State logic is contained within state classes
- ✅ **Transition Management**: Centralized handling of what can be triggered
- ✅ **Progression Control**: States know when and how to progress
- ✅ **Order Management**: Transitions are processed in priority order
- ✅ **Rate Management**: Transition rates are properly managed
- ✅ **Manager Integration**: Centralized state machine manager/handler

The implementation maintains full compatibility with the existing Starsim framework while providing significant improvements in code organization, maintainability, and extensibility.
