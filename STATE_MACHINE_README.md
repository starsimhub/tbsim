# TB State Machine Implementation

## Overview

This implementation provides a comprehensive state machine for TB disease progression that addresses all the specified requirements:

- ✅ **Treats states as objects** - Each state is encapsulated with its own behavior and properties
- ✅ **Encapsulated** - State logic is contained within state classes with clear interfaces
- ✅ **Handles what can be triggered** - Each state defines its possible transitions and conditions
- ✅ **Knows when to progress** - States manage their own transition probabilities and timing
- ✅ **Knows transition order** - Transitions are processed in priority order
- ✅ **Knows transition rates** - All rates are properly managed with units and conversions
- ✅ **Has a state machine manager/handler** - Centralized management through TBStateManager

## Architecture

### Core Components

```
tbsim/
├── state_machine.py          # Core state machine implementation
├── tb_with_state_machine.py  # Enhanced TB class with state machine integration
├── tb.py                     # Original TB implementation (for comparison)
└── examples/
    └── state_machine_demo.py # Comprehensive demonstration script
```

### Key Classes

1. **TBState (Abstract Base Class)**
   - Defines interface for all TB states
   - Manages state-specific behavior and properties
   - Handles entry/exit actions

2. **Concrete State Classes**
   - `LatentSlowState` - Latent TB with slow progression
   - `LatentFastState` - Latent TB with fast progression
   - `ActivePresympState` - Pre-symptomatic active TB
   - `ActiveSmposState` - Smear positive active TB
   - `ActiveSmnegState` - Smear negative active TB
   - `ActiveExptbState` - Extra-pulmonary active TB
   - `ClearState` - Cleared TB (returned to susceptible)
   - `DeadState` - Death from TB
   - `ProtectedState` - Protected from TB (e.g., BCG)

3. **TBTransition**
   - Represents transitions between states
   - Manages rates, conditions, and priorities
   - Handles probability calculations

4. **TBStateMachine**
   - Core state machine logic
   - Processes transitions in priority order
   - Manages state transitions and validation

5. **TBStateManager**
   - High-level interface for TB module integration
   - Handles time step processing
   - Manages transmission rates and risk modifiers

6. **TBWithStateMachine**
   - Enhanced TB class using the state machine
   - Drop-in replacement for original TB class
   - Maintains full Starsim compatibility

## Features

### State Encapsulation

Each state is a self-contained object with:

```python
class LatentSlowState(TBState):
    def is_infectious(self, uids, module):
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids, module):
        return np.zeros(len(uids))
    
    def on_entry(self, uids, sim, module):
        module.susceptible[uids] = True
        module.latent_tb_state[uids] = TBS.LATENT_SLOW
```

### Transition Management

Transitions are defined with rates, conditions, and priorities:

```python
transition = TBTransition(
    target_state=target_state,
    rate=ss.perday(3e-5),
    condition=condition_function,
    probability_func=probability_function,
    priority=1
)
```

### Priority-Based Processing

Transitions are processed in priority order:

1. **Priority 1**: Latent → Pre-symptomatic, Pre-symptomatic → Clear, Active → Clear
2. **Priority 2**: Pre-symptomatic → Active, Active → Death

### Rate Management

All rates use the Starsim TimePar system:

```python
rate_LS_to_presym = ss.perday(3e-5)  # 3e-5 per day
rate_treatment_to_clear = ss.peryear(6)  # 6 per year
prob = rate.to_prob()  # Automatic conversion based on time step
```

## Usage

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
```

### Drop-in Replacement

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

### Advanced Features

```python
# Get transition matrix
transitions = tb.get_transition_matrix()

# Validate state machine
validation = tb.validate_state_machine()

# Export configuration
config = tb.export_state_machine_config()

# Visualize transitions
fig = tb.plot_state_transitions()
```

## Demonstration

Run the comprehensive demonstration:

```bash
# Basic demonstration
python examples/state_machine_demo.py

# Compare with original implementation
python examples/state_machine_demo.py --compare

# Create visualizations
python examples/state_machine_demo.py --visualize

# Full demonstration with comparison and visualization
python examples/state_machine_demo.py --compare --visualize --duration 730
```

## Integration with Existing Code

### Backward Compatibility

The state machine implementation maintains full backward compatibility:

```python
# Use original implementation
tb = TBWithStateMachine(use_state_machine=False)

# Use state machine implementation
tb = TBWithStateMachine(use_state_machine=True)
```

### Starsim Integration

The implementation integrates seamlessly with the Starsim framework:

- Inherits from `ss.Infection`
- Uses Starsim's state management system
- Compatible with all Starsim analyzers and networks
- Maintains the same result structure

## Extensibility

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
```

2. **Add to state machine**:

```python
def _initialize_states(self):
    self.states[TBS.NEW_STATE] = NewState()
```

3. **Define transitions**:

```python
def _setup_transitions(self):
    new_state = self.states[TBS.NEW_STATE]
    new_state.add_transition(TBTransition(
        target_state=self.states[TBS.ACTIVE_SMPOS],
        rate=ss.perday(1e-3),
        priority=1
    ))
```

### Custom Conditions and Probabilities

```python
def custom_condition(uids, sim, module):
    return module.some_condition[uids]

def custom_probability(uids, sim, module):
    base_rate = 1e-4
    modifiers = module.risk_factors[uids]
    return 1 - np.exp(-base_rate * modifiers * sim.t.dt)

transition = TBTransition(
    target_state=target_state,
    rate=ss.perday(1e-4),
    condition=custom_condition,
    probability_func=custom_probability,
    priority=1
)
```

## Testing and Validation

### State Machine Validation

```python
validation = tb.validate_state_machine()
if validation['valid']:
    print("State machine is valid")
else:
    print(f"Errors: {validation['errors']}")
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
    
    def test_validation(self):
        validation = self.tb.validate_state_machine()
        self.assertTrue(validation['valid'])
```

## Performance

### Comparison with Original Implementation

The state machine implementation has minimal performance overhead:

- **Memory**: Slight increase due to state objects
- **CPU**: Negligible overhead for transition processing
- **Maintainability**: Significant improvement in code organization

### Optimization Features

- Batch processing of transitions
- Conditional evaluation of transition conditions
- Priority-based ordering for efficient processing
- Cached probability calculations

## Documentation

- **State Machine Guide**: `docs/state_machine_guide.md` - Comprehensive usage guide
- **API Documentation**: Generated from docstrings
- **Examples**: `examples/state_machine_demo.py` - Complete demonstration
- **Code Comments**: Extensive inline documentation

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

The TB State Machine implementation successfully addresses all the specified requirements:

- ✅ **States as Objects**: Each state is encapsulated with its own behavior
- ✅ **Encapsulation**: State logic is contained within state classes
- ✅ **Transition Management**: Centralized handling of what can be triggered
- ✅ **Progression Control**: States know when and how to progress
- ✅ **Order Management**: Transitions are processed in priority order
- ✅ **Rate Management**: Transition rates are properly managed
- ✅ **Manager Integration**: Centralized state machine manager/handler

The implementation provides a robust, maintainable, and extensible framework for managing TB disease progression while maintaining full compatibility with the existing Starsim framework.
