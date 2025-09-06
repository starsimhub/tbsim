# TB State Machine Architecture

## System Overview

The TB State Machine implementation provides a comprehensive, object-oriented approach to managing TB disease progression. The architecture is designed to be modular, extensible, and maintainable while integrating seamlessly with the existing Starsim framework.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TB State Machine System                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │ TBStateManager  │    │ TBStateMachine  │    │ TBTransition │ │
│  │                 │    │                 │    │              │ │
│  │ • process_time_ │    │ • process_      │    │ • target_    │ │
│  │   step()        │    │   transitions() │    │   state      │ │
│  │ • update_       │    │ • get_state()   │    │ • rate       │ │
│  │   transmission_ │    │ • validate()    │    │ • condition  │ │
│  │   rates()       │    │                 │    │ • priority   │ │
│  │ • reset_risk_   │    │                 │    │              │ │
│  │   modifiers()   │    │                 │    │              │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │      │
│           │                       │                       │      │
│           ▼                       ▼                       ▼      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    TBState (Abstract)                      │ │
│  │                                                             │ │
│  │ • is_infectious()                                          │ │
│  │ • get_transmission_rate()                                  │ │
│  │ • on_entry()                                               │ │
│  │ • on_exit()                                                │ │
│  │ • update_state_properties()                                │ │
│  │ • add_transition()                                         │ │
│  │ • get_possible_transitions()                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                  Concrete State Classes                     │ │
│  │                                                             │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │ │
│  │ │LatentSlow   │ │LatentFast   │ │ActivePresymp│ │Active   │ │ │
│  │ │State        │ │State        │ │State        │ │Smpos    │ │ │
│  │ │             │ │             │ │             │ │State    │ │ │
│  │ │• Non-       │ │• Non-       │ │• Infectious │ │• Most   │ │ │
│  │ │  infectious │ │  infectious │ │• Low trans  │ │  infec- │ │ │
│  │ │• Susceptible│ │• Non-       │ │• Pre-       │ │  tious  │ │ │
│  │ │  to reinf.  │ │  susceptible│ │  symptomatic│ │• High   │ │ │
│  │ └─────────────┘ └─────────────┘ └─────────────┘ │  trans  │ │ │
│  │                                                 └─────────┘ │ │
│  │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────┐ │ │
│  │ │Active       │ │Active       │ │Clear        │ │Dead     │ │ │
│  │ │Smneg        │ │Exptb        │ │State        │ │State    │ │ │
│  │ │State        │ │State        │ │             │ │         │ │ │
│  │ │             │ │             │ │• Non-       │ │• Non-   │ │ │
│  │ │• Moderate   │ │• Least      │ │  infectious │ │  infec- │ │ │
│  │ │  infectious │ │  infectious │ │• Susceptible│ │  tious  │ │ │
│  │ │• Moderate   │ │• Low trans  │ │• Reset all  │ │• Request│ │ │
│  │ │  trans      │ │• Extra-     │ │  TB states  │ │  death  │ │ │
│  │ └─────────────┘ │  pulmonary  │ └─────────────┘ └─────────┘ │ │
│  │                 └─────────────┘                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Integration Layer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                TBWithStateMachine                           │ │
│  │                                                             │ │
│  │ • Inherits from TB (ss.Infection)                          │ │
│  │ • Integrates with TBStateManager                           │ │
│  │ • Maintains Starsim compatibility                          │ │
│  │ • Provides enhanced state management                       │ │
│  │ • Supports backward compatibility                          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Starsim Framework                        │ │
│  │                                                             │ │
│  │ • ss.Sim - Simulation orchestration                        │ │
│  │ • ss.Infection - Base disease class                        │ │
│  │ • ss.TimePar - Rate and probability management             │ │
│  │ • ss.Result - Result tracking and analysis                 │ │
│  │ • ss.People - Agent management                             │ │
│  │ • ss.Networks - Contact network management                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## State Transition Flow

```
                    ┌─────────────────┐
                    │   SUSCEPTIBLE   │
                    │   (TBS.NONE)    │
                    └─────────┬───────┘
                              │
                              │ Infection
                              │ (beta parameter)
                              ▼
                    ┌─────────────────┐
                    │   LATENT TB     │
                    │                 │
                    │ ┌─────────────┐ │
                    │ │ LATENT_SLOW │ │ ──┐
                    │ │ (rate_LS)   │ │   │
                    │ └─────────────┘ │   │
                    │                 │   │
                    │ ┌─────────────┐ │   │
                    │ │ LATENT_FAST │ │ ──┼──┐
                    │ │ (rate_LF)   │ │   │  │
                    │ └─────────────┘ │   │  │
                    └─────────────────┘   │  │
                              │           │  │
                              │           │  │
                              ▼           ▼  ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │   PRE-SYMPTOMATIC│   │   PRE-SYMPTOMATIC│
                    │   (TBS.ACTIVE_  │   │   (TBS.ACTIVE_  │
                    │    PRESYMP)     │   │    PRESYMP)     │
                    └─────────┬───────┘   └─────────┬───────┘
                              │                     │
                              │ Treatment           │ Natural
                              │ Clearance           │ Progression
                              │ (rate_treatment)    │ (rate_presym)
                              │                     │
                              ▼                     ▼
                    ┌─────────────────┐   ┌─────────────────┐
                    │   CLEAR         │   │   ACTIVE TB     │
                    │ (TBS.NONE)      │   │                 │
                    │                 │   │ ┌─────────────┐ │
                    │ (return to      │   │ │ ACTIVE_SMPOS│ │
                    │  susceptible)   │   │ │ (most inf.) │ │
                    └─────────────────┘   │ └─────────────┘ │
                                         │                 │
                                         │ ┌─────────────┐ │
                                         │ │ ACTIVE_SMNEG│ │
                                         │ │ (mod. inf.) │ │
                                         │ └─────────────┘ │
                                         │                 │
                                         │ ┌─────────────┐ │
                                         │ │ ACTIVE_EXPTB│ │
                                         │ │ (least inf.)│ │
                                         │ └─────────────┘ │
                                         └─────────┬───────┘
                                                   │
                                                   │ Natural
                                                   │ Recovery
                                                   │ or Treatment
                                                   │
                                                   ▼
                                         ┌─────────────────┐
                                         │   CLEAR         │
                                         │ (TBS.NONE)      │
                                         │                 │
                                         │ (natural or     │
                                         │  treatment)     │
                                         └─────────────────┘
                                                   │
                                                   │ Death
                                                   │ (rate_*_to_dead)
                                                   ▼
                                         ┌─────────────────┐
                                         │     DEATH       │
                                         │   (TBS.DEAD)    │
                                         │                 │
                                         │ (from active    │
                                         │  TB states)     │
                                         └─────────────────┘
```

## Key Design Principles

### 1. Encapsulation
- Each state is a self-contained object with its own behavior
- State-specific logic is contained within state classes
- Clear interfaces define state responsibilities

### 2. Separation of Concerns
- State behavior is separated from transition logic
- Transition management is centralized in the state machine
- Integration with Starsim is handled by the manager

### 3. Extensibility
- New states can be added by extending TBState
- Custom transitions can be defined with conditions and probabilities
- State machine can be extended without modifying existing code

### 4. Maintainability
- Clear code organization and documentation
- Comprehensive testing and validation
- Backward compatibility with existing implementations

### 5. Performance
- Efficient batch processing of transitions
- Priority-based transition ordering
- Minimal overhead compared to original implementation

## Integration Points

### Starsim Framework Integration
- Inherits from `ss.Infection` for full framework compatibility
- Uses `ss.TimePar` for rate and probability management
- Integrates with `ss.Result` for tracking and analysis
- Compatible with all Starsim analyzers and networks

### TB Module Integration
- Drop-in replacement for original TB class
- Maintains identical interface and result structure
- Supports both state machine and original implementations
- Seamless migration path for existing code

### State Machine Integration
- Centralized state management through TBStateManager
- Automatic transition processing and validation
- Comprehensive state statistics and analysis
- Export/import of state machine configurations

## Benefits

### For Developers
- **Cleaner Code**: State logic is organized and encapsulated
- **Easier Testing**: Individual states can be tested in isolation
- **Better Maintainability**: Changes to state behavior are localized
- **Enhanced Debugging**: Clear state transitions and validation

### For Researchers
- **Flexible Modeling**: Easy to add new states and transitions
- **Comprehensive Analysis**: Detailed state statistics and tracking
- **Validation Tools**: Built-in state machine validation
- **Visualization**: State transition diagrams and matrices

### For Users
- **Backward Compatibility**: Existing code continues to work
- **Enhanced Features**: Additional state management capabilities
- **Better Documentation**: Comprehensive guides and examples
- **Improved Reliability**: Validated state transitions and logic

## Future Enhancements

### Planned Features
1. **Dynamic State Addition**: Runtime addition of new states
2. **State History Tracking**: Track state transition history
3. **Parallel Processing**: Multi-threaded transition processing
4. **State Machine Visualization**: Interactive state diagrams
5. **Custom State Behaviors**: Plugin system for custom logic

### Extension Points
1. **Custom State Classes**: Add new states by extending TBState
2. **Custom Transitions**: Define transitions with custom conditions
3. **Custom Probabilities**: Implement custom probability calculations
4. **Custom Managers**: Extend TBStateManager for specialized behavior
5. **Custom Integration**: Create specialized TB module variants

This architecture provides a solid foundation for TB disease modeling while maintaining flexibility for future enhancements and extensions.
