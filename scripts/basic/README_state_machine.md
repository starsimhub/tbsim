# TB State Machine Scripts

This directory contains demonstration scripts for the new TB state machine implementation.

## Scripts

### 1. `run_tb_state_machine.py` - Comprehensive Demonstration

This is the main demonstration script that shows all the features of the TB state machine implementation.

**Features:**
- Comparison between original and state machine implementations
- State machine feature demonstration
- Comprehensive visualizations
- Command-line options for different modes

**Usage:**
```bash
# Run comparison between original and state machine implementations
python run_tb_state_machine.py --mode compare --visualize

# Run single simulation with state machine
python run_tb_state_machine.py --mode state_machine

# Run single simulation with original implementation
python run_tb_state_machine.py --mode original

# Run comparison without saving plots
python run_tb_state_machine.py --mode compare --no-save
```

**Command-line options:**
- `--mode`: Choose between 'single', 'compare', 'original', 'state_machine'
- `--visualize`: Create visualizations
- `--no-save`: Do not save plots to files

### 2. `run_tb_state_machine_simple.py` - Simple Example

This is a simplified version that shows basic usage of the state machine implementation.

**Features:**
- Basic TB simulation with state machine
- Simple state machine feature demonstration
- Minimal code example

**Usage:**
```bash
python run_tb_state_machine_simple.py
```

### 3. `run_tb.py` - Original Script

This is the original TB simulation script for comparison.

**Usage:**
```bash
python run_tb.py
```

## What the Scripts Demonstrate

### State Machine Features

1. **State Statistics**: Current distribution of individuals across TB states
2. **Transition Matrix**: All possible state transitions and their rates
3. **Validation**: State machine configuration validation
4. **Configuration Export**: Export state machine configuration
5. **Visualization**: State transition diagrams and result plots

### Comparison Capabilities

1. **Result Comparison**: Side-by-side comparison of original vs state machine results
2. **Performance**: Minimal overhead of state machine implementation
3. **Compatibility**: Drop-in replacement for original TB class
4. **Validation**: Built-in validation and error checking

### Visualizations

1. **Comparison Plots**: Prevalence, infections, active cases, deaths
2. **State Distribution**: Number of individuals in each state over time
3. **Transition Matrix**: Visual representation of state transitions
4. **Standard TB Plots**: All the usual TB simulation plots

## Expected Output

### Console Output
```
TB State Machine Demonstration Script
============================================================
Mode: compare
Create visualizations: True
Save plots: True

============================================================
TB STATE MACHINE COMPARISON
============================================================

1. Running Original TB Implementation...
Running simulation...

2. Running State Machine TB Implementation...
Running simulation...

3. Comparing Results...
prevalence     : Orig=    0.12, SM=    0.12, Diff=   +0.00 (+0.1%)
new_infections : Orig=   45.00, SM=   44.00, Diff=   -1.00 (-2.2%)
new_active     : Orig=   12.00, SM=   12.00, Diff=   +0.00 (+0.0%)
new_deaths     : Orig=    3.00, SM=    3.00, Diff=   +0.00 (+0.0%)

============================================================
STATE MACHINE FEATURES DEMONSTRATION
============================================================

1. Current State Distribution:
   Latent Slow         :  234
   Latent Fast         :   45
   Active Pre-symptomatic:   12
   Active Smear Positive:    8
   Active Smear Negative:   15
   Active Extra-pulmonary:    3
   Clear               :  683

2. Available Transitions:
   Latent Slow         -> Active Pre-symptomatic: 3.00e-05
   Latent Fast         -> Active Pre-symptomatic: 6.00e-03
   Active Pre-symptomatic -> Clear               : 6.00e+00
   Active Pre-symptomatic -> Active Smear Positive: 3.00e-02
   ...

3. State Machine Validation:
   Valid: True

4. State Machine Configuration:
   States defined: 8
   Transitions defined: 12
   Parameters exported: 13
```

### Generated Files
- `tb_implementation_comparison.png` - Comparison plots
- `tb_state_transition_matrix.png` - State transition matrix
- `tb_state_distribution.png` - State distribution over time

## Key Differences from Original

### Original TB Implementation
- Manual state transitions in `step()` method
- Hard-coded transition logic
- Difficult to extend or modify
- No built-in validation

### State Machine Implementation
- Encapsulated state objects with their own behavior
- Centralized transition management
- Easy to extend with new states
- Built-in validation and error checking
- Comprehensive state statistics and analysis
- Visual transition diagrams

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're running from the correct directory
   ```bash
   cd /path/to/newtbsim/scripts/basic
   python run_tb_state_machine.py
   ```

2. **State Machine Not Available**: Check that the state machine is properly initialized
   ```python
   tb = TBWithStateMachine(use_state_machine=True)
   ```

3. **Plotting Issues**: Make sure matplotlib is properly installed
   ```bash
   pip install matplotlib
   ```

### Performance Notes

- The state machine implementation has minimal performance overhead
- Results should be nearly identical between original and state machine implementations
- Small differences are expected due to random number generation differences

## Next Steps

1. **Explore the Code**: Look at the state machine implementation in `tbsim/state_machine.py`
2. **Read Documentation**: Check `docs/state_machine_guide.md` for detailed usage
3. **Create Custom States**: Follow the guide to add your own states and transitions
4. **Integrate with Your Models**: Use `TBWithStateMachine` as a drop-in replacement

## Support

For questions or issues:
1. Check the documentation in `docs/state_machine_guide.md`
2. Look at the examples in `examples/state_machine_demo.py`
3. Review the architecture documentation in `docs/state_machine_architecture.md`
