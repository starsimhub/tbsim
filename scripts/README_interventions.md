# Multi-Intervention TB Simulations

This directory contains simplified scripts for running TB simulations with multiple interventions.

## Quick Start

### Run predefined scenarios with plots:
```bash
python run_tb_interventions.py
```

### Run simple examples:
```bash
python simple_example.py
```

## How to Use

### 1. Single Intervention
```python
scenario = {
    'bcgintervention': {
        'coverage': 0.8,
        'start': sc.date('1980-01-01'),
        'stop': sc.date('2020-12-31'),
        'age_range': (1, 5),
    }
}
```

### 2. Multiple Interventions of Same Type
```python
scenario = {
    'bcgintervention': [
        {
            'coverage': 0.9,
            'start': sc.date('1980-01-01'),
            'stop': sc.date('2020-12-31'),
            'age_range': (0, 2),
        },
        {
            'coverage': 0.3,
            'start': sc.date('1985-01-01'),
            'stop': sc.date('2015-12-31'),
            'age_range': (15, 25),
        }
    ]
}
```

### 3. Different Types of Interventions
```python
scenario = {
    'bcgintervention': {
        'coverage': 0.8,
        'start': sc.date('1980-01-01'),
        'stop': sc.date('2020-12-31'),
        'age_range': (1, 5),
    },
    'tptintervention': {
        'p_tpt': ss.bernoulli(0.7),
        'max_age': 50,
        'hiv_status_threshold': True,
        'start': sc.date('1990-01-01'),
    }
}
```

## Available Intervention Types

### BCG Interventions (`bcgintervention`)
- **coverage**: Fraction of eligible individuals vaccinated (0-1)
- **start/stop**: Date range for intervention
- **age_range**: Tuple of (min_age, max_age) for eligibility

### TPT Interventions (`tptintervention`)
- **p_tpt**: Probability of TPT initiation (use `ss.bernoulli()`)
- **max_age**: Maximum age for eligibility
- **hiv_status_threshold**: Whether to consider HIV status
- **start**: Start date for intervention

### Beta Interventions (`betabyyear`)
- **years**: List of years when to apply intervention
- **x_beta**: Multiplicative factor for transmission rate. Can be a single value (applied to all years) or a list of the same length as years (each value applied to the corresponding year).

#### Example usages:
```python
# Apply the same reduction in multiple years
scenario = {
    'betabyyear': {
        'years': [2000, 2010, 2020],
        'x_beta': 0.7  # 30% reduction applied at each year
    }
}

# Apply different reductions in different years
scenario = {
    'betabyyear': {
        'years': [2000, 2010, 2020],
        'x_beta': [0.7, 0.8, 0.9]  # 30%, 20%, and 10% reductions respectively
    }
}
```

- Each (year, x_beta) pair is only applied once during the simulation and then removed, so the same change will not be applied repeatedly if the simulation steps through the same year multiple times.

## Examples

See `simple_example.py` for complete working examples of:
1. Single BCG intervention
2. Multiple BCG interventions
3. Combined BCG + TPT interventions
4. Custom scenario with multiple intervention types

## Files

- `run_tb_interventions.py`: Main script with predefined scenarios
- `simple_example.py`: Simple examples for learning
- `README_interventions.md`: This file

## Notes

- Each intervention automatically gets a unique name to avoid conflicts
- You can mix single interventions (dict) and multiple interventions (list)
- All scenarios run successfully and generate plots
- The tuple warnings are harmless and don't affect results 