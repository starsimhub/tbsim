"""
Simple example showing how to use multiple interventions in TB simulations.

This script demonstrates the basic usage of the run_tb_interventions.py module.
"""

import tbsim as mtb
import starsim as ss
import sciris as sc

# Import the functions from our main script
from scripts.run_tb_bcg_beta import build_sim, run_scenarios

def example_single_intervention():
    """Example: Single BCG intervention"""
    print("=== Example 1: Single BCG Intervention ===")
    
    scenario = {
        'name': 'My BCG Scenario',
        'bcgintervention': {
            'coverage': 0.8,
            'start': sc.date('1980-01-01'),
            'stop': sc.date('2020-12-31'),
            'age_range': (1, 5),
        }
    }
    
    sim = build_sim(scenario=scenario)
    sim.run()
    print(f"Simulation completed with {len(sim.interventions)} intervention(s)")
    return sim

def example_multiple_interventions():
    """Example: Multiple BCG interventions"""
    print("\n=== Example 2: Multiple BCG Interventions ===")
    
    scenario = {
        'name': 'Multiple BCG Scenario',
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
    
    sim = build_sim(scenario=scenario)
    sim.run()
    print(f"Simulation completed with {len(sim.interventions)} intervention(s)")
    return sim

def example_combined_interventions():
    """Example: BCG + TPT interventions"""
    print("\n=== Example 3: BCG + TPT Combined ===")
    
    scenario = {
        'name': 'Combined Scenario',
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
    
    sim = build_sim(scenario=scenario)
    sim.run()
    print(f"Simulation completed with {len(sim.interventions)} intervention(s)")
    return sim

def example_custom_scenario():
    """Example: Create your own custom scenario"""
    print("\n=== Example 4: Custom Scenario ===")
    
    # Define your own scenario
    my_scenario = {
        'name': 'My Custom Scenario',
        'tbpars': {
            'start': sc.date('1980-01-01'),
            'stop': sc.date('2030-12-31'),
        },
        'bcgintervention': [
            {
                'coverage': 0.95,
                'start': sc.date('1980-01-01'),
                'stop': sc.date('2025-12-31'),
                'age_range': (0, 1),
            },
            {
                'coverage': 0.4,
                'start': sc.date('1985-01-01'),
                'stop': sc.date('2015-12-31'),
                'age_range': (10, 20),
            }
        ],
        'tptintervention': {
            'p_tpt': ss.bernoulli(0.8),
            'max_age': 45,
            'hiv_status_threshold': True,
            'start': sc.date('1995-01-01'),
        }
    }
    
    sim = build_sim(scenario=my_scenario)
    sim.run()
    print(f"Custom simulation completed with {len(sim.interventions)} intervention(s)")
    return sim

if __name__ == '__main__':
    print("Simple Examples of Multi-Intervention TB Simulations")
    print("=" * 50)
    
    # Run examples
    example_single_intervention()
    example_multiple_interventions()
    example_combined_interventions()
    example_custom_scenario()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("\nTo run all predefined scenarios with plots, use:")
    print("python run_tb_interventions.py") 