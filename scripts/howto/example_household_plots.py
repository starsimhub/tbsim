#!/usr/bin/env python3
"""
Example script demonstrating household plotting functionality with HouseholdNetGeneric.

This script shows how to:
1. Create household structures for HouseholdNetGeneric
2. Generate household plots during simulation creation
3. Use different plot types (basic, analysis, both)
"""

import numpy as np
import starsim as ss
import tbsim as mtb
from tbsim.utils.plots import plot_household_structure, plot_household_network_analysis


def create_sample_households(n_agents=100):
    """Create sample household structure for testing."""
    households = []
    current_uid = 0
    while current_uid < n_agents:
        household_size = min(np.random.randint(2, 7), n_agents - current_uid)
        if household_size < 2:
            break
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households


def example_basic_household_plot():
    """Example: Generate basic household structure plot."""
    print("Example 1: Basic Household Structure Plot")
    print("=" * 40)
    
    # Create sample households
    households = create_sample_households(50)
    print(f"Created {len(households)} households")
    
    # Create people object
    people = ss.People(n_agents=50, extra_states=mtb.get_extrastates())
    
    # Generate basic household plot
    plot_household_structure(
        households=households,
        people=people,
        title="Example: Basic Household Network",
        show_household_ids=True,
        show_agent_ids=False,
        max_households_to_show=15,
        dark=True,
        savefig=True,
        outdir='results/examples'
    )


def example_analysis_plot():
    """Example: Generate comprehensive household analysis plot."""
    print("\nExample 2: Comprehensive Household Analysis")
    print("=" * 40)
    
    # Create sample households
    households = create_sample_households(100)
    print(f"Created {len(households)} households")
    
    # Create people object with age data
    import pandas as pd
    age_data = pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
    })
    people = ss.People(n_agents=100, age_data=age_data, extra_states=mtb.get_extrastates())
    
    # Generate analysis plot
    plot_household_network_analysis(
        households=households,
        people=people,
        figsize=(15, 10),
        dark=True,
        savefig=True,
        outdir='results/examples'
    )


def example_with_simulation():
    """Example: Use household plotting with actual simulation."""
    print("\nExample 3: Household Plotting with Simulation")
    print("=" * 40)
    
    # Create simulation with household plotting enabled
    from scripts.run_tb_bcg_tpt import build_sim
    
    # Build simulation with household plot
    sim = build_sim(
        show_household_plot=True, 
        household_plot_type='both'  # Show both basic and analysis plots
    )
    
    print("Simulation created successfully with household plots!")
    print("Check the 'results/household_plots' directory for saved figures.")


if __name__ == '__main__':
    print("Household Plotting Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_household_plot()
    example_analysis_plot()
    example_with_simulation()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nHousehold plotting features:")
    print("- Visual representation of household structures")
    print("- Network connectivity analysis")
    print("- Household size distribution")
    print("- Agent age distribution (when available)")
    print("- Integration with HouseholdNetGeneric class")
    print("- Automatic figure saving with timestamps") 