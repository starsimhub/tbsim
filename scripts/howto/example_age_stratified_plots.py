#!/usr/bin/env python3
"""
Example script demonstrating the new age-stratified plotting functionality.

This script shows how to use the enhanced plot_results function to generate
age-stratified plots from simulation results.
"""

import tbsim as mtb
import starsim as ss
import sciris as sc
import tbsim.utils.plots as pl
import pandas as pd

def build_example_sim():
    """Build a simple TB simulation for demonstration."""
    
    # Simulation parameters
    spars = dict(
        unit='day',
        dt=7,
        start=sc.date('1975-01-01'),
        stop=sc.date('1985-12-31'),  # 10 years
        rand_seed=123,
        verbose=0,
    )

    tbpars = dict(
        beta=ss.rate_prob(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        unit='day',
        dt=7,      
        start=sc.date('1975-02-01'),
        stop=sc.date('1985-12-31'),
    )

    # Age distribution with good coverage across age groups
    age_data = pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [15, 10, 20, 15, 10, 8, 6, 5, 4, 3, 2, 1]
    })

    # Build simulation
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        mtb.HouseholdNet()
    ]
    
    return ss.Sim(
        people=pop,
        networks=networks,
        diseases=[tb],
        pars=spars,
    )

def example_traditional_plotting():
    """Example of traditional plotting (existing functionality)."""
    print("Example 1: Traditional plotting with flat_results")
    print("=" * 50)
    
    # Run simulation
    sim = build_example_sim()
    sim.run()
    
    # Flatten results manually
    flat_results = {'Overall_Population': sim.results.flatten()}
    
    # Plot using traditional method
    pl.plot_results(
        flat_results=flat_results,
        keywords=['prevalence', 'incidence'],
        n_cols=2,
        dark=True,
        savefig=False
    )

def example_age_stratified_plotting():
    """Example of age-stratified plotting (new functionality)."""
    print("\nExample 2: Age-stratified plotting with results object")
    print("=" * 50)
    
    # Run simulation
    sim = build_example_sim()
    sim.run()
    
    # Define age bins for stratification
    age_bins = [0, 5, 15, 30, 50, 200]  # Creates bins: 0-5, 5-15, 15-30, 30-50, 50+
    
    # Plot using new age-stratified method
    pl.plot_results(
        results=sim.results,
        sim=sim,
        age_bins=age_bins,
        keywords=['prevalence', 'incidence'],
        n_cols=2,
        dark=True,
        savefig=False
    )

def example_custom_age_bins():
    """Example with custom age bins."""
    print("\nExample 3: Custom age bins")
    print("=" * 50)
    
    # Run simulation
    sim = build_example_sim()
    sim.run()
    
    # Define custom age bins (children, young adults, older adults, elderly)
    age_bins = [0, 18, 35, 65, 200]
    
    # Plot with custom age bins
    pl.plot_results(
        results=sim.results,
        sim=sim,
        age_bins=age_bins,
        keywords=['prevalence_active', 'incidence_kpy'],
        n_cols=2,
        dark=False,
        cmap='viridis',
        savefig=False
    )

def example_error_handling():
    """Example of error handling."""
    print("\nExample 4: Error handling")
    print("=" * 50)
    
    # Run simulation
    sim = build_example_sim()
    sim.run()
    
    print("Testing missing sim parameter:")
    try:
        pl.plot_results(
            results=sim.results,
            age_bins=[0, 15, 50, 200],
            keywords=['prevalence'],
            savefig=False
        )
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")
    
    print("\nTesting missing results parameter:")
    try:
        pl.plot_results(
            sim=sim,
            age_bins=[0, 15, 50, 200],
            keywords=['prevalence'],
            savefig=False
        )
    except ValueError as e:
        print(f"✓ Correctly caught error: {e}")

if __name__ == '__main__':
    print("Age-Stratified Plotting Examples")
    print("=" * 60)
    
    # Run examples
    example_traditional_plotting()
    example_age_stratified_plotting()
    example_custom_age_bins()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nKey features demonstrated:")
    print("- Traditional plotting with flat_results (existing functionality)")
    print("- Age-stratified plotting with results + sim + age_bins (new functionality)")
    print("- Custom age bin definitions")
    print("- Error handling for missing parameters") 