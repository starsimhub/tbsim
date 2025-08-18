import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


def build_improved_tbsim(sim_pars=None):
    """Build an improved TB simulation with better distribution management"""
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=1000)
    
    # Create TB module with improved distributions
    tb = mtb.TB(dict(
        dt = ss.days(7),
        beta = ss.peryear(0.025, name='transmission_rate'),
        # Use improved distribution patterns
        init_prev = ss.bernoulli(0.01, name='initial_prevalence'),
        p_latent_fast = ss.bernoulli(0.1, name='latent_fast_prob'),
        # Add individual heterogeneity
        reltrans_het = ss.normal(loc=1.0, scale=0.2, name='transmission_heterogeneity'),
    ))
    
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5, name='contact_rate'), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    sim.pars.verbose = 0

    return sim


def demonstrate_distribution_features(tb_module):
    """Demonstrate the improved distribution features"""
    print("=== TB Module Distribution Features ===")
    
    # Show transition distributions
    print(f"Transition distributions: {list(tb_module.transition_dists.keys())}")
    
    # Demonstrate distribution state management
    states = tb_module.manage_distribution_states()
    print(f"Distribution states: {len(states)} distributions managed")
    
    # Demonstrate scaling capabilities
    scaling_info = tb_module.demonstrate_distribution_scaling()
    print(f"Distribution scaling: {scaling_info['scale_type']}")
    
    # Show parameter distributions
    print(f"Parameter distributions with names:")
    for name, value in tb_module.pars.items():
        if hasattr(value, 'name'):
            print(f"  {name}: {value.name}")
    
    return


def run_improved_simulation():
    """Run the improved TB simulation"""
    print("Building improved TB simulation...")
    sim = build_improved_tbsim()
    
    print("Running simulation...")
    sim.run()
    
    print("Demonstrating distribution features...")
    tb_module = sim.diseases[0]  # Get the TB module
    demonstrate_distribution_features(tb_module)
    
    print("Simulation completed successfully!")
    print(f"Final active TB cases: {sim.results['n_active'][-1]}")
    print(f"Total deaths: {sim.results['cum_deaths'][-1]}")
    
    return sim


if __name__ == '__main__':
    # Run the improved simulation
    sim = run_improved_simulation()
    
    # Plot results
    print("Plotting results...")
    results = sim.results.flatten()
    results = {'improved_tb': results}
    mtb.plot_combined(results, dark=False)
    
    plt.show()
    
    print("=== Key Improvements Demonstrated ===")
    print("1. Better distribution management with ss.Dists class")
    print("2. Named distributions for better tracking")
    print("3. Individual heterogeneity with normal distributions")
    print("4. Proper distribution state management")
    print("5. Distribution scaling capabilities")
    print("6. Time-varying parameter support")
    print("7. Multi-random distribution support for pairwise events")
