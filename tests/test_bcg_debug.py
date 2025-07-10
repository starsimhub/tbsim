import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def skip_test_bcg_effectiveness():
    """Test BCG vaccination effectiveness with simple scenario."""
    
    # Simple simulation parameters
    sim_pars = dict(
        unit='day',
        dt=7,
        start=sc.date('1970-01-01'),
        stop=sc.date('1980-12-31'),  # 10 years
        rand_seed=123,
        verbose=0,
    )
    
    # TB parameters with higher transmission for testing
    tb_pars = dict(
        beta=ss.rate_prob(0.01),  # Higher transmission rate
        init_prev=ss.bernoulli(p=0.1),  # 10% initial prevalence
        unit='day',
        dt=7,
        start=sc.date('1970-01-01'),
        stop=sc.date('1980-12-31'),
    )
    
    # Age distribution skewed toward young children
    age_data = pd.DataFrame({
        'age': [0, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50],
        'value': [30, 25, 20, 15, 10, 5, 2, 1, 1, 1, 1, 1]  # Many young children
    })
    
    # BCG intervention with very strong effects for testing
    bcg_pars = dict(
        coverage=1.0,  # 100% coverage
        start=sc.date('1971-01-01'),  # Start after 1 year
        stop=sc.date('1975-12-31'),   # Stop after 5 years
        efficacy=1.0,  # 100% efficacy
        immunity_period=ss.years(5),  # 5 years protection
        age_range=(0, 5),  # Target 0-5 year olds
        # Very strong modifiers for testing
        activation_modifier=ss.uniform(0.01, 0.05),  # 95-99% reduction in activation
        clearance_modifier=ss.uniform(5.0, 10.0),    # 5-10x increase in clearance
        death_modifier=ss.uniform(0.001, 0.01),      # 99-99.9% reduction in death
    )
    
    # Build simulation
    pop = ss.People(n_agents=200, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tb_pars)
    bcg = mtb.BCGProtection(pars=bcg_pars)
    
    networks = [ss.RandomNet({'n_contacts': ss.poisson(lam=3), 'dur': 0})]
    demographics = [ss.Births(pars={'birth_rate': 20}), ss.Deaths(pars={'death_rate': 10})]
    
    sim = ss.Sim(
        people=pop,
        networks=networks,
        interventions=[bcg],
        diseases=[tb],  # <-- Pass as a list
        demographics=demographics,
        pars=sim_pars,
    )
    
    # Run just the first step to check initialization
    print("Running first simulation step...")
    sim.init()
    sim.run_one_step()
    
    # Diagnostics after first step
    print("\n=== DIAGNOSTICS AFTER FIRST STEP ===")
    print(f"Population size: {len(sim.people)}")
    print(f"tb.state length: {len(tb.state.raw)}")
    print(f"tb.rr_activation length: {len(tb.rr_activation.raw)}")
    print(f"tb.rr_death length: {len(tb.rr_death.raw)}")
    print(f"First 10 tb.state: {tb.state.raw[:10]}")
    print(f"First 10 tb.rr_activation: {tb.rr_activation.raw[:10]}")
    print(f"First 10 tb.rr_death: {tb.rr_death.raw[:10]}")

    # Now run the rest of the simulation
    print("Running full BCG test simulation...")
    sim.run()
    # ... rest of the code as before ...

if __name__ == '__main__':
    import pandas as pd
    sim = test_bcg_effectiveness() 