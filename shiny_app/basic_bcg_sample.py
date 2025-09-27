import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Basic simulation parameters
SPARS = dict(
    dt=ss.days(7),
    start=ss.date('1975-01-01'),
    stop=ss.date('2030-12-31'),
    rand_seed=123,
    verbose=1,
)

# Basic TB parameters
TBPARS = dict(
    beta=ss.per(0.0025),
    init_prev=ss.bernoulli(p=0.25),
    dt=ss.days(7),      
    start=ss.date('1975-02-01'),
    stop=ss.date('2030-12-31'),
)

# Simple age distribution
age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
})

def create_sample_households(n_agents=500):
    """Create sample household structure."""
    households = []
    current_uid = 0
    while current_uid < n_agents:
        household_size = min(np.random.randint(3, 8), n_agents - current_uid)
        if household_size < 2:
            break
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households

def run_basic_bcg():
    """Run a basic simulation with BCG intervention only."""
    print("Creating population...")
    
    # Create population
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    
    # Create TB disease
    tb = mtb.TB(pars=TBPARS)
    
    # Create networks
    households = create_sample_households(500)
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        mtb.HouseholdNet(hhs=households, pars={'add_newborns': True})
    ]
    
    # Create BCG intervention
    bcg_intervention = mtb.BCGProtection(pars=dict(
        coverage=0.8,  # 80% coverage
        start=ss.date('1980-01-01'),
        stop=ss.date('2020-12-31'),
        age_range=[1, 5],  # Target children aged 1-5
    ))
    
    # Create simulation
    sim = ss.Sim(
        people=pop,
        networks=networks,
        interventions=[bcg_intervention],
        diseases=[tb],
        pars=SPARS,
    )
    
    print("Running simulation...")
    sim.run()
    
    print("Simulation completed!")
    print(f"Final TB prevalence: {sim.results.tb.prevalence[-1]:.3f}")
    print(f"BCG vaccination coverage: {sim.results.bcgprotection.vaccination_coverage[-1]:.3f}")
    print(f"BCG protection coverage: {sim.results.bcgprotection.protection_coverage[-1]:.3f}")
    
    return sim

def plot_results(sim):
    """Create basic plots of the results."""
    import tbsim.utils.plots as pl
    
    # Plot TB prevalence over time
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sim.results.tb.prevalence)
    plt.title('TB Prevalence Over Time')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Prevalence')
    
    plt.subplot(2, 2, 2)
    plt.plot(sim.results.bcgprotection.vaccination_coverage, label='Vaccination Coverage')
    plt.plot(sim.results.bcgprotection.protection_coverage, label='Protection Coverage')
    plt.title('BCG Coverage Over Time')
    plt.xlabel('Time (weeks)')
    plt.ylabel('Coverage')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(sim.results.tb.new_infections)
    plt.title('TB New Infections Over Time')
    plt.xlabel('Time (weeks)')
    plt.ylabel('New Infections')
    
    plt.subplot(2, 2, 4)
    plt.plot(sim.results.tb.new_deaths)
    plt.title('TB Deaths Over Time')
    plt.xlabel('Time (weeks)')
    plt.ylabel('New Deaths')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Run basic BCG simulation
    sim = run_basic_bcg()
    
    # Plot results
    plot_results(sim)
