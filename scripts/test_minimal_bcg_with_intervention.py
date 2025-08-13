import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np

# Simple test with minimal BCG intervention
def test_bcg_sim():
    """Test simulation with minimal BCG intervention"""
    print("Testing simulation with BCG intervention...")
    
    # Create a simple simulation
    pop = ss.People(n_agents=100, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=dict(
        beta=ss.probpermonth(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        start=ss.date('1990-01-01'),
        stop=ss.date('2000-12-31'),
    ))
    
    # Create a minimal BCG intervention
    bcg = mtb.BCGProtection(pars=dict(
        coverage=0.8,
        start=ss.date('1990-01-01'),
        stop=ss.date('2000-12-31'),
        age_range=[1, 5],
    ))
    
    sim = ss.Sim(
        people=pop,
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        diseases=[tb],
        interventions=[bcg],
        pars=dict(
            dt=ss.days(7),
            start=ss.date('1990-01-01'),
            stop=ss.date('2000-12-31'),
            rand_seed=123,
            verbose=0,
        ),
    )
    
    print("Running simulation with BCG...")
    sim.run()
    print("Simulation with BCG completed successfully!")
    return sim

if __name__ == '__main__':
    test_bcg_sim()
