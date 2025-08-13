import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np

# Simple test without BCG intervention
def test_basic_sim():
    """Test basic simulation without BCG intervention"""
    print("Testing basic simulation...")
    
    # Create a simple simulation
    pop = ss.People(n_agents=100, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=dict(
        beta=ss.probpermonth(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        start=ss.date('1990-01-01'),
        stop=ss.date('2000-12-31'),
    ))
    
    sim = ss.Sim(
        people=pop,
        networks=ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        diseases=[tb],
        pars=dict(
            dt=ss.days(7),
            start=ss.date('1990-01-01'),
            stop=ss.date('2000-12-31'),
            rand_seed=123,
            verbose=0,
        ),
    )
    
    print("Running simulation...")
    sim.run()
    print("Simulation completed successfully!")
    return sim

if __name__ == '__main__':
    test_basic_sim()
