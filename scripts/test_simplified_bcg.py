import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import sys
import os

# Add the tbsim directory to the path so we can import the simplified BCG
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'tbsim', 'interventions'))
from bcg_simple import BCGProtectionSimple

# Test with simplified BCG intervention
def test_simplified_bcg():
    """Test simulation with simplified BCG intervention"""
    print("Testing simulation with simplified BCG intervention...")
    
    # Create a simple simulation
    pop = ss.People(n_agents=100, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=dict(
        beta=ss.probpermonth(0.0025),
        init_prev=ss.bernoulli(p=0.25),
        start=ss.date('1990-01-01'),
        stop=ss.date('2000-12-31'),
    ))
    
    # Create the simplified BCG intervention
    bcg = BCGProtectionSimple(pars=dict(
        coverage=0.8,
        start=ss.date('1990-01-01'),
        stop=ss.date('2000-12-31'),
        age_range=[1, 5],
        efficacy=0.8,
        immunity_period=10,
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
    
    print("Running simulation with simplified BCG...")
    sim.run()
    print("Simulation with simplified BCG completed successfully!")
    
    # Print some results
    print(f"Total vaccinated: {len(bcg.vaccinated_uids)}")
    if hasattr(bcg, 'results') and 'n_protected' in bcg.results:
        print(f"Final protected: {bcg.results['n_protected'][-1]}")
    else:
        print("Results not available")
    
    return sim

if __name__ == '__main__':
    test_simplified_bcg()
