import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
from shared_functions import make_hiv_interventions, make_demographics, plot_results


# Main Simulation Setup
def sim_setup( n_agents=10_000,
            Intvs=True,
            Demgs=False,
            verbose_log=False,
        ) -> ss.Sim:

    sim_pars = dict(
        unit='day',
        dt=7,
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        verbose=verbose_log
    )

    extra_states = [ss.Arr(name="CustomField", dtype=str, default="Any Value"),]
    people = ss.People(n_agents=n_agents, extra_states=extra_states)

    hiv = mtb.HIV(pars=dict(
        init_prev=ss.bernoulli(p=0.30),
        init_onart=ss.bernoulli(p=0.50),
        dt=7,
    ))
    
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    sim = ss.Sim(
        people=people,
        diseases=hiv,
        pars=sim_pars,
        networks=network,
        interventions=make_hiv_interventions(Intvs),    
        demographics=make_demographics(Demgs),      
    )

    return sim

# HIV Basic Run
if __name__ == '__main__':
    args = []
    # args.append(dict(Intvs=False,Demgs=False))
    # args.append(dict(Intvs=False,Demgs=True))
    args.append(dict(Intvs=True,Demgs=True))
    args.append(dict(Intvs=True,Demgs=False))
    results = {}
    for i, arg in enumerate(args):
        print(f"Running scenario: {arg}")
        sim = sim_setup(**arg).run()  
        results[str(arg)] = sim.results.flatten()
        
    plot_results(results, dark=False)