import tbsim as mtb
import starsim as ss
import sciris as sc
import numpy as np
import matplotlib.pyplot as plt
import shared_functions as sf


def build_tbhiv_sim(Intvs=True, tb=True, includehiv = True, Demgs= True, simpars = None) -> ss.Sim:
    """Build a TB-HIV simulation with current disease and intervention models."""

    # --- Simulation Parameters ---
    default_simpars = dict(
        unit='day',
        dt=7,
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
    )
    extra_states = [ss.Arr(name="CustomField", dtype=str, default="Any Value"),]
    people = ss.People(n_agents=1000, extra_states=extra_states)
    
    

  
    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
        init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
    )
    hiv = mtb.HIV(pars=hiv_pars)

    # --- Network ---
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))


    # --- Assemble Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=[sf.make_tb(include=tb), hiv],
        # interventions=sf.make_interventions(include=Intvs),
        # demographics=demographics,
        networks=network,
        connectors=sf.make_tb_hiv_connector(include=tb),
        pars=default_simpars,
        verbose=0,
    )
    return sim


if __name__ == '__main__':

    


    args = []
    args.append(dict(Intvs=False,Demgs=False))
    args.append(dict(Intvs=True,Demgs=False))
    
    results = {}
    for i, arg in enumerate(args):
        print(f"Running scenario: {arg}")
        sim = build_tbhiv_sim(**arg).run()  
        results[str(arg)] = sim.results.flatten()
        
    sf.plot_results(results)