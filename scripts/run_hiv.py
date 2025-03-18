import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def build_hivsim():
    # --------- Disease ----------
    nut = mtb.HIV()
    
    # --------- People ----------
    n_agents = 200
    pop = ss.People(n_agents=n_agents)
    
    # -------- simulation -------
    sim_pars = dict(
        dt=7/365,
        start=1990,
        stop=2020,  # we dont use dur, as duration gets calculated internally.
    )
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))
    sim = ss.Sim(people=pop, 
                 diseases=nut, 
                 demographics=[deaths, births],
                 networks=net,
                 pars=sim_pars)
    
    sim.pars.verbose = 30/365
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim = build_hivsim()
    sim.run()
    sim.plot()
    plt.show()
