import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
def build_hivsim():
    # -------- simulation -------
    sim_pars = dict(
        unit = 'day',
        dt=7,
        start=sc.date('2013-01-01'),
        stop=sc.date('2020-12-31'),
        rand_seed=123,)
    
    # --------- Disease ----------
    nut = mtb.HIV()
    
    # --------- People ----------
    n_agents = 1000
    extra_states = [
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)
    
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
