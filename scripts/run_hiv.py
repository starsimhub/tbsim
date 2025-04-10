import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np


def build_hivsim():
    # -------- simulation -------
    sim_pars = dict(
        unit = 'day',
        dt=7,
        start=ss.date('2000-01-01'),
        stop=ss.date('2035-12-31'),
        )
    
    # --------- Disease ----------
    hiv_pars = dict(
        # init_prev=ss.bernoulli(p=0.20),  # Initial prevalence of HIV
        )
    # Create the HIV disease model with the specified parameters
    hiv = mtb.HIV(pars=hiv_pars)

    # --------- People ----------
    n_agents = 1000
    # For demonstration purposes only:
    extra_states = [   # People additional attributes - Cross simulation and diseases
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (e.g. low SES)
        ss.Arr(name="CustomField", dtype=str, default="Any Value"),  # Custom field for each agent
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)
    
    # --------- Network ---------
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    
    # --------- Demographics -----
    births = ss.Births(pars=dict(birth_rate=1.5))
    deaths = ss.Deaths(pars=dict(death_rate=0.08))
    

    # --------- Simulation -------
    sim = ss.Sim(people=pop, 
                 diseases=hiv, 
                 demographics=[deaths,births],
                #  interventions=[inv], 
                 networks=net,
                 pars=sim_pars)
    
    sim.pars.verbose = 7/365
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim = build_hivsim()
    sim.run()
    sim.plot()
    plt.show()
