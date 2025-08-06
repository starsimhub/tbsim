import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def make_malnutrition():
    # --------- Disease ----------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)
    
    # --------- People ----------
    n_agents = 200
    pop = ss.People(n_agents=n_agents)
    
    # -------- simulation -------
    sim_pars = dict(
        dt=ss.days(7)/365,
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
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim_n = make_malnutrition()
    sim_n.run()
    sim_n.diseases['malnutrition'].plot()
    plt.show()
    