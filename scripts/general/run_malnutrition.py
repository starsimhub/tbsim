import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def make_malnutrition():
    # --------- Disease ----------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)
    
    # --------- People ----------
    n_agents = 10000
    pop = ss.People(n_agents=n_agents)
    
    # -------- simulation -------
    sim_pars = dict(
        dt = 7/365,
        start = 1990,
        stop = 2020,  # we dont use dur, as duration gets calculated internally.
    )
    sim = ss.Sim(people=pop, diseases=nut, pars=sim_pars)
    sim.pars.verbose = sim.pars.dt / 5      # Print status every 5 years instead of every 10 steps
    return sim


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim = make_malnutrition()
    sim.run()
    sim.plot() #sim.diseases['malnutrition'].plot()
    sim.plot()
    plt.show()
