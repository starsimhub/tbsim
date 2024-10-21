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
        dt=7/365,
        start=1990,
        stop=2020,  # we dont use dur, as duration gets calculated internally.
    )
    sim = ss.Sim(people=pop, diseases=nut, pars=sim_pars)
    return sim


def make_malnutrition_02(agents=100, start=1990, dt=0.5, dur=13.5):  # dur is in years and removes the need for "stop"
    print("Running make_malnutrition_02 with agents=%d, start=%d, dur=%d, dt=%f" % (agents, start, dur, dt))
    return ss.Sim(people=ss.People(n_agents=agents), diseases=mtb.Malnutrition({}), pars=dict(dt=dt, start=start, dur=dur))


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim_n = make_malnutrition()
    sim_n.run()
    sim_n.diseases['malnutrition'].plot()
    plt.show()
    
    # Same concept, simpler parameters
    sim_n = make_malnutrition_02(agents=80000, start=2000, dur=15, dt=0.25)
    sim_n.run()
    sim_n.diseases['malnutrition'].plot()
    plt.show()