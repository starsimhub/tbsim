import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

def make_malnutrition():
    # --------- Disease ----------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)
    
    # --------- People ----------
    n_agents = 10000
    pop = ss.People(n_agents=n_agents)
    
    # -------- simulation -------
    sim_pars = dict(
        dt = 0.5,
        start = 1990,
        end = 2000,
        )
    sim = ss.Sim(people=pop, diseases=nut, pars=sim_pars)
    return sim

def make_malnutrition_02(agents=10000, start=1990, end=2000, dt=0.5):
    print("Running make_malnutrition_02 with agents=%d, start=%d, end=%d, dt=%f" % (agents, start, end, dt))
    return ss.Sim(people=ss.People(n_agents=agents), diseases=mtb.Malnutrition({}), pars=dict(dt=dt, start=start, end=end))


if __name__ == '__main__':
    # Make Malnutrition simulation
    sim_n = make_malnutrition()
    sim_n.run()
    mtb.plot_sim(sim_n)
    plt.show()
    
    # Same concept, simpler parameters
    sim_n = make_malnutrition_02(agents=80000, start=2000, end=2020, dt=0.25)
    sim_n.run()
    mtb.plot_sim(sim_n)
    plt.show()