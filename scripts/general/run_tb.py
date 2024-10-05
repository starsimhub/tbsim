import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt


def make_tb():
    # --------------- People ----------
    n_agents = 1000
    pop = ss.People(n_agents=n_agents)

    # --------------- TB disease --------
    tb_pars = dict(  # Disease parameters
        beta = 0.001, 
        init_prev = 0.25,
        )
    tb = mtb.TB(tb_pars) # Initialize

    # --------------- Network ---------
    net_pars = dict(    # Network parameters
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
        )
    net = ss.RandomNet(net_pars)  # Initialize a random network

    # --------------- Demographics --------
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # --------------- simulation -------
    sim_pars = dict(    # define simulation parameters
        dt = 7/365,
        start = 1990,
        )
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=sim_pars, demographics=dems)   # initialize the simulation
    sim.pars.verbose = sim.pars.dt / 5      # Print status every 5 years instead of every 10 steps
    return sim

def make_tb_simplified(agents=1000, start=2000, dt=7/365):
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(dict(beta = 0.001, init_prev = 0.25))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=dict(dt = dt, start = start, ), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

if __name__ == '__main__':
   
    sim_tb = make_tb()
    sim_tb.run()
    sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    plt.show()

    sim_tb = make_tb_simplified(agents=1500, start=2000, dt=7/365)
    sim_tb.run()
    sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    plt.show()