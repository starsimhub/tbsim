import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt


def make_tb():
    # --------------- People ----------
    n_agents = 1000
    pop = ss.People(n_agents=n_agents)

    # --------------- TB disease --------
    tb_pars = dict(  # Disease parameters
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(0.25),
    )
    tb = mtb.TB(tb_pars) # Initialize

    # --------------- Network ---------
    net_pars = dict(    # Network parameters
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End connections after each timestep
    )
    net = ss.RandomNet(net_pars)  # Initialize a random network

    # --------------- Demographics --------
    dems = [
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 women 15-49
    ]

    # --------------- simulation -------
    sim_pars = dict(    # define simulation parameters
        dt = 7/365,
        start = 1990,
        stop = 2010,            # Stop after 20 years
    )
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=sim_pars, demographics=dems)   # initialize the simulation
    sim.pars.verbose = sim.pars.dt / 5      # Print status every 5 years instead of every 10 steps
    return sim


if __name__ == '__main__':
    sim = make_tb()
    sim.run()
    sim.diseases['tb'].plot()
    plt.show()
