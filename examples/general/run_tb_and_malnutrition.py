import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

def make_tb_nut():
    # --------- People ----------
    n_agents = 10000
    extra_states = [
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = 0.01, 
        init_prev = 0.25,
        )
    # Initialize
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # -------- Network ---------
    # Network parameters
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
        )
    # Initialize a random network
    net = ss.RandomNet(net_pars)

    # Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # Connector
    cn_pars = dict()
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        start = 1980,
        end = 2020,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    return sim


if __name__ == '__main__':  
    sim_tbn = make_tb_nut()
    sim_tbn.run()
    mtb.plot_sim(sim_tbn)
    plt.show()
