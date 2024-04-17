import tbsim as mtb
#from .connector import TB_Nutrition_Connector
import starsim as ss
import matplotlib.pyplot as plt

def make_nutrition():
    # --------- Nutrition ----------
    nut_pars = dict(
        init_prev = 0.001,
        )
    nut = mtb.Nutrition(nut_pars)

    n_agents = 10000
    pop = ss.People(n_agents=n_agents)

    sim_pars = dict(
        dt = 0.5,
        start = 1990,
        end = 2000,
        )
    sim = ss.Sim(people=pop, diseases=nut, pars=sim_pars)

    return sim

def make_tb():
    # --------- People ----------
    n_agents = 1000
    pop = ss.People(n_agents=n_agents)

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = 0.001, 
        init_prev = 0.25,
        )
    # Initialize
    tb = mtb.TB(tb_pars)

    # -------- Network ---------
    # Network parameters
    net_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
        )
    # Initialize a random network
    net = ss.RandomNet(net_pars)

    # TODO: Add demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=15)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people
    ]

    # -------- simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        start = 1990,
        end = 2000,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=sim_pars, demographics=dems)

    return sim

def make_tb_nut():
    # --------- People ----------
    n_agents = 10000
    extra_states = [
        ss.State('SES', int, ss.bernoulli(p=0.3)), # ~30% get 0, ~70% get 1
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

    # ---------- Nutrition --------
    nut_pars = dict(
        init_prev = 0.001,
        )
    nut = mtb.Nutrition(nut_pars)

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
    cn_pars = dict(
        rel_LS_prog_risk = 2.0,
    )
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

    return sim


if __name__ == '__main__':
    if False:
        sim_n = make_nutrition()
        sim_n.run()
        sim_n.plot()
        
        sim_tb = make_tb()
        sim_tb.run()
        sim_tb.diseases['tb'].plot()
        sim_tb.plot()

    sim_tbn = make_tb_nut()
    sim_tbn.run()
    sim_tbn.plot()

    plt.show()
