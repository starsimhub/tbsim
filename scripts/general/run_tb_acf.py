import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt



def make_tb_simplified(agents=1000, start=2000, dt=7/365):
    # input modules for the simulation
    pop = ss.People(n_agents=agents)
    tb = mtb.TB(dict(beta = 0.01, init_prev = 0.25))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    intv = mtb.ActiveCaseFinding()
    dems = [ss.Deaths(pars=dict(death_rate=15)), ss.Pregnancy(pars=dict(fertility_rate=15))]
    # define the simulation
    sim = ss.Sim(people=pop, demographics=dems, networks=net, diseases=tb, interventions=[intv], pars=dict(dt = dt, start = start, dur=10))   
    # set the verbose parameter
    sim.pars.verbose = sim.pars.dt / 5
    return sim

if __name__ == '__main__':
   
    sim_tb = make_tb_simplified(agents=1500, start=2000, dt=7/365)
    sim_tb.run()
    sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    plt.show()