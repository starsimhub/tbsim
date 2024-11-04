import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

def make_tb_simplified(sim_pars = dict(unit = 'day', dt=7, 
                                       start=sc.date('2013-01-01'), 
                                       stop=sc.date('2016-12-31'), 
                                       rand_seed=123)):
    
    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(beta = 0.001, init_prev = 0.25))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur = 0))
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=15))]
    sim = ss.Sim(people=pop, networks=net, diseases=tb, demographics=dems, pars=sim_pars)   # Using duration instead of stop.
    
    sim.pars.verbose = sim.pars.dt / 365
    
    return sim

if __name__ == '__main__':
   
    sim_tb = make_tb_simplified()
    sim_tb.run()
    sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    plt.show()