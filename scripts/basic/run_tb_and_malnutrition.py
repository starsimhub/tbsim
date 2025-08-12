import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

def make_sim(agents=1000, start=1980, stop=2020, dt=7/365):
    pop = ss.People(n_agents=agents, extra_states=[ss.FloatArr('SES', default=ss.bernoulli(p=0.3))])
    tb = mtb.TB({'beta': 0.01, 'init_prev': 0.25})
    nut = mtb.Malnutrition({})
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    dems = [ss.Pregnancy(pars={'fertility_rate': 15}), ss.Deaths(pars={'death_rate': 10})]
    cn = mtb.TB_Nutrition_Connector({})
    sim_pars = {'dt': dt, 'start': start, 'stop' : stop}
    sim = ss.Sim(people=pop, networks=net, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn)
    return sim

if __name__ == '__main__':  
    sim_tbn = make_sim(agents=1500)
    sim_tbn.pars.verbose = -1
    sim_tbn.run()
    sim_tbn.plot()
    plt.show()