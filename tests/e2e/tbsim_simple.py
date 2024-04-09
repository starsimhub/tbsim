import tbsim as mtb
import starsim as ss

n_agents = 5_000
pop = ss.People(n_agents=n_agents)

tb_pars = dict(
    beta = 0.1,
    init_prev = 0.05
)
tb = mtb.TB(tb_pars)
#tb.pars['beta'] = 0.1 #{'random': [0.0008, 0.0004]}  # Specify transmissibility over the MF network
#tb.pars['init_prev'] = 0.05

net_pars = None
net = ss.RandomNet(net_pars)

sim = ss.Sim(people=pop, networks=net, diseases=tb)
sim.run()
sim.plot()
