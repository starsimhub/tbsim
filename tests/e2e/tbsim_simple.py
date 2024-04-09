import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

n_agents = 1_000
pop = ss.People(n_agents=n_agents)

tb_pars = dict(
    beta = 0.001, 
    init_prev = 0.25,
)
tb = mtb.TB(tb_pars)

net_pars = dict(
    n_contacts=ss.poisson(lam=5)
)
net = ss.RandomNet(net_pars)

sim_pars = dict(
    dt = 7/365,
    start = 1990,
    end = 2000,
)

# TODO: Add demographics

sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=sim_pars)

sim.run()
sim.diseases['tb'].plot()
plt.show()