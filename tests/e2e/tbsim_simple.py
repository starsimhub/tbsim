import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt

# --------- Nutrition ----------
nut_pars = dict(
    beta = 0.05, 
    init_prev = 0.001,
    c = 1, 
    scale = 7
    )
nut = mtb.Nutrition(nut_pars)
sim = ss.Sim(people=ss.People(n_agents=1000), diseases=nut)

sim.run()
sim.plot()
plt.show()




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
    n_contacts=ss.poisson(lam=5)
    )
# Initialize a random network
net = ss.RandomNet(net_pars)


# TODO: Add demographics
dems = mtb.Demographics()


# -------- simulation -------
# define simulation parameters
sim_pars = dict(
    dt = 7/365,
    start = 1990,
    end = 2000,
    )
# initialize the simulation
sim = ss.Sim(people=pop, networks=net, diseases=tb, pars=sim_pars, demographics=dems)

sim.run()
sim.diseases['tb'].plot()
plt.show()

sim.plot()
plt.show()