import tbsim as mtb
import starsim as ss

tb =    mtb.TB()
tb.pars['beta'] = {'mf': [0.0008, 0.0004]}  # Specify transmissibility over the MF network
tb.pars['init_prev'] = 0.05


n_agents = 5_000
sim = ss.Sim(n_agents=n_agents, diseases=[tb])
# sim = ss.Sim(n_agents=n_agents, networks=ss.RandomNet(), diseases=[tb])
sim.run()
sim.plot()
