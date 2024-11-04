import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

def make_tb(sim_pars=None):
    spars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('2013-01-01'), 
        stop = sc.date('2016-12-31'), 
        rand_seed = 123,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        unit = 'day'
    ))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=15))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    sim.pars.verbose = sim.pars.dt / 365

    return sim

if __name__ == '__main__':
    sim_tb = make_tb()
    sim_tb.run()
    sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    plt.show()
