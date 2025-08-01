import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

def build_tbsim(sim_pars=None):
    spars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('1940-01-01'),      
        stop = sc.date('2010-12-31'), 
        rand_seed = 1,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(
        unit = 'day',
        dt = 7,
        beta = ss.rate_prob(0.0025, unit='year')
    ))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    sim.pars.verbose = 0

    return sim

if __name__ == '__main__':
    sim = ss.Sim()
    
    sim = build_tbsim()
    sim.run()
    print(sim.pars)
    results = sim.results.flatten()
    results = {'basic': results}
    mtb.plot_combined(results, dark=True, n_cols=3, filter=mtb.FILTERS.important_metrics)

    
    
    plt.show()
