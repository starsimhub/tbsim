import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

def build_tbsim(sim_pars=None):
    spars = dict(
        dt = ss.days(7), 
        start = ss.date('1940-01-01'),      
        stop = ss.date('2010-12-31'), 
    )
    if sim_pars is not None:
        spars.update(sim_pars)
    
    # Set random seed using numpy instead of rand_seed parameter
    np.random.seed(1)
        
    pop = ss.People(n_agents=1000)
    tb = mtb.TB(dict(
        dt = ss.days(7),
        beta = ss.peryear(0.0025)  # Standardized transmission rate from Abu-Raddad model
    ))
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=20))
    deaths = ss.Deaths(pars=dict(death_rate=15))

    # Updated Sim constructor to use modules parameter for v3.0.1
    sim = ss.Sim(
        people=pop,
        modules=[net, tb, deaths, births],  # All modules in a single list
        pars=spars,
    )

    sim.pars.verbose = 0

    return sim

if __name__ == '__main__':
    sim = build_tbsim()
    sim.run()
    print(sim.pars)
    results = sim.results.flatten()
    results = {'basic': results}
    mtb.plot_combined(results, dark=True, n_cols=3, filter=mtb.FILTERS.important_metrics)

    plt.show()
