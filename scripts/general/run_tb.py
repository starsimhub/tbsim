import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np

def make_tb(sim_pars=None):
    spars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('1990-01-01'), 
        stop = sc.date('2016-12-31'), 
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    np.random.seed()
    pop = ss.People(n_agents=500)
    tb = mtb.TB(dict(
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        unit = 'day',
    ))
    dwell_analyzer = mtb.DTAn()

    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    births = ss.Births(pars=dict(birth_rate=5))
    deaths = ss.Deaths(pars=dict(death_rate=5))

    sim = ss.Sim(
        people=pop,
        networks=net,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
        analyzers=dwell_analyzer,
    )

    sim.pars.verbose = sim.pars.dt / 365

    return sim

if __name__ == '__main__':
    sim_tb = make_tb()
    sim_tb.run()
    # sim_tb.diseases['tb'].plot()
    # mtb.plot_sim(sim_tb)
    # plt.show()

    an = sim_tb.analyzers[0]
    an.plot_dwell_time_validation_interactive()
    # an.validate_dwell_time_distributions() # This is an optional step
    # These are two of the few plots that have been added directly to the DwellTimeAnalyzer class
    # an.plot_dwell_time_validation()        
    # an.dwell_time_analyzer.plot_dwell_time_validation_interactive()

    # Otherwise, use the function to save the dwell time distributions to a file and then plot them

    file_dwt = an.file_name
    mtb.sankey(file_dwt)
    mtb.stacked_bars_states_per_agent_clean(file_dwt)
    mtb.plot_dwell_time_lines_for_each_agent(file_dwt)
    mtb.parallel_coordinates(file_dwt)
    mtb.parallel_categories(file_dwt)
    # plt.show()
