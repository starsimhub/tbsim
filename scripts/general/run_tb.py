import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

def make_tb(sim_pars=None):
    spars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('1980-01-01'), 
        stop = sc.date('2000-12-31'), 
        rand_seed = 123,
    )
    if sim_pars is not None:
        spars.update(sim_pars)

    pop = ss.People(n_agents=500)
    tb = mtb.TB(dict(
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        unit = 'day'
    ), validate_dwell_times=True)
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
    tb = sim_tb.diseases['tb']

    # tb.validate_dwell_time_distributions() # This is an optional step
    # These are two of the few plots that have been added directly to the DwellTimeAnalyzer class
    # tb.plot_dwell_time_validation()        
    # tb.dwell_time_analyzer.plot_dwell_time_validation_interactive()

    # Otherwise, use the function to save the dwell time distributions to a file and then plot them

    file_dwt = tb.dwell_time_analyzer.save_to_file()
    mtb.stacked_bars_states_per_agent_clean(file_dwt)
    mtb.plot_dwell_time_lines_for_each_agent(file_dwt)
    # plt.show()
