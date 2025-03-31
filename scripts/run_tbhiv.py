import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt

def build_tbhiv_sim(simpars=None, tbpars=None):
    # --------- Simulation ---------
    _simpars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('2013-01-01'),      
        stop = sc.date('2016-12-31'), 
        rand_seed = 123,
    )
    
    # --------- People ----------
    n_agents = 1000
    extra_states = [
        ss.FloatArr('SES', default= ss.bernoulli(p=0.3)), # SES example: ~30% get 0, ~70% get 1 (TODO)
    ]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)

    # ------- TB  --------
    _tbpars = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        # rel_sus_latentslow=0.1,
        unit="day"
    )
    if tbpars is not None:  # Update parameters if provided
        _tbpars.update(tbpars)
    
    tb = mtb.TB(_tbpars)

    # ---------- hiv --------
    hiv = mtb.HIV() #mtb.HIV(hiv_pars)
    
    # --------- Demographics ---------
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=20)), # Per 1,000 people
        ss.Deaths(pars=dict(death_rate=20)), # Per 1,000 people
    ]

    # ----- Networks -----
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    # --------- Connectors ---------
    cn_pars = dict()
    cn = mtb.TB_HIV_Connector(cn_pars)

    # initialize the simulation
    sim = ss.Sim(
                people=pop,  
                diseases=[tb, hiv], 
                pars=_simpars, 
                demographics=dems, 
                connectors=cn
                )
    
    sim.pars.verbose = 7/365 # Print status every 5 years instead of every 10 steps

    return sim

if __name__ == '__main__':  
    sim = build_tbhiv_sim( )
    sim.run()
    sim.plot()
    plt.show()
