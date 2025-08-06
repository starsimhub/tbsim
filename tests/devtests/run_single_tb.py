import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np


def make_tb():
    # --------- People ----------
    
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1000, scale=50)))
    demog = [
        #ss.Pregnancy(pars=dict(fertility_rate=65)),
        ss.Births(pars=dict(birth_rate=17)),
        ss.Deaths(pars=dict(death_rate=14))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=3), dur = 0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta=ss.prob(0.20, ),
        init_prev=0.05,
        rate_LS_to_presym=ss.perday(3e-5),
        rate_LF_to_presym=ss.perday(6e-3),
        rate_active_to_clear=ss.perday(2.4e-4),
        rate_exptb_to_dead=ss.perday(0.15 * 4.5e-4),      
        rate_smpos_to_dead=ss.perday(4.5e-4),             
        rate_smneg_to_dead=ss.perday(0.3 * 4.5e-4), 
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10,
        rel_sus_latentslow = 0.1,
    )

    tb = mtb.TB(tb_pars)

    # for the intervention 
    
    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        dt=ss.days(30),
        start=ss.date('1100-01-01'), stop=ss.date('2018-12-31')
        )

    # build the sim object 
    sim = ss.Sim(
        people=pop, networks=nets, demographics=demog, diseases=tb,
        pars=sim_pars, verbose = 0
    )

    sim.pars.verbose = 1 / 5

    return sim


if __name__ == '__main__':  
    sim_tb = make_tb()
    sim_tb.run()
    sim_tb.plot()
    #sim_tb.plot('demographics')
    plt.show()
