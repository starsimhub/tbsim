import starsim as ss
import tbsim as mtb
import numpy as np

def test_burnin():
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1000, scale=50)))
    demog = [
        #ss.Pregnancy(pars=dict(fertility_rate=65)),
        ss.Births(pars=dict(birth_rate=14)),
        ss.Deaths(pars=dict(death_rate=14))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=3), dur = 0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta=ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        rate_LS_to_presym=ss.perday(3e-5),
        rate_LF_to_presym=ss.perday(6e-3),
        rate_exptb_to_dead=ss.perday(0.15 * 4.5e-4),      
        rate_smpos_to_dead=ss.perday(4.5e-4),             
        rate_smneg_to_dead=ss.perday(0.3 * 4.5e-4), 
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10
    )

    tb = mtb.TB(tb_pars)

    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        unit='day', dt=30,
        start=ss.date('1820-01-01'), stop=ss.date('2018-12-31')
    )

    # build the sim object 
    sim = ss.Sim(
        people=pop, networks=nets, demographics=demog, diseases=tb,
        pars=sim_pars, verbose = 0
    )

    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 

    sim.run()

if __name__ == '__main__':
    test_burnin()