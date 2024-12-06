import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

n_reps = 4
debug = False
#%% Intervention to reduce trendmssion and progression of the TB disease
class time_varying_beta(ss.Intervention):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            tb_parameter = 'beta',
            ti = [sc.date('1995-01-01'),sc.date('2014-01-01')],
            rc = [1, 0.5],
        )
        self.update_pars(pars, **kwargs)
        return
    
    def init_pre(self, sim, **kwargs):
        super().init_pre(sim, **kwargs)
        self.orignal_beta = sim.diseases.tb.pars.beta
        return
 
    def step(self):
        # make simulation and input time of the same type
        datetime_timevec =  sc.date(self.t.timevec)
        input_datetime = self.pars.ti

        # an ugly solution to find the closest time to the input in timevec
        t_index = []
        for i,t in enumerate(input_datetime):
            abs_diff = [abs(t - x) for x in datetime_timevec]
            min_index = abs_diff.index(min(abs_diff))
            t_index.append(min_index)
        
        # interpolate the values
        rc = np.interp(self.t.ti, t_index, self.pars.rc)
        self.sim.diseases.tb.pars[self.pars.tb_parameter] = self.orignal_beta * rc
        return


#%% Analyser for to pick out infections in children
class ChildInfections(ss.Analyzer):

    def init_pre(self, sim):
        super().init_pre(sim)
        self.define_results(
            ss.Result('age_5_6', dtype=int, label='[5,6) y'),
            ss.Result('age_6_15', dtype=int, label='[6,15) y'),
            )

    def step(self):
        ti = self.sim.ti
        res = self.sim.results.childinfections
        infected = self.sim.diseases.tb.infected
        age = self.sim.people.age
        
        res['age_5_6'][ti] = infected[(age>=5) & (age<6)].sum()
        res['age_6_15'][ti] = infected[(age>=6) & (age<15)].sum()
        

def make_tb():
    # --------- People ----------
    
    pop = ss.People(n_agents=np.round(np.random.normal(loc=60_000, scale=50)))
    demog = [
        #ss.Pregnancy(pars=dict(fertility_rate=65)),
        ss.Births(pars=dict(birth_rate=109)), #30
        ss.Deaths(pars=dict(death_rate=97.88)) 
        # Note -- 
        # Demographics are very senitive to the changes in the time-varying parameters. 
        # Might require individual consideration to keep demographics stable. 
        # For beta - the birth and death rates are were set to 109 and 97.88 per 1000 people respctively. 
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=3), dur = 0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta=ss.beta(1e-2), #ss.beta(1.0e-3),
        init_prev=0.05,
        rate_LS_to_presym=ss.perday(3e-6), # needs to be 10x slow
        rate_LF_to_presym=ss.perday(6e-3),
        rate_active_to_clear=ss.perday(9.5e-4),#ss.perday(2.4e-4),
        rate_exptb_to_dead=ss.perday(0.15 * 4.5e-4),      
        rate_smpos_to_dead=ss.perday(4.5e-4),             
        rate_smneg_to_dead=ss.perday(0.3 * 4.5e-4), 
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10
    )

    tb = mtb.TB(tb_pars)

    # for the intervention -- start by reducing `beta`
    decrease_beta = time_varying_beta(
        dict(ti = [sc.date('1995-01-01'),sc.date('2014-01-01')],
             rc = [1, 0.55]
             ))
        
    # analyzer to collect the infections in children
    child_infections = ChildInfections()
    
    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        unit='day', dt=30,
        start=ss.date('1850-01-01'), stop=ss.date('2013-12-31') 
        )

    # build the sim object 
    sim = ss.Sim(
        people=pop, networks=nets, demographics=demog, diseases=tb, 
        analyzers=child_infections,
        interventions=[decrease_beta],
        pars=sim_pars, verbose = 0
    )

    # sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 

    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, n_reps)), 
                     initialize=True, debug = debug, parallel=True)

    return ms




if __name__ == '__main__':  
    tic = sc.tic()
    sim_tb = make_tb()
    sim_tb.run()
    toc = sc.toc()
    sim_tb.plot()
    #sim_tb.plot('demographics')
    plt.show()

