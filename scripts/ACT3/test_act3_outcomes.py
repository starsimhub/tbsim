#%% Import the required packages and set some calibration defaults 
import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import scipy.stats as sps
import seaborn as sns
import os
import matplotlib.pyplot as plt

debug = True

n_clusters = [20, 10][debug]  # this
#%% Intervention to reduce trendmssion and progression of the TB disease
class time_varying_param(ss.Intervention):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            parameter = 'rate_presym_to_active',
            ti = [0, 30],
            rc = [1, 0.5],
        )
        self.update_pars(pars, **kwargs)
        return
 
    def step(self):
        rc = np.interp(self.t.ti, self.pars.ti, self.pars.rc)
        self.sim.diseases.tb.pars[self.pars.parameter] *= rc
        return
    


#%% Analyser for to pick out infections in children
class ChildInfections(ss.Analyzer):

    def init_pre(self, sim):
        super().init_pre(sim)
        self.define_results(
            ss.Result('age_5_6', dtype=int, label='Children between [5,6) y'),
            ss.Result('age_6_15', dtype=int, label='Children bretween [6,15) y'),
            )

    def step(self):
        ti = self.sim.ti
        res = self.sim.results.childinfections
        infected = self.sim.diseases.tb.infected
        age = self.sim.people.age
        
        res['age_5_6'][ti] = infected[(age>=5) & (age<6)].sum()
        res['age_6_15'][ti] = infected[(age>=6) & (age<15)].sum()
        
#%% a function to build a simulation for calibration
def make_sim():
    """
    Build the simulation object that will simulate the ACT3
    """

    # random seed is used when deciding the initial n_agents, so set here
    np.random.seed()

    # Retrieve intervention, TB, and simulation-related parameters from scen and skey
    # for TB
    # Create the people, networks, and demographics
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1000, scale=50)))
    demog = [
        ss.Deaths(pars=dict(death_rate=10)),
        ss.Pregnancy(pars=dict(fertility_rate=45))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=1), dur = 0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta=ss.beta(0.045),
        init_prev=0.1,
        rate_LS_to_presym=ss.perday(3e-5),
        rate_LF_to_presym=ss.perday(6e-3),
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10
    )

    tb = mtb.TB(tb_pars)

    # analyser for the simulation
    anz = ChildInfections()
    
    # for the intervention 
    intv = mtb.ActiveCaseFinding(dict(p_treat = ss.bernoulli(p=1.0)))

    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        unit='day', dt=14,
        start=ss.date('2000-01-01'), stop=ss.date('2018-12-31')
        )

    # build the sim object 
    sim = ss.Sim(
        people=pop, networks=nets, diseases=tb, demographics=demog, interventions=intv, analyzers=anz,
        pars=sim_pars, verbose = 0
    )

    return sim


#%% a function to sue the input calibration parameters to modify the simulation
def build_sim(sim, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_clusters', 1)
    
    # twice the number of reps are needed sinces there are two arms 
    total_reps = reps*2    
    
    # constrcut the multi-sim object
    ms = ss.MultiSim(
        sim, 
        iterpars=dict(rand_seed=np.random.randint(0, 1e6, total_reps)), 
        initialize=True, debug = True, parallel=False
        ) 

    # change the labels to the individual simulations, 
    # and set the control simulation tretaement probability to 0
    for i, sim in enumerate(ms.sims):
        if i >= total_reps//2:
            sim.label = 'Intervention'
        else:
            sim.label = 'Control'
            sim.pars.interventions.p_treat = ss.bernoulli(p=0.0)
    
    return  ms

def collect_outcomes(ms):
    """ Run the simulation and return the results """
    ms_run = ms.run()
    
# collect the results
# prevalence 

    prev = ms_run.results.tbprev
    inc = ms_run.results.tbinc
    inf = ms_run.results.childinfections

    return prev, inc, inf


# Run as a script
if __name__ == '__main__':

    
    T = sc.timer()
    sim = make_sim()
    msim = build_sim(sim, n_clusters=n_clusters)
    prev, inc, inf = collect_outcomes(msim)

    T.toc()

    plt.show()