import tbsim as mtb
import starsim as ss
import matplotlib.pyplot as plt
import numpy as np

'''
This function will create a TB simulation with >>>> DEFAULT RATES <<<< for age stratification
The age stratification is >>>> TURNED ON <<<< and the rates are set for different age groups
'''

def tb_with_age_stratification_defaults(agents=2000, start=2000, stop=2005, dt=7/365, age_off=False):
    
    pop = pop = ss.People(n_agents=agents)
    demog = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=3), dur = 0)

    # *** Here we will use the default age stratification for TB ***
    tb_pars = dict(
        beta=ss.beta(0.02),
        init_prev=0.25,
        age_off=False,      # <<<< TURN ON age stratification - this is the default value, so it is not necessary to set it
        rel_trans_smpos=1.0,
    )
    tb = mtb.TB(tb_pars)

    sim_pars = dict( unit='day', dt=30,
        start=ss.date('2000-01-01'), stop=ss.date('2018-12-31') )

    # build the sim object 
    sim = ss.Sim(people=pop, networks=nets, diseases=tb, pars=dict(dt=dt, start=start, stop=stop))
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 

    return sim

'''
This function will create a TB simulation with >>>> CUSTOM RATES <<<< for age stratification
The rates are illustrative and do not represent real TB rates
The age stratification is >>>> TURNED ON <<<< and the rates are set for different age groups
'''
def tb_with_custom_rates_stratification_ON(agents=2000, start=2000, stop=2005, dt=7/365, age_off=False):
    
    pop = ss.People(n_agents=agents)
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    nets = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Modify the defaults to if necessary based on the input scenario and TURN ON the age stratification
    # Please note, values below are only for illustrative purposes and do not represent real TB rates
    
    tb_pars = dict(
        beta=ss.beta(0.02),
        init_prev=0.25,
        age_off=age_off,       # <<<< TURN ON age stratification
        rate_LS_to_presym =       mtb.RateVec(cutoffs=[0, 18, 30], values=[3e-5, 2.0548e-6, 3e-5, 3e-5], off_value=3e-5),  
        rate_LF_to_presym =       mtb.RateVec(cutoffs=[0, 18, 30], values=[6e-3, 4.5e-3, 6e-3, 6e-3], off_value=6e-3),  
        rate_presym_to_active =   mtb.RateVec(cutoffs=[0, 18, 30], values=[3e-2, 5.48e-3, 3e-2, 3e-2], off_value=3e-2),  
        rate_active_to_clear =    mtb.RateVec(cutoffs=[0, 18, 30], values=[2.4e-4, 2.74e-4, 2.4e-4, 2.4e-4], off_value=2.4e-4),  
        rate_smpos_to_dead =      mtb.RateVec(cutoffs=[0, 18, 30], values=[4.5e-4, 6.85e-4, 4.5e-4, 4.5e-4], off_value=4.5e-4),  
        rate_smneg_to_dead =      mtb.RateVec(cutoffs=[0, 18, 30], values=[1.35e-4, 2.74e-4, 1.35e-4, 1.35e-4], off_value=1.35e-4),  
        rate_exptb_to_dead =      mtb.RateVec(cutoffs=[0, 18, 30], values=[6.75e-5, 2.74e-4, 6.75e-5, 6.75e-5], off_value=6.75e-5),  
        rate_treatment_to_clear = mtb.RateVec(cutoffs=[0, 18, 30], values=[1/60, 1/180, 1/60, 1/60], off_value=1/60), 
        rel_trans_smpos=1.0,
        rel_trans_presymp=0.10
    )
    tb = mtb.TB(tb_pars)
    sim = ss.Sim(people=pop, networks=nets, diseases=tb, pars=dict(dt=dt, start=start, stop=stop), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim

'''
This function will >>>> NOT USE AGE STRATIFICATION <<<<
'''
def tb_default_single_rate(agents=2000, start=2000, stop=2005, dt=7/365, age_off=True):
    pop = ss.People(n_agents=agents)
    dems = [ss.Pregnancy(pars=dict(fertility_rate=15)), ss.Deaths(pars=dict(death_rate=10))]
    nets = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Modify the defaults to if necessary based on the input scenario and TURN ON the age stratification
    # Please note, values below are only for illustrative purposes and do not represent real TB rates
    
    tb_pars = dict(
        beta=ss.beta(0.02),
        init_prev=0.25,
        age_off=age_off,      # <<<< TURN OFF age stratification as the parameter 'age_off' IS SET TO TRUE
        rel_trans_smpos=1.0,
        rel_trans_presymp=0.10
    )
    tb = mtb.TB(tb_pars)
    sim = ss.Sim(people=pop, networks=nets, diseases=tb, pars=dict(dt=dt, start=start, stop=stop), demographics=dems)
    sim.pars.verbose = sim.pars.dt / 5
    return sim



if __name__ == '__main__':  
    tb_ratesbyage_defaults = tb_with_age_stratification_defaults()
    tb_ratesbyage_defaults.run()
    tb_ratesbyage_defaults.plot()
    plt.show()
    
    sim_tb_custom_rates = tb_with_custom_rates_stratification_ON()
    sim_tb_custom_rates.run()
    sim_tb_custom_rates.plot()
    plt.show()

    tb_simple = tb_default_single_rate()
    tb_simple.run()
    tb_simple.plot()
    plt.show()

