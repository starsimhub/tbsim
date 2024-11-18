"""
Using the calibration uplift propsed by Dan Klien.
NOTE: Until the PR is merged will need to use the claib_uplift branch of starsim.
"""


#%% Import the required packages and set some calibration defaults 
import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import os

debug = False

# is subject to change later 
do_plot = 1
do_save = 0
n_agents = 2e3


#%% a function to load the data for calibration
def make_data(file_path = os.path.join('data', 'calib_data', "marks_2019_2022.xlsx"), 
              sheet_name = 'S1', arm=None):
    """
    Load and process the data for calibration
    """
    
    # load the data 
    prev_data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # select the relevant arm 
    if arm is not None:
        prev_data = prev_data.loc[prev_data['Arm'] == arm]
    
    # process the data to extract the relevant columns 
    df = (
    prev_data \
    .loc[prev_data['Group'] == 'Xpert MTB positive and Mtb culture positive'] \
    .drop(columns=['Group']) \
    .iloc[:, [0, 1, 3]]
    )
    # rename columns for clarity
    df.columns = ['time', 'arm', 'prev']
    
    # year of the trial 
    df['time'] = np.array([sc.date('2014-12-31'), 
                           sc.date('2015-12-31'), 
                           sc.date('2016-12-31'),
                           sc.date('2017-12-31')
                           ])

    return df


#%% a function to build a simulation for calibration
def make_sim(rand_seed=0):
    """
    Build the simulation object that will simulate the ACT3
    """

    # random seed is used when deciding the initial n_agents, so set here
    np.random.seed(rand_seed)

    # Retrieve intervention, TB, and simulation-related parameters from scen and skey
    # for TB
    # Create the people, networks, and demographics
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1000, scale=50)))
    demog = [
        ss.Deaths(pars=dict(death_rate=10)),
        ss.Pregnancy(pars=dict(fertility_rate=45))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=5), dur = 0)

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

    # for the intervention 
    
    intv = mtb.ActiveCaseFinding(dict(p_treat = ss.bernoulli(p=1.0)))

    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        unit='day', dt=14,
        start=ss.date('2003-01-01'), stop=ss.date('2017-12-31'),
        rand_seed=rand_seed
        )

    # build the sim object 
    sim = ss.Sim(
        people=pop, networks=nets, diseases=tb, demographics=demog, interventions=intv,
        pars=sim_pars
    )

    return sim


#%% a function to sue the input calibration parameters to modify the simulation
def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    # extract the relevant parameters from the simulation object
    # only modifying diesease, intervention and network parameters
    nets_par = sim.pars.networks
    tb_par = sim.pars.diseases

    for key, value in calib_pars.items():
        
        v = value
        
        if key == 'rand_seed':
            sim.pars.rand_seed = v
            continue
        
        # TODO: 
        # This needs to like a look-up table rather than the if-else block
        # Maybe some way of specifying the address
        if key == 'beta':
            tb_par.beta = ss.beta(v)
        elif key == 'init_prev':
            tb_par.init_prev = ss.bernoulli(v)
        elif key == 'n_contacts':
            nets_par.n_contacts = ss.poisson(v)
        else:
            raise NotImplementedError(f'Parameter {key} not recognized')
        
    return sim

 #%% a fucntion to run the calibration 
def run_calib(do_plot = False):
    """ Runs the claibration for the ACT3 model """

    # defining calibration paramters 
    # Going to start with just 1
    calib_pars = dict(
        beta = dict( low=0.01, high=0.80, guess=0.05, 
                    suggest_type='suggest_float', log=True, path=('diseases', 'tb', 'beta')),
      
    )

    # generate the data for calibration
    data_input = make_data(arm='Intervention').drop(columns='arm')

    # define a simulation object
    sim = make_sim()

    # define a component
    prevalence_comp = ss.CalibComponent(
        name = 'prevalence',
        
        real_data = pd.DataFrame({
            'prev': data_input['prev']
        } index=pd.index([ti for ti in data_input['time']], name ='t')),

       sim_data_fn = lambda sim: pd.DataFrame({
            'prev': sim.results.tb.prevalence,
        }, index=pd.Index(sim.results.timevec, name='t')),

        conform = ss.eConform.PREVALENT,
        likelihood = ss.eLikelihood.POISSON,

        weight = 1
        )
    
    # make calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,

        build_fn = build_sim,
        build_kwargs = None,
        
        components = [prevalence_comp],
        
        total_trioals = 1_000,
        n_workers = None,
        die = True
        debug = debug
        )
    
    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate(confirm_fit=False)

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.confirm_fit()
    print(f'Fit with original pars: {calib.before_fits}')
    print(f'Fit with best-fit pars: {calib.after_fits}')
    if calib.after_fits.mean() <= calib.before_fits.mean():
        print('✓ Calibration improved fit')
    else:
        print('✗ Calibration did not improve fit, but this sometimes happens stochastically and is not necessarily an error')

    if do_plot:
        calib.plot_sims()
        calib.plot_trend()

    return sim, calib    


#%% Run as a script
if __name__ == '__main__':

    # Useful for generating fake "real_data"
    if False:
        sim = make_sim()
        pars = {
            'beta'      : dict(value=0.075),
            'init_prev' : dict(value=0.02),
            'n_contacts': dict(value=4),
        }
        sim = build_sim(sim, pars)
        ms = ss.MultiSim(sim, n_runs=25)
        ms.run().plot()

    T = sc.timer()
    do_plot = True

    sim, calib = test_calibration(do_plot=do_plot)

    T.toc()

    import matplotlib.pyplot as plt
    plt.show()