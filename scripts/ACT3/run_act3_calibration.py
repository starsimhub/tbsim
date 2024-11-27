"""
Using the calibration uplift propsed by Dan Klien.
NOTE: Until the PR is merged will need to use the claib-continued branch of starsim.
"""

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

debug = False

n_clusters = [20, 10][debug]  # this
n_trials = [500, 2][debug]

# is subject to change later 
do_plot = 1
do_save = 0
n_agents = 1e3

#%% a function to load the data for calibration
def make_data(file_path = os.path.join('scripts', 'ACT3', 
                                       'data', 'calib_data', 
                                       "marks_2019_2022.xlsx"), 
              sheet_name = 'S1'):
    """
    Load and process the data for calibration
    """
    
    # load the data 
    prev_data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # process the data to extract the relevant columns 
    df = (
    prev_data \
    .loc[prev_data['Group'] == 'Xpert MTB positive and Mtb culture positive'] \
    .drop(columns=['Group']) \
    .iloc[:, [0, 1, 2, 3]]
    )
    
    # rename columns for clarity
    df.columns = ['time', 'arm', 'n_elig', 'n_found']
    
    # TODO: year of the trial -- need a better way to assign these values 
    df['time'] = np.array([sc.date('2014-06-01'), 
                           sc.date('2015-06-01'), 
                           sc.date('2016-06-01'),
                           sc.date('2017-06-01'),
                           sc.date('2017-06-01')
                           ])

    return df


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
        people=pop, networks=nets, diseases=tb, demographics=demog, interventions=intv,
        pars=sim_pars, verbose = 0
    )

    return sim


#%% a function to sue the input calibration parameters to modify the simulation
def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_clusters', 1)
    
    # extract the relevant parameters from the simulation object
    # only modifying diesease, intervention and network parameters
    nets_par = sim.pars.networks.pars
    tb_par = sim.pars.diseases.pars

    if calib_pars is not None:
        for key, config in calib_pars.items():
            
            if key == 'rand_seed':
                sim.pars.rand_seed = config
                continue
            
            v = config['value']  

            # TODO: 
            # This needs to like a look-up table rather than the if-else block
            # Maybe some way of specifying the address(?)
            if key == 'beta':
                tb_par.beta = ss.beta(v)
            elif key == 'init_prev':
                tb_par.init_prev = ss.bernoulli(v)
            elif key == 'n_contacts':
                nets_par.n_contacts = ss.poisson(v)
            else:
                raise NotImplementedError(f'Parameter {key} not recognized')
    
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

#%% objective function to calculate the likelihood of the model|data 
# make sim_data 
def make_sim_data(multi_sim, obs_data):
    """
    Make the simulation data from the calibration
    """
    sim_data = []
    for i, s in enumerate(multi_sim.sims):
        if s.label == 'Intervention':
            sim_df = pd.DataFrame({
                'replicate': [i] * 4,
                'time': obs_data['time'],
                'n_elig': s.results.activecasefinding['n_elig'][:4],
                'n_found': s.results.activecasefinding['n_found'][:4]
            })
            sim_data.append(sim_df)

    # print(sim_data[0]['beta'])
    
    return pd.concat(sim_data)

# extract beta-binomial density
def neg_beta_binom_density(row):
    # expected values
    x_e = row['n_found']
    n_e = row['n_elig']
    # observed values
    x_o = row['n_found_obs']
    n_o = row['n_elig_obs']

    return -sps.betabinom.logpmf(k=x_o, n=n_o, a=x_e+1, b=n_e-x_e+1)

# objective function
def objective_fn(sim, **kwargs):
    """
    Objective function to calculate the likelihood of the model given the data.
    This funtion expects the following inputs:
        1. the simulation object runs 
        2. the data to compare the simulation to - provided as **kwargs
    """

    data = kwargs.get('data', None)
    method = kwargs.get('method', None)
    
    # prepare the data for comparison
    data_intv = data \
        .loc[data['arm'] == 'Intervention'] \
        .drop(columns=['arm']) 
    data_intv.columns = ['time', 'n_elig_obs', 'n_found_obs']
    
    # prepare the simulation for comparison
    sim_data = make_sim_data(sim, data_intv)

    # calculate the likelihood of the model given the data
    merged = pd.merge(data_intv, sim_data, on=['time'], how='inner')  
    merged['nloglik'] = merged.apply(neg_beta_binom_density, axis=1)
    # sum over the time points within each replicate
    merged = merged[['replicate', 'nloglik']].groupby('replicate').sum()
    
    return merged['nloglik'].mean() # mean over the replicates


def plot_compare_fit(calib, **kwargs):
    """Makes comparison plots of the calibration sims to the data"""
    obs_data = kwargs.get('obs_data')
    # only extract the intervention data
    intv_data = obs_data[obs_data['arm'] == 'Intervention'].drop(columns=['arm'])
    intv_data.columns = ['time', 'n_elig', 'n_found']

    # pick simulations for comparisons 
    sim_output_before = make_sim_data(calib.before_msim, intv_data)
    sim_output_before['state'] = 'Before'
    sim_output_after = make_sim_data(calib.after_msim, intv_data)
    sim_output_after['state'] = 'After'
    
    # make one data 
    sim_output = pd.concat([sim_output_before, sim_output_after])

    # make a facte plots of binomial 
    g = sns.FacetGrid(data=sim_output, col='time', row = 'state', sharex=False)
    g.map_dataframe(plot_facet, obs_data=intv_data)

    sim_output['prevalence'] = sim_output['n_found']/sim_output['n_elig']*100_000
    sim_output_summary = sim_output.groupby(['time', 'state']).mean().reset_index()

    # compare the prevalence
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(intv_data['time'], sim_output_summary[sim_output_summary['state'] == 'Before']['prevalence'], label='Before calibration')
    ax.plot(intv_data['time'], sim_output_summary[sim_output_summary['state'] == 'After']['prevalence'], label='After calibration')
    ax.plot(intv_data['time'], intv_data['n_found']/intv_data['n_elig']*100_000, 'o', label='Observed')
    ax.set_xlabel('Time')
    ax.set_ylabel('Prevalence per 100,000')
    ax.legend()

    # just trajectories
    calib.after_msim.plot('tb')

    return fig

def plot_facet(data, obs_data, color, **kwargs):
    t = data.iloc[0]['time']
    expected = obs_data.set_index('time').loc[t]
    e_n, e_x = expected['n_elig'], expected['n_found']
    kk = np.arange(int(e_x/2), int(2*e_x))
    for idx, row in data.iterrows():
        alpha = row['n_found'] + 1
        beta = row['n_elig'] - row['n_found'] + 1
        q = sps.betabinom(n=e_n, a=alpha, b=beta)
        yy = q.pmf(kk)
        plt.step(kk, yy, label=f"{row['replicate']}")
        yy = q.pmf(e_x)
        plt.plot(e_x, yy, 'x', ms=10, color='k')
    plt.axvline(e_x, color='k', linestyle='--')

    return plt


#%%a function to run the calibration 
def run_calib(do_plot = False):
    """ Runs the claibration for the ACT3 model """

    # defining calibration paramters 
    # Going to start with just 1
    calib_pars = dict(
        beta = dict(low=0.01, high=0.3, guess=0.01, 
                    suggest_type='suggest_float', log=True), 
        init_prev = dict(low=0.01, high=0.3, guess=0.05, 
                         suggest_type='suggest_float', log=True), 
        n_contacts = dict(low=1, high=10, guess=4, 
                          suggest_type='suggest_int', log=False)                             
    )

    # generate the data for calibration
    data = make_data().reset_index(drop=True)

    # define a simulation object
    sim = make_sim()

    #test = objective_fn(msim, **kwargs)
    
    # make calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,

        build_fn = build_sim,
        build_kw = dict(n_clusters=n_clusters),
        
        eval_fn = objective_fn,
        eval_kw = dict(data=data, method='mcmc'),
        
        total_trials = n_trials,
        n_workers = 10,
        die = True,
        debug = debug,
        )
    
    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Confirm
    sc.printcyan('\nConfirming fit...')
    calib.check_fit()
    
    if do_plot:
        #plot_facet(calib, obs_data=data)
        plot_compare_fit(calib, obs_data=data)
        calib.plot_trend()

    return sim, calib    

# Run as a script
if __name__ == '__main__':

    # Useful for generating fake "real_data"
    if False:
        sim = make_sim()
        pars = {
            'beta'      : dict(value=1)#,
            #'init_prev' : dict(value=0.02),
            #'n_contacts': dict(value=4),
        }
        ms = build_sim(sim, pars)
        ms.run().plot('tb')
        plt.show()

    T = sc.timer()
    do_plot = True

    sim, calib = run_calib(do_plot=do_plot)

    T.toc()

    
    plt.show()
