#%% Imports and settings
import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import scipy.stats as sps
import seaborn as sns
import os
import matplotlib.pyplot as plt


debug = False # If true, will run in serial
n_reps = [10, 1][debug] # Per trial
total_trials = [250, 10][debug]
n_agents = 1_000
do_plot = 1


#%% Intervention to reduce transmission and progression of the TB disease
class time_varying_parameter(ss.Intervention):
    def __init__(self, pars=None, *args, **kwargs):
        super().__init__()
        self.define_pars(
            tb_parameter = 'beta', # The parameter of the TB module to change
            rc_endpoint = 0.5,     # Will linearly interpolate from 1 at start to rc_endpoint at stop
            start = sc.date('1995-01-01'),
            stop = sc.date('2014-01-01'),
        )
        self.update_pars(pars, **kwargs)
        return
    
    def init_pre(self, sim, **kwargs):
        super().init_pre(sim, **kwargs)

        # Store the original value
        self.original_value = sim.diseases.tb.pars[self.pars.tb_parameter]

        # Make simulation and input time of the same type
        self.input_year = [sc.datetoyear(t) for t in [self.pars.start, self.pars.stop]]
        return
 
    def step(self):
        # Interpolate the values and modify the parameter
        rc = np.interp(self.t.now('year'), self.input_year, [1, self.pars.rc_endpoint])
        self.sim.diseases.tb.pars[self.pars.tb_parameter] = self.original_value * rc
        return

#%% Analyzer to track age specific infections 
class AgeInfect(ss.Analyzer):

    def init_pre(self, sim):
        super().init_pre(sim)
        self.define_results(
            ss.Result('inf_5_6', dtype=int, label='[5,6) y infections'),
            ss.Result('inf_6_15', dtype=int, label='[6,15) infections'),
            ss.Result('inf_15+', dtype=int, label='>=15 infections'),
            ss.Result('pop_5_6', dtype=int, label='[5,6) y alive'),
            ss.Result('pop_6_15', dtype=int, label='[6,15) alive'),
            ss.Result('pop_15+', dtype=int, label='>=15 alive'),
        )
        return

    def step(self):
        ti = self.t.ti
        res = self.results
        infected = self.sim.diseases.tb.infected
        alive = self.sim.people.alive
        age = self.sim.people.age

        res['inf_5_6'][ti]  = np.count_nonzero(infected[(age>=5) & (age<6)])
        res['inf_6_15'][ti] = np.count_nonzero(infected[(age>=6) & (age<15)])
        res['inf_15+'][ti]   = np.count_nonzero(infected[(age>=15)])
        res['pop_5_6'][ti]  = np.count_nonzero(alive[(age>=5) & (age<6)])
        res['pop_6_15'][ti] = np.count_nonzero(alive[(age>=6) & (age<15)])
        res['pop_15+'][ti]   = np.count_nonzero(alive[(age>=15)])
        return

#%% Helper functions
def make_sim():
    """
    Build the simulation object that will simulate the ACT3
    """

    # Random seed is used when deciding the initial n_agents, so set here
    np.random.seed()

    # Retrieve intervention, TB, and simulation-related parameters from scen and skey
    # for TB
    # Create the people, networks, and demographics
    pop = ss.People(n_agents=np.round(np.random.normal(loc=n_agents, scale=50)))
    demog = [ # Note: Better to have Deaths before Births
        ss.Births(pars=dict(birth_rate=109)), 
        ss.Deaths(pars=dict(death_rate=97.88)),
    ]

    nets = ss.RandomNet(n_contacts=ss.poisson(lam=3), dur=0)

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

    # Analyzer to track age specific infections
    ageinfect = AgeInfect()

    # ACT3 intervention 
    act3 = mtb.ActiveCaseFinding(dict(p_treat = ss.bernoulli(p=1.0)))

    # Time varying parameters
    decrease_beta = time_varying_parameter(
        tb_parameter = 'beta', # The parameter of the TB module to change
        rc_endpoint = 0.5,     # Will linearly interpolate from 1 at start to rc_endpoint at stop
        start = sc.date('1995-01-01'),
        stop = sc.date('2014-01-01'),
    )

    # Simulation parameters
    sim_pars = dict(
        # Default simulation parameters
        unit='day', dt=30,
        start=ss.date('1980-01-01'), stop=ss.date('2018-12-31')
    )

    # Build the Sim object 
    sim = ss.Sim(
        people=pop, networks=nets, diseases=tb, demographics=demog, 
        interventions=[decrease_beta, act3], 
        analyzers=ageinfect,
        pars=sim_pars, verbose = 0
    )

    return sim

def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_reps', n_reps)

    sir = sim.pars.diseases # There is only one disease in this simulation and it is a SIR
    net = sim.pars.networks # There is only one network in this simulation and it is a RandomNet

    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if k == 'beta':
            sir.pars.beta = ss.beta(v)
        elif k == 'init_prev':
            sir.pars.init_prev = ss.bernoulli(v)
        elif k == 'n_contacts':
            net.pars.n_contacts = ss.poisson(v)
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    if reps == 1:
        return sim

    ms = ss.MultiSim(sim, iterpars=dict(rand_seed=np.random.randint(0, 1e6, reps)), initialize=True, debug=True, parallel=False) # Run in serial
    return ms


#%% Define the tests
def run_calibration(do_plot=False):
    sc.heading('Testing calibration')

    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.30, guess=0.15, suggest_type='suggest_float', log=True), # Log scale and no "path", will be handled by build_sim (ablve)
        init_prev = dict(low=0.01, high=0.25, guess=0.15, path=('diseases', 'hiv', 'init_prev')), # Default type is suggest_float, no need to re-specify
        n_contacts = dict(low=2, high=10, guess=3, suggest_type='suggest_int'), # Suggest int just for demo
    )

    # Make the sim and data
    sim = make_sim()
    
    # Define the components - for prevalence
    prevalence = ss.BetaBinomial(
        name = 'Number Active',
        weight = 1,
        conform = 'prevalent',

        # Need to feed in the right data  
        expected = pd.DataFrame({
            'x': [240, 169, 136, 78, 53],             # Number of individuals found to be infectious
            'n': [60000, 43425, 44082, 42150, 41680], # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['1995-12-31', '2014-12-31', 
                                                '2015-12-31', '2016-12-31', '2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.tb.new_cases,
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    incidence = ss.GammaPoisson(
        name = 'Incidence Cases',
        weight = 1,
        conform = 'incident',

        # Need to feed in the right data 
        expected = pd.DataFrame({
            'n': [28661, 24705, 28823], 
            'x': [70, 35, 26],      
            't': [ss.date(d) for d in ['2014-01-01', '2015-01-01', '2016-01-01']], # Between t and t1
            't1': [ss.date(d) for d in ['2014-12-31', '2015-12-31', '2016-12-31']],
        }).set_index(['t', 't1']),

        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.tb.new_cases, # Events
            'n': sim.results.n_alive * sim.t.dt, # Person-years at risk
        }, index=pd.Index(sim.results.timevec, name='t'))
    )

    infected_5_6 = ss.BetaBinomial(
        name = 'Number Infected Age 5-6',
        weight = 1,
        conform = 'prevalent',

        # Need to feed in the data 
        expected = pd.DataFrame({
            'x': 23,  # Number of individuals found to be infectious
            'n': 701, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.inf_5_6,
            'n': sim.results.ageinfect.pop_5_6
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    infected_6_15 = ss.BetaBinomial(
        name = 'Number Infected Age 6-15',
        weight = 1,
        conform = 'prevalent',

        # Need to feed in the data 
        expected = pd.DataFrame({
            'x': 32,  # Number of individuals found to be infectious
            'n': 779, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.inf_6_15,
            'n': sim.results.ageinfect.pop_6_15,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    infected_15plus = ss.BetaBinomial(
        name = 'Number Infected 15+',
        weight = 1,
        conform = 'prevalent',

        # Need to feed in the data 
        expected = pd.DataFrame({
            'x': 286,  # Number of individuals found to be infectious
            'n': 1319, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2007-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect['inf_15+'],
            'n': sim.results.ageinfect['pop_15+'],
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim, # Use default builder, Calibration.translate_pars
        reseed = False,
        components = [prevalence, incidence, 
                      infected_5_6, infected_6_15, infected_15plus], #infectious, incidence
        total_trials = total_trials,
        n_workers = None, # None indicates to use all available CPUs
        die = True,
        debug = debug,
    )

    # Perform the calibration
    sc.printcyan('\nPeforming calibration...')
    calib.calibrate()

    # Check
    assert calib.check_fit(), 'Calibration did not improve the fit'

    return sim, calib


#%% Run as a script
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Useful for generating fake "expected" data
    if False:
        sim = make_sim()
        pars = {
            'beta'      : dict(value=0.075),
            'init_prev' : dict(value=0.02),
            'n_contacts': dict(value=4),
        }
        ms = build_sim(sim, pars, n_reps=25)
        ms.run().plot()

        dfs = []
        for sim in ms.sims:
            df = sim.to_df()
            df['prevalence'] = df['sir_n_infected']/df['n_alive']
            df['rand_seed'] = sim.pars.rand_seed
            dfs.append(df)
        df = pd.concat(dfs)

        import seaborn as sns
        sns.relplot(data=df, x='timevec', y='prevalence', hue='rand_seed', kind='line')
        plt.show()

    T = sc.timer()
    do_plot = True

    sim, calib = run_calibration(do_plot=do_plot)

    T.toc()

    if do_plot:
        calib.plot_final()
        calib.plot_optuna(['plot_param_importances', 'plot_optimization_history'])
    plt.show()