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


debug = False
n_reps = [10, 1][debug] # Per trial (and each trial requires 2 simulations - control and intervention)
total_trials = [250, 2][debug]
n_agents = 1_000

date = sc.getdate(dateformat='%Y%b%d-%H%M%S')
# Check if the results directory exists, if not, create it
resdir = os.path.join('results', f'ACT3Calib_{date}')
os.makedirs(resdir, exist_ok=True)

storage = ["mysql://covasim_user@localhost/covasim_db", None][debug]  # Storage for calibrations
n_workers = [40, 1][debug]  # How many cores to use



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
            ss.Result('einf_5_6', dtype=int, label='[5,6) Ever Infected'),
            ss.Result('einf_6_15', dtype=int, label='[6,15) Ever Infected'),
            ss.Result('einf_15+', dtype=int, label='>=15 Ever Infected'),
            ss.Result('pop_5_6', dtype=int, label='[5,6) Alive'),
            ss.Result('pop_6_15', dtype=int, label='[6,15) Alive'),
            ss.Result('pop_15+', dtype=int, label='>=15 Alive'),
        )
        return

    def step(self):
        ti = self.t.ti
        res = self.results
        ever_infected = self.sim.diseases.tb.ever_infected
        alive = self.sim.people.alive
        age = self.sim.people.age

        res['einf_5_6'][ti]  = np.count_nonzero(ever_infected[(age>=5) & (age<6)])
        res['einf_6_15'][ti] = np.count_nonzero(ever_infected[(age>=6) & (age<15)])
        res['einf_15+'][ti]   = np.count_nonzero(ever_infected[(age>=15)])
        res['pop_5_6'][ti]  = np.count_nonzero(alive[(age>=5) & (age<6)])
        res['pop_6_15'][ti] = np.count_nonzero(alive[(age>=6) & (age<15)])
        res['pop_15+'][ti]   = np.count_nonzero(alive[(age>=15)])
        return

#%% Helper functions

def make_people(n):
    # Create the people, networks, and demographics
    age_data = pd.DataFrame({ # Data from WPP, https://population.un.org/wpp/Download/Standard/MostUsed/
        'age': np.arange(0, 101, 5),
        #'value': [3407, 2453, 2376, 2520, 2182, 2045, 1777, 1701, 1465, 1421, 1119, 907, 692, 484, 309, 160, 52, 22, 6, 1, 0], # 1950
        #'value': [8004, 7093, 6048, 5769, 5246, 4049, 2799, 2081, 1998, 2063, 1726, 1551, 1261, 1081, 774, 552, 269, 102, 23, 3, 0], #1980
        'value': [7955, 7388, 6928, 7061, 8657, 8104, 8006, 7005, 6486, 5927, 5495, 4625, 3198, 2090, 1366, 1109, 830, 402, 153, 34, 4], #2015
    })
    pop = ss.People(
        n_agents = n,
        age_data = age_data,
    )
    return pop

def make_sim():
    """
    Build the simulation object that will simulate the ACT3
    """

    # Random seed is used when deciding the initial n_agents, so set here
    np.random.seed()
    n = np.round(np.random.normal(loc=n_agents, scale=50))
    pop = make_people(n)

    demog = [
        # Crude rates for Vietnam in 2015
        ss.Births(birth_rate=ss.peryear(18.5), unit='day', dt=30),  # Matching to simulation's unit and dt, hopefully soon not necessary
        ss.Deaths(death_rate=ss.peryear(6.2), unit='day', dt=30),
    ]

    nets = ss.RandomNet(n_contacts=ss.poisson(lam=3), dur=0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta                  = ss.beta(0.045, unit='year'),
        init_prev             = ss.bernoulli(0.02),
        p_latent_fast         = ss.bernoulli(0.24),
        rate_presym_to_active = ss.peryear(1/0.3), # duration of 0.3 years (exponential mean)
        rate_LS_to_presym     = ss.peryear(3e-6),
        rate_LF_to_presym     = ss.perday(6e-3),
        rel_trans_smpos       = 1.0,
        rel_trans_smneg       = 0.2,
        rel_trans_exptb       = 0.0,
        rel_trans_presymp     = 0.3,
    )
    tb = mtb.TB(tb_pars)

    # Analyzer to track age specific infections
    ageinfect = AgeInfect()

    pcf = mtb.ActiveCaseFinding(
        name = 'Passive Care Seeking',
        p_treat = ss.bernoulli(p=1.0),
        date_cov = {
            sc.date('1994-01-01'): 0.0, # Start of DOTs in Vietnam
            # Coverage values will be multiplied by xpcf, a calibration parameter
            sc.date('2000-01-01'): ss.peryear(0.7), # 2000 reflects completion of DOTS scale-up in Vietnam
            sc.date('2020-01-01'): ss.peryear(1.0), # Coverage continued to scale up through 2020
        },
        interp = True,

        # Sensitivity also reflects care seeking behavior as coverage is agnostic to state
        test_sens = {
            mtb.TBS.ACTIVE_SMPOS: 1,
            mtb.TBS.ACTIVE_PRESYMP: 0.0,
            mtb.TBS.ACTIVE_SMNEG: 0.8,
            mtb.TBS.ACTIVE_EXPTB: 0.1, # Not feeling well, but not obviously TB
        }
    )

    # ACT3 intervention 
    act3 = mtb.ActiveCaseFinding(name='ACT3 Active Case Finding')

    # Time varying parameters
    decrease_beta = time_varying_parameter(
        tb_parameter = 'beta', # The parameter of the TB module to change
        rc_endpoint = 1.0,     # Will linearly interpolate from 1 at start to rc_endpoint at stop
        start = sc.date('1995-01-01'),
        stop = sc.date('2014-01-01'),
    )

    # Simulation parameters
    sim_pars = dict(
        # Default simulation parameters
        unit='day', dt=30,
        #start=ss.date('1800-01-01'),
        #start=ss.date('1900-01-01'),
        start=ss.date('1960-01-01'),
        #start=ss.date('1980-01-01'),
        stop=ss.date('2018-12-31')
    )

    # Build the Sim object 
    sim = ss.Sim(
        people=pop, networks=nets, diseases=tb, demographics=demog, 
        interventions=[decrease_beta, pcf, act3], 
        analyzers=ageinfect,
        pars=sim_pars, verbose = 0
    )

    return sim

def build_sim(sim, calib_pars, **kwargs):
    """ Modify the base simulation by applying calib_pars """

    reps = kwargs.get('n_reps', n_reps)

    sir = sim.pars.diseases # There is only one disease in this simulation and it is a SIR
    net = sim.pars.networks # There is only one network in this simulation and it is a RandomNet
    intv = sim.pars.interventions # There is only one network in this simulation and it is a RandomNet

    for k, pars in calib_pars.items():
        if k == 'rand_seed':
            sim.pars.rand_seed = pars
            continue

        v = pars['value']
        if k == 'beta':
            sir.pars.beta = ss.beta(v, unit='year')
        elif k == 'init_prev':
            sir.pars.init_prev = ss.bernoulli(v)
        elif k == 'n_contacts':
            net.pars.n_contacts = ss.poisson(v)
        elif k == 'beta_change':
            for intv in sim.pars.interventions:
                if isinstance(intv, time_varying_parameter):
                    intv.pars.rc_endpoint = v
        elif k == 'beta_change_year':
            for intv in sim.pars.interventions:
                if isinstance(intv, time_varying_parameter):
                    intv.pars.start = sc.date(f'{v}-01-01')
        elif k == 'xpcf':
            for intv in sim.pars.interventions:
                if intv.name == 'Passive Care Seeking':
                    for cov in intv.pars.date_cov.values():
                        cov *= v
        else:
            raise NotImplementedError(f'Parameter {k} not recognized')

    sims = []
    for seed in np.arange(sim.pars.rand_seed, sim.pars.rand_seed+reps): #np.random.randint(0, 1e6, reps):
        sim_intv = sim.copy()

        np.random.seed(seed) # Used for initial pop size
        n = np.round(np.random.normal(loc=n_agents, scale=50))
        pop = make_people(n)
        sim_intv.pars.people = pop

        sim_intv.pars.rand_seed = seed
        sim_intv.label = 'Intervention'
        sims.append(sim_intv)

        sim_ctrl = sim_intv.copy()
        sim_ctrl.label = 'Control'
        for intv in sim_ctrl.pars.interventions:
            if intv.name == 'ACT3 Active Case Finding':
                #intv.pars.p_treat = ss.bernoulli(p=0.0)
                for year, cov in intv.pars.date_cov.items():
                    if year < 2017: # Year as float
                        intv.pars.date_cov[year] = 0.0 # In control arm, no ACF until 2017
        sims.append(sim_ctrl)

    ms = ss.MultiSim(sims, initialize=True, debug=True, parallel=False)
    return ms


#%% Define the tests
def make_calibration():
    # Define the calibration parameters
    calib_pars = dict(
        beta = dict(low=0.01, high=0.70, guess=0.15, suggest_type='suggest_float', log=False), # Log scale and no "path", will be handled by build_sim (above)
        #init_prev = dict(low=0.01, high=0.25, guess=0.15), # Default type is suggest_float, no need to re-specify
        #n_contacts = dict(low=2, high=10, guess=3),
        beta_change = dict(low=0.25, high=1, guess=0.5),
        beta_change_year = dict(low=1950, high=2014, guess=2000, suggest_type='suggest_int'),
        xpcf = dict(low=0, high=1.0, guess=0.1),
    )

    # Make the sim and data
    sim = make_sim()
    
    # Define the components - for prevalence
    prevalence_intv = ss.BetaBinomial(
        name = 'Prevalence Active (Intervention)',
        include_fn = lambda sim: sim.label == 'Intervention',
        weight = 25,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': [360, 169, 136, 78, 53],             # Number of individuals found to be infectious
            'n': [60000, 43425, 44082, 42150, 42150], # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['1995-12-31', '2014-12-31', 
                                                '2015-12-31', '2016-12-31', '2017-12-31']], name='t')), # On these dates

        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.tb.n_active,
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    prevalence_ctrl = ss.BetaBinomial(
        name = 'Prevalence Active (Control)',
        include_fn = lambda sim: sim.label == 'Control',
        weight = 25,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': [360, 94],      # Number of individuals found to be infectious
            'n': [60000, 41680], # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['1995-12-31', '2017-12-31']], name='t')), # On these dates

        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.tb.n_active,
            'n': sim.results.n_alive,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    incidence = ss.GammaPoisson(
        name = 'Incidence Cases (Intervention)',
        include_fn = lambda sim: sim.label == 'Intervention',
        weight = 1,
        conform = 'incident',

        expected = pd.DataFrame({
            'n': [28661, 24705, 28823], 
            'x': [70, 35, 26],
            't': [ss.date(d) for d in ['2015-01-01', '2016-01-01', '2017-01-01']], # Between t and t1
            't1': [ss.date(d) for d in ['2015-12-31', '2016-12-31', '2017-12-31']],
        }).set_index(['t', 't1']),

        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.tb.new_active, # Events
            'n': sim.results.n_alive * sim.t.dt_year, # Person-years at risk
        }, index=pd.Index(sim.results.timevec, name='t'))
    )

    infected_5_6_intv = ss.BetaBinomial(
        name = 'Prev Ever Infected Age 5-6 (Intervention)',
        include_fn = lambda sim: sim.label == 'Intervention',
        weight = 1,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': 23,  # Number of individuals found to be infectious
            'n': 701, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.einf_5_6,
            'n': sim.results.ageinfect.pop_5_6
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    infected_5_6_ctrl = ss.BetaBinomial(
        name = 'Prev Ever Infected Age 5-6 (Control)',
        include_fn = lambda sim: sim.label == 'Control',
        weight = 1,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': 18,  # Number of individuals found to be infectious
            'n': 705, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.einf_5_6,
            'n': sim.results.ageinfect.pop_5_6
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    infected_6_15_intv = ss.BetaBinomial(
        name = 'Prev Ever Infected Age 6-15 (Intervention)',
        include_fn = lambda sim: sim.label == 'Intervention',
        weight = 1,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': 32,  # Number of individuals found to be infectious
            'n': 779, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.einf_6_15,
            'n': sim.results.ageinfect.pop_6_15,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    infected_6_15_ctrl = ss.BetaBinomial(
        name = 'Prev Ever Infected Age 6-15 (Control)',
        include_fn = lambda sim: sim.label == 'Control',
        weight = 1,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': 64,  # Number of individuals found to be infectious
            'n': 769, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2017-12-31']], name='t')), # On these dates
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect.einf_6_15,
            'n': sim.results.ageinfect.pop_6_15,
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    # TODO: Which arm?
    infected_15plus = ss.BetaBinomial(
        name = 'Prev Ever Infected 15+ (Intervention)',
        include_fn = lambda sim: sim.label == 'Intervention',
        weight = 1,
        conform = 'prevalent',

        expected = pd.DataFrame({
            'x': 286,  # Number of individuals found to be infectious
            'n': 1319, # Number of individuals sampled
        }, index=pd.Index([ss.date(d) for d in ['2016-01-01']], name='t')), # June 2015 to March 2016.
        
        extract_fn = lambda sim: pd.DataFrame({
            'x': sim.results.ageinfect['einf_15+'],
            'n': sim.results.ageinfect['pop_15+'],
        }, index=pd.Index(sim.results.timevec, name='t')),
    )

    # Make the calibration
    calib = ss.Calibration(
        calib_pars = calib_pars,
        sim = sim,
        build_fn = build_sim, # Use default builder, Calibration.translate_pars
        reseed = True,
        components = [prevalence_intv, prevalence_ctrl, incidence, 
                      infected_5_6_intv, infected_5_6_ctrl,
                      infected_6_15_intv, infected_6_15_ctrl,
                      infected_15plus ],
        total_trials = total_trials,
        db_name = f'{resdir}/calibration.db',
        keep_db = True,
        n_workers = n_workers, #None, # None indicates to use all available CPUs
        storage = storage,
        die = True,
        debug = debug,
    )

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
        import sys
        sys.exit()

    sim, calib = make_calibration()

    T = sc.timer()
    try:
        # Perform the calibration
        sc.printcyan('\nPeforming calibration...')
        calib.calibrate()
    except KeyboardInterrupt:
        print("Calibration interrupted by user, plotting final results")
    T.toc()

    # Check fit and make plots
    calib.check_fit(do_plot=False)
    figs = calib.plot()

    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(resdir, f'Component_{i}.png'), dpi=300)

    plots = ['param_importances', 'optimization_history', 'parallel_coordinate', 'contour']
    figs = calib.plot_optuna([f'plot_{lbl}' for lbl in plots])
    for fig, lbl in zip(figs, plots):
        try:
            if isinstance(fig, (list, np.ndarray)): # List of axes
                fig = fig.flatten()[0].get_figure()
            elif isinstance(fig, plt.Axes): # Single axis
                fig = fig.get_figure()
            fig.tight_layout()
            fig.set_size_inches(8, 8)
            fig.savefig(os.path.join(resdir, f'{lbl}.png'), dpi=300)
        except:
            print(f"Failed to save {lbl}.png")

    #fig = calib.plot_final()
    #fig.set_size_inches(24, 16)
    #fig.savefig(os.path.join(resdir, 'calibration.png'), dpi=300)

    dfs = []
    for sim in calib.before_msim.sims + calib.after_msim.sims:
        df = sim.to_df()
        df['seed'] = sim.pars.rand_seed
        df['arm'] = sim.label
        df['calibrated'] = 'After Calibration' if sim in calib.after_msim.sims else 'Before Calibration'
        dfs.append(df)

    df = pd.concat(dfs)

    df.to_csv(os.path.join(resdir, 'results.csv'))
    ret = df.melt(id_vars=['timevec', 'arm', 'calibrated', 'seed'], value_name='value', var_name='variable')

    g = sns.relplot(data=ret, x='timevec', y='value', hue='arm', col='variable', kind='line', style='calibrated', style_order=['Before Calibration'], col_wrap=6, errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('Before Calibration')
    g.fig.savefig(os.path.join(resdir, 'sim_before.png'), dpi=600)

    g = sns.relplot(data=ret, x='timevec', y='value', hue='arm', col='variable', kind='line', style='calibrated', style_order=['After Calibration'], col_wrap=6, errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.95)
    g.fig.suptitle('After Calibration')
    g.fig.savefig(os.path.join(resdir, 'sim_after.png'), dpi=600)

    g = sns.relplot(data=ret, x='timevec', y='value', hue='arm', col='variable', kind='line', style='calibrated', col_wrap=6, errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'sim_both.png'), dpi=600)

    plt.show()
