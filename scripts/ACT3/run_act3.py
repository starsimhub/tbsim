import starsim as ss
import tbsim as tb
import numpy as np

debug = False # NOTE: Debug runs in serial
default_n_rand_seeds = [60, 1][debug]


def build_RATIONS(skey, scen, rand_seed=0):
    '''
    Run a single simulation of the RATIONS trial
    '''

    # Set the global random seed in case any random numbers are used outside of Starsim
    np.random.seed(rand_seed)

    # Create the population with enough people for 1400 index cases and 4724
    # household contacts in the control arm and 1400 index cases and 5621
    # household contacts in the intervention arm.
    # TODO: India-like age structure, although it doesn't really matter (yet)
    pop = ss.People(n_agents = 1400 + 4724 + 1400 + 5621)

    # Create networks
    nets = mtb.HouseholdNet(add_newborns=False) # Optionally add newborns to households here

    # Create the instance of TB disease
    tb_pars = dict(
        beta = ss.beta(0.045),
        init_prev = 0, # Infections seeded by Rations class
        rate_LS_to_presym = ss.perday(3e-5),
        rate_LF_to_presym = ss.perday(6e-3),
        
        # Relative transmissibility by TB state
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    if scen is not None and 'TB' in scen.keys() and scen['TB'] is not None:
        tb_pars.update(scen['TB'])
    tb = mtb.TB(tb_pars)

    # Create malnutrition --------
    malnutrition_parameters = dict()
    if scen is not None and 'Malnutrition' in scen.keys() and scen['Malnutrition'] is not None:
        malnutrition_parameters.update(scen['Malnutrition'])
    nut = mtb.Malnutrition(malnutrition_parameters)

    # Create demographics
    dems = [
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
        ss.Pregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women
    ]

    # Create the connector between TB and malnutrition
    cn_pars = {}
    if scen is not None and 'Connector' in scen.keys() and scen['Connector'] is not None:
        cn_pars.update(scen['Connector'])
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # Most of the RATIONS trial is handled by the RATIONSTrial intervention
    rations_pars = dict()
    if scen is not None and 'RATIONS' in scen.keys() and scen['RATIONS'] is not None:
        rations_pars.update(scen['RATIONS'])
    RATIONS_trial = RATIONSTrial(rations_pars)
    intvs = [RATIONS_trial]

    # Create analyzers
    azs = None #[
        #mtb.RationsAnalyzer(),
        #mtb.GenHHAnalyzer(),
        #mtb.GenNutritionAnalyzer(track_years_arr=[2017, 2018], group_by=['Arm', 'Macro', 'Micro'])
    #]

    # Create the simulation parameters and simulation
    sim_pars = dict(
        dt = 7/365,
        start = 2019, # Dates don't matter
        stop = 2030, # Long enough that all pre-symptomatic period end + 2y
        rand_seed = rand_seed,
    )
    if scen is not None and 'Simulation' in scen.keys() and scen['Simulation'] is not None:
        sim_pars.update(scen['Simulation'])

    sim = ss.Sim(people=pop, 
        networks=nets, 
        diseases=[tb, nut], 
        pars=sim_pars, 
        demographics=dems, 
        connectors=cn, 
        interventions=intvs,
        analyzers=azs,
    )
    sim.pars.verbose = [0, sim.pars.dt / 5][debug] # Print status every 5 years instead of every 10 steps

    return sim


def run_ACF(skey, scen, rand_seed=0):

    sim = build_ACF(skey, scen, rand_seed)
    sim.run() # Run the sim

    # Build a dictionary of results
    ret = {}
    act3 = sim.analyzers['act3'].results
    '''
    dat = [
        (rtr.incident_cases_ctrl.cumsum(), 'Incident Cases', 'Control'),
        (rtr.incident_cases_intv.cumsum(), 'Incident Cases', 'Intervention'),
        (rtr.coprevalent_cases_ctrl.cumsum(), 'Co-Prevalent Cases', 'Control'),
        (rtr.coprevalent_cases_intv.cumsum(), 'Co-Prevalent Cases', 'Intervention'),
        (rtr.new_hhs_enrolled_ctrl.cumsum(), 'HHS Enrolled', 'Control'),
        (rtr.new_hhs_enrolled_intv.cumsum(), 'HHS Enrolled', 'Intervention'),
        (rtr.person_years_ctrl.cumsum(), 'Person Years', 'Control'),
        (rtr.person_years_intv.cumsum(), 'Person Years', 'Intervention'),
    ]

    dfs = []
    for d in dat:
        dfs.append(
            pd.DataFrame({'Values': d[0]}, index=pd.MultiIndex.from_product([sim.results.timevec, [d[1]], [d[2]]], names=['Year', 'Channel', 'Arm']))
        )
    df = pd.concat(dfs)
    df['Seed'] = rand_seed
    df['Scenario'] = skey
    ret['results'] = df.reset_index()

    for k, az in sim.analyzers.items():
        df = az.df
        df['Seed'] = rand_seed
        df['Scenario'] = skey
        ret[k] = df

    print(f'Finishing sim with "{skey}" seed {rand_seed} ')
    '''

    return ret



def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for skey, scen in scens.items():
        for rs in range(n_seeds):
            scen['Simulation']['n_agents'] = np.random.normal(loc=1200, scale=50)
            cfgs.append({'skey':skey, 'scen':scen, 'rand_seed':rs})

    T = sc.tic()
    results += sc.parallelize(run_ACF, iterkwargs=cfgs, die=True, serial=debug)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # Aggregate the results
    dfs = {}
    for k in results[0].keys():
        df_list = [r[k] for r in results]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))

    return dfs

if __name__ == '__main__':

    scens = {
        'Control': {
            'ACT3': None,
            'TB': None,
            'Simulation': None,
        },
        'Basic ACT3': {
            'ACT3': tbsim.ActiveCaseFinding(),
            'TB': None,
            'Simulation': None,
        }
    }

    ret = run_scenarios(scens)