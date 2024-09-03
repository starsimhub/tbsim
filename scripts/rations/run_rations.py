""" 
Run simulation scenarios associated with the RATIONS trial
"""

import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
from scripts.rations.rations import RATIONSTrial
from scripts.rations.plots import plot_rations, plot_epi, plot_hh, plot_nut, plot_active_infections
from tbsim.nutritionenums import eMacroNutrients, eMicroNutrients
import warnings
import os 

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False # NOTE: Debug runs in serial
default_n_rand_seeds = [25, 2][debug]

resdir = cfg.create_res_dir()

# TODO: Move
def compute_rel_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == eMacroNutrients.STANDARD_OR_ABOVE) & (micro == eMicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == eMacroNutrients.SLIGHTLY_BELOW_STANDARD) & (micro == eMicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == eMacroNutrients.MARGINAL)  & (micro == eMicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == eMacroNutrients.UNSATISFACTORY)   & (micro == eMicroNutrients.DEFICIENT)] = 3.0
    return ret


def run_RATIONS(skey, scen, rand_seed=0):
    '''
    Run a single simulation of the RATIONS trial
    '''

    # TODO: Handle scenario

    # Set the global random seed in case any random numbers are used outside of Starsim
    np.random.seed(rand_seed)

    # Create the population with enough people for 1400 index cases and 4724
    # household contacts in the control arm and 1400 index cases and 5621
    # household contacts in the intervention arm.
    # TODO: India-like age structure, although it doesn't really matter (yet)
    pop = ss.People(n_agents = 1400 + 4724 + 1400 + 5621)

    # Create networks
    matnet = ss.MaternalNet() # To track newborn --> household
    householdnet = mtb.HouseholdNet()
    nets = [householdnet, matnet]

    # Create the instance of TB disease
    tb_pars = dict(
        #beta = dict(householdnet=0.03, maternal=0.0),
        beta = dict(householdnet=0.5, maternal=0.0),
        init_prev = 0, # Infections seeded by Rations class
        rate_LS_to_presym = 3e-5,  # Slow down LS-->Presym as this is now the rate for healthy individuals
        rate_LF_to_presym = 6e-3,  # TODO: double check pars
        
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
        ss.Pregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women ### DJK: HouseholdNewborns
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]

    # Create the connector between TB and malnutrition
    cn_pars = dict(
        #rel_LS_prog_func = compute_rel_prog,
        #rel_LF_prog_func = compute_rel_prog,   
        #relsus_microdeficient = 1 # Increased susceptibility of those with micronutrient deficiency (could make more complex function like LS_prog)
    )
    if scen is not None and 'Connector' in scen.keys() and scen['Connector'] is not None:
        cn_pars.update(scen['Connector'])
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------

    # Most of the RATIONS trial is handled by the RATIONSTrial intervention
    rations_pars = dict()
    if scen is not None and 'RATIONS' in scen.keys() and scen['RATIONS'] is not None:
        rations_pars.update(scen['RATIONS'])
    RATIONS_trial = RATIONSTrial(rations_pars)
    intvs = [RATIONS_trial]

    # Create additional interventions, these will likely move elsewhere
    m = mtb.eMicroNutrients
    M = mtb.eMacroNutrients
    b = mtb.eBmiStatus
    #   Table S11: Weight loss in household contacts in RATIONS trial and the association with nutritional status at baseline   
    #intvs.append( mtb.BMIChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.132], from_state=b.SEVERE_THINNESS, to_state=b.MODERATE_THINNESS, 
    #                                       p_new_micro=0.0, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.VITAMIN))
    
    #intvs.append( mtb.BMIChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.168], from_state=b.MODERATE_THINNESS, to_state=b.NORMAL_WEIGHT, 
    #                                       p_new_micro=0.01, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.VITAMIN))
    
    #intvs.append( mtb.BMIChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.132], from_state=b.SEVERE_THINNESS, to_state=b.MODERATE_THINNESS, 
    #                                       p_new_micro=0.0, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.CONTROL))
    
    #intvs.append( mtb.BMIChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.168], from_state=b.MODERATE_THINNESS, to_state=b.NORMAL_WEIGHT, 
    #                                       p_new_micro=0.01, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.CONTROL))
        
    #intvs.append( mtb.MicroNutrientsSupply(year=[2017, 2017.1, 2017.2, 2017.3, 2017.4, 2017.5], rate=[0.2, 0.2,0.2, 0.2,0.2, 0.2]))
    
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
        end = 2030, # Long enough that all pre-symptomatic period end + 2y
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

        copy_inputs=False # No need to create a copy of the inputs
    )
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    sim.run() # Run the sim

    #sim.diseases['tb'].log.line_list.to_csv('linelist.csv')

    # Build a dictionary of results
    ret = {}

    rtr = RATIONS_trial.results
    dat = [
        (rtr.incident_cases_ctrl.cumsum(), 'Incident Cases', 'Control'),
        (rtr.incident_cases_intv.cumsum(), 'Incident Cases', 'Intervention'),
        (rtr.coprevalent_cases_ctrl.cumsum(), 'Co-Prevalent Cases', 'Control'),
        (rtr.coprevalent_cases_intv.cumsum(), 'Co-Prevalent Cases', 'Intervention'),
        (rtr.new_hhs_enrolled_ctrl.cumsum(), 'New HHS Enrolled', 'Control'),
        (rtr.new_hhs_enrolled_intv.cumsum(), 'New HHS Enrolled', 'Intervention'),
    ]

    dfs = []
    for d in dat:
        dfs.append(
            pd.DataFrame({'Values': d[0]}, index=pd.MultiIndex.from_product([sim.results.yearvec, [d[1]], [d[2]]], names=['Year', 'Channel', 'Arm']))
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

    return ret


def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for skey, scen in scens.items():
        for rs in range(n_seeds):
            cfgs.append({'skey':skey, 'scen':scen, 'rand_seed':rs})

    T = sc.tic()
    results += sc.parallelize(run_RATIONS, iterkwargs=cfgs, die=False, serial=debug)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # Aggregate the results
    dfs = {}
    for k in results[0].keys():
        df_list = [r[k] for r in results]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))



    return dfs


if __name__ == '__main__':
    # Define the scenarios
    scens = {
        'Baseline': None,
        'Baseline no link': {
            'Connector': dict(rr_activation_func=mtb.TB_Nutrition_Connector.ones_rr), # All ones
        },
        'Increase transmission before trial enrollment': {
            'TB': None,
            'Malnutrition': None,
            'Connector': None,
            'RATIONS': dict(dur_active_to_dx = ss.weibull(c=2, scale=6/12)),
            'Simulation': None,
        },
    }

    ret = run_scenarios(scens)

    # Create plots
    if 'results' in ret:
        plot_rations(resdir, ret['results'])

    if 'rationsanalyzer' in ret:
        # Incidence
        plot_active_infections(resdir, ret['rationsanalyzer'])

    if 'genhhanalyzer' in ret:
        # Household size distribution  
        plot_hh(resdir, ret['genhhanalyzer'])

    if 'gennutritionanalyzer' in ret:
        # Nutrition
        plot_nut(resdir, ret['gennutritionanalyzer'])

    if 'rationsanalyzer' in ret:
        # Prevalence 
        plot_epi(resdir, ret['rationsanalyzer'])

    print(f'Done, results directory is {resdir}.')
