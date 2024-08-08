import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
from examples.rations.plots import plot_epi, plot_hh, plot_nut, plot_active_infections
import examples.rations.rationsBaseClass as rBC
from tbsim.nutritionenums import eMacroNutrients, eMicroNutrients

import os 


import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True
default_n_rand_seeds = [1000, 10][debug]

def compute_rel_prog(macro, micro):
    
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == eMacroNutrients.STANDARD_OR_ABOVE) & (micro == eMicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == eMacroNutrients.SLIGHTLY_BELOW_STANDARD) & (micro == eMicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == eMacroNutrients.MARGINAL)  & (micro == eMicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == eMacroNutrients.UNSATISFACTORY)   & (micro == eMicroNutrients.DEFICIENT)] = 3.0
    return ret


def run_rations(rand_seed=0):

    np.random.seed(rand_seed)

    # -------------- Rations  ----------
    scenario_parameters = dict(hhdat = pd.DataFrame({
                'size': np.arange(1,6),
                'p': np.array([3, 17, 20, 37, 23]) / 100
                }),
                n_hhs = 2800, # Number of households to generate
                )
    rations = rBC.Rations(scenario_parameters)


    # -------------- People ----------
    pop = rations.people()

    # -------------- Network ---------
    randomnet_parameters = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
    )
    randnet = ss.RandomNet(randomnet_parameters)
    matnet = ss.MaternalNet() # To track newborn --> household
    # matnet = mtb.HouseholdNewborns(pars=dict(fertility_rate=45)) # Why is this not used here?
    householdnet = rations.net()
    nets = [householdnet, randnet, matnet]

    # -------------- TB disease --------
    tb_pars = dict(
        ####beta = dict(rations=0.03, random=0.003, maternal=0.0),
        beta = dict(householdnet=0.03, random=0.0, maternal=0.0),
        init_prev = 0, # Infections seeded by Rations class
        rate_LS_to_presym = 3e-5,  # Slow down LS-->Presym as this is now the rate for healthy individuals
        rate_LF_to_presym = 6e-3,  # TODO: double check pars
        
        # Relative transmissibility by TB state
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    tb = mtb.TB(tb_pars)


    # ---------- Malnutrition --------
    
    malnutrition_parameters = dict()
    nut = mtb.Malnutrition(malnutrition_parameters)

    # Add demographics
    dems = [
        mtb.HouseholdNewborns(pars=dict(fertility_rate=45)), # Per 1,000 women
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]


    # -------- Connector -------
    cn_pars = dict(
        rel_LS_prog_func = compute_rel_prog,
        rel_LF_prog_func = compute_rel_prog,   
        relsus_microdeficient = 1 # Increased susceptibilty of those with micronutrient deficiency (could make more complex function like LS_prog)
    )
    cn = mtb.TB_Nutrition_Connector(cn_pars)


    # -------- Interventions -------
    # Enums:
    m = mtb.eMicroNutrients
    M = mtb.eMacroNutrients
    b = mtb.eBmiStatus
    
    # Interventions array:
    intvs = []    
    #   Table S11: Weight loss in household contacts in RATIONS trial and the association with nutritional status at baseline   
    intvs.append( mtb.BmiChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.132], from_state=b.SEVERE_THINNESS, to_state=b.MODERATE_THINNESS, 
                                           p_new_micro=0.0, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.VITAMIN))
    
    intvs.append( mtb.BmiChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.168], from_state=b.MODERATE_THINNESS, to_state=b.NORMAL_WEIGHT, 
                                           p_new_micro=0.0, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.VITAMIN))
    intvs.append( mtb.MicroNutrientsSupply(year=[2017, 2017.1, 2017.2, 2017.3, 2017.4, 2017.5], rate=[0.2, 0.2,0.2, 0.2,0.2, 0.2]))
    
    intvs.append( mtb.BmiChangeIntervention(year=[2017, 2017.5], rate=[0.0, 0.05], from_state=b.MODERATE_THINNESS, to_state=b.NORMAL_WEIGHT, 
                                           p_new_micro=0.0, new_micro_state=m.NORMAL, arm=mtb.eStudyArm.CONTROL))
    
    
    # -------- Analyzer -------
    azs = [
        mtb.RationsAnalyzer(),
        mtb.GenHHAnalyzer(),
        mtb.GenNutritionAnalyzer(track_years_arr=[2017, 2018], group_by=['Arm', 'Macro', 'Micro'])
    ]


    # -------- Simulation -------
    sim_pars = dict(
        dt = 0.019230769230769232, #7/365,
        start = 2015,
        end = 2023,
        rand_seed = rand_seed,
        )
    sim = ss.Sim(people=pop, 
                 networks=nets, 
                 diseases=[tb, nut], 
                 pars=sim_pars, 
                 demographics=dems, 
                 connectors=cn, 
                 interventions=intvs,
                analyzers=azs
                 )
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    sim.initialize()

    rations.set_states(sim, target_group='all')
    
    seed_uids = rations.choose_seed_infections()
    tb = sim.diseases['tb']
    tb.set_prognoses(seed_uids)

    # After set_prognoses, seed_uids will be in latent slow or fast
    # Change to ACTIVE_PRESYMP and set time of activation to current time step
    tb.state[seed_uids] = mtb.TBS.ACTIVE_PRESYMP
    tb.ti_active[seed_uids] = sim.ti

   #  rations is at least 1 per HH # In Rations, 83% or 84% have SmPos active infections, let's fix that now
    # desired_n_active = 2800 # 0.835 * len(seed_uids)
    cur_n_active = np.count_nonzero(tb.active_tb_state[seed_uids]==mtb.TBS.ACTIVE_SMPOS)
    print(f'Current number of active infections: {cur_n_active}')
    
    
    # Setting TB active states to 2800 indexes of households
    # ------------------------------------------------------
    active_TB_states = [mtb.TBS.ACTIVE_SMPOS, 
                        mtb.TBS.ACTIVE_SMNEG, 
                        mtb.TBS.ACTIVE_EXPTB]
    idx = ss.uids(rations.hhsIndexes)
    random_distribution = np.random.choice(active_TB_states, len(idx))
    print(f'Changing {len(idx)} households to have at least one person with active infection')
    tb.active_tb_state[idx] = random_distribution
    # ------------------------------------------------------
    
    # add_n_active = desired_n_active - cur_n_active
    # non_smpos_uids = seed_uids[tb.active_tb_state[seed_uids]!=mtb.TBS.ACTIVE_SMPOS]
    # p = add_n_active / ( len(seed_uids) - cur_n_active)
    # change_to_smpos = np.random.rand(len(non_smpos_uids)) < p
    # tb.active_tb_state[non_smpos_uids[change_to_smpos]] = mtb.TBS.ACTIVE_SMPOS
    
    sim.run() # Actually run the sim

    df = sim.analyzers['rationsanalyzer'].df
    df['rand_seed'] = rand_seed

    dfhh = sim.analyzers['genhhanalyzer'].df
    dfhh['rand_seed'] = rand_seed

    dfn = sim.analyzers['gennutritionanalyzer'].df
    dfn['rand_seed'] = rand_seed

    print(f'Finishing sim with rand_seed={rand_seed} ')

    return df, dfhh, dfn


def run_sims(n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for rs in range(n_seeds):
        cfgs.append({'rand_seed':rs})
    T = sc.tic()
    results += sc.parallelize(run_rations, iterkwargs=cfgs, die=False, serial=debug)
    epi, hh, nut = zip(*results)
    
    # epi, hh, nut = run_rations(rand_seed=0)
    
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    df_epi = pd.concat(epi)
    df_epi.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"result_{cfg.FILE_POSTFIX}.csv"))

    df_hh = pd.concat(hh)
    df_hh.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"hhsizes_{cfg.FILE_POSTFIX}.csv"))

    df_nut = pd.concat(nut)
    df_nut.to_csv(os.path.join(cfg.RESULTS_DIRECTORY, f"nutrition_{cfg.FILE_POSTFIX}.csv"))

    return df_epi, df_hh, df_nut


if __name__ == '__main__':
    df_epi, df_hh, df_nut = run_sims()

    '''
    if debug:
        sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
        sim.diseases['tb'].plot()
        sim.plot()
        sim.analyzers['Rationsanalyzer'].plot()
        plt.show()
    '''
    
    
    plot_active_infections(df_epi)      #Incidence
    plot_hh(df_hh)                      #Household size distribution  
    plot_nut(df_nut)                    #Nutrition
    plot_epi(df_epi)                    #Prevalence 

    print(f"Results directory {cfg.RESULTS_DIRECTORY}\nThis run: {cfg.FILE_POSTFIX}")
    print('Done')
