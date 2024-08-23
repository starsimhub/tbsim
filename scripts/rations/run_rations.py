""" 
"""

import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
from scripts.rations.rations import RATIONSTrial
from scripts.rations.plots import plot_epi, plot_hh, plot_nut, plot_active_infections
from tbsim.nutritionenums import eMacroNutrients, eMicroNutrients
import warnings
import os 

# ______________________________________________________________
#
#                  SCRIPT TO RUN RATIONS 
#                     1 Scenario only                      
# ______________________________________________________________

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True
default_n_rand_seeds = [1000, 1][debug]

resdir = cfg.create_res_dir()

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

    pop = ss.People(n_agents = 1400 + 4724 + 1400 + 5621)

    # ---------------- Rations Class Instance Creation  -----------------
    matnet = ss.MaternalNet() # To track newborn --> household
    householdnet = ss.Network(name='householdnet') # Placeholder
    nets = [householdnet, matnet]

    # -------------- TB disease --------
    tb_pars = dict(
        beta = dict(householdnet=0.03, maternal=0.0),
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

    # ---------- Demographics --------
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women ### DJK: HouseholdNewborns
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]

    # ----------- Connector ----------
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
    intvs = [RATIONSTrial()]
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
    
    # -------- Analyzer -------
    azs = None #[
        #mtb.RationsAnalyzer(),
        #mtb.GenHHAnalyzer(),
        #mtb.GenNutritionAnalyzer(track_years_arr=[2017, 2018], group_by=['Arm', 'Macro', 'Micro'])
    #]

    # -------- Simulation -------
    sim_pars = dict(
        dt = 7/365,
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
    
    sim.run() # Actually run the sim

    ret = {}
    for k, az in sim.analyzers.items():
        df = az.df
        df['rand_seed'] = rand_seed
        ret[k] = df

    print(f'Finishing sim with rand_seed={rand_seed} ')

    return ret


def run_sims(n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for rs in range(n_seeds):
        cfgs.append({'rand_seed':rs})

    T = sc.tic()
    results += sc.parallelize(run_rations, iterkwargs=cfgs, die=False, serial=debug)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    #epi, hh, nut = zip(*results)
    dfs = {}
    for k in results[0].keys():
        df_list = [r[k] for r in results]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))

    return dfs


if __name__ == '__main__':
    df = run_sims()

    '''
    if debug:
        sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
        sim.diseases['tb'].plot()
        sim.plot()
        sim.analyzers['Rationsanalyzer'].plot()
        plt.show()
    '''
    
    plot_active_infections(resdir, df['rationsanalyzer'])      #Incidence
    plot_hh(resdir, df['genhhanalyzer'])                      #Household size distribution  
    plot_nut(resdir, df['gennutritionanalyzer'])                    #Nutrition
    plot_epi(resdir, df['rationsanalyzer'])                    #Prevalence 

    print(f'Results directory {resdir}.')
    print('Done')
