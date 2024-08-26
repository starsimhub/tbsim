""" 
Run simulation scenarios of the RATIONS trial
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

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False # NOTE: Debug runs in serial
default_n_rand_seeds = [20, 2][debug]

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


def run_RATIONS(rand_seed=0):
    '''
    Run a single simulation of the RATIONS trial
    '''

    # Set the global random seed in case any random numbers are used outside of Starsim
    np.random.seed(rand_seed)

    # Create the population with enough people for 1400 index cases and 4724
    # household contacts in the control arm and 1400 index cases and 5621
    # household contacts in the intervention arm
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
    tb = mtb.TB(tb_pars)

    # Create malnutrition --------
    malnutrition_parameters = dict()
    nut = mtb.Malnutrition(malnutrition_parameters)

    # Create demographics
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women ### DJK: HouseholdNewborns
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]

    # Create the connector between TB and malnutrition
    cn_pars = dict(
        rel_LS_prog_func = compute_rel_prog,
        rel_LF_prog_func = compute_rel_prog,   
        relsus_microdeficient = 1 # Increased susceptibility of those with micronutrient deficiency (could make more complex function like LS_prog)
    )
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------

    # Most of the RATIONS trial is handled by the RATIONSTrial intervention
    RATIONS_trial = RATIONSTrial()
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
    ret['results'] = pd.DataFrame({
        'year': sim.results.yearvec,
        'incident_cases_ctrl': RATIONS_trial.results.incident_cases_ctrl,
        'incident_cases_intv': RATIONS_trial.results.incident_cases_intv,
        'new_hhs_enrolled'   : RATIONS_trial.results.new_hhs_enrolled,
        'rand_seed': rand_seed,
    })

    for k, az in sim.analyzers.items():
        df = az.df
        df['rand_seed'] = rand_seed
        ret[k] = df

    print(f'Finishing sim with rand_seed={rand_seed} ')

    return ret


def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    for scen in scens:
        for rs in range(n_seeds):
            cfgs.append({'scen':scen, 'rand_seed':rs})

    T = sc.tic()
    results += sc.parallelize(run_RATIONS, iterkwargs=cfgs, die=False, serial=debug)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # Aggregate the results
    dfs = {}
    for k in results[0].keys():
        df_list = [r[k] for r in results]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))

    # TODO: Move to plotting
    import matplotlib.pyplot as plt
    import seaborn as sns
    import datetime as dt 

    dfr = dfs['results']
    first_year = int(dfr['year'].iloc[0])
    assert dfr['year'].iloc[0] == first_year
    dfr['date'] = pd.to_datetime(365 * (dfr['year']-first_year), unit='D', origin=dt.datetime(year=first_year, month=1, day=1))

    fig, axv = plt.subplots(1,2)

    dfm = dfr.set_index('date').groupby([pd.Grouper(freq='ME')])['new_hhs_enrolled'].sum().to_frame()
    months = ['2019-08-31', '2019-09-30', '2019-10-31', '2019-11-30', '2019-12-31', '2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31', '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31', '2020-11-30', '2020-12-31', '2021-01-31']
    enrolled = [105, 215, 244, 284, 248, 263, 265, 184, 63, 69, 122, 104, 54, 107, 112, 115, 186, 60]
    dfm.loc[months, 'RATIONS'] = enrolled
    dfm = dfm.reset_index().melt(id_vars='date', var_name='Source', value_name='Households Enrolled').replace({'Source':{'new_hhs_enrolled':'Simulation'}})
    sns.barplot(data=dfm.reset_index(), x='date', y='Households Enrolled', hue='Source', ax=axv[0])

    df = dfr[['year', 'rand_seed', 'incident_cases_ctrl', 'incident_cases_intv']] \
        .set_index(['year', 'rand_seed']) \
        .cumsum() \
        .reset_index('year') \
        .reset_index('rand_seed', drop=True) \
        .melt(id_vars='year', var_name='Arm', value_name='Incident Cases') \
        .replace({'Arm': {'incident_cases_ctrl':'Control', 'incident_cases_intv':'Intervention'}})
    sns.lineplot(data=df, x='year', y='Incident Cases', hue='Arm', ax=axv[1])

    import matplotlib.dates as mdates
    locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
    formatter = mdates.ConciseDateFormatter(locator)
    axv[0].xaxis.set_major_locator(locator)
    axv[0].xaxis.set_major_formatter(formatter)

    plt.show()
    return dfs


if __name__ == '__main__':
    ret = run_scenarios()

    # Create plots
    if 'rationsanalyzer' in ret:
        #Incidence
        plot_active_infections(resdir, ret['rationsanalyzer'])

    if 'genhhanalyzer' in ret:
        #Household size distribution  
        plot_hh(resdir, ret['genhhanalyzer'])

    
    if 'gennutritionanalyzer' in ret:
        #Nutrition
        plot_nut(resdir, ret['gennutritionanalyzer'])

    if 'rationsanalyzer' in ret:
        #Prevalence 
        plot_epi(resdir, ret['rationsanalyzer'])

    print(f'Done, results directory is {resdir}.')
