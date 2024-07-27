import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
# from examples.rations.plots import plot_epi, plot_hh, plot_nut, plot_active_infections
from tbsim.functionObjects import compute_rel_prog
import os 


import warnings
warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = True
default_n_rand_seeds = [1000, 1][debug]

def run_rations(rand_seed=0):

    np.random.seed(rand_seed)

    # --------- Rations ----------
    rations = mtb.Rations()

    # --------- People ----------
    pop = rations.people()

    # -------- Network ---------
    rationsnet = rations.net()

    # Network parameters
    randnet_pars = dict(
        n_contacts=ss.poisson(lam=5),
        dur = 0, # End after one timestep
    )
    # Initialize a random network
    randnet = ss.RandomNet(randnet_pars)
    matnet = ss.MaternalNet() # To track newborn --> household
    nets = [rationsnet, randnet, matnet]

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        ####beta = dict(rations=0.03, random=0.003, maternal=0.0),
        beta = dict(rations=0.03, random=0.0, maternal=0.0),
        init_prev = 0, # Infections seeded by Rations class
        rate_LS_to_presym = 3e-5,  # Slow down LS-->Presym as this is now the rate for healthy individuals
        rate_LF_to_presym = 6e-3,  # TODO: double chek pars
        
        # Relative transmissibility by TB state
        rel_trans_smpos     = 1.0,
        rel_trans_smneg     = 0.3,
        rel_trans_exptb     = 0.05,
        rel_trans_presymp   = 0.10,
    )
    tb = mtb.TB(tb_pars)

    # ---------- Malnutrition --------
    nut_pars = dict()
    nut = mtb.Malnutrition(nut_pars)

    # Add demographics
    dems = [
        mtb.HarlemPregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women
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
    vs = mtb.VitaminSupplementation(year=[1942, 1943], rate=[10.0, 3.0]) # Need coverage, V1 vs V2

    m = mtb.MicroNutrients
    M = mtb.MacroNutrients
    # Rates to match Appendix Table 7
    nc0 = mtb.NutritionChange(year=[1942, 1944], rate=[1.25, 0], from_state=M.UNSATISFACTORY, to_state=M.MARGINAL,
                p_new_micro=0.0, new_micro_state=m.NORMAL)
    nc1 = mtb.NutritionChange(year=[1942, 1944], rate=[1.75, 0], from_state=M.MARGINAL, to_state=M.SLIGHTLY_BELOW_STANDARD,
                p_new_micro=0.0, new_micro_state=m.NORMAL)
    nc2 = mtb.NutritionChange(year=[1942, 1944], rate=[1.75, 0], from_state=M.SLIGHTLY_BELOW_STANDARD, to_state=M.STANDARD_OR_ABOVE,
                p_new_micro=0.0, new_micro_state=m.NORMAL)
    intvs = [vs, nc0, nc1, nc2]

    # -------- Analyzer -------
    azs = [
        mtb.HarlemAnalyzer(),
        mtb.HHAnalyzer(),
        mtb.NutritionAnalyzer(),
    ]


    # -------- Simulation -------
    # define simulation parameters
    sim_pars = dict(
        dt = 7/365,
        #start = 1935, # Start early to burn-in
        start = 1941,
        end = 1947,
        rand_seed = rand_seed,
        )
    # initialize the simulation
    sim = ss.Sim(people=pop, 
                 networks=nets, 
                 diseases=[tb, nut], 
                 pars=sim_pars, 
                 demographics=dems, 
                 connectors=cn, 
                 interventions=intvs, 
                 analyzers=azs)
    sim.pars.verbose = sim.pars.dt / 5 # Print status every 5 years instead of every 10 steps

    sim.initialize()

    rations.set_states(sim)
    seed_uids = rations.choose_seed_infections()
    tb = sim.diseases['tb']
    tb.set_prognoses(seed_uids)

    # After set_prognoses, seed_uids will be in latent slow or fast
    # Change to ACTIVE_PRESYMP and set time of activation to current time step
    tb.state[seed_uids] = mtb.TBS.ACTIVE_PRESYMP
    tb.ti_active[seed_uids] = sim.ti

    # In Rations, 83% or 84% have SmPos active infections, let's fix that now
    desired_n_active = 0.835 * len(seed_uids)
    cur_n_active = np.count_nonzero(tb.active_tb_state[seed_uids]==mtb.TBS.ACTIVE_SMPOS)
    add_n_active = desired_n_active - cur_n_active
    non_smpos_uids = seed_uids[tb.active_tb_state[seed_uids]!=mtb.TBS.ACTIVE_SMPOS]
    p = add_n_active / ( len(seed_uids) - cur_n_active)
    change_to_smpos = np.random.rand(len(non_smpos_uids)) < p
    tb.active_tb_state[non_smpos_uids[change_to_smpos]] = mtb.TBS.ACTIVE_SMPOS

    sim.run() # Actually run the sim

    df = sim.analyzers['harlemanalyzer'].df
    df['rand_seed'] = rand_seed

    dfhh = sim.analyzers['hhanalyzer'].df
    dfhh['rand_seed'] = rand_seed

    dfn = sim.analyzers['nutritionanalyzer'].df
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
        sim.analyzers['harlemanalyzer'].plot()
        plt.show()
    '''
    
    
    # plot_active_infections(df_epi)      #Incidence
    # plot_hh(df_hh)                      #Household size distribution  
    # plot_nut(df_nut)                    #Nutrition
    # plot_epi(df_epi)                    #Prevalence 

    print(f"Results directory {cfg.RESULTS_DIRECTORY}\nThis run: {cfg.FILE_POSTFIX}")
    print('Done')
