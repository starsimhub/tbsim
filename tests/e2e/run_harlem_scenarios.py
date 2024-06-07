import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import tbsim.config as cfg
import os 
import warnings

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

debug = False
default_n_rand_seeds = [1000, 2][debug]
cache_from = [None, '06-07_11-35-10'][0] # Run sims if None, plot from dir if datestr provided
scen_filter = None # Put a list of scenarios here to restrict, e.g. ['Base']

def compute_rel_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE)         & (micro == mtb.MicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD)   & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == mtb.MacroNutrients.MARGINAL)                  & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY)            & (micro == mtb.MicroNutrients.DEFICIENT)] = 3.0
    return ret

def compute_rel_prog_alternate(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE)         & (micro == mtb.MicroNutrients.DEFICIENT)] = 2
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD)   & (micro == mtb.MicroNutrients.DEFICIENT)] = 5
    ret[(macro == mtb.MacroNutrients.MARGINAL)                  & (micro == mtb.MicroNutrients.DEFICIENT)] = 10
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY)            & (micro == mtb.MicroNutrients.DEFICIENT)] = 20
    return ret

def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == mtb.MacroNutrients.UNSATISFACTORY)] = 0

    return p

def p_micro_recovery_alt(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    return p

# All scens will get these, which can be overwritten
scen_default = {
    'relsus_microdeficient': 1,
    'n_hhs': 194/2,
}

def run_scen(scen, filter):
    if filter is None:
        return True
    return scen in filter

scenarios = {
    'Base': {
        'beta': 0.24,
        'active': run_scen('Base', scen_filter),
    },

    'MoreMicroDeficient': {
        'beta': 0.20,
        'p_microdeficient_given_macro': {
            mtb.MacroNutrients.UNSATISFACTORY: 1.0,
            mtb.MacroNutrients.MARGINAL: 1.0,
            mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD: 0.75,
            mtb.MacroNutrients.STANDARD_OR_ABOVE: 0.5,
        },
        'active': run_scen('MoreMicroDeficient', scen_filter),
    },

    'LatentSeeding': {
        'beta': 0.16,
        'init_prev': 0.33,
        'active': run_scen('LatentSeeding', scen_filter),
    },

    'RelSus': {
        'relsus_microdeficient': 5,
        'beta': 0.08,
        'active': run_scen('RelSus', scen_filter),
    },

    'LSProgAlt': {
        'beta': 0.16,
        'rel_LS_prog_func': compute_rel_prog_alternate,
        'active': run_scen('LSProgAlt', scen_filter),
    },

    'LatentFast': {
        'beta': 0.16,
        'rel_LF_prog_func': compute_rel_prog,
        'active': run_scen('LatentFast', scen_filter),
    },

    'FastSlowAlt': {
        'beta': 0.16,
        'rel_LF_prog_func': compute_rel_prog_alternate,
        'rel_LS_prog_func': compute_rel_prog_alternate,
        'active': run_scen('LatentFast', scen_filter),
    },

    'AllVitamin': {
        'beta': 0.24,
        'p_micro_recovery_func': p_micro_recovery_alt,
        'active': run_scen('AllVitamin', scen_filter),
    },

    'NoSecular': {
        'beta': 0.22,
        'secular_trend': False,
        'active': False, # Disable
    },

    'SecularMicro': {
        'beta': 0.26,
        'p_new_micro': 0.5,
        'active': False, # Disable
    },
}

# Create matching CONTROL and VITAMIN arms for each of the above scenarios
scens = {}
for skey, scn in scenarios.items():
    if 'active' in scn and not scn['active']:
        continue
    control = scen_default.copy() | scn.copy()
    control['p_control'] = 1
    control['vitamin_year_rate'] = None
    control['skey'] = skey
    control['arm'] = 'CONTROL'
    scens[f'{skey} CONTROL'] = control

    vitamin = control.copy()
    vitamin['p_control'] = 0
    vitamin['vitamin_year_rate'] = [(1942, 10.0), (1943, 3.0)]
    vitamin['arm'] = 'VITAMIN'
    vitamin['ref'] = f'{skey} CONTROL'
    scens[f'{skey} VITAMIN'] = vitamin

def run_harlem(scen, rand_seed=0, idx=0, n_hhs=194, p_control=0.5,
               vitamin_year_rate=None, relsus_microdeficient=1, beta=0.1,
               secular_trend=True, p_new_micro=0.0,
               rel_LS_prog_func=compute_rel_prog, rel_LF_prog_func=mtb.TB_Nutrition_Connector.compute_rel_LF_prog,
               init_prev=0.0, p_microdeficient_given_macro=None,
               p_micro_recovery_func=p_micro_recovery_default, **kwargs):
    # vitamin_year_rate is a list of tuples like [(1942, 10.0), (1943, 3.0)] or None if CONTROL
    lbl = f'sim {idx}: {scen} with rand_seed={rand_seed}, p_control={p_control}, vitamin_year_rate={vitamin_year_rate}'
    print(f'Starting {lbl}')

    np.random.seed(rand_seed)

    # --------- Harlem ----------
    harlem_pars = dict(n_hhs=n_hhs, p_control=p_control)
    if p_microdeficient_given_macro is not None:
        harlem_pars['p_microdeficient_given_macro'] = p_microdeficient_given_macro
    harlem = mtb.Harlem(harlem_pars)

    # --------- People ----------
    pop = harlem.people()

    # -------- Network ---------
    harlemnet = harlem.net()

    # Network parameters
    matnet = ss.MaternalNet() # To track newborn --> household
    nets = [harlemnet, matnet] # randnet, 

    # ------- TB disease --------
    # Disease parameters
    tb_pars = dict(
        beta = dict(harlem=beta, maternal=0.0),
        init_prev = init_prev, # Infections seeded by Harlem class
        rate_LS_to_presym = 3e-5, # Slow down LS-->Presym as this is now the rate for healthy individuals

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
        #mtb.HarlemPregnancy(pars=dict(fertility_rate=45)), # Per 1,000 women
        mtb.HarlemPregnancy(pars=dict(fertility_rate=150)), # Per 1,000 women aged 15-49
        ss.Deaths(pars=dict(death_rate=10)), # Per 1,000 people (background deaths, excluding TB-cause)
    ]

    # -------- Connector -------
    cn_pars = dict(
        rel_LS_prog_func = rel_LS_prog_func,
        rel_LF_prog_func = rel_LF_prog_func,
        relsus_microdeficient = relsus_microdeficient # Increased susceptibilty of those with micronutrient deficiency (could make more complex function like LS_prog)
    )
    cn = mtb.TB_Nutrition_Connector(cn_pars)

    # -------- Interventions -------
    m = mtb.MicroNutrients
    M = mtb.MacroNutrients
    intvs = []
    if secular_trend:
        # Rates hand picked to match Appendix Table 7
        nc0 = mtb.NutritionChange(year=[1942, 1944], rate=[1.25, 0], from_state=M.UNSATISFACTORY, to_state=M.MARGINAL,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        nc1 = mtb.NutritionChange(year=[1942, 1944], rate=[2.50, 0], from_state=M.MARGINAL, to_state=M.SLIGHTLY_BELOW_STANDARD,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        nc2 = mtb.NutritionChange(year=[1942, 1944], rate=[1.6, 0], from_state=M.SLIGHTLY_BELOW_STANDARD, to_state=M.STANDARD_OR_ABOVE,
                    p_new_micro=p_new_micro, new_micro_state=m.NORMAL)
        intvs = [nc0, nc1, nc2]

    vs = []
    if vitamin_year_rate is not None:
        years, rates = zip(*vitamin_year_rate)
        vs = mtb.VitaminSupplementation(year=years, rate=rates, p_micro_recovery_func=p_micro_recovery_func) # Need coverage, V1 vs V2
        intvs += [vs]


    # -------- Analyzer -------
    azs = [
        mtb.HarlemAnalyzer(),
        mtb.HHAnalyzer(),
        mtb.NutritionAnalyzer(),
    ]

    # -------- Simulation -------
    sim_pars = dict(
        dt = 7/365,
        start = 1941,
        end = 1947,
        rand_seed = rand_seed,
    )
    sim = ss.Sim(people=pop, networks=nets, diseases=[tb, nut], pars=sim_pars, demographics=dems, connectors=cn, interventions=intvs, analyzers=azs)
    sim.pars.verbose = 0 # change to sim.pars.dt / 5 to print status every 5 years instead of every 10 steps

    sim.initialize()

    harlem.set_states(sim)
    seed_uids = harlem.choose_seed_infections()
    tb = sim.diseases['tb']
    tb.set_prognoses(seed_uids)

    # After set_prognoses, seed_uids will be in latent slow or fast
    # Change to ACTIVE_PRESYMP and set time of activation to current time step
    tb.state[seed_uids] = mtb.TBS.ACTIVE_PRESYMP
    tb.ti_active[seed_uids] = sim.ti

    # In Harlem, 83% or 84% have SmPos active infections, let's fix that now
    desired_n_active = 0.835 * len(seed_uids)
    cur_n_active = np.count_nonzero(tb.active_tb_state[seed_uids]==mtb.TBS.ACTIVE_SMPOS)
    add_n_active = desired_n_active - cur_n_active
    non_smpos_uids = seed_uids[tb.active_tb_state[seed_uids]!=mtb.TBS.ACTIVE_SMPOS]
    p = add_n_active / ( len(seed_uids) - cur_n_active)
    change_to_smpos = np.random.rand(len(non_smpos_uids)) < p
    tb.active_tb_state[non_smpos_uids[change_to_smpos]] = mtb.TBS.ACTIVE_SMPOS

    sim.run() # Actually run the sim

    df = sim.analyzers['harlemanalyzer'].df
    assert p_control == 0 or p_control == 1, f'p_control should be 0 or 1, but input was p_control={p_control}'
    # Remove "arm" as it's not useful
    df = df.loc[ df['arm'] == kwargs['arm']]

    # Could us df.attrs here?
    df['p_control'] = p_control
    df['rand_seed'] = rand_seed
    df['Scenario'] = scen
    df['Scen'] = kwargs['skey'] # Raises a SettingWithCopyWarning?!
    #df['Arm'] = kwargs['arm'] # Not needed because we have 'arm' from the analyzer

    dfhh = sim.analyzers['hhanalyzer'].df
    dfhh['rand_seed'] = rand_seed
    dfhh['Scenario'] = scen

    dfn = sim.analyzers['nutritionanalyzer'].df
    dfn['rand_seed'] = rand_seed
    dfn['Scenario'] = scen

    #print(f'Finishing {lbl}')

    return df, dfhh, dfn


def run_scenarios(n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []

    for skey, scen in scens.items():
        for rs in range(n_seeds):
            cfgs.append({'scen': skey,'rand_seed':rs, 'idx':len(cfgs)} | scen) # Merge dicts with pipe operators
    T = sc.tic()
    results += sc.parallelize(run_harlem, iterkwargs=cfgs, die=False, serial=False)
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
    if cache_from is None:
        df_epi, df_hh, df_nut = run_scenarios()
    else:
        cfg.FILE_POSTFIX = cache_from
        cfg.RESULTS_DIRECTORY = os.path.join('figs', 'TB', cfg.FILE_POSTFIX)
        df_epi = pd.read_csv(os.path.join(cfg.RESULTS_DIRECTORY, f'result_{cfg.FILE_POSTFIX}.csv'), index_col=0)
        df_hh = pd.read_csv(os.path.join(cfg.RESULTS_DIRECTORY, f'hhsizes_{cfg.FILE_POSTFIX}.csv'), index_col=0)
        df_nut = pd.read_csv(os.path.join(cfg.RESULTS_DIRECTORY, f'nutrition_{cfg.FILE_POSTFIX}.csv'), index_col=0)
        #df_epi = df_epi.loc[df_epi['Scenario'].isin(['Base CONTROL', 'Base VITAMIN', 'RelSus5 CONTROL', 'RelSus5 VITAMIN'])]

    '''
    if debug:
        sim.diseases['tb'].log.line_list.to_csv('linelist.csv')
        sim.diseases['tb'].plot()
        sim.plot()
        sim.analyzers['harlemanalyzer'].plot()
        plt.show()
    '''

    skeys = df_hh['Scenario'].apply(lambda x: x.split(' ')[0]).unique()

    mtb.plot_calib(df_epi, scens, channel='cum_active_infections')
    mtb.plot_diff(df_epi, scens, channel='cum_active_infections')
    mtb.plot_active_infections(df_epi)
    mtb.plot_epi(df_epi)
    mtb.plot_hh(df_hh)

    for skey in ['Base', 'MoreMicroDeficient']:
        if skey in skeys:
            print('plotting', skey)
            mtb.plot_nut(df_nut, scenarios = [f'{skey} CONTROL', f'{skey} VITAMIN'], lbl=skey)

    print(f'Results directory {cfg.RESULTS_DIRECTORY}\nThis run: {cfg.FILE_POSTFIX}')
    print('Done')
