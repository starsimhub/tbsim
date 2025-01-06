import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os

from run_act3_calibration import make_sim, build_sim

do_run = True
debug = False #NOTE: Debug runs in serial

# EACH SEED WILL RUN R_REPS SIMULATIONS
default_n_rand_seeds = [10, 2][debug]
n_reps = [30, 3][debug]


# Check if the results directory exists, if not, create it
resdir = os.path.join('results', 'ACT3')
os.makedirs(resdir, exist_ok=True)

def run_ACF(base_sim, skey, scen, rand_seed=0):
    """
    Run n_reps of control and intervention simulations in a single multisim for seed rand_seed
    """
    #print(skey)
    sim = base_sim.copy()
    # MODIFY THE SIMULATION OBJECT BASED ON THE SCENARIO HERE
    # skey, scen

    scen['CalibPars']['rand_seed'] = rand_seed # This is the base seed that build_sim will increment from for n_reps
    #########################################################

    ms = build_sim(sim, calib_pars=scen['CalibPars'], n_reps=n_reps)
    ms.run()

    tb_res = []
    acf_res = []
    for s in ms.sims:
        tb_df = pd.DataFrame({
            'time_year': s.results.timevec,
            'on_treatment': s.results.tb.n_on_treatment, 
            'prevalence': s.results.tb.prevalence,
            'active_presymp': s.results.tb.n_active_presymp,
            'active_smpos': s.results.tb.n_active_smpos,
            'active_exptb': s.results.tb.n_active_exptb,
        })
        first_seed = ms.sims[0].pars.rand_seed
        tb_df['scenario'] = skey
        tb_df['arm'] = s.label
        tb_df['rand_seed'] = s.pars.rand_seed
        tb_df['replicate'] = s.pars.rand_seed - first_seed + 1
        tb_res.append(tb_df)

        # get the indices of the time points where the prevalence surveys were conducted
        # removing hard coding: enable scenarios that extend beyond 2017
        date_cov_keys = list(s.interventions['ACT3 Active Case Finding'].pars.date_cov.keys())
        
        # this should default to -- ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']]
        act3_dates = [ss.date(t) for t in date_cov_keys]
        inds = np.searchsorted(s.results.timevec, act3_dates, side='left')

        acf_df = pd.DataFrame({
            'time_year': s.results.timevec[inds],
            'n_elig': s.results['ACT3 Active Case Finding'].n_elig[inds],
            'n_tested': s.results['ACT3 Active Case Finding'].n_tested[inds],
            'n_positive': s.results['ACT3 Active Case Finding'].n_positive[inds],
            'prev_active': s.results.tb.prevalence_active[inds],
            'inc_kpy': s.results.tb.incidence_kpy[inds]
        })
        acf_df['scenario'] = skey
        acf_df['arm'] = s.label
        acf_df['rand_seed'] = s.pars.rand_seed
        acf_df['replicate'] = s.pars.rand_seed - first_seed + 1
        acf_res.append(acf_df)

    tb_res = pd.concat(tb_res)
    acf_res = pd.concat(acf_res)

    # process the results to calculate the effect of the intervention - 
    # compare the gradients betwen the control and the intervention arms
    eff_cols = ['time_year', 'prev_active', 'scenario', 'rand_seed', 'replicate']
    
    # calculate the change in prevalnce between the first and last time points - control
    tb_control_prev_res = acf_res[(acf_res['arm'] == 'Control')][eff_cols]
    tb_control_prev_res = (
        tb_control_prev_res \
            .groupby(['rand_seed', 'replicate', 'scenario']) \
            # for prevalence
            .apply(lambda group: 
                   group.loc[group['time_year'].idxmax(), 'prev_active'] - group.loc[group['time_year'].idxmin(), 'prev_active'], 
                   include_groups=False) \
            .reset_index(name='prev_gradient_ctrl')
    )
    
    # calculate the change in prevalnce between the first and last time points - intervention
    tb_intervention_prev_res = acf_res[(acf_res['arm'] == 'Intervention')][eff_cols]
    tb_intervention_prev_res = (
        tb_intervention_prev_res \
            .groupby(['rand_seed', 'scenario']) \
            .apply(lambda group: 
                   group.loc[group['time_year'].idxmax(), 'prev_active'] - group.loc[group['time_year'].idxmin(), 'prev_active'], 
                   include_groups=False) \
            .reset_index(name='prev_gradient_intv')
    )
    
    # merge the two dataframes to calculate the relative change in prevalence - intervention relative to control
    acf_effect_res = pd.merge(tb_control_prev_res, tb_intervention_prev_res, on=['rand_seed', 'scenario'])
    acf_effect_res['relative_prev'] = (acf_effect_res['prev_gradient_intv'] - acf_effect_res['prev_gradient_ctrl'])*100_000

    return {'TB': tb_res, 'ACT3': acf_res, 'ACT3 Effect': acf_effect_res}


def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []

    seeds = np.random.randint(0, 1e6, n_seeds)

    # Iterate over scenarios and random seeds
    for skey, scen in scens.items():
        for seed in seeds:
            # Append configuration for parallel execution
            seed = np.random.randint(0, 1e6) # Use a random seed because the multisim will increment from this and we don't want to reuse
            cfgs.append({'skey': skey, 'scen': scen, 'rand_seed': seed})

    # Run simulations in parallel
    T = sc.tic()

    sim = make_sim()
    results += sc.parallelize(run_ACF, iterkwargs=cfgs, kwargs=dict(base_sim=sim), die=True, serial=debug)

    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # separate the results for each component of the simulation (TB and ACT3)
    dfs = {}
    for k in results[0].keys():
        df_list = [r.get(k) 
                   for r in results 
                   if r.get(k) is not None
                   ]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))
    return dfs


if __name__ == '__main__':
    # hard coding ain't cool: there should be a way to pull parameter values from a calibration run and feed them into the base scenarios
    scens = {
        'Basic ACT3': {
            # default has been set to basic ACT3
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                # {'beta': 0.4016615352455662, 'beta_change': 0.7075221406739112, 'beta_change_year': 2001, 'xpcf': 0.14812009041601049, 'rand_seed': 172264}. Best is trial 88 with value: 11479.499964475886.
                beta = dict(low=0.01, high=0.70, value=0.4016615352455662, suggest_type='suggest_float', log=False), # Log scale and no "path", will be handled by build_sim (above)
                beta_change = dict(low=0.25, high=1, value=0.7075221406739112),
                beta_change_year = dict(low=1950, high=2014, value=2001, suggest_type='suggest_int'),
                xpcf = dict(low=0, high=1.0, value=0.14812009041601049),
                stop = dict(value=sc.date('2027-12-31')),
                date_cov = dict(value = { 
                    sc.date('2014-06-01'): 0.844,
                    sc.date('2015-06-01'): 0.80,
                    sc.date('2016-06-01'): 0.779,
                    sc.date('2017-06-01'): 0.743})
            )
        }, 
        'Low Transmission ACT3': {
            # reduce the beta by 50% to simulate a low transmission scenario
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                # keys other than 'value' are optional when simulating scnearios
                beta = dict(value=0.4016615352455662*0.5), 
                beta_change = dict(value=0.7075221406739112),
                beta_change_year = dict(value=2001),
                xpcf = dict(value=0.14812009041601049),
                stop = dict(value=sc.date('2027-12-31')),
            )
        },
        'High Transmission ACT3': {
            # increase the beta by 50% to simulate a high transmission scenario
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                beta = dict(value=0.4016615352455662*1.5), # Log scale and no "path", will be handled by build_sim (above)
                beta_change = dict(value=0.7075221406739112),
                beta_change_year = dict(value=2001),
                xpcf = dict(value=0.14812009041601049),
                stop = dict(value=sc.date('2027-12-31')),
            )
        },
        "ACT3 Basic Missed Sub-clinical": {
            # active case finding is run for 3 years like the basic scenario but we miss the sub-clinical cases
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                beta = dict(value=0.4016615352455662), # Log scale and no "path", 
                beta_change = dict(value=0.7075221406739112),
                beta_change_year = dict(value=2001),
                xpcf = dict(value=0.14812009041601049),
                test_sens = dict(value = {
                    mtb.TBS.ACTIVE_SMPOS: 1,
                    # reduce the ability to detect sub-clinical cases by ~75%
                    mtb.TBS.ACTIVE_PRESYMP: 0.25,
                    mtb.TBS.ACTIVE_SMNEG: 0.8,
                    mtb.TBS.ACTIVE_EXPTB: 0.1
                    }) 
            )
        },
        'ACT5': {
            # active case finding is run for 5 years instead of 3 years and the prevalence survey conducted between 2018 and 2019
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                beta = dict(value=0.4016615352455662), 
                beta_change = dict(value=0.7075221406739112),
                beta_change_year = dict(value=2001),
                xpcf = dict(value=0.14812009041601049),
                stop = dict(value=sc.date('2030-12-31')),
                date_cov = dict(value = 
                    {sc.date('2014-06-01'): 0.6,
                     sc.date('2015-06-01'): 0.7,
                     sc.date('2016-06-01'): 0.64,
                     sc.date('2017-06-01'): 0.64, 
                     sc.date('2018-06-01'): 0.64,
                     sc.date('2019-06-01'): 0.64
                     })
            )
        }, 
        "ACT7": {
            # active case finding is run for 7 years instead of 3 years and the prevalence survey conducted between 2020 and 2021
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': dict(
                beta = dict(value=0.4016615352455662), # Log scale and no "path", 
                beta_change = dict(value=0.7075221406739112),
                beta_change_year = dict(value=2001),
                xpcf = dict(value=0.14812009041601049),
                stop = dict(value=sc.date('2031-12-31')),
                date_cov = dict(value = {
                    sc.date('2014-06-01'): 0.6,
                    sc.date('2015-06-01'): 0.7,
                    sc.date('2016-06-01'): 0.64,
                    sc.date('2017-06-01'): 0.64, 
                    sc.date('2018-06-01'): 0.64,
                    sc.date('2019-06-01'): 0.64,
                    sc.date('2020-06-01'): 0.64,
                    sc.date('2021-06-01'): 0.64})
            )
        }
    }

    if do_run:
        df_result = run_scenarios(scens)
    else:
        try:
            df_result = {}
            for k in ['TB', 'ACT3']:
                df_result[k] = pd.read_csv(os.path.join(resdir, f'{k}.csv'))
        except FileNotFoundError:
            print('No results found, please set do_run to True')
            raise

    # plot the results
    df_result.get('ACT3')


    # MOVE TO aplt:

    # TB time series
    # import seaborn as sns
    # ret = df_result.get('TB').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    # g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    # g.set_titles(col_template="{col_name}")
    # g.fig.tight_layout()
    # g.fig.savefig(os.path.join(resdir, 'figs', 'timeseries.png'), dpi=600)

    # # ACT3 time series
    # ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    # g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    # g.set_titles(col_template="{col_name}")
    # g.fig.tight_layout()
    # g.fig.savefig(os.path.join(resdir, 'figs', 'act3.png'), dpi=600)

    # # ACT3 cases found, scaled to trial
    # ret = df_result.get('ACT3').reset_index(drop=True).melt(id_vars=['scenario', 'time_year', 'arm', 'rand_seed'], value_name='value', var_name='variable')
    # g = sns.relplot(data=ret, x='time_year', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='rand_seed'
    # g.set_titles(col_template="{col_name}")
    # g.fig.tight_layout()
    # g.fig.savefig(os.path.join(resdir, 'figs', 'act3.png'), dpi=600)
    ################

    aplt.plot_scenarios(results=df_result.get('TB'))