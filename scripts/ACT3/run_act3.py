import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os

from run_act3_calibration import make_sim, build_sim

do_run = True
debug = True #NOTE: Debug runs in serial

# Each scenario will be run n_seeds times for each of intervention and control.
n_seeds = [60, 2][debug]
n_reps = 1 # Default, leave as 1 for now, results will be combined by bootstrapping across seeds

# Check if the results directory exists, if not, create it
resdir = os.path.join('results', 'ACT3')
os.makedirs(resdir, exist_ok=True)

class prev_by_age(ss.Intervention):
    def __init__(self, year, **kwargs):
        self.year = year
        super().__init__(**kwargs)
        return
    
    def step(self):
        if self.year >= self.t.now('year') and  self.year < self.t.now('year')+self.t.dt_year:
            self.age_bins = np.arange(0, 101, 5)
            self.n, _        = np.histogram(self.sim.people.age, bins=self.age_bins)
            self.ever, _     = np.histogram(self.sim.people.age[self.sim.diseases.tb.ever_infected], bins=self.age_bins)
            self.infected, _ = np.histogram(self.sim.people.age[self.sim.diseases.tb.infected], bins=self.age_bins)
            self.active, _   = np.histogram(self.sim.people.age[self.sim.diseases.tb.infectious], bins=self.age_bins)
        return

def run_ACF(base_sim, skey, scen, rand_seed=0):
    """
    Run n_reps of control and intervention simulations in a single multisim for seed rand_seed
    """
    #print(skey)
    sim = base_sim.copy()
    # MODIFY THE SIMULATION OBJECT BASED ON THE SCENARIO HERE
    # skey, scen

    sim.pars.interventions += [prev_by_age(year=2013)]

    sim.pars.rand_seed = rand_seed # This is the base seed that build_sim will increment from for n_reps
    #########################################################

    cp = scen['CalibPars'].copy()
    cp['rand_seed'] = rand_seed
    ms = build_sim(sim, calib_pars=cp, n_reps=n_reps)
    ms.run()

    tb_res = []
    acf_res = []
    pba_res = []
    for s in ms.sims:
        df = pd.DataFrame({
            'time_year': s.results.timevec,
            'on_treatment': s.results.tb.n_on_treatment, 
            'prevalence': s.results.tb.prevalence,
            'prevalence_active': s.results.tb.prevalence_active,
            'active_presymp': s.results.tb.n_active_presymp,
            'active_smpos': s.results.tb.n_active_smpos,
            'active_smneg': s.results.tb.n_active_smneg,
            'active_exptb': s.results.tb.n_active_exptb,
            'incidence_kpy': s.results.tb.incidence_kpy,
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['rand_seed'] = s.pars.rand_seed
        df['include'] = np.any(s.results.tb.n_infected[s.timevec >= ss.date('2013-01-01')] > 0)
        tb_res.append(df)

        date_cov_keys = list(s.interventions['ACT3 Active Case Finding'].pars.date_cov.keys())
        act3_dates = [ss.date(t) for t in date_cov_keys]
        inds = np.searchsorted(s.results.timevec, act3_dates, side='left')
        df = pd.DataFrame({
            'time_year': s.results.timevec[inds],
            'n_elig': s.results['ACT3 Active Case Finding'].n_elig[inds],
            'n_tested': s.results['ACT3 Active Case Finding'].n_tested[inds],
            'n_positive': s.results['ACT3 Active Case Finding'].n_positive[inds],
            'prev_active': s.results.tb.prevalence_active[inds],

            'n_positive_presymp': s.results['ACT3 Active Case Finding'].n_positive_presymp[inds],
            'n_positive_smpos': s.results['ACT3 Active Case Finding'].n_positive_smpos[inds],
            'n_positive_smneg': s.results['ACT3 Active Case Finding'].n_positive_smneg[inds],
            'n_positive_exp': s.results['ACT3 Active Case Finding'].n_positive_exp[inds],

            'n_positive_via_LF': s.results['ACT3 Active Case Finding'].n_positive_via_LF[inds],
            'n_positive_via_LS': s.results['ACT3 Active Case Finding'].n_positive_via_LS[inds],
            'n_positive_via_LF_dur': s.results['ACT3 Active Case Finding'].n_positive_via_LF_dur[inds],
            'n_positive_via_LS_dur': s.results['ACT3 Active Case Finding'].n_positive_via_LS_dur[inds],
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['include'] = np.any(s.results.tb.n_infected[s.timevec >= ss.date('2013-01-01')] > 0)
        df['rand_seed'] = s.pars.rand_seed
        acf_res.append(df)

        intv = s.interventions.get('prev_by_age', None)
        if intv is None:
            continue
        df = pd.DataFrame({
            'n': intv.n,
            'ever': intv.ever,
            'infected': intv.infected,
            'active': intv.active,
        }, index = pd.Index([f'{b}-{e}' for b,e in zip(intv.age_bins[:-1], intv.age_bins[1:])], name='age bin'))
        df['year'] = ss.date(intv.year)
        df['scenario'] = skey
        df['arm'] = s.label
        df['rand_seed'] = s.pars.rand_seed
        pba_res.append(df)

    tb_res = pd.concat(tb_res)
    acf_res = pd.concat(acf_res)
    pba_res = pd.concat(pba_res)

    # Additional calculations - move elsewhere ################################################################

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

    return {'TB': tb_res, 'ACT3': acf_res, 'PBA': pba_res, 'ACT3 Effect': acf_effect_res}


def run_scenarios(scens, n_seeds=n_seeds):
    results = []
    cfgs = []

    seeds = np.random.randint(0, 1e6, n_seeds)

    # Iterate over scenarios and random seeds
    for skey, scen in scens.items():
        for si, seed in enumerate(seeds):
            if 'rand_seed' in scen['CalibPars']:
                seed = scen['CalibPars']['rand_seed'] + si
            else:
                seed = seeds[si] # Use a random seed because the multisim will increment from this and we don't want to reuse
            # Append configuration for parallel execution
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
    calib_base = {
        'beta': 0.5138842296279839,
        'x_pcf1': 0.16269807089242358,
        'x_pcf2': 0.9658136632225554,
        'x_acf_cov': 0.34117557685886396,
        'p_fast': 0.4128032502420532,
    }

    calib_lowtrans = calib_base.copy() # TODO
    calib_lowtrans['beta'] *= 0.5

    calib_hightrans = calib_base.copy() # TODO
    calib_hightrans['beta'] *= 2


    # Convert to format expected by the builder
    calib_base = {k:dict(value=v) for k,v in calib_base.items()} # | {'rand_seed': 165568}
    calib_lowtrans = {k:dict(value=v) for k,v in calib_lowtrans.items()} # | {'rand_seed': 165568}
    calib_hightrans = {k:dict(value=v) for k,v in calib_hightrans.items()} # | {'rand_seed': 165568}

    scens = {
        'Basic ACT3': {
            # default has been set to basic ACT3
            'ACT3': None,
            'TB': None,
            'Simulation': dict(stop=sc.date('2027-12-31')),
            'CalibPars': calib_base,
        }, 
        'Low Transmission ACT3': {
            # reduce the beta by 50% to simulate a low transmission scenario
            'ACT3': None,
            'TB': None,
            'Simulation': dict(stop=sc.date('2027-12-31')),
            'CalibPars': calib_lowtrans
        },
        'High Transmission ACT3': {
            # increase the beta by 50% to simulate a high transmission scenario
            'ACT3': None,
            'TB': None,
            'Simulation': None,
            'CalibPars': calib_hightrans,
        },
        "ACT3 Basic Missed Sub-clinical": {
            # active case finding is run for 3 years like the basic scenario but we miss the sub-clinical cases
            'ACT3': {
                'test_sens': {
                    mtb.TBS.ACTIVE_SMPOS: 1,
                    # reduce the ability to detect sub-clinical cases by ~75%
                    mtb.TBS.ACTIVE_PRESYMP: 0.25,
                    mtb.TBS.ACTIVE_SMNEG: 0.8,
                    mtb.TBS.ACTIVE_EXPTB: 0.1
                    },
            },
            'TB': None,
            'Simulation': None,
            'CalibPars': calib_base,
        },
        'ACT5': {
            # active case finding is run for 5 years instead of 3 years and the prevalence survey conducted between 2018 and 2019
            'ACT3': {
                'date_cov': {
                    sc.date('2014-06-01'): 0.6,
                     sc.date('2015-06-01'): 0.7,
                     sc.date('2016-06-01'): 0.64,
                     sc.date('2017-06-01'): 0.64, 
                     sc.date('2018-06-01'): 0.64,
                     sc.date('2019-06-01'): 0.64
                }
            },
            'TB': None,
            'Simulation': dict(stop=sc.date('2030-12-31')),
            'CalibPars': calib_base
        }, 
        "ACT7": {
            # active case finding is run for 7 years instead of 3 years and the prevalence survey conducted between 2020 and 2021
            'ACT3': {
                'date_cov': {
                    sc.date('2014-06-01'): 0.6,
                    sc.date('2015-06-01'): 0.7,
                    sc.date('2016-06-01'): 0.64,
                    sc.date('2017-06-01'): 0.64, 
                    sc.date('2018-06-01'): 0.64,
                    sc.date('2019-06-01'): 0.64,
                    sc.date('2020-06-01'): 0.64,
                    sc.date('2021-06-01'): 0.64,
                }
            },
            'TB': None,
            'Simulation': dict(stop=sc.date('2031-12-31')),
            'CalibPars': calib_base,
        }
    }

    if do_run:
        df_result = run_scenarios(scens)
    else:
        try:
            df_result = {}
            for k in ['TB', 'ACT3', 'PBA']:
                df_result[k] = pd.read_csv(os.path.join(resdir, f'{k}.csv'), index_col=0)
                if 'time_year' in df_result[k]:
                    df_result[k]['time_year'] = pd.to_datetime(df_result[k]['time_year'])
        except FileNotFoundError:
            print('No results found, please set do_run to True')
            raise

    # plot the results
    df_result.get('ACT3')


    plot_TODO(df_result, resdir)

    # APLT PLOTS #######################################################
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