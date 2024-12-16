import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os

from run_act3_calibration import make_sim, build_sim


debug = True #NOTE: Debug runs in serial

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

    sim = base_sim.copy()
    # MODIFY THE SIMULATION OBJECT BASED ON THE SCENARIO HERE
    # skey, scen, rand_seed
    #########################################################

    scen['CalibPars']['rand_seed'] = rand_seed # This is the base seed that build_sim will increment from for n_reps
    ms = build_sim(sim, calib_pars=scen['CalibPars'], n_reps=n_reps)
    ms.run()

    tb_res = []
    acf_res = []
    for s in ms.sims:
        df = pd.DataFrame({
            'timevec': s.results.timevec,
            'on_treatment': s.results.tb.n_on_treatment, 
            'prevalence': s.results.tb.prevalence,
            'active_presymp': s.results.tb.n_active_presymp,
            'active_smpos': s.results.tb.n_active_smpos,
            'active_exptb': s.results.tb.n_active_exptb,
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['seed'] = s.pars.rand_seed
        tb_res.append(df)

        act3_dates = [ss.date(t) for t in ['2014-06-01', '2015-06-01', '2016-06-01', '2017-06-01']]
        inds = np.searchsorted(s.results.timevec, act3_dates, side='left')
        df = pd.DataFrame({
            'timevec': s.results.timevec[inds],
            'n_elig': s.results['ACT3 Active Case Finding'].n_elig[inds],
            'n_tested': s.results['ACT3 Active Case Finding'].n_tested[inds],
            'n_positive': s.results['ACT3 Active Case Finding'].n_positive[inds],
        })
        df['scenario'] = skey
        df['arm'] = s.label
        df['seed'] = s.pars.rand_seed
        acf_res.append(df)

    tb_res = pd.concat(tb_res)
    acf_res = pd.concat(acf_res)

    return {'TB': tb_res, 'ACT3': acf_res}


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
            )
        }
    }

    df_result = run_scenarios(scens)

    # plot the results
    df_result.get('ACT3')


    # MOVE TO aplt:
    import seaborn as sns
    ret = df_result.get('TB').reset_index(drop=True).melt(id_vars=['timevec', 'arm', 'seed'], value_name='value', var_name='variable')
    g = sns.relplot(data=ret, x='timevec', y='value', hue='arm', col='variable', kind='line', row='scenario', errorbar='sd', facet_kws={'sharey': False}, height=3, aspect=1.4) # SD for speed, units='seed'
    g.set_titles(col_template="{col_name}")
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(resdir, 'figs', 'timeseries.png'), dpi=600)
    ################

    aplt.plot_scenarios(results=df_result.get('TB'))