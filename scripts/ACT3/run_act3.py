import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import act3_plots as aplt
import os


debug = True #NOTE: Debug runs in serial
default_n_rand_seeds = [60, 1][debug]


# Check if the results directory exists, if not, create it
resdir = os.path.join('results', 'ACT3')

os.makedirs(resdir, exist_ok=True)

def build_ACF(skey, scen, rand_seed=0):
    """
    Build the simulation object that will simulate the ACT3
    """

    # random seed is used when deciding the initial n_agents, so set here
    np.random.seed(rand_seed)

    # Retrieve intervention, TB, and simulation-related parameters from scen and skey
    # for TB
    # Create the people, networks, and demographics
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1000, scale=50)))
    demog = [
        ss.Deaths(pars=dict(death_rate=10)),
        ss.Pregnancy(pars=dict(fertility_rate=45))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=5), dur = 0)

    # Modify the defaults to if necessary based on the input scenario 
    # for the TB module
    tb_pars = dict(
        beta=ss.beta(0.045),
        init_prev=0.1,
        rate_LS_to_presym=ss.perday(3e-5),
        rate_LF_to_presym=ss.perday(6e-3),
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10
    )

    if scen is not None and 'TB' in scen.keys() and scen['TB'] is not None:
        tb_pars.update(scen['TB'])

    tb = mtb.TB(tb_pars)

    # for the intervention 
    intv_pars = {}
    if scen is not None and 'ACT3' in scen.keys() and scen['ACT3'] is not None:
        intv_pars.update(scen['ACT3'])

    intv = mtb.ActiveCaseFinding(intv_pars)

    # for the simulation parameters
    sim_pars = dict(
        # default simulation parameters
        unit='day', dt=14,
        start=ss.date('2013-01-01'), stop=ss.date('2016-12-31'),
        rand_seed=rand_seed
        )

    if scen is not None and 'Simulation' in scen.keys() and scen['Simulation'] is not None:
        sim_pars.update(scen['Simulation'])

    anz = None
    # some scenarios can also alter the simulation parameters 
    # TODO: 
        # What happens when when you rerun ACT3 for longer? 
        # What happens when you run ACT3 more frequently?
    sim = ss.Sim(
        people=pop, networks=nets, diseases=tb, demographics=demog, interventions=intv,
        analyzers=anz, pars=sim_pars
    )

    # Print status every 1 year outside of debug mode
    sim.pars.verbose = [0, sim.pars.dt/365][debug] 

    return sim


def run_ACF(skey, scen, rand_seed=0):
    """
    Run the pick simulation for the ACT3 under a single scenario - pick out the results
    """

    sim = build_ACF(skey, scen, rand_seed)
    sim.run()

    sim.plot('tb')
    plt.show()

    tb_res = pd.DataFrame({
        'time_year': sim.results.timevec,
        'on_treatment': sim.results.tb.n_on_treatment, 
        'prevalence': sim.results.tb.prevalence,
        'active_presymp': sim.results.tb.n_active_presymp,
        'active_smpos': sim.results.tb.n_active_smpos,
        'active_exptb': sim.results.tb.n_active_exptb,
    })

    acf_res = pd.DataFrame({
        'time_year': sim.results.activecasefinding.time,
        'n_elig': sim.results.activecasefinding.n_elig,
        'n_found': sim.results.activecasefinding.n_found,
        'n_treated': sim.results.activecasefinding.n_treated,
    })

    # add the scenarios label to the results
    tb_res['scenario'] = skey
    acf_res['scenario'] = skey

    tb_res['rand_seed'] = rand_seed
    acf_res['rand_seed'] = rand_seed

    return {'TB': tb_res, 'ACT3': acf_res}


def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []

    # Iterate over scenarios and random seeds
    for skey, scen in scens.items():
        for rs in range(n_seeds):
            # Append configuration for parallel execution
            cfgs.append({'skey': skey, 'scen': scen, 'rand_seed': rs})

    # Run simulations in parallel
    T = sc.tic()

    results += sc.parallelize(run_ACF, iterkwargs=cfgs, die=True, serial=debug)

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
        'Control': {
            # Turn off the tretatment for the control scenario
            'ACT3': dict(p_treat=ss.bernoulli(p=0.0)),
            'TB': None,
            'Simulation': None
        },
        'Basic ACT3': {
            # default has been set to basic ACT3
            'ACT3': dict(p_treat=ss.bernoulli(p=1.0)),
            'TB': None,
            'Simulation': None
        }
    }

    df_result = run_scenarios(scens)

    # plot the results
    df_result.get('ACT3')


    aplt.plot_scenarios(results=df_result.get('TB'))