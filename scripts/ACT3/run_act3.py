import starsim as ss
import tbsim as mtb
import numpy as np
import pandas as pd
import sciris as sc
import os


debug = True # NOTE: Debug runs in serial
default_n_rand_seeds = [60, 1][debug]


def build_ACF(skey, scen, random_seed=0):
    
    """
    Build the simulation object that will simulate the ACT3
        - set a random seed
        - start with roughly 1200 or so people 
        TODO: Think about the age distribution of the population, Vietnam??
        - define demographics 
        - right now use random nets and define a network of individuals 
        - define the tb parameters 
        - define the interevention
        - make room of the analyzers, currently set it to None 
        TODO: add analyzers to do stuff
        - Age distribution of incidence and prevalence? 
        Define the parameters for simulation 
            - Set to start and stop?
            - Set the time step? 
    """
    
    # this needs to be run based on scenarios -- 
    # will come back to this - default is ON! 
    # TODO: need a spot to pass on the scenarios thereby updating the
    # simulation parameters
    np.random.seed(random_seed)
    
    # TODO: Need to pick the right simulation scene to pass to the build.
    # Retrieve intervention, TB, and simulation-related parameters from scen and skey
    # for TB
    # Create the people, networks, and demographics
    pop = ss.People(n_agents=np.round(np.random.normal(loc=1200, scale=50)))
    demog = [
        ss.Deaths(pars=dict(death_rate=10)),
        # is pregnanacy really required? or can I just have births here?
        ss.Pregnancy(pars=dict(fertility_rate=45))
    ]
    nets = ss.RandomNet(n_contacts = ss.poisson(lam=5), dur = 0)
    
    # Modify the defaults to if necesary based on the input scenario 
    # for the TB module
    tb = scen.get('TB') or mtb.TB(
        # setting the default TB parameters if TB input is not provided 
        # TODO: If these are are the same in the TB module maybe setting it to mtb.TB() should suffice?
        beta=ss.beta(0.045),
        init_prev=0.1,
        rate_LS_to_presym=ss.perday(3e-5),
        rate_LF_to_presym=ss.perday(6e-3),
        rel_trans_smpos=1.0,
        rel_trans_smneg=0.3,
        rel_trans_exptb=0.05,
        rel_trans_presymp=0.10
        )        
    
    # for the intervention 
    intv = scen.get('ACT3') or mtb.ActiveCaseFinding()

    # for the simulation parameters
    sim_input = scen.get('Simulation') or {
        # default simulation parameters
        'unit': 'day', 'dt': 7, 
        'start': sc.date('2013-01-01'), 'stop': sc.date('2016-12-31')
        }
    
    anz = None
    # some scenarios can also alter the simulation parameters 
    # TODO: 
        # What happens when when you reun ACT3 for longer? 
        # What happens when you run ACT3 more frequently?
    sim = ss.Sim(
        **sim_input,
        people=pop, networks=nets, diseases=[tb], demographics=demog, interventions=[intv], analyzers=anz 
        )
    
    # Print status every 5 years instead of every 10 steps
    sim.pars.verbose = [0, sim.pars.dt / 5][debug] 
    
    return sim
    

def run_ACF(skey=None, scen=None, rand_seed=0):

    """
    Run the pick simulation for the ACT3 under a single scenario - pick out the results
    """

    sim = build_ACF(skey, scen, rand_seed)
    sim.run()

    tb_res = pd.DataFrame({
        'time': sim.results.timevec,
        'on_treatment': sim.results.tb, 
        'prevalence': sim.results.tb.prevalence,
        'active_presymp': sim.results.tb.n_active_presymp,
        'active_smpos': sim.results.tb.n_active_presymp,
        'active_presymp': sim.results.tb.n_active_presymp}
        )
    
    acf_res = pd.DataFrame({
        'time': sim.results.timevec,
        'n_elig': sim.results.activecasefinding[0].n_elig,
        'n_found': sim.results.interventions[0].n_found
        })
        
    return tb_res, acf_res
    
    

def run_scenarios(scens, n_seeds=default_n_rand_seeds):
    results = []
    cfgs = []
    
    # Iterate over scenarios and random seeds
    for skey, scen in scens.items():
        for rs in range(n_seeds):
            # Append configuration for parallel execution
            # scen['Simulation']['n_agents'] = np.random.normal(loc=1200, scale=50)
            cfgs.append({'skey': skey, 'scen': scen, 'rand_seed': rs})

    # Run simulations in parallel
    T = sc.tic()
    results += sc.parallelize(run_ACF, iterkwargs=cfgs, die=True, serial=debug)
    print(f'That took: {sc.toc(T, output=True):.1f}s')

    # Aggregate the results
    dfs = {}
    for k in results[0].keys():
        df_list = [r[k] for r in results]
        dfs[k] = pd.concat(df_list)
        dfs[k].to_csv(os.path.join(resdir, f'{k}.csv'))

    return dfs
    
   
if __name__ == '__main__':
    
    scens = {
        'Control': {
            # turn off the tretament for the control scenario
            'ACT3': mtb.ActiveCaseFinding(p_treat=ss.bernoulli(p=0.0)),
            'TB': None,
            'Simulation': None
        },
        'Basic ACT3': {
            # default has been set to basic ACT3
            'ACT3': None,
            'TB': None,
            'Simulation': None
        }
    }

    run_scenarios(scens)