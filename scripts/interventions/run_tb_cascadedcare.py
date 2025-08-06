import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


def build_sim(spars=None, scenario=None, **kwargs):
    spars = dict(
        dt=ss.days(7), 
        start = sc.date('1965-01-01'),      
        stop = sc.date('2035-12-31'), 
        rand_seed = 123,
    )
    inv = []
    pop = ss.People(n_agents=100, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=scenario['tbpars'])
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    hhnet = mtb.HouseholdNet()
    mat = ss.MaternalNet()
 
    births = ss.Births(pars=dict(birth_rate=15))
    deaths = ss.Deaths(pars=dict(death_rate=15))
    
    if scenario['cascadedcare'] is not None: inv.append(mtb.TbCascadeIntervention(pars=scenario['cascadedcare']))

    
    sim = ss.Sim(
        people=pop,
        networks=[net, hhnet, mat],
        interventions=inv,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    return sim

def get_scenarios():
    scenarios = {
        'CascadedCare': {
            'name': 'Cascade Care',
            'pars': {
                'tbpars' : {
                    'start' : ss.date('1975-01-01'),
                    'stop' : ss.date('2030-12-31'),  
                },
                'cascadedcare': {
                    'start': ss.date('1975-01-01'),
                    'stop': ss.date('2020-12-31')
                },
            },
        },          
    }
    return scenarios


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import plots as pl
    results = {}
    scenarios = get_scenarios()
    for name, scen in scenarios.items():
        print(f"\nRunning scenario: {name}")
        sim = build_sim(scenario=scen['pars'])
        sim.run()
        results[name] = sim.results.flatten()
    
    pl.plot_results(results, n_cols=5,
        dark=True, cmap='viridis', heightfold=2)
    
    
    plt.show()
