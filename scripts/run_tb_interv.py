import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np


def build_sim(spars=None, scenario=None, **kwargs):
    spars = dict(
        unit = 'day',
        dt = 7, 
        start = sc.date('1940-01-01'),      
        stop = sc.date('2035-12-31'), 
        rand_seed = 123,
    )
    inv = []
    pop = ss.People(n_agents=500, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=scenario['tbpars'])
    net = ss.RandomNet(dict(n_contacts=ss.poisson(lam=5), dur=0))
    hhnet = mtb.HouseholdNet(
        n_contacts=ss.poisson(lam=5),
        n_households=ss.poisson(lam=5),
        household_size=ss.poisson(lam=5),
        household_size_min=1,
        household_size_max=5,
    )
    net.add_network(hhnet)
    births = ss.Births(pars=dict(birth_rate=15))
    deaths = ss.Deaths(pars=dict(death_rate=15))
    
    if scenario['tptintervention'] is not None: inv.append(mtb.TPTInitiation(pars=scenario['tptintervention']))
    if scenario['bcgintervention'] is not None: inv.append(mtb.BCGProtection(pars=scenario['bcgintervention']))
    
    sim = ss.Sim(
        people=pop,
        networks=hhnet,
        interventions=inv,
        diseases=tb,
        demographics=[deaths, births],
        pars=spars,
    )

    return sim

def get_scenarios():
    scenarios = {
        'TPT': {
            'name': 'TPT INITIATION',
            'pars': {
                'tbpars' : {
                    'start' : sc.date('1990-01-01'),
                    'stop' : sc.date('2030-12-31'),  
                },
                'tptintervention': {
                    'p_tpt':ss.bernoulli(1.0),
                    'tpt_duration':2.0,
                    'max_age':25,
                    'hiv_status_threshold':True,
                    'p_3HP':0.8,
                    'start': ss.date('1970-01-01'),
                },
                'bcgintervention': None
            },
        },
        # 'BCG': {
        #     'name': 'BCG PROTECTIOB',
        #     'pars': {
        #         'tbpars' : {
        #             'start' : sc.date('1990-01-01'),
        #             'stop' : sc.date('2030-12-31'),  
        #         },
        #         'tptintervention': None,
        #         'bcgintervention': {
        #             'coverage':0.60,
        #             'target_age':18,
        #         }
        #     },
        # },
                
    }
    return scenarios

# def get_extrastates():
#     exs = [ss.State('sought_care', default=False),
#         ss.State('returned_to_community', default=False),
#         ss.State('received_tpt', default=False),
#         ss.State('tb_treatment_success', default=False),
#         ss.State('tested', default=False),
#         ss.State('test_result', default=np.nan),
#         ss.State('diagnosed', default=False),
#         ss.State('on_tpt', default=True),
#         ss.State('tb_smear', default=False),
#         ss.State('hiv_positive', default=False),
#         ss.State('eptb', default=False),
#         ss.State('symptomatic', default=False),
#         ss.State('presymptomatic', default=False),
#         ss.State('non_symptomatic', default=True),
#         ss.State('screen_negative', default=True),
#         ss.State('household_contact', default=False),
#         ss.FloatArr('vaccination_year', default=np.nan),]
#     return exs

if __name__ == '__main__':
    import plots as pl
    results = {}
    scenarios = get_scenarios()
    for name, scen in scenarios.items():
        print(f"\nRunning scenario: {name}")
        sim = build_sim(scenario=scen['pars'])
        sim.run()
        results[name] = sim.results.flatten()
    
    pl.plot_results(results, n_cols=5,
        dark=True, cmap='tab20', heightfold=3, style='default')
    
    
    plt.show()
