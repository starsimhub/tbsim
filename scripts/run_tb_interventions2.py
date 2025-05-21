import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import numpy as np

# Default simulation parameters
DEFAULT_SPARS = dict(
    unit='day',
    dt=7,
    start=sc.date('1965-01-01'),
    stop=sc.date('2035-12-31'),
    rand_seed=123,
)
DEFAULT_TBPARS = dict(
        beta = ss.beta(0.1),
        init_prev = ss.bernoulli(p=0.25),
        unit = 'day',
        dt=7,
        start = sc.date('2004-05-05')
    )

def build_sim(scenario=None, spars=None):
    scenario = scenario or {}
    tbpars = {}

    spars = {**DEFAULT_SPARS, **(spars or {})}  # Merge user spars with default
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})} 
    
    # Set up interventions safely
    inv = []
    for key, cls in [('tptintervention', mtb.TPTInitiation), 
                     ('bcgintervention', mtb.BCGProtection)]:
        params = scenario.get(key)
        if params:
            inv.append(cls(pars=params))

    # Core sim components
    pop = ss.People(n_agents=100, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    networks = [ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
                mtb.HouseholdNet(),
                ss.MaternalNet()]
    
    demographics = [ss.Births(pars={'birth_rate': 15}),
                    ss.Deaths(pars={'death_rate': 15})]

    # Create and return simulation
    return ss.Sim(
        people=pop,
        networks=networks,
        interventions=inv,
        diseases=tb,
        demographics=demographics,
        pars=spars,
    )


def get_scenarios():
    return {
        'BCG': {
            'name': 'BCG PROTECTION',
            'tbpars': dict(start=sc.date('1975-02-15'), 
                           stop=sc.date('2030-12-31')),
            'tptintervention': None,
            'bcgintervention': dict(
                coverage=0.60,
                target_age=18,
                year= [1987]  #sc.date('1970-01-01'),
            ),
        },
        'TPT': {
            'name': 'TPT INITIATION',
            'tbpars': dict(start=sc.date('1975-02-01'), 
                           stop=sc.date('2030-12-31')),
            'tptintervention': dict(
                p_tpt=ss.bernoulli(1.0),
                tpt_duration=2.0,
                max_age=25,
                hiv_status_threshold=True,
                p_3HP=0.8,
                start=sc.date('1970-01-01'),
            ),
            'bcgintervention': None,
        },
    }


def run_scenarios(plot=True):
    import plots as pl

    results = {}
    for name, scenario in get_scenarios().items():
        print(f"\nRunning scenario: {name}")
        sim = build_sim(scenario=scenario)
        sim.run()
        results[name] = sim.results.flatten()

    if plot:
        pl.plot_results(results, n_cols=5, dark=True, cmap='viridis', heightfold=2)
        plt.show()


if __name__ == '__main__':
    run_scenarios()
