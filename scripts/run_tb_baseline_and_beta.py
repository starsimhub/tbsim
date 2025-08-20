import tbsim as mtb
import starsim as ss
import sciris as sc
import tbsim.utils.plots as pl
import pandas as pd

# Simple default parameters
DEFAULT_SPARS = dict(
    dt=ss.days(7),
    start=ss.date('1975-01-01'),
    stop=ss.date('2030-12-31'),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    beta=ss.peryear(0.0025),
    init_prev=ss.bernoulli(p=0.25),
    dt=ss.days(7),      
    start=ss.date('1975-02-01'),
    stop=ss.date('2030-12-31'),
)

# Simple age distribution
age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def build_sim(scenario=None, spars=None):
    scenario = scenario or {}
    spars = {**DEFAULT_SPARS, **(spars or {})}
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}
    interventions = []
    beta_params = scenario.get('betabyyear')
    if beta_params:
        if isinstance(beta_params, dict):
            interventions.append(mtb.BetaByYear(pars=beta_params))
        elif isinstance(beta_params, list):
            for i, params in enumerate(beta_params):
                params['name'] = f'Beta_{i}'
                interventions.append(mtb.BetaByYear(pars=params))
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        mtb.HouseholdNet()
    ]
    return ss.Sim(
        people=pop,
        networks=networks,
        interventions=interventions,
        diseases=[tb],
        pars=spars,
    )

def get_scenarios():
    return {
        'Baseline': {
            'name': 'No interventions',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
        },
        'Baseline and BetaByYear': {
            'name': 'BetaByYear intervention',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
            'betabyyear': dict(years=[1990, 2000], x_beta=[0.5, 1.4])
        },
    }

def run_scenarios(plot=True):
    scenarios = get_scenarios()
    results = {}
    for name, scenario in scenarios.items():
        print(f"\nRunning: {name}")
        sim = build_sim(scenario=scenario)
        sim.run()
        results[name] = sim.results.flatten()
    if plot:
        pl.plot_combined(results, dark=True, cmap='viridis')

if __name__ == "__main__":
    run_scenarios() 