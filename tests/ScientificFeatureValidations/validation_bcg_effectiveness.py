"""
Compares baseline TB simulation with BCG intervention to validate effectiveness.
"""

import tbsim as mtb
import starsim as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import sciris as sc

warnings.filterwarnings('ignore')

# Simulation parameters
SIM_PARAMS = {
    'n_agents': 5000,
    'dt': ss.days(7),
    'start': ss.date('2000-01-01'),
    'stop': ss.date('2025-12-31'),
}

TB_PARAMS = {
    'beta': ss.peryear(0.01),
    'init_prev': 0.25,
}

BCG_PARAMS = {
    'coverage': 0.95,
    'efficacy': 0.98,
    'age_range': (0, 50),
    'immunity_period': ss.years(10),
    'start': ss.date('2004-01-01'),
    'stop': ss.date('2025-12-31'),
    # 'delivery': ss.weibull(scale=1.2, c=1.5), 
    'delivery': ss.uniform(0, 10),
}

def create_population(n_agents=5000):
    """Create test population with age distribution"""
    age_data = pd.DataFrame({
        'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
        'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]
    })
    return ss.People(n_agents=n_agents, age_data=age_data)


def run_baseline_simulation():
    """Run baseline simulation without BCG"""
    pop = create_population(SIM_PARAMS['n_agents'])
    tb = mtb.TB(pars=TB_PARAMS)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        pars={**SIM_PARAMS, 'verbose': 0}
    )
    sim.run()
    return sim.results.flatten()

def run_bcg_simulation():
    """Run simulation with BCG intervention"""
    pop = create_population(SIM_PARAMS['n_agents'])
    tb = mtb.TB(pars=TB_PARAMS)
    net = ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0})
    
    bcg = mtb.BCGProtection(pars=BCG_PARAMS)    
    sim = ss.Sim(
        people=pop,
        diseases=[tb],
        networks=[net],
        interventions=[bcg],
        pars={**SIM_PARAMS, 'verbose': 0}
    )
    sim.run()
    return sim.results.flatten()
    



def main():
    baseline = run_baseline_simulation()
    bcg = run_bcg_simulation()
    return baseline, bcg

if __name__ == '__main__':
    baseline, bcg_results = main()
    
    # Prepare flattened results for plotting and saving
    results = {}
    results['baseline'] = baseline
    results['bcg'] = bcg_results
    
    # Save flattened results to file
    os.makedirs('results/interventions', exist_ok=True)
    results_file = 'results/interventions/bcg_validation_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Saved flattened results to {results_file}")
    
    # Save as CSV
    for scenario_name, flat_results in results.items():
        data = {}
        for key, val in flat_results.items():
            if hasattr(val, 'values'):
                data[key] = val.values
            else:
                data[key] = np.asarray(val)
        df = pd.DataFrame(data)
        csv_file = f'results/interventions/bcg_validation_{scenario_name}.csv'
        df.to_csv(csv_file, index=False)
        print(f"Saved CSV: {csv_file}")
    
    mtb.plot_combined(results, title='BCG Intervention Effectiveness: Baseline vs Intervention', outdir='results/interventions')
    plt.show()