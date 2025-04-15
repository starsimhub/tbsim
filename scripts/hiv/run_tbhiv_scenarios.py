import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import tbsim as mtb
import starsim as ss


def build_tbhiv_sim(simpars=None, tbpars=None, hivinv_pars=None) -> ss.Sim:
    """Build a TB-HIV simulation with current disease and intervention models."""

    # --- Simulation Parameters ---
    default_simpars = dict(
        unit='day',
        dt=7,
        start=ss.date('1980-01-01'),
        stop=ss.date('2035-12-31'),
        rand_seed=123,
    )
    if simpars:
        default_simpars.update(simpars)

    # --- Population ---
    n_agents = 1000
    extra_states = [ss.FloatArr('SES', default=ss.bernoulli(p=0.3))]
    people = ss.People(n_agents=n_agents, extra_states=extra_states)

    # --- TB Model ---
    pars = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        rel_sus_latentslow=0.1,
    )
    if tbpars:
        pars.update(tbpars)
    tb = mtb.TB(pars=pars)

    # --- HIV Disease Model ---
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.10),
        init_onart=ss.bernoulli(p=0.50),
    )
    hiv = mtb.HIV(pars=hiv_pars)
    
    # --- Network ---
    network = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=2), dur=0))

    # --- Connector ---
    connector = mtb.TB_HIV_Connector()
    
    # --- HIV Intervention ---
    hiv_intervention = None
    if hiv_pars is not None:
        hiv_intervention = mtb.HivInterventions(pars=hivinv_pars)



    # --- Assemble Simulation ---
    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        interventions=[hiv_intervention],
        networks=network,
        connectors=[connector],
        pars=default_simpars,
    )

    return sim


def run_scenarios():
    scenarios = {
        'baseline': None,
            
        'early_low_coverage': dict(
            mode='both',
            prevalence=0.10,
            percent_on_ART=0.10,
            minimum_age=15,
            max_age=49,
            start=ss.date('1990-01-01'),
            stop=ss.date('2000-12-31'),
        ),
        'mid_coverage_mid_years': dict(
            mode='both',
            prevalence=0.20,
            percent_on_ART=0.40,
            minimum_age=20,
            max_age=60,
            start=ss.date('2000-01-01'),
            stop=ss.date('2010-12-31'),
        ),
        'high_coverage_recent': dict(
            mode='both',
            prevalence=0.25,
            percent_on_ART=0.75,
            minimum_age=10,
            max_age=60,
            start=ss.date('2010-01-01'),
            stop=ss.date('2025-12-31'),
        ),
        'youth_targeted': dict(
            mode='both',
            prevalence=0.15,
            percent_on_ART=0.60,
            minimum_age=10,
            max_age=24,
            start=ss.date('2005-01-01'),
            stop=ss.date('2020-12-31'),
        ),
    }

    flat_results = {}

    for name, hivinv_pars in scenarios.items():
        print(f'Running scenario: {name}')
        sim = build_tbhiv_sim(hivinv_pars=hivinv_pars)
        sim.run()
        flat_results[name] = sim.results.flatten()

    return flat_results

def plot_results(flat_results, keywords=None, exclude=['15']):
    # Automatically identify all unique metrics across all scenarios
    metrics = []
    if keywords is None:
        metrics = sorted({key for flat in flat_results.values() for key in flat.keys()}, reverse=True)
        
    else:
        metrics = sorted({
            k for flat in flat_results.values() for k in flat
            if any(kw in k for kw in keywords) 
        })
        # Exclude specified metrics
    
    metrics = [m for m in metrics if not any(excl in m for excl in exclude)]
        
    n_metrics = len(metrics)
    if n_metrics > 0:
        # If there are more than 5 metrics, use a grid of 5 columns
        n_cols = 5
        n_rows = int(np.ceil(n_metrics / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*2))
        axs = axs.flatten()
        
    cmap = plt.cm.get_cmap('tab10', len(flat_results))

    for i, metric in enumerate(metrics):
        ax = axs[i] if n_metrics > 1 else axs
        for j, (scenario, flat) in enumerate(flat_results.items()):
            if metric in flat:
                result = flat[metric]
                ax.plot(result.timevec, result.values, label=scenario, color=cmap(j))
        ax.set_title(metric)
        if max(result.values) < 1:
            # identify the max value of result.values
            v = max(result.values)
            ax.set_ylim(0, max(0.5, v)) 
            ax.set_ylabel('%')
        else:
            ax.set_ylabel('Value')
        ax.set_xlabel('Time')
        
        ax.grid(True)
        ax.legend()
        # reduce the legend font size if there are many scenarios
        if len(flat_results) > 5:
            leg = ax.legend(loc='upper right', fontsize=5)
        else:
            leg = ax.legend(loc='upper right', fontsize=6)
            
        # Handle legend positioning for crowded plots
        if leg:
            leg.get_frame().set_alpha(0.5)
            

    # Ensure the layout is tight
    plt.tight_layout()
    
    # add an option to change the background color of the plot for better visibility
    for ax in axs:
        # Change the background color of each axis for better visibility
        ax.set_facecolor('#f0f0f0')  # Light gray background for better contrast

    # save the plot in the current directory
    dirname = sc.thisdir()
    plt.savefig(f'{dirname}/tbhiv_scenarios.png', dpi=300)
    # Show the plot
    plt.show()

if __name__ == '__main__':
    flat_results = run_scenarios()
    plot_results(flat_results)
