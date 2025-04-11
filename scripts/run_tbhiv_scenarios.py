import matplotlib.pyplot as plt
import numpy as np
import sciris as sc
import tbsim as mtb
import starsim as ss

def build_tbhiv_sim(hiv_pars=None, tb_pars=None):
    # Simulation parameters
    simpars = dict(
        unit='day',
        dt=7,
        start=sc.date('2000-01-01'),
        stop=sc.date('2025-12-31'),
        rand_seed=123,
    )

    # People and states
    n_agents = 10_000
    extra_states = [ss.FloatArr('SES', default=ss.bernoulli(p=0.3))]
    pop = ss.People(n_agents=n_agents, extra_states=extra_states)

    # TB
    default_tbpars = dict(
        beta=ss.beta(0.1),
        init_prev=ss.bernoulli(p=0.25),
        unit="day"
    )
    if tb_pars:
        default_tbpars.update(tb_pars)
    tb = mtb.TB(default_tbpars)

    # HIV
    default_hivpars = dict(
        init_prev=ss.bernoulli(p=0.3),  # Initial prevalence of HIV
    )
    if hiv_pars:
        default_hivpars.update(hiv_pars)
    hiv = mtb.HIV(pars=default_hivpars)

    # Demographics and connectors
    dems = [
        ss.Pregnancy(pars=dict(fertility_rate=10)),
        ss.Deaths(pars=dict(death_rate=10)),
    ]
    
    # Connectors
    cn = mtb.TB_HIV_Connector(pars=dict(art_effectiveness=0.30))

    # Build sim
    sim = ss.Sim(
        people=pop,
        diseases=[tb, hiv],
        pars=simpars,
        demographics=dems,
        connectors=cn,
    )
    sim.pars.verbose = 0
    return sim

def run_scenarios():
    # Define scenarios
    scenarios = {
        'base': {},
        '50_perc_ART': dict(art_coverage=0.50),
        '75_perc_ART': dict(art_coverage=0.75),
        'art effectiveness 0.2': dict(art_effectiveness=0.2),
        'art effectiveness 0.1': dict(art_effectiveness=0.1),  # This is the default in the base case
        'art effectiveness 0.001': dict(art_effectiveness=0.001),
    }

    # Store flattened results for each scenario
    flat_results = {}

    for name, hiv_pars in scenarios.items():
        print(f'Running scenario: {name}')
        sim = build_tbhiv_sim(hiv_pars=hiv_pars)
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
            # ax.set_yscale('log')   
            ax.set_ylim(0, max(result.values) * 1.1) 
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
            leg = ax.legend(loc='upper right')
        # Handle legend positioning for crowded plots
        if leg:
            leg.get_frame().set_alpha(0.5)
    # Handle x-axis label for the last subplot
    if n_metrics > 0:
        # Only set xlabel for the last subplot in the grid
        if isinstance(axs, np.ndarray):
            axs[-1].set_xlabel('Time')
        else:
            # for single ax
            axs.set_xlabel('Time')
    else:
        # In case there are no metrics to plot, set xlabel on axs
        if isinstance(axs, np.ndarray):
            axs.set_xlabel('Time')
        else:
            axs.set_xlabel('Time')
    # Ensure the layout is tight to avoid overlap
    if n_metrics > 0:
        if isinstance(axs, np.ndarray):
            axs[-1].set_xlabel('Time')  # Ensure the last subplot has the x-label
        else:
            axs.set_xlabel('Time')
    else:
        # In case there are no metrics to plot, set xlabel on axs
        axs.set_xlabel('Time')
    if n_metrics > 0:
        plt.tight_layout()
    else:
        # In case no metrics were plotted, ensure layout is tight
        plt.tight_layout()
    # Final layout adjustments
    if n_metrics > 0:
        plt.tight_layout()
    else:
        plt.tight_layout()  # Ensure layout is tight even with no metrics

    axs[-1].set_xlabel('Time')
    
    # add an option to change the background color of the plot for better visibility
    for ax in axs:
        # Change the background color of each axis for better visibility
        ax.set_facecolor('#f0f0f0')  # Light gray background for better contrast
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    flat_results = run_scenarios()
    plot_results(flat_results)
    
