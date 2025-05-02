import starsim as ss 
import tbsim as mtb 
import numpy as np
import matplotlib.pyplot as plt
import sciris as sc


#- - - - - - MAKE INTERVENTIONS - - - - - -
def make_interventions(include:bool=True, pars=None):
    if not include: return None
    if pars is None:
        pars=dict(
                mode='both',
                prevalence=0.30,
                percent_on_ART=0.5,
                minimum_age=15,
                max_age=49,)
        
    return [mtb.HivInterventions(pars=pars),]
    
# - - - - - - MAKE HIV - - - - - -
def make_hiv(include:bool=True, hiv_pars=None):
    if hiv_pars is None:
        hiv_pars = dict(
            init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
            init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
        )
    return mtb.HIV(pars=hiv_pars)

#- - - - - - - - MAKE TB - - - - - - - -
def make_tb(include:bool=True, tb_pars=None):
    if tb_pars is None:
        pars = dict(
            beta=ss.beta(0.1),
            init_prev=ss.bernoulli(p=0.25),
            rel_sus_latentslow=0.1,
        )
    return mtb.TB(pars=pars)

# - - - - - - - MAKE TB-HIV CONNECTOR - - - - - -
def make_tb_hiv_connector(include:bool=True, pars=None):
    if not include: return None
    return mtb.TB_HIV_Connector(pars=pars)  

# - - - - - -  MAKE DEMOGRAPHICS - - - - - -
def make_demographics(include:bool=False):
    if not include: return None
    return [ss.Births(pars=dict(birth_rate=8.4)),
            ss.Deaths(pars=dict(death_rate=8.4)),]


# - - - - - -  PLOT RESULTS - - - - - -
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
        
        leg = ax.legend(loc='upper right', fontsize=(6 if len(flat_results) <= 5 else 5))
        
        if leg:
            leg.get_frame().set_alpha(0.5)
            
    plt.tight_layout()
    for ax in axs:
        ax.set_facecolor('#f0f0f0')  # Light gray background for better contrast
    dirname = sc.thisdir()
    plt.savefig(f'{dirname}/scenarios.png', dpi=300)
    plt.show()


def uncertanty_plot():
    import matplotlib.pyplot as plt
    import numpy as np

    # Example data generation
    np.random.seed(0)
    timesteps = np.arange(8)  # Number of timepoints

    # Simulate multiple runs for two groups (blue and orange)
    n_runs = 5

    # Create fake datasets
    blue_data = [np.random.rand(len(timesteps)) + np.linspace(1, 2, len(timesteps)) for _ in range(n_runs)]
    orange_data = [np.random.rand(len(timesteps)) + np.linspace(0.5, 1.5, len(timesteps)) for _ in range(n_runs)]

    # Organize data
    variables = ['n_positive_smpos', 'n_positive_smneg', 'n_positive_via_LS', 'n_positive_via_LF_dur']

    # Setup plot
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    linestyles = ['-.', ':', '--']  # Various styles

    for idx, var in enumerate(variables):
        ax = axs[idx // 2, idx % 2]

        # Plot blue group
        for run_idx, run in enumerate(blue_data):
            ax.plot(timesteps, run + np.random.normal(0, 0.2, size=len(timesteps)), 
                    linestyle=linestyles[run_idx % len(linestyles)], 
                    color='steelblue', alpha=0.9)
            ax.fill_between(timesteps,
                            run - 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            run + 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            color='steelblue', alpha=0.3)

        # Plot orange group
        for run_idx, run in enumerate(orange_data):
            ax.plot(timesteps, run + np.random.normal(0, 0.2, size=len(timesteps)),
                    linestyle=linestyles[run_idx % len(linestyles)], 
                    color='darkorange', alpha=0.9)
            ax.fill_between(timesteps,
                            run - 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            run + 0.5 + np.random.normal(0, 0.1, size=len(timesteps)),
                            color='darkorange', alpha=0.3)

        ax.set_title(var, fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlim(timesteps[0], timesteps[-1])
        ax.set_ylim(0, None)

    plt.tight_layout()
    plt.show()

