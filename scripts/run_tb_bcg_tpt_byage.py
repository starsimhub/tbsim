import tbsim as mtb
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
import pprint as pprint
import pandas as pd
import numpy as np
from tbsim.utils.plots import plot_household_structure, plot_household_network_analysis

# Simple default parameters
DEFAULT_SPARS = dict(
    unit='day',
    dt=7,
    start=sc.date('1975-01-01'),
    stop=sc.date('2030-12-31'),
    rand_seed=123,
    verbose=0,
)

DEFAULT_TBPARS = dict(
    beta=ss.rate_prob(0.0025),
    init_prev=ss.bernoulli(p=0.25),
    unit='day',
    dt=7,      
    start=sc.date('1975-02-01'),
    stop=sc.date('2030-12-31'),
)

# Simple age distribution
age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Skewed toward younger ages
})

def create_sample_households(n_agents=500):
    """Create sample household structure for testing."""
    households = []
    current_uid = 0
    while current_uid < n_agents:
        household_size = min(np.random.randint(3, 8), n_agents - current_uid)
        if household_size < 2:
            break
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households

def build_sim(scenario=None, spars=None, show_household_plot=False, household_plot_type='basic'):
    """
    Build and return a complete Starsim-based simulation instance for TB modeling,
    incorporating optional interventions and user-defined parameters.

    Args:
        scenario (dict, optional): A dictionary defining scenario-specific components,
            such as intervention parameters and TB simulation settings. Expected keys:
                - 'tbpars' (dict): TB-specific simulation parameters.
                - 'tptintervention' (dict, optional): Parameters for TPT intervention.
                - 'bcgintervention' (dict, optional): Parameters for BCG intervention.
        spars (dict, optional): General simulation parameters (e.g., timestep, duration).
            These override values in the DEFAULT_SPARS global dictionary.

    Returns:
        ss.Sim: A fully initialized simulation object containing:
            - A population (`People`) with TB-related extra states.
            - A TB disease module initialized with merged parameters.
            - A list of social and household network layers.
            - Optional interventions (TPT, BCG or Beta) as defined by the scenario.
            - Demographic processes like births and deaths.
            - Core simulation parameters merged from defaults and user inputs.

    Notes:
        - If no parameters are provided, it will use the default values of the participating
          simulation components.
    
    Example:
        sim = build_sim(scenario=my_scenario, spars={'n_steps': 200})
        sim.run()
    """
    scenario = scenario or {}
    
    # Merge parameters
    spars = {**DEFAULT_SPARS, **(spars or {})}
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}
    
    # Create interventions list
    interventions = []
    
    # Add BCG interventions (can be single or multiple)
    bcg_params = scenario.get('bcgintervention')
    if bcg_params:
        if isinstance(bcg_params, dict):
            # Single BCG intervention
            interventions.append(mtb.BCGProtection(pars=bcg_params))
        elif isinstance(bcg_params, list):
            # Multiple BCG interventions
            for i, params in enumerate(bcg_params):
                params['name'] = f'BCG_{i}'  # Give unique name
                interventions.append(mtb.BCGProtection(pars=params))
    
    # Add TPT interventions (can be single or multiple)
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        if isinstance(tpt_params, dict):
            # Single TPT intervention
            interventions.append(mtb.TPTInitiation(pars=tpt_params))
        elif isinstance(tpt_params, list):
            # Multiple TPT interventions
            for i, params in enumerate(tpt_params):
                params['name'] = f'TPT_{i}'  # Give unique name
                interventions.append(mtb.TPTInitiation(pars=params))
    
    # Add Beta interventions (can be single or multiple)
    beta_params = scenario.get('betabyyear')
    if beta_params:
        if isinstance(beta_params, dict):
            # Single Beta intervention
            interventions.append(mtb.BetaByYear(pars=beta_params))
        elif isinstance(beta_params, list):
            # Multiple Beta interventions
            for i, params in enumerate(beta_params):
                params['name'] = f'Beta_{i}'  # Give unique name
                interventions.append(mtb.BetaByYear(pars=params))
    
    # Create simulation components
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    tb = mtb.TB(pars=tbpars)
    
    # Create household structure for HouseholdNetGeneric
    households = create_sample_households(500)
    
    # Show household plot if requested
    if show_household_plot:
        # Initialize people object for plotting
        temp_sim = ss.Sim(
            people=pop,
            networks=[],
            diseases=[],
            pars=spars,
        )
        temp_sim.init()
        
        print(f"\nGenerating household plot (type: {household_plot_type})...")
        if household_plot_type == 'basic':
            plot_household_structure(
                households=households,
                people=pop,
                title="HouseholdNetGeneric Structure",
                show_household_ids=True,
                show_agent_ids=False,
                max_households_to_show=30,
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        elif household_plot_type == 'analysis':
            plot_household_network_analysis(
                households=households,
                people=pop,
                figsize=(15, 10),
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        elif household_plot_type == 'both':
            # Show both plots
            plot_household_structure(
                households=households,
                people=pop,
                title="HouseholdNetGeneric Structure",
                show_household_ids=True,
                show_agent_ids=False,
                max_households_to_show=30,
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
            plot_household_network_analysis(
                households=households,
                people=pop,
                figsize=(15, 10),
                dark=True,
                savefig=True,
                outdir='results/household_plots'
            )
        print("Household plot generation completed!")
    
    networks = [
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        mtb.HouseholdNet(),
        mtb.HouseholdNetGeneric(hhs=households, pars={'add_newborns': True})
    ]
    
    # Create and return simulation
    return ss.Sim(
        people=pop,
        networks=networks,
        interventions=interventions,
        diseases=[tb],
        pars=spars,
    )

def get_scenarios():
    """ HELP
    Define a set of simulation scenarios for evaluating TB interventions.

    Returns:
        dict: A dictionary where each key is the name of a scenario and the value is 
        a dictionary of simulation parameters. Each scenario may include:
            - 'name' (str): A human-readable scenario name.
            - 'tbpars' (dict, optional): Parameters controlling the simulation timeframe.
            - 'bcgintervention' (dict, optional): BCG vaccine intervention settings.
            - 'tptintervention' (dict, optional): Tuberculosis Preventive Therapy settings.
            - 'betabyyear' : (dict, optional): For changing the value of beta during the same simulation period.
    
    Scenarios included:
        - 'Baseline': No intervention, default simulation window.
        - 'BCG': BCG vaccination with 90% coverage.
        - 'TPT': TPT with full eligibility, conditional on HIV status.
    """
    
    return {
        'Baseline': {
            'name': 'No interventions',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        },
        'TPT with Household Network': {
            'name': 'TPT intervention with optimized household network',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
            'tptintervention': dict(
                p_tpt=0.8,
                age_range=[0, 100],
                hiv_status_threshold=False,
                tpt_treatment_duration=ss.peryear(0.25),  # 3 months
                tpt_protection_duration=ss.peryear(2.0),  # 2 years
                start=sc.date('1980-01-01'),
                stop=sc.date('2020-12-31'),
            ),
        },
        # 'Baseline and BetaByYear': {
        #     'name': 'No interventions',
        #     'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        #     'betabyyear':dict(years=[1990, 2000], x_beta=[0.5, 1.4])
        # },
        'Single BCG': {
            'name': 'Single BCG intervention',
            'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
            'bcgintervention': dict(
                coverage=0.8,
                start=sc.date('1980-01-01'),
                stop=sc.date('2020-12-31'),
                age_range=[1, 5],
            ),
        },
        
        # 'Multiple BCG': {
        #     'name': 'Multiple BCG interventions',
        #     'tbpars': dict(start=sc.date('1975-01-01'), stop=sc.date('2030-12-31')),
        #     'bcgintervention': [
        #         dict(
        #             coverage=0.9,
        #             start=sc.date('1980-01-01'),
        #             stop=sc.date('2020-12-31'),
        #             age_range=[0, 2],           # For children
        #         ),
        #         dict(
        #             coverage=0.3,
        #             start=sc.date('1985-01-01'),
        #             stop=sc.date('2015-12-31'),
        #             age_range=[25, 40],         # For adults
        #         ),
        #     ],
        # },
    }

def run_scenarios(plot=True, show_household_plot=False, household_plot_type='basic', age_bins=None):
    """Run all scenarios and optionally plot results with age stratification."""
    import tbsim.utils.plots as pl
    
    scenarios = get_scenarios()
    results = {}
    simulations = {}  # Store simulation objects for age stratification
    
    for name, scenario in scenarios.items():
        print(f"\nRunning: {name}")
        sim = build_sim(scenario=scenario, show_household_plot=show_household_plot, household_plot_type=household_plot_type)
        sim.run()
        results[name] = sim.results.flatten()
        simulations[name] = sim  # Store simulation object
    
    if plot:
        # Traditional plotting (overall population)
        print("\nGenerating traditional plots (overall population)...")
        pl.plot_combined(results, 
                        heightfold=2, outdir='results/interventions')
        
        # Age-stratified plotting if age_bins provided
        if age_bins is not None:
            print(f"\nGenerating age-stratified plots with age bins: {age_bins}")
            
            # Generate age-stratified results for each scenario
            age_stratified_results = {}
            
            for name, sim in simulations.items():
                print(f"  Processing age stratification for: {name}")
                
                # Generate age-stratified results
                stratified = pl._generate_age_stratified_results(
                    sim, sim.results.flatten(), age_bins
                )
                
                # Add scenario prefix to age bin names
                for age_bin, metrics in stratified.items():
                    age_stratified_results[f"{name}_{age_bin}"] = metrics
            
            # Create age bin labels for display
            age_labels = []
            for i in range(len(age_bins) - 1):
                if age_bins[i+1] == float('inf'):
                    age_labels.append(f"{int(age_bins[i])}+")
                else:
                    age_labels.append(f"{int(age_bins[i])}-{int(age_bins[i+1])}")
            
            print(f"  Age groups: {age_labels}")
            
            # Plot 1: TB Prevalence by age group
            print("  Plot 1: TB Prevalence by Age Group")
            pl.plot_results(
                flat_results=age_stratified_results,
                keywords=['prevalence_active'],
                n_cols=2,
                dark=True,
                cmap='viridis',
                savefig=True,
                outdir='results/interventions/age_stratified'
            )
            
            # Plot 2: TB Incidence by age group
            print("  Plot 2: TB Incidence by Age Group")
            pl.plot_results(
                flat_results=age_stratified_results,
                keywords=['incidence_kpy'],
                n_cols=2,
                dark=True,
                cmap='plasma',
                savefig=True,
                outdir='results/interventions/age_stratified'
            )
            
            # Plot 3: Latent TB by age group
            print("  Plot 3: Latent TB by Age Group")
            pl.plot_results(
                flat_results=age_stratified_results,
                keywords=['n_latent_slow', 'n_latent_fast'],
                n_cols=2,
                dark=True,
                cmap='Blues',
                savefig=True,
                outdir='results/interventions/age_stratified'
            )
            
            # Plot 4: All metrics comparison by age
            print("  Plot 4: All Metrics Comparison by Age")
            pl.plot_results(
                flat_results=age_stratified_results,
                keywords=['prevalence_active', 'incidence_kpy', 'n_latent_slow', 'n_latent_fast'],
                n_cols=2,
                dark=False,
                cmap='tab20',
                savefig=True,
                outdir='results/interventions/age_stratified'
            )
            
            # Plot 5: Scenario comparison by age group
            print("  Plot 5: Scenario Comparison by Age Group")
            pl.plot_combined(
                age_stratified_results,
                heightfold=2,
                outdir='results/interventions/age_stratified',
                filter=['prevalence_active', 'incidence_kpy']
            )
            
            print("✓ All age-stratified plots generated and saved to results/interventions/age_stratified/")
        
        plt.show()
    
    return results, simulations


def test_household_plots_only():
    """Test only the household plotting functionality without running full simulations."""
    print("Testing HouseholdNetGeneric Plotting Functionality")
    print("=" * 50)
    
    # Create a simple simulation just for testing household plots
    sim = build_sim(show_household_plot=True, household_plot_type='both')
    
    print("\nHousehold plot testing completed!")
    print("Check the 'results/household_plots' directory for saved figures.")


def analyze_age_stratified_results(results, simulations, age_bins):
    """
    Analyze age-stratified results and provide summary statistics.
    
    Args:
        results: Dictionary of scenario results
        simulations: Dictionary of simulation objects
        age_bins: List of age bin boundaries
    """
    import tbsim.utils.plots as pl
    
    print("\n" + "="*60)
    print("AGE-STRATIFIED ANALYSIS SUMMARY")
    print("="*60)
    
    # Create age bin labels
    age_labels = []
    for i in range(len(age_bins) - 1):
        if age_bins[i+1] == float('inf'):
            age_labels.append(f"{int(age_bins[i])}+")
        else:
            age_labels.append(f"{int(age_bins[i])}-{int(age_bins[i+1])}")
    
    print(f"Age groups analyzed: {age_labels}")
    print()
    
    # Analyze each scenario
    for scenario_name, sim in simulations.items():
        print(f"Scenario: {scenario_name}")
        print("-" * 40)
        
        # Get overall results
        overall_results = results[scenario_name]
        final_prevalence = overall_results['tb_prevalence_active'].values[-1]
        final_incidence = overall_results['tb_incidence_kpy'].values[-1]
        
        print(f"Overall population (final year):")
        print(f"  TB Prevalence: {final_prevalence:.4f}")
        print(f"  TB Incidence: {final_incidence:.2f} per 1000 person-years")
        print()
        
        # Get age-stratified results
        stratified = pl._generate_age_stratified_results(
            sim, sim.results.flatten(), age_bins
        )
        
        print("Age-stratified results (final year):")
        for age_bin in age_labels:
            if age_bin in stratified:
                metrics = stratified[age_bin]
                if 'tb_prevalence_active' in metrics:
                    prev = metrics['tb_prevalence_active'].values[-1]
                    print(f"  Age {age_bin}: Prevalence = {prev:.4f}")
        
        print()
    
    # Compare scenarios
    print("Scenario Comparison:")
    print("-" * 40)
    
    baseline_name = 'Baseline'
    if baseline_name in results:
        baseline_prev = results[baseline_name]['tb_prevalence_active'].values[-1]
        baseline_inc = results[baseline_name]['tb_incidence_kpy'].values[-1]
        
        for scenario_name, scenario_results in results.items():
            if scenario_name != baseline_name:
                scenario_prev = scenario_results['tb_prevalence_active'].values[-1]
                scenario_inc = scenario_results['tb_incidence_kpy'].values[-1]
                
                prev_reduction = ((baseline_prev - scenario_prev) / baseline_prev) * 100
                inc_reduction = ((baseline_inc - scenario_inc) / baseline_inc) * 100 if baseline_inc > 0 else 0
                
                print(f"{scenario_name} vs Baseline:")
                print(f"  Prevalence reduction: {prev_reduction:+.1f}%")
                print(f"  Incidence reduction: {inc_reduction:+.1f}%")
                print()
    
    print("="*60)


if __name__ == '__main__':
    # Define age bins for analysis
    age_bins = [0,5, 15, 200]  # 0-2, 2-5, 5-10, 10-15, 15+ years
    
    print("TB BCG TPT Interventions with Age-Stratified Analysis")
    print("=" * 60)
    print(f"Age bins: {age_bins}")
    print("Age groups: 0-5, 10-15, 15+ years")
    print()
    
    # Run all scenarios with household plot option and age stratification
    # Options for household_plot_type: 'basic', 'analysis', 'both', or None
    # Set show_household_plot=False to disable household plotting
    results, simulations = run_scenarios(
        plot=True, 
        show_household_plot=False, 
        household_plot_type='analysis',
        age_bins=age_bins
    )
    
    # Analyze age-stratified results
    analyze_age_stratified_results(results, simulations, age_bins)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETED")
    print("="*60)
    print("Generated plots:")
    print("1. Traditional plots (overall population) -> results/interventions/")
    print("2. Age-stratified plots -> results/interventions/age_stratified/")
    print("   - TB Prevalence by Age Group")
    print("   - TB Incidence by Age Group") 
    print("   - Latent TB by Age Group")
    print("   - All Metrics Comparison by Age")
    print("   - Scenario Comparison by Age Group")
    print()
    print("Age-stratified analysis reveals intervention effectiveness across different age groups!")
    print("Key insights:")
    print("- Compare intervention effectiveness across age groups")
    print("- Identify age-specific intervention impacts")
    print("- Understand demographic patterns in TB transmission")
    print("- Optimize intervention targeting by age")
