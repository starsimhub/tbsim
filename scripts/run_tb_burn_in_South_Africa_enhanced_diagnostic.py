"""
Enhanced TB Diagnostic Integration with South Africa Burn-in Script

This script demonstrates how to integrate the new EnhancedTBDiagnostic intervention
with the existing South Africa calibration framework, showing how to use the
detailed parameter stratification from interventions_updated.py while maintaining
compatibility with the health-seeking and treatment cascade.
"""

import numpy as np
import starsim as ss
import tbsim as mtb
import matplotlib.pyplot as plt
from tbsim.interventions.enhanced_tb_diagnostic import EnhancedTBDiagnostic, create_enhanced_diagnostic_scenarios


def run_enhanced_diagnostic_scenario(scenario_name, diagnostic_params, 
                                   beta=0.025, rel_sus_latentslow=0.15, 
                                   tb_mortality=3.0e-4, seed=0, years=200, n_agents=1000):
    """
    Run a simulation with enhanced diagnostic intervention.
    
    Parameters:
    -----------
    scenario_name : str
        Name of the diagnostic scenario (e.g., 'baseline', 'oral_swab', 'fujilam')
    diagnostic_params : dict
        Parameters for the enhanced diagnostic intervention
    beta : float
        TB transmission rate
    rel_sus_latentslow : float
        Relative susceptibility of latent slow individuals
    tb_mortality : float
        TB mortality rate (used to set rate_smpos_to_dead)
    seed : int
        Random seed
    years : int
        Number of years to simulate
    n_agents : int
        Number of agents in simulation
    """
    
    # Set random seed
    np.random.seed(seed)
    
    # Create people with extra states
    people = ss.People(n_agents=n_agents, extra_states=mtb.get_extrastates())
    
    # Create TB disease with parameters
    tb_params = {
        'init_prev': ss.bernoulli(0.25),
        'beta': beta,
        'rel_sus_latentslow': rel_sus_latentslow,
        'rate_smpos_to_dead': ss.perday(tb_mortality),
        'rate_exptb_to_dead': ss.perday(0.15 * tb_mortality),
        'rate_smneg_to_dead': ss.perday(0.3 * tb_mortality),
    }
    tb = mtb.TB(tb_params)
    
    # Create HIV disease (if needed for FujiLAM)
    hiv = None
    if diagnostic_params.get('use_fujilam', False):
        from tbsim.comorbidities.hiv.hiv import HIV
        hiv = HIV({'init_prev': ss.bernoulli(0.15)})
    
    # Create diseases list
    diseases = [tb]
    if hiv is not None:
        diseases.append(hiv)
    
    # Create interventions
    interventions = [
        # Health-seeking behavior
        mtb.HealthSeekingBehavior(pars={
            'initial_care_seeking_rate': ss.perday(0.25)
        }),
        
        # Enhanced diagnostic intervention
        EnhancedTBDiagnostic(pars=diagnostic_params),
        
        # Treatment intervention
        mtb.TBTreatment(pars={
            'treatment_success_rate': 0.85,
            'reseek_multiplier': 2.0,
            'reset_flags': True,
        }),
    ]
    
    # Create network
    network = ss.RandomNet({'n_contacts': ss.poisson(lam=2), 'dur': 0})
    
    # Create simulation
    sim = ss.Sim(
        people=people,
        diseases=diseases,
        interventions=interventions,
        networks=network,
        pars=dict(start=1850, stop=1850+years, dt=1/12),  # Monthly timesteps
    )
    
    # Run simulation
    sim.run()
    
    return sim


def compare_diagnostic_scenarios():
    """
    Compare different diagnostic scenarios using the enhanced intervention.
    """
    
    # Get predefined scenarios
    scenarios = create_enhanced_diagnostic_scenarios()
    
    # Add some custom scenarios
    scenarios.update({
        'high_sensitivity': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': False,
            'sensitivity_adult_smearpos': 0.95,
            'sensitivity_adult_smearneg': 0.85,
            'sensitivity_child': 0.80,
        },
        'low_sensitivity': {
            'use_oral_swab': False,
            'use_fujilam': False,
            'use_cadcxr': False,
            'sensitivity_adult_smearpos': 0.70,
            'sensitivity_adult_smearneg': 0.60,
            'sensitivity_child': 0.50,
        }
    })
    
    # Run simulations for each scenario
    results = {}
    
    for scenario_name, diagnostic_params in scenarios.items():
        print(f"Running scenario: {scenario_name}")
        
        # Add common parameters
        diagnostic_params.update({
            'coverage': ss.bernoulli(0.8, strict=False),
            'care_seeking_multiplier': 2.0,
        })
        
        sim = run_enhanced_diagnostic_scenario(
            scenario_name=scenario_name,
            diagnostic_params=diagnostic_params,
            beta=0.025,
            rel_sus_latentslow=0.15,
            tb_mortality=3.0e-4,  # This will be converted to proper rate parameters
            seed=0,
            years=50,  # Shorter for comparison
            n_agents=1000
        )
        
        results[scenario_name] = sim
    
    return results


def plot_diagnostic_comparison(results):
    """
    Plot comparison of different diagnostic scenarios.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced TB Diagnostic Scenarios Comparison', fontsize=16)
    
    # Colors for different scenarios
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (scenario_name, sim) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # Get diagnostic results
        if 'enhancedtbdiagnostic' in sim.results:
            diag_results = sim.results['enhancedtbdiagnostic']
            
            # Plot 1: Cumulative positive tests
            axes[0, 0].plot(diag_results['cum_test_positive'].timevec, 
                           diag_results['cum_test_positive'].values, 
                           label=scenario_name, color=color, linewidth=2)
            
            # Plot 2: Diagnostic methods used
            if 'n_xpert_baseline' in diag_results:
                axes[0, 1].plot(diag_results['n_xpert_baseline'].timevec, 
                               diag_results['n_xpert_baseline'].values, 
                               label=f'{scenario_name} (Xpert)', color=color, linestyle='-')
            if 'n_oral_swab' in diag_results:
                axes[0, 1].plot(diag_results['n_oral_swab'].timevec, 
                               diag_results['n_oral_swab'].values, 
                               label=f'{scenario_name} (Oral)', color=color, linestyle='--')
            if 'n_fujilam' in diag_results:
                axes[0, 1].plot(diag_results['n_fujilam'].timevec, 
                               diag_results['n_fujilam'].values, 
                               label=f'{scenario_name} (FujiLAM)', color=color, linestyle=':')
        
        # Plot 3: Active TB prevalence
        if 'tb' in sim.results:
            tb_results = sim.results['tb']
            axes[0, 2].plot(tb_results['n_active'].timevec, 
                           tb_results['n_active'].values / len(sim.people) * 100, 
                           label=scenario_name, color=color, linewidth=2)
        
        # Plot 4: Treatment outcomes
        if 'tbtreatment' in sim.results:
            tx_results = sim.results['tbtreatment']
            axes[1, 0].plot(tx_results['cum_treatment_success'].timevec, 
                           tx_results['cum_treatment_success'].values, 
                           label=f'{scenario_name} (Success)', color=color, linestyle='-')
            axes[1, 0].plot(tx_results['cum_treatment_failure'].timevec, 
                           tx_results['cum_treatment_failure'].values, 
                           label=f'{scenario_name} (Failure)', color=color, linestyle='--')
        
        # Plot 5: Health-seeking behavior
        if 'healthseekingbehavior' in sim.results:
            hsb_results = sim.results['healthseekingbehavior']
            axes[1, 1].plot(hsb_results['new_sought_care'].timevec, 
                           hsb_results['new_sought_care'].values, 
                           label=scenario_name, color=color, linewidth=2)
        
        # Plot 6: Latent TB prevalence
        if 'tb' in sim.results:
            tb_results = sim.results['tb']
            # Calculate total latent from slow and fast components
            latent_slow = tb_results['n_latent_slow'].values
            latent_fast = tb_results['n_latent_fast'].values
            latent_total = latent_slow + latent_fast
            axes[1, 2].plot(tb_results['n_latent_slow'].timevec, 
                           latent_total / len(sim.people) * 100, 
                           label=scenario_name, color=color, linewidth=2)
    
    # Set labels and titles
    axes[0, 0].set_title('Cumulative Positive Tests')
    axes[0, 0].set_ylabel('Cumulative Tests')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title('Diagnostic Methods Used')
    axes[0, 1].set_ylabel('Number of Tests')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[0, 2].set_title('Active TB Prevalence')
    axes[0, 2].set_ylabel('Prevalence (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    axes[1, 0].set_title('Cumulative Treatment Outcomes')
    axes[1, 0].set_ylabel('Cumulative Treatments')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].set_title('New Health-Seeking Behavior')
    axes[1, 1].set_ylabel('New Seekers')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    axes[1, 2].set_title('Latent TB Prevalence')
    axes[1, 2].set_ylabel('Prevalence (%)')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    # Set x-axis labels
    for ax in axes.flat:
        ax.set_xlabel('Time (years)')
    
    plt.tight_layout()
    plt.show()


def print_scenario_summary(results):
    """
    Print summary statistics for each scenario.
    """
    
    print("\n" + "="*80)
    print("ENHANCED DIAGNOSTIC SCENARIOS SUMMARY")
    print("="*80)
    
    for scenario_name, sim in results.items():
        print(f"\n{scenario_name.upper()}:")
        print("-" * 40)
        
        # Diagnostic results
        if 'enhancedtbdiagnostic' in sim.results:
            diag = sim.results['enhancedtbdiagnostic']
            print(f"  Total tests: {np.sum(diag['n_tested'].values)}")
            print(f"  Total positive: {np.sum(diag['n_test_positive'].values)}")
            print(f"  Total negative: {np.sum(diag['n_test_negative'].values)}")
            
            # Diagnostic methods breakdown
            if 'n_xpert_baseline' in diag:
                print(f"  Xpert baseline tests: {np.sum(diag['n_xpert_baseline'].values)}")
            if 'n_oral_swab' in diag:
                print(f"  Oral swab tests: {np.sum(diag['n_oral_swab'].values)}")
            if 'n_fujilam' in diag:
                print(f"  FujiLAM tests: {np.sum(diag['n_fujilam'].values)}")
            if 'n_cadcxr' in diag:
                print(f"  CAD CXR tests: {np.sum(diag['n_cadcxr'].values)}")
        
        # Treatment results
        if 'tbtreatment' in sim.results:
            tx = sim.results['tbtreatment']
            print(f"  Total treated: {np.sum(tx['n_treated'].values)}")
            print(f"  Treatment successes: {np.sum(tx['n_treatment_success'].values)}")
            print(f"  Treatment failures: {np.sum(tx['n_treatment_failure'].values)}")
        
        # TB prevalence - use correct keys
        if 'tb' in sim.results:
            tb = sim.results['tb']
            final_active = tb['n_active'].values[-1] / len(sim.people) * 100
            
            # Calculate latent prevalence from separate slow and fast components
            final_latent_slow = tb['n_latent_slow'].values[-1] / len(sim.people) * 100
            final_latent_fast = tb['n_latent_fast'].values[-1] / len(sim.people) * 100
            final_latent_total = final_latent_slow + final_latent_fast
            
            print(f"  Final active TB prevalence: {final_active:.2f}%")
            print(f"  Final latent TB prevalence (total): {final_latent_total:.2f}%")
            print(f"    - Latent slow: {final_latent_slow:.2f}%")
            print(f"    - Latent fast: {final_latent_fast:.2f}%")
        
        # Health-seeking behavior
        if 'healthseekingbehavior' in sim.results:
            hsb = sim.results['healthseekingbehavior']
            total_sought = np.sum(hsb['new_sought_care'].values)
            print(f"  Total who sought care: {total_sought}")


if __name__ == '__main__':
    """
    Main execution: Run comparison of enhanced diagnostic scenarios.
    """
    
    print("Running Enhanced TB Diagnostic Scenarios Comparison...")
    
    # Run all scenarios
    results = compare_diagnostic_scenarios()
    
    # Print summary
    print_scenario_summary(results)
    
    # Plot comparison
    plot_diagnostic_comparison(results)
    
    print("\nEnhanced diagnostic scenarios completed!")
    print("\nKey Features Demonstrated:")
    print("1. Age and TB state-specific sensitivity/specificity")
    print("2. HIV-stratified parameters for FujiLAM")
    print("3. Integration with health-seeking behavior")
    print("4. False negative handling with care-seeking multipliers")
    print("5. Comprehensive result tracking by diagnostic method")
    print("6. Compatibility with existing treatment cascade") 