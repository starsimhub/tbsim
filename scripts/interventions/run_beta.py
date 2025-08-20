import scenarios as scen
from scripts.interventions.functions import build_sim as _fn_build_sim
import tbsim.utils.plots as pl


def run_beta_scenarios():
    """Run only scenarios that have beta interventions and plot results."""
    scenarios = scen.Scenarios
    results = {}
    
    # Find and run scenarios with beta interventions
    for scenario_name, scenario_config in scenarios.items():
        if 'betabyyear' in scenario_config:
            print(f"Running: {scenario_name}")
            print(f"Beta params: {scenario_config['betabyyear']}")
            
            try:
                sim = _fn_build_sim(scenario=scenario_config)
                sim.run()
                results[scenario_name] = sim.results.flatten()
                print(f"✓ Completed: {scenario_name}\n")
            except Exception as e:
                print(f"✗ Failed: {scenario_name} - {e}\n")
                
    # run baseline
    sim = _fn_build_sim(scenario=scenarios.baseline)
    sim.run()
    results['baseline'] = sim.results.flatten()
    
    # Plot combined results
    if results:
        print("Plotting combined results...")
        pl.plot_combined(results, dark=True, cmap='viridis')
        print("✓ Plots completed")
    else:
        print("No results to plot")


if __name__ == "__main__":
    run_beta_scenarios()








