"""
Generalized TB Model Calibration Framework

This script provides a generalized framework for TB model calibration that can be
used with different countries, data sources, and calibration scenarios. It uses
the centralized utilities from tbsim.calibration and tbsim.plotting for maximum reusability and flexibility.
"""

import numpy as np
import pandas as pd
import datetime
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import centralized utilities
from tbsim.calibration import (
    CalibrationData,
    CalibrationTarget,
    create_south_africa_data,
    create_country_data,
    calculate_calibration_score,
    create_calibration_report,
    run_calibration_simulation_suite,
    SimulationConfig,
    DiseaseConfig,
    InterventionConfig,
)
from tbsim.plotting import CalibrationPlotter
from tbsim.simulation.factory import (
    make_tb,
    make_hiv,
    make_tb_hiv_connector,
    make_hiv_interventions
)


def create_vietnam_data():
    """
    Create synthetic Vietnam TB data for calibration (example for different country)
    
    Returns:
        CalibrationData: Calibration data object for Vietnam
    """
    
    # Vietnam case notification data (example)
    case_notification_data = {
        'year': [2000, 2005, 2010, 2015, 2020],
        'rate_per_100k': [450, 520, 580, 520, 380],  # Different trend than South Africa
        'total_cases': [180000, 220000, 250000, 230000, 180000],
        'source': ['WHO Global TB Report'] * 5
    }
    
    # Vietnam age-stratified prevalence (example)
    age_prevalence_data = {
        'age_group': ['15-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'prevalence_per_100k': [650, 900, 1100, 1200, 1300, 1500],  # Different pattern
        'prevalence_percent': [0.65, 0.90, 1.10, 1.20, 1.30, 1.50],
        'sample_size': [4000, 3500, 3000, 2500, 2000, 1500],
        'source': ['Vietnam TB Prevalence Survey 2018'] * 6
    }
    
    # Vietnam calibration targets
    targets = {
        'overall_prevalence_2018': CalibrationTarget(
            name='overall_prevalence_2018',
            value=0.65,  # Different from South Africa
            year=2018,
            description='Overall TB prevalence from 2018 survey',
            source='Vietnam TB Prevalence Survey 2018'
        ),
        'hiv_coinfection_rate': CalibrationTarget(
            name='hiv_coinfection_rate',
            value=0.15,  # Much lower than South Africa
            description='HIV coinfection rate among TB cases',
            source='WHO Global TB Report'
        ),
        'case_detection_rate': CalibrationTarget(
            name='case_detection_rate',
            value=0.75,  # Higher than South Africa
            description='Case detection rate',
            source='WHO Global TB Report'
        ),
        'treatment_success_rate': CalibrationTarget(
            name='treatment_success_rate',
            value=0.85,  # Higher than South Africa
            description='Treatment success rate',
            source='WHO Global TB Report'
        ),
        'mortality_rate': CalibrationTarget(
            name='mortality_rate',
            value=0.08,  # Lower than South Africa
            description='Case fatality rate',
            source='WHO Global TB Report'
        )
    }
    
    return CalibrationData(
        case_notifications=pd.DataFrame(case_notification_data),
        age_prevalence=pd.DataFrame(age_prevalence_data),
        targets=targets,
        country='Vietnam',
        description='TB model calibration data for Vietnam'
    )


def run_calibration_analysis(country_name="South Africa", custom_data=None, 
                           disease_config=None, intervention_config=None, 
                           sim_config=None, save_results=True):
    """
    Run a complete calibration analysis for any country
    
    Args:
        country_name: Name of the country
        custom_data: Custom CalibrationData object (if None, uses default for country)
        disease_config: Custom disease configuration
        intervention_config: Custom intervention configuration
        sim_config: Custom simulation configuration
        save_results: Whether to save results to files
    
    Returns:
        tuple: (sim, calibration_data, report, plotter)
    """
    
    print(f"=== TB Model Calibration Analysis for {country_name} ===")
    
    # Create calibration data
    if custom_data is None:
        if country_name.lower() == "south africa":
            calibration_data = create_south_africa_data()
        elif country_name.lower() == "vietnam":
            calibration_data = create_vietnam_data()
        else:
            # Create generic data structure - user should provide custom_data
            raise ValueError(f"No default data available for {country_name}. Please provide custom_data.")
    else:
        calibration_data = custom_data
    
    print(f"✓ Created calibration data for {calibration_data.country}")
    
    # Set default configurations if not provided
    if disease_config is None:
        disease_config = DiseaseConfig()
    
    if intervention_config is None:
        intervention_config = InterventionConfig()
    
    if sim_config is None:
        sim_config = SimulationConfig()
    
    # Run simulation
    print("Running calibration simulation...")
    try:
        sim = run_calibration_simulation_suite(
            country_name=country_name,
            disease_config=disease_config,
            intervention_config=intervention_config,
            sim_config=sim_config
        )
        print("✓ Simulation completed")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print(f"Falling back to South Africa demographic data for simulation...")
        sim = run_calibration_simulation_suite(
            country_name="South Africa",  # Use South Africa data
            disease_config=disease_config,
            intervention_config=intervention_config,
            sim_config=sim_config
        )
        print("✓ Simulation completed with South Africa demographics")
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    
    # Create plotter
    plotter = CalibrationPlotter()
    
    # Create calibration plots
    if save_results:
        print("Creating calibration plots...")
        comparison_fig = plotter.plot_calibration_comparison(
            sim, 
            calibration_data, 
            timestamp,
            save_path=f"calibration_comparison_{country_name.replace(' ', '_')}_{timestamp}.pdf"
        )
        print("✓ Calibration plots created")
    
    # Create calibration report
    if save_results:
        print("Creating calibration report...")
        report = create_calibration_report(
            sim, 
            calibration_data, 
            timestamp,
            save_path=f"calibration_report_{country_name.replace(' ', '_')}_{timestamp}.json"
        )
        print("✓ Calibration report created")
    else:
        report = create_calibration_report(sim, calibration_data, timestamp, save_path=None)
    
    # Save calibration data for reference
    if save_results:
        calibration_data.case_notifications.to_csv(
            f"case_notifications_{country_name.replace(' ', '_')}_{timestamp}.csv", 
            index=False
        )
        calibration_data.age_prevalence.to_csv(
            f"age_prevalence_{country_name.replace(' ', '_')}_{timestamp}.csv", 
            index=False
        )
        print("✓ Calibration data saved")
    
    print(f"\nCalibration analysis completed for {country_name}!")
    print(f"Files created with timestamp: {timestamp}")
    
    return sim, calibration_data, report, plotter


def run_parameter_sweep(country_name="South Africa", parameter_ranges=None, 
                       max_simulations=50, save_results=True):
    """
    Run a parameter sweep for calibration
    
    Args:
        country_name: Name of the country
        parameter_ranges: Dictionary of parameter ranges to sweep
        max_simulations: Maximum number of simulations to run
        save_results: Whether to save results to files
    
    Returns:
        tuple: (sweep_summary, results_df, best_sim, calibration_data)
    """
    
    print(f"=== TB Model Parameter Sweep for {country_name} ===")
    
    # Default parameter ranges
    if parameter_ranges is None:
        parameter_ranges = {
            'beta': np.array([0.010, 0.015, 0.020, 0.025, 0.030]),
            'rel_sus_latentslow': np.array([0.05, 0.10, 0.15, 0.20, 0.25]),
            'tb_mortality': np.array([1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
        }
    
    # Create calibration data
    if country_name.lower() == "south africa":
        calibration_data = create_south_africa_data()
    elif country_name.lower() == "vietnam":
        calibration_data = create_vietnam_data()
    else:
        raise ValueError(f"No default data available for {country_name}")
    
    # Initialize results storage
    results = []
    best_score = float('inf')
    best_params = None
    best_sim = None
    
    # Calculate total combinations
    total_combinations = (len(parameter_ranges['beta']) * 
                         len(parameter_ranges['rel_sus_latentslow']) * 
                         len(parameter_ranges['tb_mortality']))
    simulations_run = 0
    
    print(f"Parameter ranges:")
    for param, values in parameter_ranges.items():
        print(f"  {param}: {values}")
    print(f"Max simulations: {max_simulations}")
    print(f"Total combinations: {total_combinations}")
    print("Running simulations...")
    
    start_time = time.time()
    
    # Run parameter sweep
    for beta in parameter_ranges['beta']:
        for rel_sus in parameter_ranges['rel_sus_latentslow']:
            for tb_mortality in parameter_ranges['tb_mortality']:
                
                if simulations_run >= max_simulations:
                    print(f"\nReached maximum simulations ({max_simulations})")
                    break
                
                simulations_run += 1
                print(f"Simulation {simulations_run}/{min(total_combinations, max_simulations)}: "
                      f"β={beta:.4f}, rel_sus={rel_sus:.3f}, mort={tb_mortality:.1e}")
                
                try:
                    # Create disease configuration
                    disease_config = DiseaseConfig(
                        beta=beta,
                        rel_sus_latentslow=rel_sus,
                        tb_mortality=tb_mortality
                    )
                    
                    # Create simulation configuration
                    sim_config = SimulationConfig(
                        n_agents=500,  # Smaller for faster sweep
                        years=150,     # Shorter for faster sweep
                        seed=simulations_run  # Different seed for each simulation
                    )
                    
                    # Run simulation
                    sim = run_calibration_simulation_suite(
                        country_name=country_name,
                        disease_config=disease_config,
                        sim_config=sim_config
                    )
                    
                    # Calculate calibration score
                    score_metrics = calculate_calibration_score(sim, calibration_data)
                    
                    # Store results
                    result = {
                        'beta': beta,
                        'rel_sus_latentslow': rel_sus,
                        'tb_mortality': tb_mortality,
                        'simulation_number': simulations_run,
                        'score_metrics': score_metrics,
                        'composite_score': score_metrics['composite_score']
                    }
                    results.append(result)
                    
                    # Update best if better
                    if score_metrics['composite_score'] < best_score:
                        best_score = score_metrics['composite_score']
                        best_params = (beta, rel_sus, tb_mortality)
                        best_sim = sim
                        print(f"  ✓ New best score: {best_score:.2f}")
                    
                except Exception as e:
                    print(f"  ✗ Simulation failed: {e}")
                    continue
            
            if simulations_run >= max_simulations:
                break
        
        if simulations_run >= max_simulations:
            break
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Create results summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('composite_score')
    
    # Create sweep summary
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
    sweep_summary = {
        'timestamp': timestamp,
        'country': country_name,
        'parameter_ranges': {k: v.tolist() for k, v in parameter_ranges.items()},
        'simulation_settings': {
            'max_simulations': max_simulations,
            'simulations_run': simulations_run,
            'elapsed_time_minutes': elapsed_time / 60
        },
        'best_parameters': {
            'beta': best_params[0] if best_params else None,
            'rel_sus_latentslow': best_params[1] if best_params else None,
            'tb_mortality': best_params[2] if best_params else None,
            'composite_score': best_score
        },
        'top_10_results': results_df.head(10).to_dict('records')
    }
    
    # Save results
    if save_results:
        results_df.to_csv(f"calibration_sweep_results_{country_name.replace(' ', '_')}_{timestamp}.csv", index=False)
        
        with open(f"calibration_sweep_summary_{country_name.replace(' ', '_')}_{timestamp}.json", 'w') as f:
            json.dump(sweep_summary, f, indent=2, default=str)
        
        # Create plots
        plotter = CalibrationPlotter()
        sweep_fig = plotter.plot_sweep_results(
            results_df, 
            timestamp,
            save_path=f"calibration_sweep_results_{country_name.replace(' ', '_')}_{timestamp}.pdf"
        )
        
        violin_fig = plotter.plot_violin_plots(
            results_df, 
            timestamp,
            save_path=f"calibration_violin_plots_{country_name.replace(' ', '_')}_{timestamp}.pdf"
        )
        
        print(f"✓ Sweep results saved")
    
    # Print summary
    print(f"\n=== BEST CALIBRATION PARAMETERS ===")
    print(f"Beta: {sweep_summary['best_parameters']['beta']:.4f}")
    print(f"Relative Susceptibility: {sweep_summary['best_parameters']['rel_sus_latentslow']:.3f}")
    print(f"TB Mortality: {sweep_summary['best_parameters']['tb_mortality']:.1e}")
    print(f"Composite Score: {sweep_summary['best_parameters']['composite_score']:.2f}")
    print(f"================================")
    
    return sweep_summary, results_df, best_sim, calibration_data


def main():
    """
    Main function demonstrating the generalized calibration framework
    """
    
    print("Generalized TB Model Calibration Framework")
    print("=" * 50)
    
    # Example 1: South Africa calibration
    print("\n1. Running South Africa calibration...")
    sim_sa, data_sa, report_sa, plotter_sa = run_calibration_analysis("South Africa")
    
    # Example 2: Vietnam calibration
    print("\n2. Running Vietnam calibration...")
    sim_vn, data_vn, report_vn, plotter_vn = run_calibration_analysis("Vietnam")
    
    # Example 3: Parameter sweep for South Africa
    print("\n3. Running parameter sweep for South Africa...")
    sweep_summary, results_df, best_sim, calibration_data = run_parameter_sweep(
        "South Africa", 
        max_simulations=25  # Reduced for demonstration
    )
    
    print("\n" + "=" * 50)
    print("All calibration analyses completed!")
    print("Check the generated files for detailed results.")
    
    return sim_sa, data_sa, report_sa, sweep_summary, results_df


if __name__ == '__main__':
    sim_sa, data_sa, report_sa, sweep_summary, results_df = main() 