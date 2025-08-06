#!/usr/bin/env python3
"""
Comprehensive Beta Calibration Sweep for TB Model

This script performs systematic calibration of the beta parameter (transmission rate)
for different transmission contexts based on literature evidence and calibration targets.

Literature-Based Beta Contexts:
- Household transmission: β_HH: 0.3-0.7 (high exposure)
- Community transmission: β_community: 0.005-0.02 (low exposure)
- High burden settings: 0.01-0.05/day ≈ 0.3-1.5/month
- Calibrated settings: Often calibrated to match incidence ~250-350/100,000

Features:
- Context-specific beta ranges
- Multi-metric calibration scoring
- Comprehensive visualization
- Literature comparison
- Export of results

Author: TB Simulation Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import tbsim utilities
import tbsim as mtb
from tbsim.calibration import (
    CalibrationData, 
    CalibrationTarget,
    compute_age_stratified_prevalence,
    compute_case_notifications,
    calculate_calibration_score,
    create_calibration_report,
    create_south_africa_data,
    run_calibration_simulation_suite
)
from tbsim.plotting import CalibrationPlotter
from tbsim.calibration import SimulationConfig, DiseaseConfig, InterventionConfig


class BetaCalibrationSweep:
    """
    Comprehensive beta calibration sweep for TB model.
    
    This class provides systematic calibration of the beta parameter across
    different transmission contexts with literature-informed ranges and
    multi-metric evaluation.
    """
    
    def __init__(self, calibration_data: CalibrationData, context='community'):
        """
        Initialize beta calibration sweep.
        
        Parameters:
            calibration_data: CalibrationData object with targets and data
            context: Transmission context ('household', 'community', 'high_burden', 'calibrated')
        """
        self.calibration_data = calibration_data
        self.context = context
        self.plotter = CalibrationPlotter()
        
        # Define beta ranges based on literature
        self.beta_ranges = {
            'household': {
                'range': [0.3, 0.4, 0.5, 0.6, 0.7],
                'description': 'Household transmission (high exposure, repeated contacts)',
                'literature_source': 'Household transmission studies: β_HH: 0.3-0.7'
            },
            'community': {
                'range': [0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
                'description': 'Community transmission (low exposure, diffuse contacts)',
                'literature_source': 'Community settings: β_community: 0.005-0.02'
            },
            'high_burden': {
                'range': [0.3, 0.5, 0.7, 1.0, 1.3, 1.5],
                'description': 'High burden settings (e.g., India: 0.01-0.05/day)',
                'literature_source': 'HIV-negative, high burden: 0.01-0.05/day ≈ 0.3-1.5/month'
            },
            'calibrated': {
                'range': [0.015, 0.020, 0.025, 0.030, 0.035, 0.040],
                'description': 'Calibrated for incidence ~250-350/100,000',
                'literature_source': 'Starsim calibrations for incidence ~250-350/100,000'
            }
        }
        
        # Literature comparison data
        self.literature_comparison = {
            'Styblo_rule': {
                'value': 10,  # secondary infections/year per smear+ case
                'description': 'Styblo rule (historical)',
                'context': 'historical'
            },
            'High_burden_daily': {
                'value': 0.03,  # 0.01-0.05/day, using middle value
                'description': 'HIV-negative, high burden (e.g. India)',
                'context': 'high_burden'
            },
            'Household_studies': {
                'value': 0.5,  # β_HH: 0.3-0.7, using middle value
                'description': 'Household transmission studies',
                'context': 'household'
            },
            'Community_settings': {
                'value': 0.0125,  # β_community: 0.005-0.02, using middle value
                'description': 'Community settings',
                'context': 'community'
            }
        }
    
    def run_beta_sweep(self, n_people=10000, years=10, n_trials=3, 
                      target_year=2018, save_results=True):
        """
        Run comprehensive beta parameter sweep.
        
        Parameters:
            n_people: Population size for simulation
            years: Simulation duration in years
            n_trials: Number of trials per beta value
            target_year: Target year for calibration
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with sweep results
        """
        print(f"Starting beta calibration sweep for {self.context} context...")
        print(f"Beta range: {self.beta_ranges[self.context]['range']}")
        print(f"Literature source: {self.beta_ranges[self.context]['literature_source']}")
        
        results = []
        beta_values = self.beta_ranges[self.context]['range']
        
        for i, beta in enumerate(beta_values):
            print(f"\nProgress: {i+1}/{len(beta_values)} - Testing beta = {beta}")
            
            # Run multiple trials for each beta value
            trial_results = []
            for trial in range(n_trials):
                try:
                    # Create disease configuration with current beta
                    disease_config = DiseaseConfig(
                        beta=beta,
                        rel_sus_latentslow=0.15,
                        init_prev=0.25
                    )
                    
                    # Create simulation configuration
                    sim_config = SimulationConfig(
                        n_people=n_people,
                        years=years,
                        country=self.calibration_data.country
                    )
                    
                    # Create intervention configuration
                    intervention_config = InterventionConfig(
                        include_health_seeking=True,
                        include_tb_diagnostic=True,
                        include_tb_treatment=True,
                        include_hiv=False
                    )
                    
                    # Run simulation
                    sim = run_calibration_simulation_suite(
                        country=self.calibration_data.country,
                        disease_config=disease_config,
                        intervention_config=intervention_config,
                        sim_config=sim_config
                    )
                    
                    # Compute calibration metrics
                    age_prevalence = compute_age_stratified_prevalence(
                        sim, target_year=target_year
                    )
                    case_notifications = compute_case_notifications(
                        sim, target_years=[target_year]
                    )
                    
                    # Calculate calibration score
                    score = calculate_calibration_score(
                        sim, self.calibration_data, target_year=target_year
                    )
                    
                    # Extract key metrics
                    overall_prevalence = age_prevalence['prevalence_per_100k'].sum() / 100000
                    case_rate = case_notifications['rate_per_100k'].iloc[0] if len(case_notifications) > 0 else 0
                    
                    trial_result = {
                        'beta': beta,
                        'trial': trial,
                        'calibration_score': score,
                        'overall_prevalence': overall_prevalence,
                        'case_rate_per_100k': case_rate,
                        'target_prevalence': self.calibration_data.targets.get('overall_prevalence', CalibrationTarget('overall_prevalence', 0.852)).value,
                        'prevalence_error': abs(overall_prevalence - self.calibration_data.targets.get('overall_prevalence', CalibrationTarget('overall_prevalence', 0.852)).value),
                        'context': self.context,
                        'literature_source': self.beta_ranges[self.context]['literature_source']
                    }
                    
                    trial_results.append(trial_result)
                    
                except Exception as e:
                    print(f"Error in trial {trial} for beta {beta}: {e}")
                    continue
            
            # Aggregate trial results
            if trial_results:
                avg_result = {
                    'beta': beta,
                    'calibration_score_mean': np.mean([r['calibration_score'] for r in trial_results]),
                    'calibration_score_std': np.std([r['calibration_score'] for r in trial_results]),
                    'overall_prevalence_mean': np.mean([r['overall_prevalence'] for r in trial_results]),
                    'overall_prevalence_std': np.std([r['overall_prevalence'] for r in trial_results]),
                    'case_rate_mean': np.mean([r['case_rate_per_100k'] for r in trial_results]),
                    'case_rate_std': np.std([r['case_rate_per_100k'] for r in trial_results]),
                    'prevalence_error_mean': np.mean([r['prevalence_error'] for r in trial_results]),
                    'n_trials_successful': len(trial_results),
                    'context': self.context,
                    'literature_source': self.beta_ranges[self.context]['literature_source']
                }
                results.append(avg_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if save_results:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            filename = f"beta_calibration_sweep_{self.context}_{timestamp}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
        
        return results_df
    
    def analyze_results(self, results_df):
        """
        Analyze beta sweep results and provide recommendations.
        
        Parameters:
            results_df: DataFrame with sweep results
            
        Returns:
            Dict with analysis results
        """
        if results_df.empty:
            return {}
        
        # Find best beta values
        best_score_idx = results_df['calibration_score_mean'].idxmin()
        best_prevalence_idx = results_df['prevalence_error_mean'].idxmin()
        
        analysis = {
            'best_beta_by_score': {
                'beta': results_df.loc[best_score_idx, 'beta'],
                'score': results_df.loc[best_score_idx, 'calibration_score_mean'],
                'prevalence': results_df.loc[best_score_idx, 'overall_prevalence_mean'],
                'case_rate': results_df.loc[best_score_idx, 'case_rate_mean']
            },
            'best_beta_by_prevalence': {
                'beta': results_df.loc[best_prevalence_idx, 'beta'],
                'score': results_df.loc[best_prevalence_idx, 'calibration_score_mean'],
                'prevalence': results_df.loc[best_prevalence_idx, 'overall_prevalence_mean'],
                'case_rate': results_df.loc[best_prevalence_idx, 'case_rate_mean']
            },
            'literature_comparison': self._compare_with_literature(results_df),
            'context_info': {
                'context': self.context,
                'description': self.beta_ranges[self.context]['description'],
                'literature_source': self.beta_ranges[self.context]['literature_source']
            }
        }
        
        return analysis
    
    def _compare_with_literature(self, results_df):
        """
        Compare calibration results with literature values.
        
        Parameters:
            results_df: DataFrame with sweep results
            
        Returns:
            Dict with literature comparison
        """
        comparison = {}
        
        for lit_name, lit_data in self.literature_comparison.items():
            if lit_data['context'] == self.context:
                # Find closest beta in our results
                closest_idx = (results_df['beta'] - lit_data['value']).abs().idxmin()
                closest_beta = results_df.loc[closest_idx, 'beta']
                closest_score = results_df.loc[closest_idx, 'calibration_score_mean']
                
                comparison[lit_name] = {
                    'literature_value': lit_data['value'],
                    'closest_calibrated_beta': closest_beta,
                    'calibration_score': closest_score,
                    'difference': abs(closest_beta - lit_data['value']),
                    'description': lit_data['description']
                }
        
        return comparison
    
    def plot_results(self, results_df, analysis=None, save_plots=True):
        """
        Create comprehensive visualization of beta sweep results.
        
        Parameters:
            results_df: DataFrame with sweep results
            analysis: Analysis results from analyze_results()
            save_plots: Whether to save plots to files
        """
        if results_df.empty:
            print("No results to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Beta Calibration Sweep Results - {self.context.title()} Context\n{self.beta_ranges[self.context]["description"]}', 
                    fontsize=16, fontweight='bold')
        
        # 1. Calibration score vs beta
        axes[0, 0].errorbar(results_df['beta'], results_df['calibration_score_mean'], 
                           yerr=results_df['calibration_score_std'], marker='o', capsize=5)
        axes[0, 0].set_xlabel('Beta (monthly transmission rate)')
        axes[0, 0].set_ylabel('Calibration Score (lower is better)')
        axes[0, 0].set_title('Calibration Score vs Beta')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Mark best beta if analysis provided
        if analysis and 'best_beta_by_score' in analysis:
            best_beta = analysis['best_beta_by_score']['beta']
            best_score = analysis['best_beta_by_score']['score']
            axes[0, 0].axvline(x=best_beta, color='red', linestyle='--', alpha=0.7, 
                              label=f'Best: β={best_beta:.3f}')
            axes[0, 0].legend()
        
        # 2. Prevalence vs beta
        axes[0, 1].errorbar(results_df['beta'], results_df['overall_prevalence_mean'], 
                           yerr=results_df['overall_prevalence_std'], marker='s', capsize=5)
        axes[0, 1].set_xlabel('Beta (monthly transmission rate)')
        axes[0, 1].set_ylabel('Overall Prevalence')
        axes[0, 1].set_title('Prevalence vs Beta')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add target prevalence line
        target_prevalence = self.calibration_data.targets.get('overall_prevalence', 
                                                             CalibrationTarget('overall_prevalence', 0.852)).value
        axes[0, 1].axhline(y=target_prevalence, color='red', linestyle='--', alpha=0.7, 
                           label=f'Target: {target_prevalence:.3f}')
        axes[0, 1].legend()
        
        # 3. Case rate vs beta
        axes[0, 2].errorbar(results_df['beta'], results_df['case_rate_mean'], 
                           yerr=results_df['case_rate_std'], marker='^', capsize=5)
        axes[0, 2].set_xlabel('Beta (monthly transmission rate)')
        axes[0, 2].set_ylabel('Case Rate (per 100,000)')
        axes[0, 2].set_title('Case Rate vs Beta')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Prevalence error vs beta
        axes[1, 0].plot(results_df['beta'], results_df['prevalence_error_mean'], marker='o')
        axes[1, 0].set_xlabel('Beta (monthly transmission rate)')
        axes[1, 0].set_ylabel('Prevalence Error (absolute)')
        axes[1, 0].set_title('Prevalence Error vs Beta')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mark best beta by prevalence if analysis provided
        if analysis and 'best_beta_by_prevalence' in analysis:
            best_beta = analysis['best_beta_by_prevalence']['beta']
            axes[1, 0].axvline(x=best_beta, color='green', linestyle='--', alpha=0.7, 
                              label=f'Best: β={best_beta:.3f}')
            axes[1, 0].legend()
        
        # 5. Literature comparison
        if analysis and 'literature_comparison' in analysis:
            lit_data = analysis['literature_comparison']
            if lit_data:
                lit_names = list(lit_data.keys())
                lit_values = [lit_data[name]['literature_value'] for name in lit_names]
                cal_values = [lit_data[name]['closest_calibrated_beta'] for name in lit_names]
                
                x = np.arange(len(lit_names))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, lit_values, width, label='Literature', alpha=0.7)
                axes[1, 1].bar(x + width/2, cal_values, width, label='Calibrated', alpha=0.7)
                axes[1, 1].set_xlabel('Literature Source')
                axes[1, 1].set_ylabel('Beta Value')
                axes[1, 1].set_title('Literature vs Calibrated Beta Values')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels([name.replace('_', '\n') for name in lit_names], rotation=45)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary statistics
        axes[1, 2].axis('off')
        summary_text = f"""
Beta Calibration Summary
{self.context.title()} Context

Best Beta by Score:
β = {analysis['best_beta_by_score']['beta']:.3f}
Score = {analysis['best_beta_by_score']['score']:.3f}
Prevalence = {analysis['best_beta_by_score']['prevalence']:.3f}

Best Beta by Prevalence:
β = {analysis['best_beta_by_prevalence']['beta']:.3f}
Score = {analysis['best_beta_by_prevalence']['score']:.3f}
Prevalence = {analysis['best_beta_by_prevalence']['prevalence']:.3f}

Literature Source:
{self.beta_ranges[self.context]['literature_source']}
        """
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            filename = f"beta_calibration_plots_{self.context}_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {filename}")
        
        plt.show()
    
    def generate_report(self, results_df, analysis, save_report=True):
        """
        Generate comprehensive calibration report.
        
        Parameters:
            results_df: DataFrame with sweep results
            analysis: Analysis results from analyze_results()
            save_report: Whether to save report to file
            
        Returns:
            Dict with report data
        """
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        
        report = {
            'timestamp': timestamp,
            'context': self.context,
            'context_description': self.beta_ranges[self.context]['description'],
            'literature_source': self.beta_ranges[self.context]['literature_source'],
            'calibration_data': {
                'country': self.calibration_data.country,
                'target_prevalence': self.calibration_data.targets.get('overall_prevalence', 
                                                                     CalibrationTarget('overall_prevalence', 0.852)).value
            },
            'sweep_parameters': {
                'beta_range': self.beta_ranges[self.context]['range'],
                'n_trials_per_beta': results_df['n_trials_successful'].iloc[0] if not results_df.empty else 0
            },
            'best_results': analysis,
            'all_results': results_df.to_dict('records') if not results_df.empty else [],
            'recommendations': self._generate_recommendations(analysis)
        }
        
        if save_report:
            filename = f"beta_calibration_report_{self.context}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {filename}")
        
        return report
    
    def _generate_recommendations(self, analysis):
        """
        Generate recommendations based on analysis results.
        
        Parameters:
            analysis: Analysis results from analyze_results()
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if not analysis:
            return recommendations
        
        best_score = analysis['best_beta_by_score']
        best_prevalence = analysis['best_beta_by_prevalence']
        
        recommendations.append(f"Best beta for {self.context} context: {best_score['beta']:.3f} (by calibration score)")
        recommendations.append(f"Best beta for prevalence match: {best_prevalence['beta']:.3f}")
        
        # Compare with literature
        if 'literature_comparison' in analysis:
            for lit_name, lit_data in analysis['literature_comparison'].items():
                recommendations.append(f"Literature comparison ({lit_data['description']}): "
                                    f"literature β={lit_data['literature_value']:.3f}, "
                                    f"calibrated β={lit_data['closest_calibrated_beta']:.3f}")
        
        # Context-specific recommendations
        if self.context == 'household':
            recommendations.append("For household transmission modeling, consider using β=0.5 as default")
        elif self.context == 'community':
            recommendations.append("For community transmission modeling, consider using β=0.025 as default")
        elif self.context == 'high_burden':
            recommendations.append("For high burden settings, consider using β=0.7 as default")
        elif self.context == 'calibrated':
            recommendations.append("For calibrated settings, consider using β=0.025 as default")
        
        return recommendations


def main():
    """Main function to run beta calibration sweep."""
    print("=== Beta Calibration Sweep for TB Model ===")
    print("Based on literature evidence and calibration targets")
    
    # Create South Africa calibration data
    calibration_data = create_south_africa_data()
    
    # Test different contexts
    contexts = ['community', 'calibrated', 'high_burden', 'household']
    
    for context in contexts:
        print(f"\n{'='*60}")
        print(f"Running calibration for {context} context")
        print(f"{'='*60}")
        
        # Create calibration sweep object
        sweep = BetaCalibrationSweep(calibration_data, context=context)
        
        # Run beta sweep
        results_df = sweep.run_beta_sweep(
            n_people=5000,  # Smaller population for faster testing
            years=5,        # Shorter simulation for faster testing
            n_trials=2,     # Fewer trials for faster testing
            target_year=2018
        )
        
        # Analyze results
        analysis = sweep.analyze_results(results_df)
        
        # Print summary
        if analysis:
            print(f"\nBest beta by calibration score: {analysis['best_beta_by_score']['beta']:.3f}")
            print(f"Best beta by prevalence match: {analysis['best_beta_by_prevalence']['beta']:.3f}")
            
            if 'literature_comparison' in analysis:
                print("\nLiterature comparison:")
                for lit_name, lit_data in analysis['literature_comparison'].items():
                    print(f"  {lit_data['description']}: "
                          f"literature β={lit_data['literature_value']:.3f}, "
                          f"calibrated β={lit_data['closest_calibrated_beta']:.3f}")
        
        # Generate plots
        sweep.plot_results(results_df, analysis)
        
        # Generate report
        report = sweep.generate_report(results_df, analysis)
        
        print(f"\nCalibration completed for {context} context")


if __name__ == "__main__":
    main() 