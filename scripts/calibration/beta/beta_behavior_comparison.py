#!/usr/bin/env python3
"""
Beta Behavior Comparison for TB Model

This script compares the behavior of the TB model with various beta values
across different transmission contexts, providing detailed analysis of how
beta affects key model outputs and dynamics.

Features:
- Systematic comparison of beta values across contexts
- Analysis of transmission dynamics, prevalence patterns, and case rates
- Visualization of model behavior differences
- Literature-informed beta value selection
- Export of comparative analysis results

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
from tbsim.utils import (
    CalibrationPlotter, 
    CalibrationData, 
    CalibrationTarget,
    compute_age_stratified_prevalence,
    compute_case_notifications,
    calculate_calibration_score,
    create_south_africa_data,
    run_generalized_simulation
)
from tbsim.utils.simulation_utils import SimulationConfig, DiseaseConfig, InterventionConfig


class BetaBehaviorComparison:
    """
    Comprehensive comparison of TB model behavior with different beta values.
    
    This class provides systematic analysis of how the beta parameter affects
    model dynamics across different transmission contexts and time periods.
    """
    
    def __init__(self, calibration_data: CalibrationData):
        """
        Initialize beta behavior comparison.
        
        Parameters:
            calibration_data: CalibrationData object with targets and data
        """
        self.calibration_data = calibration_data
        self.plotter = CalibrationPlotter()
        
        # Define beta values for comparison based on literature
        self.beta_comparison_sets = {
            'literature_based': {
                'name': 'Literature-Based Beta Values',
                'description': 'Beta values from epidemiological literature',
                'values': {
                    'household_high': 0.7,      # β_HH: 0.3-0.7 (high end)
                    'household_mid': 0.5,       # β_HH: 0.3-0.7 (middle)
                    'household_low': 0.3,       # β_HH: 0.3-0.7 (low end)
                    'community_high': 0.02,     # β_community: 0.005-0.02 (high end)
                    'community_mid': 0.0125,    # β_community: 0.005-0.02 (middle)
                    'community_low': 0.005,     # β_community: 0.005-0.02 (low end)
                    'high_burden_high': 1.5,    # 0.01-0.05/day ≈ 0.3-1.5/month (high end)
                    'high_burden_mid': 0.9,     # 0.01-0.05/day ≈ 0.3-1.5/month (middle)
                    'high_burden_low': 0.3,     # 0.01-0.05/day ≈ 0.3-1.5/month (low end)
                    'calibrated_high': 0.04,    # Calibrated range (high end)
                    'calibrated_mid': 0.025,    # Calibrated range (middle)
                    'calibrated_low': 0.015     # Calibrated range (low end)
                }
            },
            'context_specific': {
                'name': 'Context-Specific Beta Values',
                'description': 'Representative beta values for different transmission contexts',
                'values': {
                    'household': 0.5,           # Household transmission
                    'community': 0.025,         # Community transmission
                    'high_burden': 0.7,         # High burden settings
                    'calibrated': 0.025,        # Calibrated settings
                    'very_low': 0.001,          # Very low transmission
                    'very_high': 2.0            # Very high transmission
                }
            },
            'sensitivity_analysis': {
                'name': 'Sensitivity Analysis Beta Values',
                'description': 'Beta values for sensitivity analysis around literature values',
                'values': {
                    'baseline': 0.025,          # Baseline (current default)
                    'half_baseline': 0.0125,    # Half of baseline
                    'double_baseline': 0.05,    # Double of baseline
                    'quarter_baseline': 0.00625, # Quarter of baseline
                    'quadruple_baseline': 0.1   # Quadruple of baseline
                }
            }
        }
    
    def run_behavior_comparison(self, comparison_set='context_specific', 
                               n_people=10000, years=10, n_trials=2,
                               save_results=True):
        """
        Run comprehensive behavior comparison across beta values.
        
        Parameters:
            comparison_set: Which set of beta values to use ('literature_based', 'context_specific', 'sensitivity_analysis')
            n_people: Population size for simulation
            years: Simulation duration in years
            n_trials: Number of trials per beta value
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Starting beta behavior comparison using {comparison_set} set...")
        
        beta_set = self.beta_comparison_sets[comparison_set]
        print(f"Set: {beta_set['name']}")
        print(f"Description: {beta_set['description']}")
        print(f"Beta values: {list(beta_set['values'].keys())}")
        
        results = []
        beta_values = beta_set['values']
        
        for i, (beta_name, beta_value) in enumerate(beta_values.items()):
            print(f"\nProgress: {i+1}/{len(beta_values)} - Testing {beta_name} (β={beta_value})")
            
            # Run multiple trials for each beta value
            trial_results = []
            for trial in range(n_trials):
                try:
                    # Create disease configuration with current beta
                    disease_config = DiseaseConfig(
                        beta=beta_value,
                        rel_sus_latentslow=0.15,
                        init_prev=0.25
                    )
                    
                    # Create simulation configuration
                    sim_config = SimulationConfig(
                        n_agents=n_people,
                        years=years
                    )
                    
                    # Create intervention configuration
                    intervention_config = InterventionConfig(
                        include_health_seeking=True,
                        include_diagnostic=True,
                        include_treatment=True,
                        include_hiv=False
                    )
                    
                    # Run simulation
                    sim = run_generalized_simulation(
                        country_name=self.calibration_data.country,
                        disease_config=disease_config,
                        intervention_config=intervention_config,
                        sim_config=sim_config
                    )
                    
                    # Extract comprehensive metrics
                    metrics = self._extract_comprehensive_metrics(sim, beta_name, beta_value, trial)
                    trial_results.append(metrics)
                    
                except Exception as e:
                    print(f"Error in trial {trial} for {beta_name}: {e}")
                    continue
            
            # Aggregate trial results
            if trial_results:
                avg_result = self._aggregate_trial_results(trial_results, beta_name, beta_value)
                results.append(avg_result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        if save_results and not results_df.empty:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            filename = f"beta_behavior_comparison_{comparison_set}_{timestamp}.csv"
            results_df.to_csv(filename, index=False)
            print(f"\nResults saved to: {filename}")
        elif save_results and results_df.empty:
            print("\nNo results to save - all simulations failed")
        
        return results_df
    
    def _extract_comprehensive_metrics(self, sim, beta_name, beta_value, trial):
        """
        Extract comprehensive metrics from simulation results.
        
        Parameters:
            sim: Simulation object
            beta_name: Name of the beta value
            beta_value: Actual beta value
            trial: Trial number
            
        Returns:
            Dict with comprehensive metrics
        """
        # Basic metrics
        metrics = {
            'beta_name': beta_name,
            'beta_value': beta_value,
            'trial': trial
        }
        
        # Prevalence metrics
        age_prevalence = compute_age_stratified_prevalence(sim, target_year=2018)
        
        # Calculate overall prevalence from age groups
        total_prevalence_per_100k = sum(data['prevalence_per_100k'] for data in age_prevalence.values())
        overall_prevalence = total_prevalence_per_100k / 100000
        
        metrics.update({
            'overall_prevalence': overall_prevalence,
            'prevalence_0_14': age_prevalence.get('0-14', {}).get('prevalence_per_100k', 0) / 100000,
            'prevalence_15_24': age_prevalence.get('15-24', {}).get('prevalence_per_100k', 0) / 100000,
            'prevalence_25_34': age_prevalence.get('25-34', {}).get('prevalence_per_100k', 0) / 100000,
            'prevalence_35_44': age_prevalence.get('35-44', {}).get('prevalence_per_100k', 0) / 100000,
            'prevalence_45_54': age_prevalence.get('45-54', {}).get('prevalence_per_100k', 0) / 100000,
            'prevalence_55_plus': age_prevalence.get('55+', {}).get('prevalence_per_100k', 0) / 100000
        })
        
        # Case notification metrics
        case_notifications = compute_case_notifications(sim, target_years=[2018])
        case_rate = case_notifications.get(2018, {}).get('rate_per_100k', 0) if case_notifications else 0
        
        metrics.update({
            'case_rate_per_100k': case_rate,
            'case_rate_per_100k_15_plus': case_rate  # Use same rate for now, can be enhanced later
        })
        
        # Transmission dynamics metrics
        if hasattr(sim, 'diseases') and 'tb' in sim.diseases:
            tb_disease = sim.diseases['tb']
            
            # Calculate transmission metrics
            total_infections = np.sum(tb_disease.results['new_infections'])
            total_active_cases = np.sum(tb_disease.results['new_active'])
            total_deaths = np.sum(tb_disease.results['new_deaths'])
            
            # Calculate rates per year
            years_simulated = len(tb_disease.results['new_infections'])
            population_size = len(sim.people.alive)
            annual_infection_rate = total_infections / years_simulated / population_size
            annual_active_rate = total_active_cases / years_simulated / population_size
            annual_death_rate = total_deaths / years_simulated / population_size
            
            metrics.update({
                'total_infections': total_infections,
                'total_active_cases': total_active_cases,
                'total_deaths': total_deaths,
                'annual_infection_rate': annual_infection_rate,
                'annual_active_rate': annual_active_rate,
                'annual_death_rate': annual_death_rate,
                'infection_to_active_ratio': total_active_cases / total_infections if total_infections > 0 else 0,
                'active_to_death_ratio': total_deaths / total_active_cases if total_active_cases > 0 else 0
            })
        
        # Calibration metrics
        try:
            calibration_score = calculate_calibration_score(sim, self.calibration_data, target_year=2018)
            target_prevalence = self.calibration_data.targets.get('overall_prevalence', 
                                                               CalibrationTarget('overall_prevalence', 0.852)).value
            prevalence_error = abs(overall_prevalence - target_prevalence)
            
            metrics.update({
                'calibration_score': calibration_score,
                'target_prevalence': target_prevalence,
                'prevalence_error': prevalence_error
            })
        except Exception as e:
            print(f"Warning: Could not calculate calibration score: {e}")
            metrics.update({
                'calibration_score': np.nan,
                'target_prevalence': np.nan,
                'prevalence_error': np.nan
            })
        
        return metrics
    
    def _aggregate_trial_results(self, trial_results, beta_name, beta_value):
        """
        Aggregate results from multiple trials.
        
        Parameters:
            trial_results: List of trial result dictionaries
            beta_name: Name of the beta value
            beta_value: Actual beta value
            
        Returns:
            Dict with aggregated results
        """
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(trial_results)
        
        # Calculate means and standard deviations for numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        aggregated = {
            'beta_name': beta_name,
            'beta_value': beta_value,
            'n_trials': len(trial_results)
        }
        
        for col in numeric_columns:
            if col not in ['trial']:
                aggregated[f'{col}_mean'] = df[col].mean()
                aggregated[f'{col}_std'] = df[col].std()
                aggregated[f'{col}_min'] = df[col].min()
                aggregated[f'{col}_max'] = df[col].max()
        
        return aggregated
    
    def analyze_behavior_patterns(self, results_df):
        """
        Analyze patterns in model behavior across beta values.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            Dict with analysis results
        """
        if results_df.empty:
            return {}
        
        analysis = {
            'beta_impact_analysis': self._analyze_beta_impact(results_df),
            'transmission_context_analysis': self._analyze_transmission_contexts(results_df),
            'sensitivity_analysis': self._analyze_sensitivity(results_df),
            'literature_comparison': self._compare_with_literature(results_df),
            'recommendations': self._generate_behavior_recommendations(results_df)
        }
        
        return analysis
    
    def _analyze_beta_impact(self, results_df):
        """
        Analyze the impact of beta on key model outputs.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            Dict with impact analysis
        """
        # Calculate correlations with beta
        beta_correlations = {}
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col not in ['beta_value', 'n_trials'] and col.endswith('_mean'):
                base_col = col.replace('_mean', '')
                correlation = results_df['beta_value'].corr(results_df[col])
                beta_correlations[base_col] = correlation
        
        # Find beta values that achieve target prevalence
        target_prevalence = self.calibration_data.targets.get('overall_prevalence', 
                                                             CalibrationTarget('overall_prevalence', 0.852)).value
        
        closest_to_target_idx = (results_df['overall_prevalence_mean'] - target_prevalence).abs().idxmin()
        closest_beta = results_df.loc[closest_to_target_idx, 'beta_value']
        closest_prevalence = results_df.loc[closest_to_target_idx, 'overall_prevalence_mean']
        
        return {
            'beta_correlations': beta_correlations,
            'closest_to_target': {
                'beta': closest_beta,
                'prevalence': closest_prevalence,
                'beta_name': results_df.loc[closest_to_target_idx, 'beta_name']
            },
            'prevalence_range': {
                'min': results_df['overall_prevalence_mean'].min(),
                'max': results_df['overall_prevalence_mean'].max(),
                'range': results_df['overall_prevalence_mean'].max() - results_df['overall_prevalence_mean'].min()
            }
        }
    
    def _analyze_transmission_contexts(self, results_df):
        """
        Analyze behavior across different transmission contexts.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            Dict with context analysis
        """
        context_analysis = {}
        
        # Group by context (extract from beta_name)
        contexts = {
            'household': results_df[results_df['beta_name'].str.contains('household', case=False)],
            'community': results_df[results_df['beta_name'].str.contains('community', case=False)],
            'high_burden': results_df[results_df['beta_name'].str.contains('high_burden', case=False)],
            'calibrated': results_df[results_df['beta_name'].str.contains('calibrated', case=False)]
        }
        
        for context, context_df in contexts.items():
            if not context_df.empty:
                context_analysis[context] = {
                    'n_values': len(context_df),
                    'beta_range': [context_df['beta_value'].min(), context_df['beta_value'].max()],
                    'prevalence_range': [context_df['overall_prevalence_mean'].min(), context_df['overall_prevalence_mean'].max()],
                    'case_rate_range': [context_df['case_rate_per_100k_mean'].min(), context_df['case_rate_per_100k_mean'].max()],
                    'avg_calibration_score': context_df['calibration_score_mean'].mean() if 'calibration_score_mean' in context_df.columns else np.nan
                }
        
        return context_analysis
    
    def _analyze_sensitivity(self, results_df):
        """
        Analyze sensitivity of model outputs to beta changes.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            Dict with sensitivity analysis
        """
        # Calculate elasticities (percentage change in output per percentage change in beta)
        baseline_idx = results_df[results_df['beta_name'] == 'baseline'].index
        if len(baseline_idx) == 0:
            baseline_idx = results_df['beta_value'].idxmin()  # Use lowest beta as baseline
        
        baseline_beta = results_df.loc[baseline_idx[0] if isinstance(baseline_idx, pd.Index) else baseline_idx, 'beta_value']
        baseline_prevalence = results_df.loc[baseline_idx[0] if isinstance(baseline_idx, pd.Index) else baseline_idx, 'overall_prevalence_mean']
        
        elasticities = {}
        for idx, row in results_df.iterrows():
            if row['beta_value'] != baseline_beta:
                beta_change = (row['beta_value'] - baseline_beta) / baseline_beta
                prevalence_change = (row['overall_prevalence_mean'] - baseline_prevalence) / baseline_prevalence
                
                if beta_change != 0:
                    elasticity = prevalence_change / beta_change
                    elasticities[row['beta_name']] = elasticity
        
        return {
            'baseline_beta': baseline_beta,
            'baseline_prevalence': baseline_prevalence,
            'elasticities': elasticities,
            'avg_elasticity': np.mean(list(elasticities.values())) if elasticities else np.nan
        }
    
    def _compare_with_literature(self, results_df):
        """
        Compare model behavior with literature expectations.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            Dict with literature comparison
        """
        literature_expectations = {
            'household': {
                'expected_beta_range': [0.3, 0.7],
                'expected_prevalence_range': [0.1, 0.3],  # High prevalence in households
                'description': 'Household transmission studies'
            },
            'community': {
                'expected_beta_range': [0.005, 0.02],
                'expected_prevalence_range': [0.001, 0.01],  # Lower prevalence in community
                'description': 'Community settings'
            },
            'high_burden': {
                'expected_beta_range': [0.3, 1.5],
                'expected_prevalence_range': [0.01, 0.05],  # High prevalence in high burden
                'description': 'High burden settings'
            }
        }
        
        comparison = {}
        for context, expectations in literature_expectations.items():
            context_df = results_df[results_df['beta_name'].str.contains(context, case=False)]
            if not context_df.empty:
                comparison[context] = {
                    'expected_beta_range': expectations['expected_beta_range'],
                    'model_beta_range': [context_df['beta_value'].min(), context_df['beta_value'].max()],
                    'expected_prevalence_range': expectations['expected_prevalence_range'],
                    'model_prevalence_range': [context_df['overall_prevalence_mean'].min(), context_df['overall_prevalence_mean'].max()],
                    'description': expectations['description']
                }
        
        return comparison
    
    def _generate_behavior_recommendations(self, results_df):
        """
        Generate recommendations based on behavior analysis.
        
        Parameters:
            results_df: DataFrame with comparison results
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if results_df.empty:
            return recommendations
        
        # Find best performing beta values
        if 'calibration_score_mean' in results_df.columns:
            best_calibration_idx = results_df['calibration_score_mean'].idxmin()
            best_calibration_beta = results_df.loc[best_calibration_idx, 'beta_value']
            best_calibration_name = results_df.loc[best_calibration_idx, 'beta_name']
            recommendations.append(f"Best calibration performance: {best_calibration_name} (β={best_calibration_beta:.3f})")
        
        # Analyze beta impact
        beta_correlations = results_df['beta_value'].corr(results_df['overall_prevalence_mean'])
        recommendations.append(f"Beta-prevalence correlation: {beta_correlations:.3f}")
        
        # Context-specific recommendations
        for context in ['household', 'community', 'high_burden']:
            context_df = results_df[results_df['beta_name'].str.contains(context, case=False)]
            if not context_df.empty:
                avg_prevalence = context_df['overall_prevalence_mean'].mean()
                recommendations.append(f"Average prevalence for {context} context: {avg_prevalence:.4f}")
        
        return recommendations
    
    def plot_behavior_comparison(self, results_df, analysis=None, save_plots=True):
        """
        Create comprehensive visualization of behavior comparison.
        
        Parameters:
            results_df: DataFrame with comparison results
            analysis: Analysis results from analyze_behavior_patterns()
            save_plots: Whether to save plots to files
        """
        if results_df.empty:
            print("No results to plot")
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('TB Model Behavior Comparison Across Beta Values', fontsize=16, fontweight='bold')
        
        # 1. Prevalence vs Beta
        axes[0, 0].errorbar(results_df['beta_value'], results_df['overall_prevalence_mean'], 
                           yerr=results_df['overall_prevalence_std'], marker='o', capsize=5)
        axes[0, 0].set_xlabel('Beta (monthly transmission rate)')
        axes[0, 0].set_ylabel('Overall Prevalence')
        axes[0, 0].set_title('Prevalence vs Beta')
        axes[0, 0].set_xscale('log')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add target prevalence line
        target_prevalence = self.calibration_data.targets.get('overall_prevalence', 
                                                             CalibrationTarget('overall_prevalence', 0.852)).value
        axes[0, 0].axhline(y=target_prevalence, color='red', linestyle='--', alpha=0.7, 
                           label=f'Target: {target_prevalence:.3f}')
        axes[0, 0].legend()
        
        # 2. Case Rate vs Beta
        axes[0, 1].errorbar(results_df['beta_value'], results_df['case_rate_per_100k_mean'], 
                           yerr=results_df['case_rate_per_100k_std'], marker='s', capsize=5)
        axes[0, 1].set_xlabel('Beta (monthly transmission rate)')
        axes[0, 1].set_ylabel('Case Rate (per 100,000)')
        axes[0, 1].set_title('Case Rate vs Beta')
        axes[0, 1].set_xscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Calibration Score vs Beta
        if 'calibration_score_mean' in results_df.columns:
            axes[0, 2].errorbar(results_df['beta_value'], results_df['calibration_score_mean'], 
                               yerr=results_df['calibration_score_std'], marker='^', capsize=5)
            axes[0, 2].set_xlabel('Beta (monthly transmission rate)')
            axes[0, 2].set_ylabel('Calibration Score (lower is better)')
            axes[0, 2].set_title('Calibration Score vs Beta')
            axes[0, 2].set_xscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Age-stratified Prevalence
        age_groups = ['0_14', '15_24', '25_34', '35_44', '45_54', '55_plus']
        age_labels = ['0-14', '15-24', '25-34', '35-44', '45-54', '55+']
        
        for i, (age_group, label) in enumerate(zip(age_groups, age_labels)):
            col_name = f'prevalence_{age_group}_mean'
            if col_name in results_df.columns:
                axes[1, 0].plot(results_df['beta_value'], results_df[col_name], 
                               marker='o', label=label, alpha=0.7)
        
        axes[1, 0].set_xlabel('Beta (monthly transmission rate)')
        axes[1, 0].set_ylabel('Prevalence')
        axes[1, 0].set_title('Age-Stratified Prevalence vs Beta')
        axes[1, 0].set_xscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Transmission Dynamics
        if 'annual_infection_rate_mean' in results_df.columns:
            axes[1, 1].plot(results_df['beta_value'], results_df['annual_infection_rate_mean'], 
                           marker='o', label='Infection Rate', alpha=0.7)
            axes[1, 1].plot(results_df['beta_value'], results_df['annual_active_rate_mean'], 
                           marker='s', label='Active Rate', alpha=0.7)
            axes[1, 1].plot(results_df['beta_value'], results_df['annual_death_rate_mean'], 
                           marker='^', label='Death Rate', alpha=0.7)
            axes[1, 1].set_xlabel('Beta (monthly transmission rate)')
            axes[1, 1].set_ylabel('Annual Rate')
            axes[1, 1].set_title('Transmission Dynamics vs Beta')
            axes[1, 1].set_xscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Ratios
        if 'infection_to_active_ratio_mean' in results_df.columns:
            axes[1, 2].plot(results_df['beta_value'], results_df['infection_to_active_ratio_mean'], 
                           marker='o', label='Infection to Active', alpha=0.7)
            axes[1, 2].plot(results_df['beta_value'], results_df['active_to_death_ratio_mean'], 
                           marker='s', label='Active to Death', alpha=0.7)
            axes[1, 2].set_xlabel('Beta (monthly transmission rate)')
            axes[1, 2].set_ylabel('Ratio')
            axes[1, 2].set_title('Disease Progression Ratios vs Beta')
            axes[1, 2].set_xscale('log')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Context Comparison
        contexts = ['household', 'community', 'high_burden', 'calibrated']
        context_data = []
        context_labels = []
        
        for context in contexts:
            context_df = results_df[results_df['beta_name'].str.contains(context, case=False)]
            if not context_df.empty:
                context_data.append(context_df['overall_prevalence_mean'].values)
                context_labels.append(context.title())
        
        if context_data:
            axes[2, 0].boxplot(context_data, labels=context_labels)
            axes[2, 0].set_ylabel('Overall Prevalence')
            axes[2, 0].set_title('Prevalence by Transmission Context')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Beta Distribution
        axes[2, 1].hist(results_df['beta_value'], bins=10, alpha=0.7, edgecolor='black')
        axes[2, 1].set_xlabel('Beta (monthly transmission rate)')
        axes[2, 1].set_ylabel('Frequency')
        axes[2, 1].set_title('Distribution of Beta Values')
        axes[2, 1].set_xscale('log')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Summary Statistics
        axes[2, 2].axis('off')
        summary_text = f"""
Behavior Comparison Summary

Total Beta Values Tested: {len(results_df)}
Prevalence Range: {results_df['overall_prevalence_mean'].min():.4f} - {results_df['overall_prevalence_mean'].max():.4f}
Case Rate Range: {results_df['case_rate_per_100k_mean'].min():.0f} - {results_df['case_rate_per_100k_mean'].max():.0f} per 100k

Best Calibration:
{results_df.loc[results_df['calibration_score_mean'].idxmin(), 'beta_name'] if 'calibration_score_mean' in results_df.columns else 'N/A'}

Target Prevalence: {target_prevalence:.3f}
        """
        axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
            filename = f"beta_behavior_comparison_{timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {filename}")
        
        plt.show()
    
    def generate_behavior_report(self, results_df, analysis, save_report=True):
        """
        Generate comprehensive behavior comparison report.
        
        Parameters:
            results_df: DataFrame with comparison results
            analysis: Analysis results from analyze_behavior_patterns()
            save_report: Whether to save report to file
            
        Returns:
            Dict with report data
        """
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M")
        
        report = {
            'timestamp': timestamp,
            'calibration_data': {
                'country': self.calibration_data.country,
                'target_prevalence': self.calibration_data.targets.get('overall_prevalence', 
                                                                     CalibrationTarget('overall_prevalence', 0.852)).value
            },
            'comparison_parameters': {
                'n_beta_values': len(results_df) if not results_df.empty else 0,
                'beta_range': [results_df['beta_value'].min(), results_df['beta_value'].max()] if not results_df.empty else [0, 0],
                'n_trials_per_beta': results_df['n_trials'].iloc[0] if not results_df.empty else 0
            },
            'behavior_analysis': analysis,
            'all_results': results_df.to_dict('records') if not results_df.empty else [],
            'literature_contexts': self.beta_comparison_sets
        }
        
        if save_report:
            filename = f"beta_behavior_report_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Report saved to: {filename}")
        
        return report


def main():
    """Main function to run beta behavior comparison."""
    print("=== Beta Behavior Comparison for TB Model ===")
    print("Comparing model behavior across different beta values and transmission contexts")
    
    # Create South Africa calibration data
    calibration_data = create_south_africa_data()
    
    # Create behavior comparison object
    comparison = BetaBehaviorComparison(calibration_data)
    
    # Run comparison for different beta sets
    comparison_sets = ['context_specific', 'sensitivity_analysis']
    
    for comparison_set in comparison_sets:
        print(f"\n{'='*60}")
        print(f"Running behavior comparison for {comparison_set}")
        print(f"{'='*60}")
        
        # Run behavior comparison
        results_df = comparison.run_behavior_comparison(
            comparison_set=comparison_set,
            n_people=5000,  # Smaller population for faster testing
            years=5,        # Shorter simulation for faster testing
            n_trials=2      # Fewer trials for faster testing
        )
        
        # Analyze behavior patterns
        analysis = comparison.analyze_behavior_patterns(results_df)
        
        # Print summary
        if analysis and 'beta_impact_analysis' in analysis:
            impact = analysis['beta_impact_analysis']
            print(f"\nBeta Impact Analysis:")
            print(f"  Closest to target: β={impact['closest_to_target']['beta']:.3f} "
                  f"({impact['closest_to_target']['beta_name']})")
            print(f"  Prevalence range: {impact['prevalence_range']['min']:.4f} - {impact['prevalence_range']['max']:.4f}")
            
            if 'beta_correlations' in impact:
                print(f"  Beta correlations:")
                for metric, corr in impact['beta_correlations'].items():
                    print(f"    {metric}: {corr:.3f}")
        else:
            print("\nNo analysis results available - all simulations failed")
        
        # Generate plots only if we have results
        if not results_df.empty:
            comparison.plot_behavior_comparison(results_df, analysis)
            
            # Generate report
            report = comparison.generate_behavior_report(results_df, analysis)
        else:
            print("Skipping plots and report generation - no results available")
        
        print(f"\nBehavior comparison completed for {comparison_set}")


if __name__ == "__main__":
    main() 