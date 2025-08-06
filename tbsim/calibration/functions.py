"""
Calibration utilities for TB model calibration analysis.

This module provides comprehensive utilities for TB model calibration including
computation of calibration metrics, processing of simulation results, and
support for calibration plotting functionality. The functions are designed to
be generalized for different countries and scenarios.

Key Features:
- Age-stratified prevalence computation with flexible age group definitions
- Case notification analysis with fallback mechanisms
- Calibration scoring using weighted composite metrics
- Comprehensive calibration report generation
- Multi-country data structure support
- File I/O for calibration data persistence
- Generalized disease state handling

The module uses dataclasses for structured data representation and provides
both high-level convenience functions and low-level analysis utilities.

Example Usage:
    from tbsim.calibration import (
        compute_age_stratified_prevalence,
        calculate_calibration_score,
        create_calibration_report,
        CalibrationData,
        CalibrationTarget
    )
    
    # Compute age-stratified prevalence
    prevalence_data = compute_age_stratified_prevalence(sim, target_year=2018)
    
    # Calculate calibration score
    score = calculate_calibration_score(sim, calibration_data)
    
    # Generate calibration report
    report = create_calibration_report(sim, calibration_data, timestamp)

Dependencies:
    - numpy: Numerical computations
    - pandas: Data manipulation and analysis
    - tbsim: TB simulation framework
    - dataclasses: Structured data representation
    - typing: Type hints for function signatures

Author: TB Simulation Team
Version: 1.0.0
Last Updated: 2024
"""

import numpy as np
import pandas as pd
import tbsim as mtb
import os
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class CalibrationTarget:
    """
    Data class for calibration targets.
    
    This dataclass represents a single calibration target with metadata
    about the target value, timing, and source information.
    
    Attributes:
        name (str): Unique identifier for the calibration target.
        value (float): Target value for calibration (e.g., prevalence rate).
        year (Optional[int]): Year associated with the target value.
            Defaults to None.
        age_group (Optional[str]): Age group associated with the target.
            Defaults to None.
        description (Optional[str]): Human-readable description of the target.
            Defaults to None.
        source (Optional[str]): Source of the target data (e.g., survey, report).
            Defaults to None.
    
    Example:
        # Create a prevalence target
        target = CalibrationTarget(
            name="overall_prevalence_2018",
            value=0.852,
            year=2018,
            description="Overall TB prevalence from 2018 survey",
            source="South Africa TB Prevalence Survey 2018"
        )
        
        # Create an age-specific target
        age_target = CalibrationTarget(
            name="prevalence_25_34",
            value=0.012,
            year=2018,
            age_group="25-34",
            description="TB prevalence in 25-34 age group",
            source="WHO Global TB Report"
        )
    
    Notes:
        - name should be unique within a calibration dataset
        - value should be in appropriate units (e.g., prevalence as fraction)
        - year and age_group are optional but useful for validation
        - description and source help with documentation and reproducibility
    """
    name: str
    value: float
    year: Optional[int] = None
    age_group: Optional[str] = None
    description: Optional[str] = None
    source: Optional[str] = None


@dataclass
class CalibrationData:
    """
    Data class for comprehensive calibration data.
    
    This dataclass represents a complete calibration dataset including
    case notification data, age-stratified prevalence data, and calibration
    targets for a specific country or region.
    
    Attributes:
        case_notifications (pd.DataFrame): DataFrame containing case notification
            data with columns for year, rate_per_100k, total_cases, and source.
        age_prevalence (pd.DataFrame): DataFrame containing age-stratified
            prevalence data with columns for age_group, prevalence_per_100k,
            prevalence_percent, sample_size, and source.
        targets (Dict[str, CalibrationTarget]): Dictionary mapping target names
            to CalibrationTarget objects for various calibration metrics.
        country (str): Name of the country or region for this calibration data.
        description (str): Human-readable description of the calibration dataset.
    
    Example:
        # Create calibration data for South Africa
        calibration_data = CalibrationData(
            case_notifications=sa_case_data,
            age_prevalence=sa_age_data,
            targets={
                "overall_prevalence": CalibrationTarget("overall_prevalence", 0.852),
                "hiv_coinfection": CalibrationTarget("hiv_coinfection", 0.60)
            },
            country="South Africa",
            description="South Africa TB calibration data from 2018 survey"
        )
        
        # Access data
        print(f"Country: {calibration_data.country}")
        print(f"Number of targets: {len(calibration_data.targets)}")
        print(f"Case notification years: {calibration_data.case_notifications['year'].tolist()}")
    
    Notes:
        - case_notifications should include time series data for trend analysis
        - age_prevalence should cover relevant age groups for the disease
        - targets should include key metrics for model validation
        - country and description help with data organization and documentation
        - All data should be consistent in terms of units and time periods
    """
    case_notifications: pd.DataFrame
    age_prevalence: pd.DataFrame
    targets: Dict[str, CalibrationTarget]
    country: str
    description: str


def compute_age_stratified_prevalence(sim, target_year=2018, age_groups=None, disease_name='tb'):
    """
    Compute age-stratified disease prevalence from simulation results
    
    Args:
        sim: Simulation object
        target_year: Year to compute prevalence for
        age_groups: List of (min_age, max_age) tuples, defaults to standard groups
        disease_name: Name of the disease module (default: 'tb')
    
    Returns:
        dict: Age-stratified prevalence data
    """
    
    if age_groups is None:
        # Default age groups
        age_groups = [(15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
        age_group_labels = ['15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    else:
        # Generate labels from age groups
        age_group_labels = []
        for min_age, max_age in age_groups:
            if max_age == 200:
                age_group_labels.append(f'{min_age}+')
            else:
                age_group_labels.append(f'{min_age}-{max_age}')
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get disease states - generalize for different diseases
    if disease_name == 'tb':
        disease_states = sim.diseases.tb.state
        active_states = [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]
    elif disease_name == 'hiv':
        disease_states = sim.diseases.hiv.state
        active_states = [1, 2, 3]  # HIV positive states
    else:
        # Generic approach - assume disease has a 'state' attribute
        disease_states = getattr(sim.diseases, disease_name).state
        active_states = [1, 2, 3]  # Default active states
    
    active_disease_mask = np.isin(disease_states, active_states)
    
    # Get ages at target time
    ages = people.age[alive_mask]
    active_disease_ages = people.age[alive_mask & active_disease_mask]
    
    prevalence_by_age = {}
    
    for i, (min_age, max_age) in enumerate(age_groups):
        # Count people in age group
        age_mask = (ages >= min_age) & (ages <= max_age)
        total_in_age_group = np.sum(age_mask)
        
        # Count active disease cases in age group
        age_disease_mask = (active_disease_ages >= min_age) & (active_disease_ages <= max_age)
        disease_in_age_group = np.sum(age_disease_mask)
        
        # Calculate prevalence
        if total_in_age_group > 0:
            prevalence = disease_in_age_group / total_in_age_group
            prevalence_per_100k = prevalence * 100000
        else:
            prevalence = 0
            prevalence_per_100k = 0
        
        prevalence_by_age[age_group_labels[i]] = {
            'prevalence': prevalence,
            'prevalence_per_100k': prevalence_per_100k,
            'total_people': total_in_age_group,
            'disease_cases': disease_in_age_group
        }
    
    return prevalence_by_age


def compute_case_notifications(sim, target_years=None, intervention_name='tbdiagnostic', 
                              fallback_detection_rate=0.7, disease_name='tb'):
    """
    Compute case notifications from simulation results
    
    Args:
        sim: Simulation object
        target_years: Years to compute notifications for
        intervention_name: Name of diagnostic intervention
        fallback_detection_rate: Detection rate to use if intervention not present
        disease_name: Name of the disease module
    
    Returns:
        dict: Case notification data by year
    """
    
    if target_years is None:
        target_years = [2000, 2005, 2010, 2015, 2020]
    
    time_years = np.array([d.year for d in sim.results['timevec']])
    notifications_by_year = {}
    
    for target_year in target_years:
        # Find the time index closest to target year
        target_idx = np.argmin(np.abs(time_years - target_year))
        
        # Get diagnostic results for that year
        if intervention_name in sim.results:
            intervention_results = sim.results[intervention_name]
            if 'n_test_positive' in intervention_results:
                n_diagnosed = intervention_results['n_test_positive'].values[target_idx]
            elif 'n_diagnosed' in intervention_results:
                n_diagnosed = intervention_results['n_diagnosed'].values[target_idx]
            else:
                # Try to find any diagnostic-related column
                diagnostic_cols = [col for col in intervention_results.columns if 'positive' in col or 'diagnosed' in col]
                if diagnostic_cols:
                    n_diagnosed = intervention_results[diagnostic_cols[0]].values[target_idx]
                else:
                    n_diagnosed = 0
        else:
            # Fallback: estimate from active disease cases
            if disease_name == 'tb':
                disease_states = sim.diseases.tb.state
                active_states = [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]
            else:
                disease_states = getattr(sim.diseases, disease_name).state
                active_states = [1, 2, 3]  # Default active states
            
            active_disease_mask = np.isin(disease_states, active_states)
            n_diagnosed = np.sum(active_disease_mask) * fallback_detection_rate
        
        # Get population size for rate calculation
        n_alive = sim.results['n_alive'][target_idx]
        
        # Calculate rate per 100,000
        rate_per_100k = (n_diagnosed / n_alive) * 100000 if n_alive > 0 else 0
        
        notifications_by_year[target_year] = {
            'diagnosed_cases': n_diagnosed,
            'rate_per_100k': rate_per_100k,
            'population': n_alive
        }
    
    return notifications_by_year


def calculate_calibration_score(sim, calibration_data, weights=None, target_year=2018):
    """
    Calculate a composite calibration score based on multiple metrics
    
    Args:
        sim: Simulation object
        calibration_data: CalibrationData object containing targets and data
        weights: Dictionary of weights for different components
        target_year: Target year for overall prevalence
    
    Returns:
        dict: Calibration metrics and composite score
    """
    
    if weights is None:
        weights = {
            'case_notifications': 0.4,
            'age_prevalence': 0.4,
            'overall_prevalence': 0.2
        }
    
    # Compute model outputs
    notifications = compute_case_notifications(sim)
    age_prevalence = compute_age_stratified_prevalence(sim, target_year)
    
    # Case notification fit
    years = list(notifications.keys())
    model_rates = np.array([notifications[year]['rate_per_100k'] for year in years])
    data_rates = calibration_data.case_notifications['rate_per_100k'].values
    
    notification_rmse = np.sqrt(np.mean((model_rates - data_rates)**2))
    notification_mape = np.mean(np.abs((model_rates - data_rates) / data_rates)) * 100
    
    # Age prevalence fit
    age_groups = list(age_prevalence.keys())
    model_age_prev = np.array([age_prevalence[group]['prevalence_per_100k'] for group in age_groups])
    data_age_prev = calibration_data.age_prevalence['prevalence_per_100k'].values
    
    age_prev_rmse = np.sqrt(np.mean((model_age_prev - data_age_prev)**2))
    age_prev_mape = np.mean(np.abs((model_age_prev - data_age_prev) / data_age_prev)) * 100
    
    # Overall prevalence fit
    time_years = np.array([d.year for d in sim.results['timevec']])
    active_prev = sim.results['tb']['prevalence_active']
    target_idx = np.argmin(np.abs(time_years - target_year))
    model_overall_prev = active_prev[target_idx] * 100
    
    # Get target from calibration data
    target_overall_prev = None
    for target_name, target in calibration_data.targets.items():
        if 'prevalence' in target_name.lower() and target.year == target_year:
            target_overall_prev = target.value
            break
    
    if target_overall_prev is None:
        target_overall_prev = 0.852  # Default fallback
    
    overall_prev_error = abs(model_overall_prev - target_overall_prev)
    
    # Composite score (lower is better)
    composite_score = (
        weights['case_notifications'] * notification_mape +
        weights['age_prevalence'] * age_prev_mape +
        weights['overall_prevalence'] * (overall_prev_error * 100)
    )
    
    return {
        'notification_rmse': notification_rmse,
        'notification_mape': notification_mape,
        'age_prev_rmse': age_prev_rmse,
        'age_prev_mape': age_prev_mape,
        'overall_prev_error': overall_prev_error,
        'model_overall_prev': model_overall_prev,
        'target_overall_prev': target_overall_prev,
        'composite_score': composite_score
    }


def create_calibration_report(sim, calibration_data, timestamp, save_path=None):
    """
    Create a detailed calibration report with metrics
    
    Args:
        sim: Simulation object
        calibration_data: CalibrationData object
        timestamp: Timestamp for file naming
        save_path: Optional path to save the report
    
    Returns:
        dict: Calibration metrics
    """
    
    # Compute model outputs
    notifications = compute_case_notifications(sim)
    age_prevalence = compute_age_stratified_prevalence(sim)
    
    # Calculate fit metrics
    years = list(notifications.keys())
    model_rates = np.array([notifications[year]['rate_per_100k'] for year in years])
    data_rates = calibration_data.case_notifications['rate_per_100k'].values
    
    notification_rmse = np.sqrt(np.mean((model_rates - data_rates)**2))
    notification_mape = np.mean(np.abs((model_rates - data_rates) / data_rates)) * 100
    
    age_groups = list(age_prevalence.keys())
    model_age_prev = np.array([age_prevalence[group]['prevalence_per_100k'] for group in age_groups])
    data_age_prev = calibration_data.age_prevalence['prevalence_per_100k'].values
    
    age_prev_rmse = np.sqrt(np.mean((model_age_prev - data_age_prev)**2))
    age_prev_mape = np.mean(np.abs((model_age_prev - data_age_prev) / data_age_prev)) * 100
    
    # Overall prevalence fit
    time_years = np.array([d.year for d in sim.results['timevec']])
    active_prev = sim.results['tb']['prevalence_active']
    target_idx = np.argmin(np.abs(time_years - 2018))
    model_overall_prev = active_prev[target_idx] * 100
    
    target_overall_prev = 0.852  # Default
    for target_name, target in calibration_data.targets.items():
        if 'prevalence' in target_name.lower() and target.year == 2018:
            target_overall_prev = target.value
            break
    
    overall_prev_error = abs(model_overall_prev - target_overall_prev)
    
    # Create report
    report = {
        'timestamp': timestamp,
        'country': calibration_data.country,
        'description': calibration_data.description,
        'case_notifications': {
            'model_rates': model_rates.tolist(),
            'data_rates': data_rates.tolist(),
            'rmse': notification_rmse,
            'mape': notification_mape,
            'years': years
        },
        'age_prevalence': {
            'model_rates': model_age_prev.tolist(),
            'data_rates': data_age_prev.tolist(),
            'rmse': age_prev_rmse,
            'mape': age_prev_mape,
            'age_groups': age_groups
        },
        'overall_prevalence': {
            'model_2018': model_overall_prev,
            'target_2018': target_overall_prev,
            'error': overall_prev_error
        },
        'model_parameters': {
            'beta': sim.diseases.tb.pars.beta,
            'rel_sus_latentslow': sim.diseases.tb.pars.rel_sus_latentslow,
            'tb_mortality': sim.diseases.tb.pars.rate_smpos_to_dead,
            'hiv_prevalence': sim.results['hiv']['hiv_prevalence'][-1] if 'hiv' in sim.results else 0.0
        }
    }
    
    # Save report
    if save_path is None:
        save_path = f"calibration_report_{timestamp}.json"
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print summary
    print(f"\n=== TB Model Calibration Report ({calibration_data.country}) ===")
    print(f"Timestamp: {timestamp}")
    print(f"Description: {calibration_data.description}")
    print(f"\nCase Notification Fit:")
    print(f"  RMSE: {notification_rmse:.1f} per 100,000")
    print(f"  MAPE: {notification_mape:.1f}%")
    print(f"\nAge Prevalence Fit:")
    print(f"  RMSE: {age_prev_rmse:.1f} per 100,000")
    print(f"  MAPE: {age_prev_mape:.1f}%")
    print(f"\nOverall Prevalence (2018):")
    print(f"  Model: {model_overall_prev:.3f}%")
    print(f"  Target: {target_overall_prev:.3f}%")
    print(f"  Error: {overall_prev_error:.3f} percentage points")
    print(f"\nModel Parameters:")
    print(f"  Beta: {report['model_parameters']['beta']}")
    print(f"  Rel Sus Latent: {report['model_parameters']['rel_sus_latentslow']}")
    print(f"  TB Mortality: {report['model_parameters']['tb_mortality']}")
    print(f"  HIV Prevalence: {report['model_parameters']['hiv_prevalence']:.3f}")
    print(f"================================")
    
    return report


def create_south_africa_data():
    """
    Create synthetic South Africa TB data for calibration
    
    Returns:
        CalibrationData: Calibration data object
    """
    
    # Synthetic case notification data (per 100,000 population)
    case_notification_data = {
        'year': [2000, 2005, 2010, 2015, 2020],
        'rate_per_100k': [650, 950, 980, 834, 554],
        'total_cases': [280000, 450000, 490000, 450000, 320000],
        'source': ['WHO Global TB Report'] * 5
    }
    
    # Age-stratified active TB prevalence from 2018 survey (per 100,000)
    age_prevalence_data = {
        'age_group': ['15-24', '25-34', '35-44', '45-54', '55-64', '65+'],
        'prevalence_per_100k': [850, 1200, 1400, 1600, 1800, 2200],
        'prevalence_percent': [0.85, 1.20, 1.40, 1.60, 1.80, 2.20],
        'sample_size': [5000, 4500, 4000, 3500, 3000, 2500],
        'source': ['SA TB Prevalence Survey 2018'] * 6
    }
    
    # Create calibration targets
    targets = {
        'overall_prevalence_2018': CalibrationTarget(
            name='overall_prevalence_2018',
            value=0.852,
            year=2018,
            description='Overall TB prevalence from 2018 survey',
            source='SA TB Prevalence Survey 2018'
        ),
        'hiv_coinfection_rate': CalibrationTarget(
            name='hiv_coinfection_rate',
            value=0.60,
            description='HIV coinfection rate among TB cases',
            source='WHO Global TB Report'
        ),
        'case_detection_rate': CalibrationTarget(
            name='case_detection_rate',
            value=0.65,
            description='Case detection rate',
            source='WHO Global TB Report'
        ),
        'treatment_success_rate': CalibrationTarget(
            name='treatment_success_rate',
            value=0.78,
            description='Treatment success rate',
            source='WHO Global TB Report'
        ),
        'mortality_rate': CalibrationTarget(
            name='mortality_rate',
            value=0.12,
            description='Case fatality rate',
            source='WHO Global TB Report'
        )
    }
    
    return CalibrationData(
        case_notifications=pd.DataFrame(case_notification_data),
        age_prevalence=pd.DataFrame(age_prevalence_data),
        targets=targets,
        country='South Africa',
        description='TB model calibration data for South Africa'
    )


def create_country_data(country_name, case_notifications_data, age_prevalence_data, targets_dict):
    """
    Create calibration data for any country
    
    Args:
        country_name: Name of the country
        case_notifications_data: Dictionary with case notification data
        age_prevalence_data: Dictionary with age prevalence data
        targets_dict: Dictionary of calibration targets
    
    Returns:
        CalibrationData: Calibration data object
    """
    
    # Convert targets dict to CalibrationTarget objects
    targets = {}
    for target_name, target_info in targets_dict.items():
        if isinstance(target_info, dict):
            targets[target_name] = CalibrationTarget(
                name=target_name,
                value=target_info['value'],
                year=target_info.get('year'),
                age_group=target_info.get('age_group'),
                description=target_info.get('description'),
                source=target_info.get('source')
            )
        else:
            # Simple value
            targets[target_name] = CalibrationTarget(
                name=target_name,
                value=target_info
            )
    
    return CalibrationData(
        case_notifications=pd.DataFrame(case_notifications_data),
        age_prevalence=pd.DataFrame(age_prevalence_data),
        targets=targets,
        country=country_name,
        description=f'TB model calibration data for {country_name}'
    )


def load_calibration_data_from_file(file_path):
    """
    Load calibration data from a JSON file
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        CalibrationData: Calibration data object
    """
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert targets back to CalibrationTarget objects
    targets = {}
    for target_name, target_info in data['targets'].items():
        targets[target_name] = CalibrationTarget(**target_info)
    
    return CalibrationData(
        case_notifications=pd.DataFrame(data['case_notifications']),
        age_prevalence=pd.DataFrame(data['age_prevalence']),
        targets=targets,
        country=data['country'],
        description=data['description']
    )


def save_calibration_data_to_file(calibration_data, file_path):
    """
    Save calibration data to a JSON file
    
    Args:
        calibration_data: CalibrationData object
        file_path: Path to save the file
    """
    
    data = {
        'country': calibration_data.country,
        'description': calibration_data.description,
        'case_notifications': calibration_data.case_notifications.to_dict('records'),
        'age_prevalence': calibration_data.age_prevalence.to_dict('records'),
        'targets': {name: target.__dict__ for name, target in calibration_data.targets.items()}
    }
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str) 