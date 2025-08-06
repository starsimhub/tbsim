"""
TBsim Calibration Module

This module provides core calibration features for TB modeling including
data structures, functions, plotting, and simulation capabilities.
"""

from .functions import (
    compute_age_stratified_prevalence,
    compute_case_notifications,
    calculate_calibration_score,
    create_calibration_report,
    create_south_africa_data,
    create_country_data,
    load_calibration_data_from_file,
    save_calibration_data_to_file,
    CalibrationTarget,
    CalibrationData
)

from .simulation import (
    run_calibration_simulation_suite,
    run_calibration_simulation,
    create_demographics,
    create_tb_disease,
    create_hiv_disease,
    create_interventions,
    find_demographic_data,
    SimulationConfig,
    DiseaseConfig,
    InterventionConfig
)

__all__ = [
    # Calibration functions
    'compute_age_stratified_prevalence',
    'compute_case_notifications',
    'calculate_calibration_score',
    'create_calibration_report',
    'create_south_africa_data',
    'create_country_data',
    'load_calibration_data_from_file',
    'save_calibration_data_to_file',
    'CalibrationTarget',
    'CalibrationData',
    
    # Simulation functions
    'run_calibration_simulation_suite',
    'run_calibration_simulation',
    'create_demographics',
    'create_tb_disease',
    'create_hiv_disease',
    'create_interventions',
    'find_demographic_data',
    'SimulationConfig',
    'DiseaseConfig',
    'InterventionConfig',
] 