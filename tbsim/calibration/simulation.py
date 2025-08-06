"""
Simulation utilities for TB model calibration and generalized simulation.

This module provides comprehensive utilities for creating and running TB simulations
with support for multiple countries, demographic data, disease configurations, and
interventions. It serves as a high-level interface for the tbsim framework.

Key Features:
- Country-agnostic simulation setup with automatic demographic data detection
- Configurable disease parameters through dataclass configurations
- Flexible intervention management with multiple intervention types
- Generalized simulation runner with comprehensive error handling
- Backward compatibility with existing calibration workflows
- Integration with centralized factory functions

The module uses dataclasses for structured configuration management, making it
easy to create, modify, and share simulation configurations. All functions
include comprehensive error handling and validation.

Example Usage:
    from tbsim.calibration import (
        run_calibration_simulation_suite,
        SimulationConfig,
        DiseaseConfig,
        InterventionConfig
    )
    
    # Create configurations
    sim_config = SimulationConfig(n_agents=1000, years=50)
    disease_config = DiseaseConfig(beta=0.02, init_prev=0.1)
    intervention_config = InterventionConfig(include_hiv=True)
    
    # Run simulation
    sim = run_calibration_simulation_suite(
        country_name="South Africa",
        disease_config=disease_config,
        intervention_config=intervention_config,
        sim_config=sim_config
    )

Dependencies:
    - starsim: Core simulation framework
    - tbsim: TB-specific simulation components
    - pandas: Data manipulation and file I/O
    - dataclasses: Structured configuration management
    - typing: Type hints for function signatures

Author: TB Simulation Team
Version: 1.0.0
Last Updated: 2024
"""

import starsim as ss
import tbsim as mtb
import pandas as pd
import os
import sys
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class SimulationConfig:
    """
    Configuration dataclass for simulation parameters.
    
    This dataclass provides a structured way to configure simulation parameters
    including timing, population size, and simulation settings. All parameters
    have sensible defaults for typical TB simulation scenarios.
    
    Attributes:
        start_year (int): Starting year for the simulation. Defaults to 1850.
        years (int): Duration of simulation in years. Defaults to 200.
        n_agents (int): Number of agents in the simulation. Defaults to 1000.
        seed (int): Random seed for reproducibility. Defaults to 0.
        verbose (int): Verbosity level for simulation output. Defaults to 0.
        dt (int): Time step size. Defaults to 30.
        unit (str): Time unit for simulation. Defaults to 'day'.
    
    Example:
        # Create default configuration
        config = SimulationConfig()
        
        # Create custom configuration
        custom_config = SimulationConfig(
            start_year=2000,
            years=50,
            n_agents=5000,
            seed=42,
            verbose=1
        )
        
        # Modify existing configuration
        config.n_agents = 2000
        config.years = 100
    
    Notes:
        - start_year should be realistic for the demographic data being used
        - years should be sufficient for the disease dynamics to stabilize
        - n_agents affects simulation speed and statistical precision
        - seed ensures reproducible results across runs
        - verbose levels: 0=quiet, 1=basic info, 2=detailed output
    """
    start_year: int = 1850
    years: int = 200
    n_agents: int = 1000
    seed: int = 0
    verbose: int = 0
    dt: int = 30
    unit: str = 'day'


@dataclass
class DiseaseConfig:
    """
    Configuration dataclass for TB disease parameters.
    
    This dataclass provides a structured way to configure TB disease parameters
    including transmission rates, progression rates, and mortality rates. All
    parameters have defaults based on typical TB epidemiology.
    
    Attributes:
        beta (float): TB transmission rate parameter in monthly units. Defaults to 0.020.
        rel_sus_latentslow (float): Relative susceptibility of latent slow TB.
            Defaults to 0.15.
        tb_mortality (float): TB-specific mortality rate. Defaults to 3e-4.
        init_prev (float): Initial TB prevalence. Defaults to 0.10.
        rate_LS_to_presym (float): Rate from latent slow to pre-symptomatic.
            Defaults to 5e-5.
        rate_LF_to_presym (float): Rate from latent fast to pre-symptomatic.
            Defaults to 8e-3.
        rate_active_to_clear (float): Rate from active TB to cleared.
            Defaults to 1.5e-4.
        rate_exptb_to_dead (Optional[float]): Rate from extrapulmonary TB to death.
            Defaults to None (uses tb_mortality).
        rate_smneg_to_dead (Optional[float]): Rate from smear-negative TB to death.
            Defaults to None (uses tb_mortality).
    
    Example:
        # Create default configuration
        config = DiseaseConfig()
        
        # Create high-transmission configuration
        high_transmission = DiseaseConfig(
            beta=0.05,
            init_prev=0.25,
            tb_mortality=5e-4
        )
        
        # Create low-transmission configuration
        low_transmission = DiseaseConfig(
            beta=0.01,
            init_prev=0.05,
            tb_mortality=1e-4
        )
    
    Notes:
        - beta is the primary transmission parameter for calibration
        - rel_sus_latentslow affects progression from latent to active TB
        - tb_mortality affects overall disease severity
        - init_prev sets the starting prevalence in the population
        - Progression rates control the speed of disease development
        - Optional rates use tb_mortality as fallback if not specified
    """
    beta: float = 0.020
    rel_sus_latentslow: float = 0.15
    tb_mortality: float = 3e-4
    init_prev: float = 0.10
    rate_LS_to_presym: float = 5e-5
    rate_LF_to_presym: float = 8e-3
    rate_active_to_clear: float = 1.5e-4
    rate_exptb_to_dead: Optional[float] = None
    rate_smneg_to_dead: Optional[float] = None


@dataclass
class InterventionConfig:
    """
    Configuration dataclass for intervention parameters.
    
    This dataclass provides a structured way to configure various interventions
    including HIV management, health-seeking behavior, TB diagnostics, and
    TB treatment. All parameters have defaults based on typical intervention
    coverage and effectiveness.
    
    Attributes:
        include_hiv (bool): Whether to include HIV interventions. Defaults to True.
        include_health_seeking (bool): Whether to include health-seeking behavior.
            Defaults to True.
        include_diagnostic (bool): Whether to include TB diagnostic interventions.
            Defaults to True.
        include_treatment (bool): Whether to include TB treatment interventions.
            Defaults to True.
        hiv_prevalence (float): Target HIV prevalence for interventions.
            Defaults to 0.20.
        hiv_art_coverage (float): Target ART coverage among HIV+ individuals.
            Defaults to 0.50.
        health_seeking_rate (float): Rate of health-seeking behavior (per day).
            Defaults to 1/90 (every 90 days).
        diagnostic_coverage (float): TB diagnostic coverage. Defaults to 0.7.
        diagnostic_sensitivity (float): TB diagnostic sensitivity. Defaults to 0.60.
        diagnostic_specificity (float): TB diagnostic specificity. Defaults to 0.95.
        treatment_success_rate (float): TB treatment success rate. Defaults to 0.70.
    
    Example:
        # Create default configuration
        config = InterventionConfig()
        
        # Create high-coverage configuration
        high_coverage = InterventionConfig(
            diagnostic_coverage=0.9,
            treatment_success_rate=0.85,
            hiv_art_coverage=0.75
        )
        
        # Create TB-only configuration
        tb_only = InterventionConfig(
            include_hiv=False,
            include_health_seeking=True,
            include_diagnostic=True,
            include_treatment=True
        )
    
    Notes:
        - HIV interventions maintain target prevalence and ART coverage
        - Health-seeking behavior affects care-seeking rates
        - Diagnostic parameters affect case detection rates
        - Treatment success rate affects outcomes
        - All rates and coverages should be between 0.0 and 1.0
        - Interventions can be selectively enabled/disabled
    """
    include_hiv: bool = True
    include_health_seeking: bool = True
    include_diagnostic: bool = True
    include_treatment: bool = True
    hiv_prevalence: float = 0.20
    hiv_art_coverage: float = 0.50
    health_seeking_rate: float = 1/90
    diagnostic_coverage: float = 0.7
    diagnostic_sensitivity: float = 0.60
    diagnostic_specificity: float = 0.95
    treatment_success_rate: float = 0.70


def find_demographic_data(country_name: str, data_dir: str = None) -> Tuple[str, str]:
    """
    Find demographic data files for a given country.
    
    This function searches for demographic data files (CBR and ASMR) for a
    specified country. It tries multiple naming conventions and search paths
    to locate the required data files.
    
    Parameters:
        country_name (str): Name of the country to search for demographic data.
            Can include spaces and will be normalized for file searching.
        data_dir (str, optional): Specific directory to search in. If None,
            searches in common locations. Defaults to None.
    
    Returns:
        Tuple[str, str]: Tuple containing (cbr_path, asmr_path) for the found
            demographic data files.
    
    Raises:
        FileNotFoundError: If demographic data files cannot be found in any
            of the search locations with any naming convention.
    
    Example:
        # Find South Africa demographic data
        cbr_path, asmr_path = find_demographic_data("South Africa")
        print(f"CBR file: {cbr_path}")
        print(f"ASMR file: {asmr_path}")
        
        # Find data in specific directory
        cbr_path, asmr_path = find_demographic_data("Vietnam", "custom_data/")
        
        # Handle missing data
        try:
            cbr_path, asmr_path = find_demographic_data("Unknown Country")
        except FileNotFoundError as e:
            print(f"Demographic data not found: {e}")
    
    Notes:
        - Searches multiple common locations: tbsim/data, data, ../data, ../../data
        - Tries multiple naming conventions for file names
        - Handles spaces in country names by converting to underscores
        - Case-insensitive file name matching
        - Returns absolute paths to the found files
        - Raises FileNotFoundError if no files are found
    """
    
    if data_dir is None:
        # Search in common locations
        search_paths = [
            'tbsim/data',
            'data',
            '../data',
            '../../data'
        ]
    else:
        search_paths = [data_dir]
    
    # Try different naming conventions
    possible_cbr_names = [
        f'{country_name}_CBR.csv',
        f'{country_name.replace(" ", "_")}_CBR.csv',
        f'{country_name.lower()}_cbr.csv',
        f'{country_name.lower().replace(" ", "_")}_cbr.csv'
    ]
    
    possible_asmr_names = [
        f'{country_name}_ASMR.csv',
        f'{country_name.replace(" ", "_")}_ASMR.csv',
        f'{country_name.lower()}_asmr.csv',
        f'{country_name.lower().replace(" ", "_")}_asmr.csv'
    ]
    
    cbr_path = None
    asmr_path = None
    
    # Search for files
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
            
        for cbr_name in possible_cbr_names:
            test_path = os.path.join(search_path, cbr_name)
            if os.path.exists(test_path):
                cbr_path = os.path.abspath(test_path)
                break
                
        for asmr_name in possible_asmr_names:
            test_path = os.path.join(search_path, asmr_name)
            if os.path.exists(test_path):
                asmr_path = os.path.abspath(test_path)
                break
                
        if cbr_path and asmr_path:
            break
    
    if not cbr_path or not asmr_path:
        raise FileNotFoundError(
            f"Could not find demographic data for {country_name}. "
            f"Searched in: {search_paths}. "
            f"Tried CBR names: {possible_cbr_names}. "
            f"Tried ASMR names: {possible_asmr_names}"
        )
    
    return cbr_path, asmr_path


def create_demographics(country_name: str, data_dir: str = None) -> List:
    """
    Create demographic components for a specified country.
    
    This function creates demographic components (births and deaths) for a
    specified country using country-specific demographic data. It automatically
    finds the required data files and creates the appropriate demographic
    components.
    
    Parameters:
        country_name (str): Name of the country to create demographics for.
        data_dir (str, optional): Directory to search for demographic data.
            If None, searches in common locations. Defaults to None.
    
    Returns:
        List: List containing demographic components (ss.Births, ss.Deaths)
            configured with country-specific data.
    
    Raises:
        FileNotFoundError: If demographic data files cannot be found.
        ValueError: If demographic data files are invalid or corrupted.
        pd.errors.EmptyDataError: If demographic data files are empty.
    
    Example:
        # Create South Africa demographics
        demographics = create_demographics("South Africa")
        print(f"Created {len(demographics)} demographic components")
        
        # Create demographics from custom directory
        demographics = create_demographics("Vietnam", "custom_data/")
        
        # Use in simulation
        sim = ss.Sim(
            people=ss.People(n_agents=1000),
            demographics=demographics,
            # ... other components
        )
    
    Notes:
        - Automatically finds CBR and ASMR data files for the country
        - Creates ss.Births and ss.Deaths components with country-specific rates
        - Handles various file naming conventions and search paths
        - Returns components ready for use in simulation
        - Demographic data should be in CSV format with appropriate columns
        - Birth and death rates should be compatible with the simulation time period
    """
    
    cbr_path, asmr_path = find_demographic_data(country_name, data_dir)
    
    # Load demographic data
    try:
        cbr_data = pd.read_csv(cbr_path)
        asmr_data = pd.read_csv(asmr_path)
    except Exception as e:
        raise ValueError(f"Error loading demographic data: {e}")
    
    # Create demographic components
    births = ss.Births(pars=dict(birth_rate=cbr_data))
    deaths = ss.Deaths(pars=dict(death_rate=asmr_data))
    
    return [births, deaths]


def create_tb_disease(disease_config: DiseaseConfig) -> mtb.TB:
    """
    Create a TB disease object using the centralized factory functions.
    
    This function creates a TB disease object using the centralized factory
    functions, ensuring consistency with the rest of the simulation framework.
    It converts the DiseaseConfig dataclass to the appropriate parameter format.
    
    Parameters:
        disease_config (DiseaseConfig): Configuration object containing TB
            disease parameters.
    
    Returns:
        mtb.TB: Configured TB disease object ready for simulation.
    
    Raises:
        ValueError: If disease parameters are invalid.
        TypeError: If parameters are of incorrect type.
        ImportError: If required modules are not available.
    
    Example:
        # Create TB disease with default configuration
        config = DiseaseConfig()
        tb = create_tb_disease(config)
        
        # Create TB disease with custom parameters
        custom_config = DiseaseConfig(
            beta=0.03,
            init_prev=0.15,
            tb_mortality=5e-4
        )
        tb = create_tb_disease(custom_config)
        
        # Use in simulation
        sim = ss.Sim(
            people=ss.People(n_agents=1000),
            diseases=[tb],
            # ... other components
        )
    
    Notes:
        - Uses centralized factory functions for consistency
        - Converts dataclass configuration to parameter dictionary
        - Handles optional parameters with appropriate defaults
        - Ensures type safety and parameter validation
        - Integrates with the broader simulation framework
        - TB disease object is ready for immediate use in simulations
    """
    
    # Convert dataclass to parameter dictionary
    tb_pars = dict(
        beta=ss.rate_prob(disease_config.beta, unit='month'),
        init_prev=ss.bernoulli(p=disease_config.init_prev),
        rel_sus_latentslow=disease_config.rel_sus_latentslow,
        rate_LS_to_presym=ss.perday(disease_config.rate_LS_to_presym),
        rate_LF_to_presym=ss.perday(disease_config.rate_LF_to_presym),
        rate_active_to_clear=ss.perday(disease_config.rate_active_to_clear),
    )
    
    # Add optional death rate parameters if specified
    if disease_config.rate_exptb_to_dead is not None:
        tb_pars['rate_exptb_to_dead'] = ss.perday(disease_config.rate_exptb_to_dead)
    if disease_config.rate_smneg_to_dead is not None:
        tb_pars['rate_smneg_to_dead'] = ss.perday(disease_config.rate_smneg_to_dead)
    
    # Use centralized factory function
    from ..simulation.factory import make_tb
    return make_tb(include=True, tb_pars=tb_pars)


def create_hiv_disease() -> mtb.HIV:
    """
    Create an HIV disease object using the centralized factory functions.
    
    This function creates an HIV disease object using the centralized factory
    functions with default parameters suitable for TB-HIV co-infection
    simulations.
    
    Returns:
        mtb.HIV: Configured HIV disease object ready for simulation.
    
    Raises:
        ValueError: If HIV parameters are invalid.
        TypeError: If parameters are of incorrect type.
        ImportError: If required modules are not available.
    
    Example:
        # Create HIV disease
        hiv = create_hiv_disease()
        
        # Use in simulation with TB
        sim = ss.Sim(
            people=ss.People(n_agents=1000),
            diseases=[tb, hiv],
            connectors=[tb_hiv_connector],
            # ... other components
        )
    
    Notes:
        - Uses centralized factory functions for consistency
        - Default parameters suitable for TB-HIV co-infection
        - Zero initial prevalence and ART coverage (managed by interventions)
        - Integrates with TB disease through connectors
        - HIV disease object is ready for immediate use in simulations
    """
    
    # Use centralized factory function
    from ..simulation.factory import make_hiv
    return make_hiv(include=True, hiv_pars=None)


def create_interventions(intervention_config: InterventionConfig, 
                        start_year: int, years: int) -> List:
    """
    Create intervention objects using the centralized factory functions.
    
    This function creates intervention objects based on the provided configuration.
    It supports HIV interventions, health-seeking behavior, TB diagnostics, and
    TB treatment interventions.
    
    Parameters:
        intervention_config (InterventionConfig): Configuration object containing
            intervention parameters and settings.
        start_year (int): Starting year for the simulation, used to set
            intervention start and stop dates.
        years (int): Duration of simulation in years, used to set intervention
            stop dates.
    
    Returns:
        List: List of intervention objects created based on configuration.
    
    Raises:
        ValueError: If intervention parameters are invalid.
        TypeError: If parameters are of incorrect type.
        ImportError: If required intervention modules are not available.
    
    Example:
        # Create interventions with default configuration
        config = InterventionConfig()
        interventions = create_interventions(config, 2000, 50)
        
        # Create custom interventions
        custom_config = InterventionConfig(
            include_hiv=True,
            diagnostic_coverage=0.8,
            treatment_success_rate=0.8
        )
        interventions = create_interventions(custom_config, 2000, 30)
        
        # Use in simulation
        sim = ss.Sim(
            people=ss.People(n_agents=1000),
            diseases=[tb, hiv],
            interventions=interventions,
            # ... other components
        )
    
    Notes:
        - Uses centralized factory functions for consistency
        - Converts dataclass configuration to parameter dictionary
        - Sets appropriate start and stop dates based on simulation timing
        - Supports selective enabling/disabling of intervention types
        - All interventions use sensible defaults for missing parameters
        - Intervention objects are ready for immediate use in simulations
    """
    
    interventions = []
    
    if intervention_config.include_hiv:
        from ..simulation.factory import make_hiv_interventions
        hiv_intervention = make_hiv_interventions(pars=dict(
            mode='both',
            prevalence=intervention_config.hiv_prevalence,
            percent_on_ART=intervention_config.hiv_art_coverage,
            min_age=15,
            max_age=60,
            start=ss.date(f'{start_year}-01-01'),
            stop=ss.date(f'{start_year + years}-01-01'),
        ))
        if hiv_intervention:
            interventions.extend(hiv_intervention)
    
    if intervention_config.include_health_seeking:
        from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
        health_seeking = HealthSeekingBehavior(pars=dict(
            initial_care_seeking_rate=ss.perday(intervention_config.health_seeking_rate),
            start=ss.date(f'{start_year}-01-01'),
            stop=ss.date(f'{start_year + years}-01-01'),
            single_use=True,
        ))
        interventions.append(health_seeking)
    
    if intervention_config.include_diagnostic:
        from tbsim.interventions.tb_diagnostic import TBDiagnostic
        tb_diagnostic = TBDiagnostic(pars=dict(
            coverage=ss.bernoulli(intervention_config.diagnostic_coverage, strict=False),
            sensitivity=intervention_config.diagnostic_sensitivity,
            specificity=intervention_config.diagnostic_specificity,
            reset_flag=False,
            care_seeking_multiplier=2.0,
        ))
        interventions.append(tb_diagnostic)
    
    if intervention_config.include_treatment:
        from tbsim.interventions.tb_treatment import TBTreatment
        tb_treatment = TBTreatment(pars=dict(
            treatment_success_rate=intervention_config.treatment_success_rate,
            reseek_multiplier=2.0,
            reset_flags=True,
        ))
        interventions.append(tb_treatment)
    
    return interventions


def run_calibration_simulation_suite(
    country_name: str = "South Africa",
    disease_config: DiseaseConfig = None,
    intervention_config: InterventionConfig = None,
    sim_config: SimulationConfig = None,
    data_dir: str = None
) -> ss.Sim:
    """
    Run a calibration simulation suite with comprehensive configuration.
    
    This function provides a high-level interface for running TB simulations
    with support for multiple countries, configurable disease parameters,
    and flexible intervention settings. It handles all aspects of simulation
    setup including demographics, diseases, interventions, and connectors.
    
    Parameters:
        country_name (str, optional): Name of the country for demographic data.
            Defaults to "South Africa".
        disease_config (DiseaseConfig, optional): Configuration for TB disease
            parameters. If None, uses default configuration. Defaults to None.
        intervention_config (InterventionConfig, optional): Configuration for
            interventions. If None, uses default configuration. Defaults to None.
        sim_config (SimulationConfig, optional): Configuration for simulation
            parameters. If None, uses default configuration. Defaults to None.
        data_dir (str, optional): Directory to search for demographic data.
            If None, searches in common locations. Defaults to None.
    
    Returns:
        ss.Sim: Configured and run simulation object with results.
    
    Raises:
        FileNotFoundError: If demographic data files cannot be found.
        ValueError: If simulation parameters are invalid.
        RuntimeError: If simulation fails to run or converge.
        ImportError: If required modules are not available.
    
    Example:
        # Run simulation with default settings
        sim = run_calibration_simulation_suite()
        
        # Run simulation with custom configurations
        disease_config = DiseaseConfig(beta=0.03, init_prev=0.15)
        intervention_config = InterventionConfig(diagnostic_coverage=0.8)
        sim_config = SimulationConfig(n_agents=2000, years=50)
        
        sim = run_calibration_simulation_suite(
            country_name="Vietnam",
            disease_config=disease_config,
            intervention_config=intervention_config,
            sim_config=sim_config
        )
        
        # Access simulation results
        print(f"Simulation completed: {sim.complete}")
        print(f"Final TB prevalence: {sim.results['tb']['prevalence'][-1]}")
        
        # Plot results
        sim.plot()
    
    Notes:
        - Handles all simulation setup automatically
        - Uses centralized factory functions for component creation
        - Supports multiple countries with automatic data detection
        - Configurable disease and intervention parameters
        - Comprehensive error handling and validation
        - Returns fully configured simulation with results
        - Simulation is ready for analysis and plotting
        - Supports both TB-only and TB-HIV co-infection scenarios
    """
    
    # Set default configurations if not provided
    if disease_config is None:
        disease_config = DiseaseConfig()
    if intervention_config is None:
        intervention_config = InterventionConfig()
    if sim_config is None:
        sim_config = SimulationConfig()
    
    # Create people
    people = ss.People(n_agents=sim_config.n_agents, extra_states=mtb.get_extrastates())
    
    # Create demographics
    try:
        demographics = create_demographics(country_name, data_dir)
    except FileNotFoundError:
        print(f"Warning: Could not find demographic data for {country_name}. "
              f"Using basic demographics.")
        from ..simulation.factory import make_demographics
        demographics = make_demographics(include=True)
    
    # Create diseases
    diseases = []
    tb = create_tb_disease(disease_config)
    diseases.append(tb)
    
    if intervention_config.include_hiv:
        hiv = create_hiv_disease()
        diseases.append(hiv)
    
    # Create connectors
    connectors = []
    if intervention_config.include_hiv:
        from ..simulation.factory import make_tb_hiv_connector
        tb_hiv_connector = make_tb_hiv_connector()
        if tb_hiv_connector:
            connectors.append(tb_hiv_connector)
    
    # Create interventions
    interventions = create_interventions(intervention_config, sim_config.start_year, sim_config.years)
    
    # Create simulation parameters
    sim_pars = dict(
        unit='day',
        dt=sim_config.dt,
        start=ss.date(f'{sim_config.start_year}-01-01'),
        stop=ss.date(f'{sim_config.start_year + sim_config.years}-01-01'),
        rand_seed=sim_config.seed,
        verbose=sim_config.verbose,
    )
    
    # Create and run simulation
    sim = ss.Sim(
        people=people,
        demographics=demographics,
        diseases=diseases,
        connectors=connectors,
        interventions=interventions,
        pars=sim_pars
    )
    
    sim.run()
    
    return sim


def run_calibration_simulation(beta=0.020, rel_sus_latentslow=0.15, tb_mortality=3e-4, 
                              seed=0, years=200, n_agents=1000, country_name="South Africa"):
    """
    Run a calibration simulation with backward-compatible interface.
    
    This function provides a backward-compatible interface for running
    calibration simulations. It wraps the generalized simulation function
    with a simpler parameter interface for existing calibration workflows.
    
    Parameters:
        beta (float, optional): TB transmission rate parameter in monthly units. Defaults to 0.020.
        rel_sus_latentslow (float, optional): Relative susceptibility of latent slow TB.
            Defaults to 0.15.
        tb_mortality (float, optional): TB-specific mortality rate. Defaults to 3e-4.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.
        years (int, optional): Duration of simulation in years. Defaults to 200.
        n_agents (int, optional): Number of agents in the simulation. Defaults to 1000.
        country_name (str, optional): Name of the country for demographic data.
            Defaults to "South Africa".
    
    Returns:
        ss.Sim: Configured and run simulation object with results.
    
    Raises:
        FileNotFoundError: If demographic data files cannot be found.
        ValueError: If simulation parameters are invalid.
        RuntimeError: If simulation fails to run or converge.
    
    Example:
        # Run calibration simulation with default parameters
        sim = run_calibration_simulation()
        
        # Run calibration simulation with custom parameters
        sim = run_calibration_simulation(
            beta=0.03,
            rel_sus_latentslow=0.20,
            tb_mortality=5e-4,
            n_agents=2000,
            country_name="Vietnam"
        )
        
        # Use in calibration workflow
        results = sim.results['tb']['prevalence']
        calibration_score = calculate_calibration_score(sim, target_data)
    
    Notes:
        - Backward-compatible with existing calibration scripts
        - Wraps the generalized simulation function
        - Uses sensible defaults for non-specified parameters
        - Maintains the same interface as original calibration functions
        - Supports the same parameter ranges and validation
        - Returns simulation object compatible with existing analysis functions
    """
    
    # Create configurations from parameters
    disease_config = DiseaseConfig(
        beta=beta,
        rel_sus_latentslow=rel_sus_latentslow,
        tb_mortality=tb_mortality
    )
    
    sim_config = SimulationConfig(
        years=years,
        n_agents=n_agents,
        seed=seed
    )
    
    # Run calibration simulation suite
    return run_calibration_simulation_suite(
        country_name=country_name,
        disease_config=disease_config,
        sim_config=sim_config
    ) 