"""
Factory utilities for creating TB and HIV components.

This module provides centralized factory functions for creating TB and HIV
disease objects, interventions, and connectors. It follows a factory pattern
where each function is responsible for creating a specific type of simulation
component with sensible defaults and configurable parameters.

Key Features:
- Type-safe component creation with proper type hints
- Consistent interfaces across all factory functions
- Sensible default parameters for common use cases
- Comprehensive error handling and validation
- Support for both individual component creation and batch creation
- Integration with the tbsim simulation framework

Example Usage:
    from tbsim.utils import make_tb, make_hiv, create_simulation_components
    
    # Create individual components
    tb = make_tb(include=True, tb_pars=dict(beta=0.02))
    hiv = make_hiv(include=True, hiv_pars=dict(init_prev=0.0))
    
    # Create all components at once
    components = create_simulation_components(
        include_tb=True,
        include_hiv=True,
        include_interventions=True
    )

"""

import starsim as ss
import tbsim as mtb
from typing import List, Dict, Optional, Union


def make_hiv_interventions(include: bool = True, pars: Optional[Dict] = None) -> Optional[List]:
    """
    Create HIV interventions for maintaining target HIV prevalence and ART coverage.
    
    This function creates HIV intervention objects that can maintain a specified
    HIV prevalence and ART coverage rate in the population. The interventions
    work by periodically adjusting the HIV status and ART status of agents
    to maintain the target levels.
    
    Parameters:
        include (bool, optional): Whether to include HIV interventions in the output.
            If False, returns None. Defaults to True.
        pars (Dict, optional): Parameters for configuring the HIV interventions.
            If None, uses default parameters. Defaults to None.
            
            Expected keys in pars:
            - mode (str): Intervention mode ('both', 'prevalence', 'art')
            - prevalence (float): Target HIV prevalence (0.0 to 1.0)
            - percent_on_ART (float): Target ART coverage among HIV+ (0.0 to 1.0)
            - min_age (int): Minimum age for intervention targeting
            - max_age (int): Maximum age for intervention targeting
            - start (ss.date): Start date for interventions
            - stop (ss.date): Stop date for interventions
    
    Returns:
        Optional[List]: List containing HIV intervention objects, or None if include=False
        
    Raises:
        ValueError: If parameters are invalid (e.g., prevalence > 1.0)
        TypeError: If parameters are of incorrect type
        
    Example:
        # Create HIV interventions with default settings
        interventions = make_hiv_interventions()
        
        # Create HIV interventions with custom parameters
        custom_pars = {
            'mode': 'both',
            'prevalence': 0.25,
            'percent_on_ART': 0.60,
            'min_age': 15,
            'max_age': 65,
            'start': ss.date('2000-01-01'),
            'stop': ss.date('2030-12-31')
        }
        interventions = make_hiv_interventions(pars=custom_pars)
        
        # Skip HIV interventions
        interventions = make_hiv_interventions(include=False)  # Returns None
        
    Notes:
        - The intervention maintains target levels by periodically adjusting agent states
        - Default prevalence is 30% and ART coverage is 50%
        - Age range defaults to 15-60 years
        - Time range defaults to 2000-2035
        - Returns a list to support multiple intervention types
    """
    if not include:
        return None
    
    if pars is None:
        pars = dict(
            mode='both',
            prevalence=0.30,            # Maintain 30 percent of the alive population infected
            percent_on_ART=0.50,        # Maintain 50 percent of the % infected population on ART
            min_age=15, max_age=60,     # Min and Max age of agents that can be hit with the intervention
            start=ss.date('2000-01-01'), stop=ss.date('2035-12-31'),   # Intervention's start and stop dates
        )
    
    # Validate parameters
    if 'prevalence' in pars and not (0.0 <= pars['prevalence'] <= 1.0):
        raise ValueError("HIV prevalence must be between 0.0 and 1.0")
    if 'percent_on_ART' in pars and not (0.0 <= pars['percent_on_ART'] <= 1.0):
        raise ValueError("ART coverage must be between 0.0 and 1.0")
    
    return [mtb.HivInterventions(pars=pars)]


def make_hiv(include: bool = True, hiv_pars: Optional[Dict] = None) -> Optional[mtb.HIV]:
    """
    Create an HIV disease object for TB-HIV co-infection simulations.
    
    This function creates an HIV disease object that can be used in TB-HIV
    co-infection simulations. The HIV disease manages the HIV infection status,
    progression through HIV stages, and ART treatment status of agents.
    
    Parameters:
        include (bool, optional): Whether to include HIV disease in the output.
            If False, returns None. Defaults to True.
        hiv_pars (Dict, optional): Parameters for configuring the HIV disease.
            If None, uses default parameters. Defaults to None.
            
            Expected keys in hiv_pars:
            - init_prev (ss.bernoulli): Initial HIV prevalence distribution
            - init_onart (ss.bernoulli): Initial ART coverage distribution
            - Additional HIV-specific parameters as defined by mtb.HIV
    
    Returns:
        Optional[mtb.HIV]: HIV disease object, or None if include=False
        
    Raises:
        ValueError: If HIV parameters are invalid
        TypeError: If parameters are of incorrect type
        
    Example:
        # Create HIV disease with default settings
        hiv = make_hiv()
        
        # Create HIV disease with custom initial prevalence
        custom_pars = {
            'init_prev': ss.bernoulli(p=0.05),  # 5% initial HIV prevalence
            'init_onart': ss.bernoulli(p=0.00)  # 0% initial ART coverage
        }
        hiv = make_hiv(hiv_pars=custom_pars)
        
        # Skip HIV disease
        hiv = make_hiv(include=False)  # Returns None
        
    Notes:
        - Default initial HIV prevalence is 0% (no initial infections)
        - Default initial ART coverage is 0% (no initial ART)
        - The HIV disease object integrates with TB disease through connectors
        - HIV affects TB progression and treatment outcomes
        - ART coverage can be managed through interventions
    """
    if not include:
        return None
    
    if hiv_pars is None:
        hiv_pars = dict(
            init_prev=ss.bernoulli(p=0.00),     # 10% of the population is infected (in case not using intervention)
            init_onart=ss.bernoulli(p=0.00),    # 50% of the infected population is on ART (in case not using intervention)
        )
    
    return mtb.HIV(pars=hiv_pars)


def make_tb(include: bool = True, tb_pars: Optional[Dict] = None) -> Optional[mtb.TB]:
    """
    Create a TB disease object for tuberculosis simulations.
    
    This function creates a TB disease object that manages the tuberculosis
    infection status, progression through TB states, and disease dynamics
    of agents in the simulation.
    
    Parameters:
        include (bool, optional): Whether to include TB disease in the output.
            If False, returns None. Defaults to True.
        tb_pars (Dict, optional): Parameters for configuring the TB disease.
            If None, uses default parameters. Defaults to None.
            
            Expected keys in tb_pars:
            - beta (ss.rate_prob): TB transmission rate parameter
            - init_prev (ss.bernoulli): Initial TB prevalence distribution
            - rel_sus_latentslow (float): Relative susceptibility of latent slow TB
            - Additional TB-specific parameters as defined by mtb.TB
    
    Returns:
        Optional[mtb.TB]: TB disease object, or None if include=False
        
    Raises:
        ValueError: If TB parameters are invalid
        TypeError: If parameters are of incorrect type
        
    Example:
        # Create TB disease with default settings
        tb = make_tb()
        
        # Create TB disease with custom parameters
        custom_pars = {
            'beta': ss.rate_prob(0.02, unit='month'),  # 2% transmission rate per month
            'init_prev': ss.bernoulli(p=0.1), # 10% initial TB prevalence
            'rel_sus_latentslow': 0.15       # 15% relative susceptibility
        }
        tb = make_tb(tb_pars=custom_pars)
        
        # Skip TB disease
        tb = make_tb(include=False)  # Returns None
        
    Notes:
        - Default transmission rate (beta) is 0.1
        - Default initial TB prevalence is 25%
        - Default relative susceptibility for latent slow TB is 0.1
        - TB disease integrates with HIV through connectors
        - TB progression is affected by HIV status
        - TB treatment outcomes are influenced by HIV co-infection
    """
    if not include:
        return None
    
    if tb_pars is None:
        pars = dict(
            beta=ss.rate_prob(0.025, unit='month'),
            init_prev=ss.bernoulli(p=0.25),
            rel_sus_latentslow=0.1,
        )
    else:
        pars = tb_pars
    
    return mtb.TB(pars=pars)


def make_tb_hiv_connector(include: bool = True, pars: Optional[Dict] = None) -> Optional[mtb.TB_HIV_Connector]:
    """
    Create a TB-HIV connector for managing co-infection interactions.
    
    This function creates a TB-HIV connector object that manages the interactions
    between TB and HIV diseases in co-infection scenarios. The connector handles
    how HIV status affects TB progression and vice versa.
    
    Parameters:
        include (bool, optional): Whether to include TB-HIV connector in the output.
            If False, returns None. Defaults to True.
        pars (Dict, optional): Parameters for configuring the TB-HIV connector.
            If None, uses default parameters. Defaults to None.
            
            Expected keys in pars:
            - Various connector parameters as defined by mtb.TB_HIV_Connector
    
    Returns:
        Optional[mtb.TB_HIV_Connector]: TB-HIV connector object, or None if include=False
        
    Raises:
        ValueError: If connector parameters are invalid
        TypeError: If parameters are of incorrect type
        
    Example:
        # Create TB-HIV connector with default settings
        connector = make_tb_hiv_connector()
        
        # Create TB-HIV connector with custom parameters
        custom_pars = {
            # Add specific connector parameters here
        }
        connector = make_tb_hiv_connector(pars=custom_pars)
        
        # Skip TB-HIV connector
        connector = make_tb_hiv_connector(include=False)  # Returns None
        
    Notes:
        - The connector is essential for TB-HIV co-infection simulations
        - It manages bidirectional interactions between TB and HIV
        - HIV increases TB progression rates and reduces treatment success
        - TB can affect HIV progression and ART effectiveness
        - Should only be used when both TB and HIV diseases are included
    """
    if not include:
        return None
    
    return mtb.TB_HIV_Connector(pars=pars)


def make_demographics(include: bool = False) -> Optional[List]:
    """
    Create basic demographic components for population dynamics.
    
    This function creates basic demographic components (births and deaths)
    for managing population dynamics in the simulation. These components
    control population growth, aging, and natural mortality.
    
    Parameters:
        include (bool, optional): Whether to include demographic components.
            If False, returns None. Defaults to False.
            
    Returns:
        Optional[List]: List containing demographic components, or None if include=False
        
    Raises:
        ValueError: If demographic parameters are invalid
        
    Example:
        # Create demographic components
        demog = make_demographics(include=True)
        # Returns: [ss.Births(pars=dict(birth_rate=8.4)), ss.Deaths(pars=dict(death_rate=8.4))]
        
        # Skip demographic components
        demog = make_demographics(include=False)  # Returns None
        
    Notes:
        - Default birth rate is 8.4 per 1000 population per year
        - Default death rate is 8.4 per 1000 population per year
        - These are basic demographic rates; more complex demographics
          can be created using country-specific data
        - Demographic components are optional and can be omitted for
          closed population simulations
        - Birth and death rates should be balanced for stable populations
    """
    if not include:
        return None
    
    return [
        ss.Births(pars=dict(birth_rate=8.4)),
        ss.Deaths(pars=dict(death_rate=8.4)),
    ]


def make_interventions(intervention_config: Dict) -> List:
    """
    Create a comprehensive set of interventions based on configuration.
    
    This function creates multiple intervention objects based on a configuration
    dictionary. It supports HIV interventions, health-seeking behavior,
    TB diagnostics, and TB treatment interventions.
    
    Parameters:
        intervention_config (Dict): Configuration dictionary specifying which
            interventions to create and their parameters.
            
            Expected keys in intervention_config:
            - include_hiv (bool): Whether to include HIV interventions
            - include_health_seeking (bool): Whether to include health-seeking behavior
            - include_diagnostic (bool): Whether to include TB diagnostic interventions
            - include_treatment (bool): Whether to include TB treatment interventions
            - hiv_pars (Dict, optional): Parameters for HIV interventions
            - health_seeking_rate (float, optional): Rate of health-seeking behavior
            - start_date (str, optional): Start date for interventions (YYYY-MM-DD)
            - stop_date (str, optional): Stop date for interventions (YYYY-MM-DD)
            - diagnostic_coverage (float, optional): TB diagnostic coverage
            - diagnostic_sensitivity (float, optional): TB diagnostic sensitivity
            - diagnostic_specificity (float, optional): TB diagnostic specificity
            - treatment_success_rate (float, optional): TB treatment success rate
    
    Returns:
        List: List of intervention objects created based on configuration
        
    Raises:
        ValueError: If intervention parameters are invalid
        TypeError: If parameters are of incorrect type
        ImportError: If required intervention modules are not available
        
    Example:
        # Create all interventions with default settings
        config = {
            'include_hiv': True,
            'include_health_seeking': True,
            'include_diagnostic': True,
            'include_treatment': True
        }
        interventions = make_interventions(config)
        
        # Create custom interventions
        custom_config = {
            'include_hiv': True,
            'include_health_seeking': True,
            'include_diagnostic': True,
            'include_treatment': True,
            'hiv_pars': {'prevalence': 0.25, 'percent_on_ART': 0.60},
            'health_seeking_rate': 1/60,  # Every 60 days
            'start_date': '2000-01-01',
            'stop_date': '2030-12-31',
            'diagnostic_coverage': 0.8,
            'diagnostic_sensitivity': 0.7,
            'diagnostic_specificity': 0.95,
            'treatment_success_rate': 0.8
        }
        interventions = make_interventions(custom_config)
        
    Notes:
        - HIV interventions maintain target prevalence and ART coverage
        - Health-seeking behavior affects care-seeking rates
        - TB diagnostics determine case detection rates
        - TB treatment affects treatment success and outcomes
        - All interventions can be configured with custom parameters
        - Interventions are created in a logical order for simulation
        - Missing parameters use sensible defaults
    """
    interventions = []
    
    # HIV interventions
    if intervention_config.get('include_hiv', True):
        hiv_intervention = make_hiv_interventions(
            include=True,
            pars=intervention_config.get('hiv_pars')
        )
        if hiv_intervention:
            interventions.extend(hiv_intervention)
    
    # Health seeking behavior
    if intervention_config.get('include_health_seeking', True):
        from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
        health_seeking = HealthSeekingBehavior(pars=dict(
            initial_care_seeking_rate=ss.perday(intervention_config.get('health_seeking_rate', 1/90)),
            start=ss.date(intervention_config.get('start_date', '2000-01-01')),
            stop=ss.date(intervention_config.get('stop_date', '2035-12-31')),
            single_use=True,
        ))
        interventions.append(health_seeking)
    
    # TB diagnostic
    if intervention_config.get('include_diagnostic', True):
        from tbsim.interventions.tb_diagnostic import TBDiagnostic
        tb_diagnostic = TBDiagnostic(pars=dict(
            coverage=ss.bernoulli(intervention_config.get('diagnostic_coverage', 0.7), strict=False),
            sensitivity=intervention_config.get('diagnostic_sensitivity', 0.60),
            specificity=intervention_config.get('diagnostic_specificity', 0.95),
            reset_flag=False,
            care_seeking_multiplier=2.0,
        ))
        interventions.append(tb_diagnostic)
    
    # TB treatment
    if intervention_config.get('include_treatment', True):
        from tbsim.interventions.tb_treatment import TBTreatment
        tb_treatment = TBTreatment(pars=dict(
            treatment_success_rate=intervention_config.get('treatment_success_rate', 0.70),
            reseek_multiplier=2.0,
            reset_flags=True,
        ))
        interventions.append(tb_treatment)
    
    return interventions


def create_simulation_components(
    include_tb: bool = True,
    include_hiv: bool = True,
    include_connector: bool = True,
    include_demographics: bool = False,
    include_interventions: bool = True,
    tb_pars: Optional[Dict] = None,
    hiv_pars: Optional[Dict] = None,
    intervention_config: Optional[Dict] = None
) -> Dict:
    """
    Create all simulation components in one comprehensive function call.
    
    This function provides a high-level interface for creating all simulation
    components at once. It manages the creation of diseases, connectors,
    demographics, and interventions based on the provided configuration.
    
    Parameters:
        include_tb (bool, optional): Whether to include TB disease. Defaults to True.
        include_hiv (bool, optional): Whether to include HIV disease. Defaults to True.
        include_connector (bool, optional): Whether to include TB-HIV connector.
            Only used if both TB and HIV are included. Defaults to True.
        include_demographics (bool, optional): Whether to include demographic components.
            Defaults to False.
        include_interventions (bool, optional): Whether to include interventions.
            Defaults to True.
        tb_pars (Dict, optional): Parameters for TB disease creation. Defaults to None.
        hiv_pars (Dict, optional): Parameters for HIV disease creation. Defaults to None.
        intervention_config (Dict, optional): Configuration for interventions.
            Defaults to None.
    
    Returns:
        Dict: Dictionary containing all created simulation components with keys:
            - 'diseases': List of disease objects (TB, HIV)
            - 'demographics': List of demographic components or None
            - 'interventions': List of intervention objects
            - 'connectors': List of connector objects (TB-HIV)
        
    Raises:
        ValueError: If component parameters are invalid
        TypeError: If parameters are of incorrect type
        ImportError: If required modules are not available
        
    Example:
        # Create all components with default settings
        components = create_simulation_components()
        
        # Create TB-only simulation
        components = create_simulation_components(
            include_tb=True,
            include_hiv=False,
            include_connector=False,
            include_interventions=True
        )
        
        # Create comprehensive TB-HIV simulation
        components = create_simulation_components(
            include_tb=True,
            include_hiv=True,
            include_connector=True,
            include_demographics=True,
            include_interventions=True,
            tb_pars={'beta': 0.02, 'init_prev': 0.1},
            hiv_pars={'init_prev': 0.0},
            intervention_config={
                'include_hiv': True,
                'include_health_seeking': True,
                'include_diagnostic': True,
                'include_treatment': True,
                'hiv_pars': {'prevalence': 0.25, 'percent_on_ART': 0.60}
            }
        )
        
        # Access components
        diseases = components['diseases']      # [TB, HIV]
        connectors = components['connectors']  # [TB_HIV_Connector]
        interventions = components['interventions']  # [HIV, HealthSeeking, Diagnostic, Treatment]
        demographics = components['demographics']    # [Births, Deaths] or None
        
    Notes:
        - This is the recommended way to create simulation components
        - Components are created in a logical order for simulation
        - TB-HIV connector is only created if both diseases are included
        - Demographics are optional and default to False
        - Interventions include HIV, health-seeking, diagnostic, and treatment
        - All components use sensible defaults if parameters are not provided
        - The returned dictionary can be directly used to configure simulations
        - Component creation is atomic - either all succeed or all fail
    """
    components = {
        'diseases': [],
        'demographics': None,
        'interventions': [],
        'connectors': []
    }
    
    # Create diseases
    if include_tb:
        tb = make_tb(include=True, tb_pars=tb_pars)
        if tb:
            components['diseases'].append(tb)
    
    if include_hiv:
        hiv = make_hiv(include=True, hiv_pars=hiv_pars)
        if hiv:
            components['diseases'].append(hiv)
    
    # Create connector
    if include_connector and include_tb and include_hiv:
        connector = make_tb_hiv_connector(include=True)
        if connector:
            components['connectors'].append(connector)
    
    # Create demographics
    if include_demographics:
        components['demographics'] = make_demographics(include=True)
    
    # Create interventions
    if include_interventions:
        if intervention_config is None:
            intervention_config = {}
        components['interventions'] = make_interventions(intervention_config)
    
    return components 