"""
End-to-End TB Care Cascade Simulation

This script implements a complete end-to-end TB care cascade simulation that models the
natural progression of TB from infection through care seeking, diagnosis, treatment,
and preventive therapy for household contacts.

The End-to-End TB Care Cascade follows this sequence:
1. **TB Infection**: People get infected with TB through natural transmission
2. **Health Seeking**: Symptomatic individuals seek care at healthcare facilities
3. **Diagnostics**: Care-seeking individuals get tested for TB
4. **Treatment**: Diagnosed TB cases receive treatment
5. **TPT Initiation**: Household contacts of care-seeking individuals receive
   Tuberculosis Preventive Therapy (TPT)

**Ongoing TB Transmission**: The simulation includes continuous TB transmission throughout
the simulation period via the `beta` parameter, which controls the annual transmission rate.
This means new TB cases are constantly being introduced through:
- Person-to-person transmission from active TB cases
- Reinfection of previously infected individuals (slow progressors)
- Natural progression from latent to active TB states

Key Features:
- Realistic household contact tracing using the updated TPT intervention
- Multiple intervention scenarios with varying coverage levels
- Comprehensive analysis of care cascade efficiency
- Integration of BCG vaccination for enhanced protection

Scenarios:
- Baseline: No interventions (natural TB spread)
- High Transmission: Higher transmission rate (5% annually) with more ongoing cases
- Low Transmission: Lower transmission rate (1% annually) with fewer ongoing cases
- End-to-End TB Care Cascade: Complete care cascade with moderate coverage
- High Coverage End-to-End: Enhanced coverage across all interventions
- End-to-End with BCG: Complete cascade plus BCG vaccination
- High Transmission + Interventions: High transmission with full intervention package

Usage:
    python run_tb_e2e.py

The script will run all scenarios, display detailed analysis for each, and generate
comparative plots showing the impact of different intervention combinations.
"""

# Import required modules for TB simulation and analysis
import tbsim as mtb  # TB simulation module
import starsim as ss  # Core simulation framework
import sciris as sc   # Utility functions
import matplotlib.pyplot as plt  # Plotting library
import pprint as pprint  # Pretty printing for debugging
import pandas as pd  # Data manipulation
import numpy as np  # Numerical computations
from tbsim.interventions.immigration import SimpleImmigration  # Immigration intervention

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# These parameters define the basic structure of our TB simulation
# Default simulation parameters (spars) control timing and general settings
DEFAULT_SPARS = dict(
    dt=ss.days(7),                    # Time step: 7 days (weekly updates)
    start=ss.date('1975-01-01'),      # Simulation start date
    stop=ss.date('2030-12-31'),       # Simulation end date (55 years)
    rand_seed=123,                    # Random seed for reproducibility
    verbose=0,                        # Suppress detailed output during simulation
)

# Default TB disease parameters (tbpars) control TB-specific behavior
DEFAULT_TBPARS = dict(
    beta=ss.peryear(0.025),           # Base transmission rate: 2.5% annual infection rate
    init_prev=ss.bernoulli(p=0.10),   # Initial TB prevalence: 10% of population starts infected
    dt=ss.days(7),                    # TB module time step: weekly updates
    start=ss.date('1975-02-01'),      # TB module starts 1 month after simulation
    stop=ss.date('2030-12-31'),       # TB module runs until end of simulation
    
    # TB progression and transmission parameters
    rel_sus_latentslow=0.3,           # Slow progressors: 30% susceptibility to reinfection
    reltrans_het=ss.constant(v=1.0),  # Individual transmission heterogeneity (constant factor)
)

# =============================================================================
# POPULATION STRUCTURE
# =============================================================================
# Age distribution for the simulated population
# This creates a realistic age pyramid with higher proportions of younger individuals
age_data = pd.DataFrame({
    'age': [0, 2, 4, 10, 15, 20, 30, 40, 50, 60, 70, 80],           # Age groups
    'value': [20, 10, 25, 15, 10, 5, 4, 3, 2, 1, 1, 1]  # Population proportions (skewed toward younger ages)
})

# =============================================================================
# DEMOGRAPHIC MODULES
# =============================================================================
# Functions to create demographic processes that affect population dynamics

def make_demographics(include_immigration=True, include_births=True, include_deaths=True):
    """
    Create demographic modules for the simulation.
    
    This function sets up the basic demographic processes that govern how the population
    changes over time. These processes are essential for realistic TB modeling because:
    - Immigration can introduce new TB cases
    - Births add susceptible individuals
    - Deaths remove individuals (both infected and uninfected)
    
    Parameters:
    -----------
    include_immigration : bool
        Whether to include immigration (new individuals entering population)
    include_births : bool
        Whether to include births (new individuals born into population)
    include_deaths : bool
        Whether to include deaths (individuals leaving population)
        
    Returns:
    --------
    list
        List of demographic module objects ready for simulation
    """
    demographics = []
    
    # Immigration: New individuals entering the population
    # This is important for TB modeling as immigrants may have different TB exposure histories
    if include_immigration:
        demographics.append(SimpleImmigration(pars=dict(
            immigration_rate=20,  # 20 immigrants per year (4% of 500-person population)
        )))
    
    # Births: New individuals born into the population
    # Newborns are typically TB-free and may receive BCG vaccination
    if include_births:
        demographics.append(ss.Births(pars=dict(
            birth_rate=ss.peryear(25),  # 25 births per 1000 population per year (5% annual growth)
            rel_birth=1.0,              # No difference in birth rates by TB status
        )))
    
    # Deaths: Individuals leaving the population
    # TB can increase mortality, but we use background mortality rates here
    if include_deaths:
        demographics.append(ss.Deaths(pars=dict(
            death_rate=ss.peryear(8),   # 8 deaths per 1000 population per year (1.6% annual mortality)
            rel_death=1.0,              # No difference in death rates by TB status (TB-specific mortality handled elsewhere)
        )))
    
    return demographics

# =============================================================================
# HOUSEHOLD STRUCTURE
# =============================================================================
# Households are important for TB modeling because TB transmission often occurs
# within households, and household contact tracing is a key intervention strategy

def create_sample_households(n_agents=500):
    """
    Create sample household structure for testing.
    
    This function creates a realistic household structure where TB transmission
    is more likely to occur. Each household contains 3-7 individuals, which is
    typical for TB modeling scenarios.
    
    Household structure is crucial for:
    - TB transmission modeling (higher transmission within households)
    - Contact tracing interventions
    - TPT (Tuberculosis Preventive Therapy) targeting household contacts
    
    Parameters:
    -----------
    n_agents : int
        Total number of individuals in the population
        
    Returns:
    --------
    list
        List of household lists, where each household contains agent IDs
    """
    households = []
    current_uid = 0
    while current_uid < n_agents:
        # Create households with 3-7 members (typical household size)
        household_size = min(np.random.randint(3, 8), n_agents - current_uid)
        if household_size < 2:  # Need at least 2 people for a household
            break
        household = list(range(current_uid, current_uid + household_size))
        households.append(household)
        current_uid += household_size
    return households

# =============================================================================
# SIMULATION BUILDER
# =============================================================================
# This is the core function that assembles all components into a complete simulation
def build_sim(scenario=None, spars=None):
    """
    Build and return a complete Starsim-based simulation instance for TB modeling,
    incorporating optional interventions and user-defined parameters.

    This function is the main simulation factory that:
    1. Merges default and user-provided parameters
    2. Creates the population with appropriate TB states
    3. Sets up TB disease dynamics
    4. Creates social and household networks for transmission
    5. Adds interventions in logical order (care seeking → diagnostics → treatment → TPT)
    6. Configures demographic processes (births, deaths, immigration)
    7. Returns a ready-to-run simulation object

    Args:
        scenario (dict, optional): A dictionary defining scenario-specific components.
            Expected keys:
                - 'tbpars' (dict): TB-specific simulation parameters (transmission, progression)
                - 'healthseeking' (dict, optional): Parameters for health seeking behavior
                - 'diagnostic' (dict, optional): Parameters for TB diagnostic intervention
                - 'treatment' (dict, optional): Parameters for TB treatment intervention
                - 'tptintervention' (dict, optional): Parameters for TPT intervention
                - 'bcgintervention' (dict, optional): Parameters for BCG intervention
                - 'demographics' (dict, optional): Which demographic processes to include
        spars (dict, optional): General simulation parameters (timestep, duration, etc.).
            These override values in the DEFAULT_SPARS global dictionary.

    Returns:
        ss.Sim: A fully initialized simulation object containing:
            - A population (`People`) with TB-related extra states
            - A TB disease module with realistic progression dynamics
            - Social networks (random contacts) and household networks (close contacts)
            - Interventions in logical care cascade order
            - Demographic processes (births, deaths, immigration)
            - Core simulation parameters merged from defaults and user inputs

    Notes:
        - The intervention order matters: Health Seeking → Diagnostics → Treatment → TPT
        - Each intervention builds on the previous one in the care cascade
        - Multiple interventions of the same type can be added with unique names
    
    Example:
        # Create a basic simulation
        sim = build_sim()
        
        # Create a simulation with custom TB parameters
        custom_scenario = {'tbpars': {'beta': ss.peryear(0.05)}}
        sim = build_sim(scenario=custom_scenario)
        
        # Run the simulation
        sim.run()
    """
    scenario = scenario or {}
    
    # =============================================================================
    # PARAMETER MERGING
    # =============================================================================
    # Merge default parameters with user-provided overrides
    # This allows customization while maintaining sensible defaults
    spars = {**DEFAULT_SPARS, **(spars or {})}  # Simulation parameters
    tbpars = {**DEFAULT_TBPARS, **(scenario.get('tbpars') or {})}  # TB disease parameters
    
    # =============================================================================
    # INTERVENTION SETUP
    # =============================================================================
    # Create interventions list in logical order for end-to-end care cascade
    # The order matters: each intervention builds on the previous one
    interventions = []
    
    # 1. Health Seeking Behavior - People seek care when they have TB symptoms
    # This is the first step in the care cascade: symptomatic individuals decide to seek care
    # Parameters control the probability and timing of care-seeking behavior
    healthseeking_params = scenario.get('healthseeking')
    if healthseeking_params:
        if isinstance(healthseeking_params, dict):
            # Single health seeking intervention
            interventions.append(mtb.HealthSeekingBehavior(pars=healthseeking_params))
        elif isinstance(healthseeking_params, list):
            # Multiple health seeking interventions (e.g., different time periods)
            for i, params in enumerate(healthseeking_params):
                params['name'] = f'HealthSeeking_{i}'  # Unique name for each intervention
                interventions.append(mtb.HealthSeekingBehavior(pars=params))
    
    # 2. TB Diagnostics - People who sought care get tested
    # This step depends on health seeking: only care-seekers can be tested
    # Parameters control test coverage, sensitivity, specificity, and timing
    diagnostic_params = scenario.get('diagnostic')
    if diagnostic_params:
        if isinstance(diagnostic_params, dict):
            # Single diagnostic intervention
            interventions.append(mtb.TBDiagnostic(pars=diagnostic_params))
        elif isinstance(diagnostic_params, list):
            # Multiple diagnostic interventions (e.g., different test types or time periods)
            for i, params in enumerate(diagnostic_params):
                params['name'] = f'Diagnostic_{i}'  # Unique name for each intervention
                interventions.append(mtb.TBDiagnostic(pars=params))
    
    # 3. TB Treatment - People diagnosed with TB get treated
    # This step depends on diagnostics: only diagnosed individuals can be treated
    # Parameters control treatment success rates, duration, and timing
    treatment_params = scenario.get('treatment')
    if treatment_params:
        if isinstance(treatment_params, dict):
            # Single treatment intervention
            interventions.append(mtb.TBTreatment(pars=treatment_params))
        elif isinstance(treatment_params, list):
            # Multiple treatment interventions (e.g., different treatment regimens or time periods)
            for i, params in enumerate(treatment_params):
                params['name'] = f'Treatment_{i}'  # Unique name for each intervention
                interventions.append(mtb.TBTreatment(pars=params))
    
    # 4. TPT Initiation - Household contacts of care-seeking individuals get TPT
    # This step depends on health seeking: TPT targets household contacts of care-seekers
    # Parameters control TPT coverage, eligibility criteria, and protection duration
    # TPT (Tuberculosis Preventive Therapy) prevents latent TB from progressing to active TB
    tpt_params = scenario.get('tptintervention')
    if tpt_params:
        if isinstance(tpt_params, dict):
            # Single TPT intervention
            interventions.append(mtb.TPTInitiation(pars=tpt_params))
        elif isinstance(tpt_params, list):
            # Multiple TPT interventions (e.g., different age groups or time periods)
            for i, params in enumerate(tpt_params):
                params['name'] = f'TPT_{i}'  # Unique name for each intervention
                interventions.append(mtb.TPTInitiation(pars=params))
    
    # 5. BCG interventions (can be single or multiple)
    # BCG (Bacille Calmette-Guérin) vaccination provides partial protection against TB
    # This intervention can run independently of the care cascade
    # Parameters control vaccination coverage, age targeting, and protection duration
    bcg_params = scenario.get('bcgintervention')
    if bcg_params:
        if isinstance(bcg_params, dict):
            # Single BCG intervention (e.g., childhood vaccination)
            interventions.append(mtb.BCGProtection(pars=bcg_params))
        elif isinstance(bcg_params, list):
            # Multiple BCG interventions (e.g., different age groups or time periods)
            for i, params in enumerate(bcg_params):
                params['name'] = f'BCG_{i}'  # Give unique name
                interventions.append(mtb.BCGProtection(pars=params))
    
    # 6. Beta interventions (can be single or multiple)
    # Beta interventions control TB transmission rates over time
    # This can simulate changes in transmission due to public health measures, social distancing, etc.
    # Parameters define transmission rates for different time periods
    beta_params = scenario.get('betabyyear')
    if beta_params:
        if isinstance(beta_params, dict):
            # Single Beta intervention (e.g., gradual transmission reduction over time)
            interventions.append(mtb.BetaByYear(pars=beta_params))
        elif isinstance(beta_params, list):
            # Multiple Beta interventions (e.g., different transmission scenarios)
            for i, params in enumerate(beta_params):
                params['name'] = f'Beta_{i}'  # Give unique name
                interventions.append(mtb.BetaByYear(pars=params))
    
    # =============================================================================
    # SIMULATION COMPONENTS
    # =============================================================================
    # Create the core simulation components: population, disease, and networks
    
    # Population: 500 individuals with realistic age distribution and TB-related states
    pop = ss.People(n_agents=500, age_data=age_data, extra_states=mtb.get_extrastates())
    
    # TB Disease: Core TB dynamics with transmission, progression, and treatment
    tb = mtb.TB(pars=tbpars)
    
    # Networks: Define how people interact and transmit TB
    # Create household structure for household-based transmission and contact tracing
    households = create_sample_households(500)
    
    networks = [
        # Random contacts: General population mixing (5 contacts per person)
        ss.RandomNet({'n_contacts': ss.poisson(lam=5), 'dur': 0}),
        # Household contacts: Close contacts within households (for TB transmission and TPT)
        mtb.HouseholdNet(hhs=households, pars={'add_newborns': True})
    ]
    
    # =============================================================================
    # DEMOGRAPHIC PROCESSES
    # =============================================================================
    # Add demographic processes if specified in scenario
    # These control population dynamics: immigration, births, and deaths
    demographics = scenario.get('demographics', {})
    if demographics:
        demo_modules = make_demographics(
            include_immigration=demographics.get('include_immigration', True),
            include_births=demographics.get('include_births', True),
            include_deaths=demographics.get('include_deaths', True)
        )
    else:
        # No demographic processes if not specified
        demo_modules = None
    
    # =============================================================================
    # SIMULATION ASSEMBLY
    # =============================================================================
    # Assemble all components into a complete, ready-to-run simulation
    return ss.Sim(
        people=pop,              # Population with TB states
        networks=networks,       # Social and household networks
        interventions=interventions,  # Care cascade interventions
        diseases=[tb],           # TB disease dynamics
        demographics=demo_modules,    # Population dynamics
        pars=spars,              # Simulation parameters
    )

# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================
# This function defines all the different scenarios we want to test
def get_scenarios():
    """
    Define all TB simulation scenarios for comparison.
    
    Each scenario represents a different approach to TB control:
    - Baseline: Natural TB spread without interventions
    - Transmission scenarios: Different transmission rates
    - Care cascade scenarios: Complete end-to-end TB care
    - Coverage scenarios: Different levels of intervention coverage
    - Combined scenarios: Multiple interventions working together
    
    Returns:
    --------
    dict
        Dictionary mapping scenario names to their parameter configurations
    """
    return {
        # =============================================================================
        # BASELINE SCENARIOS
        # =============================================================================
        # These scenarios show natural TB dynamics without interventions
        
        'Baseline': {
            'name': 'No interventions - TB spreads naturally',
            'description': 'This scenario shows how TB spreads naturally without any interventions. '
                           'It serves as a control to measure the impact of other scenarios.',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
        },
        'High Transmission': {
            'name': 'High transmission rate - more ongoing TB cases',
            'description': 'This scenario simulates high TB transmission settings, such as crowded urban areas '
                           'or populations with high TB prevalence. It helps understand TB dynamics under '
                           'challenging conditions.',
            'tbpars': dict(
                start=ss.date('1975-01-01'), 
                stop=ss.date('2030-12-31'),
                beta=ss.peryear(0.05),           # 5% annual transmission rate (doubled from baseline)
                init_prev=ss.bernoulli(p=0.15),  # Higher initial prevalence (15% vs 10%)
            ),
        },
        'Low Transmission': {
            'name': 'Low transmission rate - fewer ongoing TB cases',
            'description': 'This scenario simulates low TB transmission settings, such as rural areas or '
                           'populations with good public health infrastructure. It shows TB dynamics '
                           'under favorable conditions.',
            'tbpars': dict(
                start=ss.date('1975-01-01'), 
                stop=ss.date('2030-12-31'),
                beta=ss.peryear(0.01),           # 1% annual transmission rate (reduced from baseline)
                init_prev=ss.bernoulli(p=0.05),  # Lower initial prevalence (5% vs 10%)
            ),
        },
        # =============================================================================
        # END-TO-END CARE CASCADE SCENARIOS
        # =============================================================================
        # These scenarios implement the complete TB care cascade:
        # 1. Health Seeking: Symptomatic individuals seek care
        # 2. Diagnostics: Care-seekers get tested for TB
        # 3. Treatment: Diagnosed individuals receive treatment
        # 4. TPT: Household contacts receive preventive therapy
        
        'End-to-End TB Care Cascade': {
            'name': 'Complete TB care cascade: Health Seeking → Diagnostics → Treatment → TPT',
            'description': 'This scenario implements the complete TB care cascade with moderate coverage levels. '
                           'It shows how a systematic approach to TB control can reduce transmission '
                           'and prevent new cases through household contact tracing and TPT.',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
            
            # Health Seeking: Symptomatic individuals seek care at healthcare facilities
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.15),  # 15% daily probability for symptomatic individuals
                start=ss.date('1980-01-01'),                # Interventions start in 1980
                stop=ss.date('2030-12-31'),
            ),
            
            # Diagnostics: Care-seekers get tested for TB
            'diagnostic': dict(
                coverage=0.85,        # 85% of care-seekers get tested
                sensitivity=0.80,     # 80% sensitivity (20% false negatives)
                specificity=0.95,     # 95% specificity (5% false positives)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            
            # Treatment: Diagnosed individuals receive treatment
            'treatment': dict(
                treatment_success_rate=0.60,  # 60% treatment success rate
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            
            # TPT: Household contacts of care-seekers receive preventive therapy
            'tptintervention': dict(
                p_tpt=0.75,                                    # 75% of eligible household contacts get TPT
                age_range=[1, 100],                           # TPT for ages 1-100 years
                hiv_status_threshold=False,                   # No HIV status restrictions
                tpt_treatment_duration=ss.peryear(0.25),      # 3 months treatment (3HP regimen)
                tpt_protection_duration=ss.peryear(2.0),      # 2 years of protection after completion
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
        'High Coverage End-to-End': {
            'name': 'High coverage TB care cascade with enhanced TPT',
            'description': 'This scenario represents an optimized TB care cascade with high coverage across '
                           'all interventions. It shows the potential impact when resources are available '
                           'for comprehensive TB control programs.',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
            
            # Enhanced health seeking with higher rates
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.25),  # 25% daily probability (increased from 15%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            
            # Enhanced diagnostics with higher coverage and accuracy
            'diagnostic': dict(
                coverage=0.95,        # 95% of care-seekers get tested (increased from 85%)
                sensitivity=0.85,     # 85% sensitivity (increased from 80%)
                specificity=0.98,     # 98% specificity (increased from 95%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            
            # Enhanced treatment with higher success rates
            'treatment': dict(
                treatment_success_rate=0.90,  # 90% treatment success (increased from 60%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            
            # Enhanced TPT with higher coverage and broader eligibility
            'tptintervention': dict(
                p_tpt=0.90,                                    # 90% of eligible household contacts get TPT (increased from 75%)
                age_range=[0, 100],                           # TPT for all ages (increased from 1-100)
                hiv_status_threshold=False,                   # No HIV status restrictions
                tpt_treatment_duration=ss.peryear(0.25),      # 3 months treatment (3HP regimen)
                tpt_protection_duration=ss.peryear(2.0),      # 2 years of protection after completion
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
        
        # =============================================================================
        # COMBINED INTERVENTION SCENARIOS
        # =============================================================================
        # These scenarios combine multiple types of interventions
        
        'End-to-End with BCG': {
            'name': 'Complete TB care cascade plus BCG vaccination',
            'description': 'This scenario combines the complete TB care cascade with BCG vaccination. '
                           'BCG provides partial protection against TB, especially in children, '
                           'and works synergistically with the care cascade interventions.',
            'tbpars': dict(start=ss.date('1975-01-01'), stop=ss.date('2030-12-31')),
            
            # BCG vaccination (currently commented out but available for testing)
            'bcgintervention': dict(
                coverage=0.50,                          # 80% BCG vaccination coverage
                start=ss.date('1975-01-01'),           # BCG starts early (before other interventions)
                stop=ss.date('2030-12-31'),            # BCG ends before care cascade begins
                age_range=[0, 5],                      # Early childhood vaccination (0-2 years)
            ),
            # Standard care cascade parameters (same as End-to-End TB Care Cascade)
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.15),
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'diagnostic': dict(
                coverage=0.85,
                sensitivity=0.80,
                specificity=0.95,
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'treatment': dict(
                treatment_success_rate=0.85,  # Slightly higher than standard (85% vs 60%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'tptintervention': dict(
                p_tpt=0.75,
                age_range=[0, 100],  # All ages eligible (vs 1-100 in standard)
                hiv_status_threshold=False,
                tpt_treatment_duration=ss.peryear(0.25),
                tpt_protection_duration=ss.peryear(2.0),
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
        'High Transmission + Interventions': {
            'name': 'High transmission with full intervention package',
            'description': 'This challenging scenario combines high TB transmission with comprehensive '
                           'interventions. It tests whether even high-quality interventions can '
                           'control TB in high-transmission settings, such as crowded urban areas.',
            'tbpars': dict(
                start=ss.date('1975-01-01'), 
                stop=ss.date('2030-12-31'),
                beta=ss.peryear(0.05),           # High transmission rate (5% annually)
                init_prev=ss.bernoulli(p=0.15),  # High initial prevalence (15%)
            ),
            
            # Enhanced interventions to combat high transmission
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.20),  # Higher care-seeking rate (20% vs 15%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'diagnostic': dict(
                coverage=0.90,        # Higher diagnostic coverage (90% vs 85%)
                sensitivity=0.85,     # Higher sensitivity (85% vs 80%)
                specificity=0.98,     # Higher specificity (98% vs 95%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'treatment': dict(
                treatment_success_rate=0.90,  # Higher treatment success (90% vs 60%)
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'tptintervention': dict(
                p_tpt=0.85,                                    # Higher TPT coverage (85% vs 75%)
                age_range=[1, 100],                           # TPT for ages 1-100
                hiv_status_threshold=False,                   # No HIV status restrictions
                tpt_treatment_duration=ss.peryear(0.25),      # 3 months treatment (3HP regimen)
                tpt_protection_duration=ss.peryear(2.0),      # 2 years of protection
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
        # =============================================================================
        # DEMOGRAPHIC SCENARIOS
        # =============================================================================
        # These scenarios include demographic processes (immigration, births, deaths)
        # to test how population dynamics affect TB control
        
        'With Immigration': {
            'name': 'TB simulation with immigration and demographics',
            'description': 'This scenario includes realistic demographic processes: immigration, births, '
                           'and deaths. Immigration can introduce new TB cases, while births add '
                           'susceptible individuals and deaths remove people from the population.',
            'tbpars': dict(
                start=ss.date('1975-01-01'), 
                stop=ss.date('2030-12-31'),
                beta=ss.peryear(0.03),           # Moderate transmission rate (3% annually)
                init_prev=ss.bernoulli(p=0.10),  # 10% initial prevalence
            ),
            
            # Include all demographic processes
            'demographics': dict(
                include_immigration=True,  # 20 immigrants per year
                include_births=True,       # 25 births per 1000 population per year
                include_deaths=True,       # 8 deaths per 1000 population per year
            ),
            
            # Moderate intervention coverage (realistic for resource-limited settings)
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.10),  # 10% daily care-seeking rate
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'diagnostic': dict(
                coverage=0.70,        # 70% of care-seekers get tested
                sensitivity=0.75,     # 75% sensitivity
                specificity=0.95,     # 95% specificity
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'treatment': dict(
                treatment_success_rate=0.80,  # 80% treatment success
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'tptintervention': dict(
                p_tpt=0.60,                                    # 60% of eligible household contacts get TPT
                age_range=[0, 100],                           # TPT for all ages
                hiv_status_threshold=False,                   # No HIV status restrictions
                tpt_treatment_duration=ss.peryear(0.25),      # 3 months treatment (3HP regimen)
                tpt_protection_duration=ss.peryear(2.0),      # 2 years of protection
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
        'High Immigration + Interventions': {
            'name': 'High immigration scenario with full TB care cascade',
            'description': 'This scenario tests TB control in a high-immigration setting with comprehensive '
                           'interventions. It simulates scenarios where immigration brings new TB cases '
                           'but the health system responds with high-quality interventions.',
            'tbpars': dict(
                start=ss.date('1975-01-01'), 
                stop=ss.date('2030-12-31'),
                beta=ss.peryear(0.04),           # Higher transmission due to immigration (4% annually)
                init_prev=ss.bernoulli(p=0.12),  # Higher initial prevalence (12%)
            ),
            
            # Include all demographic processes (same as With Immigration)
            'demographics': dict(
                include_immigration=True,  # 20 immigrants per year
                include_births=True,       # 25 births per 1000 population per year
                include_deaths=True,       # 8 deaths per 1000 population per year
            ),
            
            # Enhanced interventions to handle high immigration
            'healthseeking': dict(
                initial_care_seeking_rate=ss.perday(0.15),  # 15% daily care-seeking rate
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'diagnostic': dict(
                coverage=0.85,        # 85% of care-seekers get tested
                sensitivity=0.80,     # 80% sensitivity
                specificity=0.95,     # 95% specificity
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'treatment': dict(
                treatment_success_rate=0.85,  # 85% treatment success
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
            'tptintervention': dict(
                p_tpt=0.80,                                    # 80% of eligible household contacts get TPT
                age_range=[0, 100],                           # TPT for all ages
                hiv_status_threshold=False,                   # No HIV status restrictions
                tpt_treatment_duration=ss.peryear(0.25),      # 3 months treatment (3HP regimen)
                tpt_protection_duration=ss.peryear(2.0),      # 2 years of protection
                start=ss.date('1980-01-01'),
                stop=ss.date('2030-12-31'),
            ),
        },
    }

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
# Functions to analyze simulation results and extract key metrics

def analyze_e2e_results(sim):
    """
    Analyze end-to-end TB care cascade results from a completed simulation.
    
    This function extracts key metrics from the simulation to evaluate the
    effectiveness of the TB care cascade interventions. It provides insights
    into:
    - Care-seeking behavior
    - Diagnostic performance
    - Treatment outcomes
    - TPT coverage and effectiveness
    - TB transmission dynamics
    - Population health metrics
    
    Parameters:
    -----------
    sim : ss.Sim
        A completed simulation object with results
        
    Returns:
    --------
    dict
        Dictionary containing key metrics and analysis results
    """
    ppl = sim.people
    tb = sim.diseases.tb
    
    print("\n=== End-to-End TB Care Cascade Analysis ===")
    
    # =============================================================================
    # CARE CASCADE METRICS
    # =============================================================================
    # These metrics track the flow of individuals through the TB care cascade
    
    # Health seeking behavior: How many symptomatic individuals sought care?
    total_sought_care = np.sum(ppl.sought_care)
    print(f"Total individuals who sought care: {total_sought_care}")
    
    # Diagnostics: How many care-seekers got tested and diagnosed?
    total_tested = np.sum(ppl.tested)
    total_diagnosed = np.sum(ppl.diagnosed)
    print(f"Total individuals tested: {total_tested}")
    print(f"Total individuals diagnosed: {total_diagnosed}")
    
    # Treatment: How many diagnosed individuals are receiving treatment?
    total_treated = np.sum(tb.on_treatment)
    print(f"Total individuals currently on treatment: {total_treated}")
    
    # TPT: How many household contacts received preventive therapy?
    total_on_tpt = np.sum(ppl.on_tpt)
    total_received_tpt = np.sum(ppl.received_tpt)
    print(f"Total individuals currently on TPT: {total_on_tpt}")
    print(f"Total individuals who received TPT: {total_received_tpt}")
    
    # Protection: How many individuals are protected by TPT or BCG?
    total_protected = np.sum(tb.state == mtb.TBS.PROTECTED)
    print(f"Total individuals protected (TPT or BCG): {total_protected}")
    
    # =============================================================================
    # TB DISEASE STATES
    # =============================================================================
    # These metrics track the current TB disease burden in the population
    
    # Active TB: Individuals currently infectious with TB
    active_tb = np.sum(np.isin(tb.state, [mtb.TBS.ACTIVE_PRESYMP, mtb.TBS.ACTIVE_SMPOS, 
                                         mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB]))
    print(f"Total active TB cases: {active_tb}")
    
    # Latent TB: Individuals infected but not currently infectious
    latent_tb = np.sum(np.isin(tb.state, [mtb.TBS.LATENT_SLOW, mtb.TBS.LATENT_FAST]))
    print(f"Total latent TB cases: {latent_tb}")
    
    # =============================================================================
    # CARE CASCADE EFFICIENCY
    # =============================================================================
    # These metrics measure how effectively the care cascade functions
    
    if total_sought_care > 0:
        testing_rate = total_tested / total_sought_care
        diagnosis_rate = total_diagnosed / total_tested if total_tested > 0 else 0
        print(f"Testing rate (tested/sought care): {testing_rate:.2%}")
        print(f"Diagnosis rate (diagnosed/tested): {diagnosis_rate:.2%}")
    
    # =============================================================================
    # POPULATION HEALTH METRICS
    # =============================================================================
    # These metrics provide population-level health indicators
    
    total_population = len(ppl)
    active_tb_rate = active_tb / total_population if total_population > 0 else 0
    latent_tb_rate = latent_tb / total_population if total_population > 0 else 0
    print(f"Active TB rate: {active_tb_rate:.2%}")
    print(f"Latent TB rate: {latent_tb_rate:.2%}")
    
    # Transmission control effectiveness
    if total_protected > 0:
        protection_rate = total_protected / total_population if total_population > 0 else 0
        print(f"Protection rate (TPT + BCG): {protection_rate:.2%}")
    
    # =============================================================================
    # DEMOGRAPHIC METRICS
    # =============================================================================
    # These metrics track population dynamics (immigration, births, deaths)
    
    # Immigration: New individuals entering the population
    total_immigrants = 0
    if hasattr(sim.results, 'immigration') and hasattr(sim.results.immigration, 'n_immigrants'):
        total_immigrants = sim.results.immigration.n_immigrants.sum()
    
    # Births and deaths: Natural population change
    total_births = 0
    total_deaths = 0
    if hasattr(sim.results, 'births') and hasattr(sim.results.births, 'n_births'):
        total_births = sim.results.births.n_births.sum()
    if hasattr(sim.results, 'deaths') and hasattr(sim.results.deaths, 'n_deaths'):
        total_deaths = sim.results.deaths.n_deaths.sum()
    
    # =============================================================================
    # RETURN RESULTS
    # =============================================================================
    # Package all metrics into a dictionary for further analysis
    return {
        # Care cascade metrics
        'sought_care': total_sought_care,
        'tested': total_tested,
        'diagnosed': total_diagnosed,
        'on_treatment': total_treated,
        'on_tpt': total_on_tpt,
        'received_tpt': total_received_tpt,
        'protected': total_protected,
        
        # Disease burden metrics
        'active_tb': active_tb,
        'latent_tb': latent_tb,
        'active_tb_rate': active_tb_rate,
        'latent_tb_rate': latent_tb_rate,
        'protection_rate': protection_rate if total_protected > 0 else 0,
        
        # Demographic metrics
        'immigrants': total_immigrants,
        'births': total_births,
        'deaths': total_deaths,
        'population_growth': total_births + total_immigrants - total_deaths,
    }

def run_scenarios(plot=True, analyze=True, store_results=True):
    """
    Run all TB simulation scenarios and optionally analyze and plot results.
    
    This function:
    1. Gets all scenario definitions
    2. Runs each scenario
    3. Analyzes results if requested
    4. Creates comparative plots if requested
    5. Returns results for further analysis
    
    Parameters:
    -----------
    plot : bool
        Whether to create comparative plots of all scenarios
    analyze : bool
        Whether to analyze results for each scenario
        
    Returns:
    --------
    tuple
        (results, analysis_results) where:
        - results: Dictionary mapping scenario names to flattened simulation results
        - analysis_results: Dictionary mapping scenario names to analysis metrics
    """
    import tbsim.utils.plots as pl
    
    scenarios = get_scenarios()
    results = {}
    analysis_results = {}
    
    # =============================================================================
    # SCENARIO EXECUTION
    # =============================================================================
    # Run each scenario and collect results
    for name, scenario in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"Description: {scenario['name']}")
        print(f"{'='*60}")
        
        # Build and run the simulation
        sim = build_sim(scenario=scenario)
        sim.run()
        
        # Store flattened results for plotting
        results[name] = sim.results.flatten()
        
        # Analyze results if requested
        if analyze:
            analysis_results[name] = analyze_e2e_results(sim)
    
    # =============================================================================
    # STORE THE SIMULATION RESULTS IN AN OBJECT
    # =============================================================================
    # After the simulation is run, store the results in an object
    if store_results:
        results_object = sim 
        results_object.save('./results.pkl')
        print(f"Simulation results stored in ./results.pkl")
        
        
    
    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    # Create comparative plots if requested
    if plot:
        print("\nCreating comparative plots...")
        pl.plot_combined(results, 
                         dark=False,
                         heightfold=1.9, 
                         outdir='results/interventions', 
                         shared_legend=True)
        plt.show()
    
    return results, analysis_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================
# This section runs when the script is executed directly

if __name__ == '__main__':
    """
    Main execution block for the TB End-to-End Care Cascade simulation.
    
    This script will:
    1. Run all defined scenarios
    2. Analyze results for each scenario
    3. Create comparative plots
    4. Print a summary of all scenarios
    """
    
    # Run all scenarios with analysis and plotting
    results, analysis = run_scenarios(plot=True, analyze=True)
    
    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================
    # Print comprehensive summary of all scenarios
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL SCENARIOS")
    print(f"{'='*80}")
    
    # Print detailed results for each scenario
    for scenario_name, analysis_data in analysis.items():
        print(f"\n{scenario_name}:")
        print(f"  Care Cascade:")
        print(f"    - Sought care: {analysis_data['sought_care']}")
        print(f"    - Tested: {analysis_data['tested']}")
        print(f"    - Diagnosed: {analysis_data['diagnosed']}")
        print(f"    - On treatment: {analysis_data['on_treatment']}")
        print(f"    - Received TPT: {analysis_data['received_tpt']}")
        print(f"    - Protected: {analysis_data['protected']}")
        print(f"  Disease Burden:")
        print(f"    - Active TB: {analysis_data['active_tb']} ({analysis_data['active_tb_rate']:.1%})")
        print(f"    - Latent TB: {analysis_data['latent_tb']} ({analysis_data['latent_tb_rate']:.1%})")
        print(f"    - Protection rate: {analysis_data['protection_rate']:.1%}")
        
        # Add demographic metrics if available
        if analysis_data.get('immigrants', 0) > 0 or analysis_data.get('births', 0) > 0 or analysis_data.get('deaths', 0) > 0:
            print(f"  Demographics:")
            print(f"    - Immigrants: {analysis_data.get('immigrants', 0)}")
            print(f"    - Births: {analysis_data.get('births', 0)}")
            print(f"    - Deaths: {analysis_data.get('deaths', 0)}")
            print(f"    - Population growth: {analysis_data.get('population_growth', 0)}")
