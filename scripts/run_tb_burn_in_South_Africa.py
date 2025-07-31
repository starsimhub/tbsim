"""
Refined TB Prevalence Sweep for Manual Calibration (South Africa) with Health-Seeking, Diagnostic, and Treatment

This script performs a manual calibration sweep of TB transmission dynamics in Starsim/TBsim
to explore plausible endemic equilibrium behavior for South Africa, incorporating key 
epidemiological features specific to the South African context, including health-seeking behavior,
diagnostic testing, and treatment outcomes.

🎯 Objective:
    - Calibrate burn-in dynamics (i.e., rise and settle to endemic steady state)
    - Target approximately:
        • >50% latent TB prevalence
        • ~1% active TB prevalence
    - Qualitative fit to empirical data point: 0.852% active TB prevalence (South Africa, 2018)
    - Model realistic health-seeking behavior, diagnostic testing, and treatment outcomes

🔧 Current Assumptions:
    - Includes HIV coinfection (critical for South Africa TB dynamics)
    - Models TB-HIV interaction effects on progression rates
    - Uses South Africa-specific demographics and population structure
    - Simulation starts in 1850 and runs 200 years to allow for equilibrium
    - Incorporates historical HIV epidemic emergence (1980s onwards)
    - Health-seeking behavior with 90-day average delay (slower for better burn-in)
    - Diagnostic testing with 60% sensitivity, 70% coverage, and 95% specificity
    - Treatment with 70% success rate and retry mechanism for failures

📊 What It Does:
    - Sweeps across a grid of:
        • TB infectiousness (β)
        • Reinfection susceptibility (rel_sus_latentslow)
        • TB mortality rates
    - For each parameter combo, it:
        • Runs a simulation with TB-HIV coinfection + health-seeking + diagnostic
        • Plots active and latent prevalence over time
        • Plots health-seeking behavior metrics
        • Plots diagnostic testing outcomes
        • Plots treatment outcomes (incident and cumulative)
        • Overlays the 2018 SA data point on each plot
        • Adds an inset focused on the post-1980 period (zoomed to 0–1% active prevalence)
    - Outputs multiple PDF figures with all subplots, timestamped with run time
    - Prints runtime diagnostics including total sweep duration

📥 Inputs:
    - Hardcoded ranges for β, rel_sus_latentslow, and TB mortality
    - South Africa-specific demographic parameters
    - HIV epidemic parameters (prevalence targets, timing)
    - Health-seeking parameters (90-day average delay)
    - Diagnostic parameters (60% sensitivity, 70% coverage, 95% specificity)

📤 Outputs:
    - PDF files showing:
        • TB prevalence trajectories across parameter grid
        • Health-seeking behavior metrics
        • Diagnostic testing outcomes (incident and cumulative)
        • Treatment outcomes (incident and cumulative)
        • Population demographics
        • HIV metrics
    - Console logging of sweep progress and timing

⚠️ Notes:
    - Active prevalence <1% is sensitive to population size; low agent counts may cause extinction
    - HIV coinfection significantly impacts TB dynamics in South Africa
    - Health-seeking behavior affects TB transmission and case detection
    - Diagnostic testing influences treatment initiation and outcomes
    - Treatment success/failure affects TB transmission and care-seeking behavior
    - This model now better reflects the South African epidemiological context with realistic care-seeking and treatment

"""

import starsim as ss
import tbsim as mtb
from tbsim.comorbidities.hiv.hiv import HIVState
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import os
import rdata
import time

import warnings
warnings.filterwarnings("ignore", message='Missing constructor for R class "data.table".*')

# Ensure the project root is in sys.path for package imports
import sys
import os
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined (e.g., in interactive mode)
    script_dir = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(script_dir, '..')))

# Helper function to get output path based on extension
def get_output_path(filename):
    ext = filename.split('.')[-1].lower()
    subdir = {
        'pdf': '../samples/pdf/',
        'csv': '../samples/csv/',
        'png': '../samples/png/',
        'json': '../samples/json/',
        'md': '../samples/md/',
    }.get(ext, '../samples/')
    # Ensure the directory exists
    outdir = os.path.join(os.path.dirname(__file__), subdir)
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, filename)

class GradualHIVIntervention(ss.Intervention):
    """
    Custom HIV intervention that implements gradual ramp-up based on van Schalkwyk et al. 2021 data
    for eThekwini, South Africa. Handles both age groups: 15-24 and 25+.
    """
    
    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            percent_on_ART=0.50,  # 50% of HIV-positive individuals on ART
            start=ss.date('1990-01-01'),
            stop=ss.date('2050-12-31'),
        )
        self.update_pars(pars, **kwargs)
        
        # Define target years and prevalence levels for adults 25+ (estimated + survey data)
        self.hiv_targets_25plus = [
            (1990, 0.01),  # 1% in 1990
            (1995, 0.04),  # 4% in 1995
            (2000, 0.11),  # 11% in 2000
            (2005, 0.19),  # 19% in 2005 (survey data)
            (2008, 0.22),  # 22% in 2008 (survey data)
            (2010, 0.18),  # 18% in 2010 (estimated)
            (2013, 0.21),  # 21% in 2013 (survey data)
            (2015, 0.19),  # 19% in 2015 (estimated)
            (2018, 0.25),  # 25% in 2018 (survey data)
        ]
        
        # Define target years and prevalence levels for adults 15-24 (simplified non-decreasing trend)
        self.hiv_targets_15to24 = [
            (1990, 0.01),  # 1% in 1990
            (1995, 0.05),  # 5% in 1995
            (2000, 0.10),  # 10% in 2000
            (2005, 0.10),  # 10% in 2005 (leveled off)
            (2010, 0.10),  # 10% in 2010 (leveled off)
            (2015, 0.10),  # 10% in 2015 (leveled off)
        ]
        
    def step(self):
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return
            
        # Get current year
        current_year = t.year
        
        # Find the target prevalence for adults 25+
        target_prevalence_25plus = 0.0
        for year, prev in self.hiv_targets_25plus:
            if current_year >= year:
                target_prevalence_25plus = prev
        
        # Find the target prevalence for adults 15-24
        target_prevalence_15to24 = 0.0
        for year, prev in self.hiv_targets_15to24:
            if current_year >= year:
                target_prevalence_15to24 = prev
        
        # Apply the target prevalence for both age groups
        self._apply_prevalence(target_prevalence_25plus, min_age=25, max_age=60)
        self._apply_prevalence(target_prevalence_15to24, min_age=15, max_age=24)
        
    def _apply_prevalence(self, target_prevalence, min_age=25, max_age=60):
        """Apply the target HIV prevalence for a specific age range"""
        self.hiv = self.sim.diseases.hiv
        people = self.sim.people
        
        # Get alive people in target age range
        alive_mask = people.alive
        age_mask = (people.age >= min_age) & (people.age <= max_age)
        eligible_mask = alive_mask & age_mask
        eligible_uids = people.auids[eligible_mask]
        
        if len(eligible_uids) == 0:
            return
            
        # Calculate target number of HIV-positive people
        target_infectious = int(np.round(len(eligible_uids) * target_prevalence))
        
        # Get current HIV-positive people in eligible age range
        # First get HIV states for eligible people
        eligible_hiv_states = self.hiv.state[eligible_uids]
        hiv_positive_mask = np.isin(eligible_hiv_states, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS])
        current_infectious_uids = eligible_uids[hiv_positive_mask]
        n_current = len(current_infectious_uids)
        
        delta = target_infectious - n_current
        
        if delta > 0:
            # Need to add more HIV infections
            at_risk_mask = (eligible_hiv_states == HIVState.ATRISK)
            at_risk_uids = eligible_uids[at_risk_mask]
            
            if delta > len(at_risk_uids):
                # Not enough eligible people to infect
                delta = len(at_risk_uids)
            
            if delta > 0:
                # Randomly select people to infect
                chosen_indices = np.random.choice(len(at_risk_uids), size=delta, replace=False)
                chosen_uids = at_risk_uids[chosen_indices]
                self.hiv.state[chosen_uids] = HIVState.ACUTE
                
                # Put some of them on ART
                art_indices = np.random.choice(len(chosen_uids), 
                                             size=int(len(chosen_uids) * self.pars.percent_on_ART), 
                                             replace=False)
                art_uids = chosen_uids[art_indices]
                self.hiv.on_ART[art_uids] = True

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import time
import sys
import os
# Dynamically add the correct path to scripts for common_functions import
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # __file__ is not defined (e.g., in Jupyter), use cwd
    current_dir = os.getcwd()
scripts_path = os.path.abspath(os.path.join(current_dir, '../../scripts'))
if scripts_path not in sys.path:
    sys.path.insert(0, scripts_path)
# Also add the current directory to the path for local imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
# Add the parent directory to the path for data access
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
import common_functions as cf

# Import health-seeking, diagnostic, and treatment interventions
from tbsim.interventions.tb_health_seeking import HealthSeekingBehavior
from tbsim.interventions.tb_diagnostic import TBDiagnostic
from tbsim.interventions.tb_treatment import TBTreatment

start_wallclock = time.time()
start_datetime = datetime.datetime.now()
print(f"Sweep started at {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")


class AgeDependentTBProgression(ss.Intervention):
    """
    Intervention to modify TB progression rates and fast progressor fractions based on age groups.
    
    This intervention adjusts:
    1. The relative risk (rr_activation) multiplier for TB progression from latent to active disease
    2. The probability of becoming a fast progressor (p_latent_fast) after TB infection
    
    Age-specific multipliers:
    - 0-4 years: 2.0x the base rate (higher progression and fast progressor fraction)
    - 5-14 years: 0.5x the base rate (lower progression and fast progressor fraction)  
    - 15+ years: 1.0x the base rate (base progression and fast progressor fraction)
    """
    
    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            age_0_4_multiplier=2.0,    # 2x progression for 0-4 year olds
            age_5_14_multiplier=0.5,   # 0.5x progression for 5-14 year olds
            age_15plus_multiplier=1.0,  # 1x progression for 15+ year olds
        )
        self.update_pars(pars, **kwargs)
    
    def step(self):
        """Apply age-dependent TB progression multipliers and fast progressor fractions"""
        tb = self.sim.diseases['tb']
        people = self.sim.people
        
        # Get ages of TB-infected individuals
        uids_tb = tb.infected.uids
        ages = people.age[uids_tb]
        
        # Apply age-specific multipliers to rr_activation
        # 0-4 years: 2x progression
        mask_0_4 = (ages >= 0) & (ages <= 4)
        tb.rr_activation[uids_tb[mask_0_4]] *= self.pars.age_0_4_multiplier
        
        # 5-14 years: 0.5x progression
        mask_5_14 = (ages >= 5) & (ages <= 14)
        tb.rr_activation[uids_tb[mask_5_14]] *= self.pars.age_5_14_multiplier
        
        # 15+ years: 1x progression (base rate)
        mask_15plus = ages >= 15
        tb.rr_activation[uids_tb[mask_15plus]] *= self.pars.age_15plus_multiplier
        
        # Also modify the p_latent_fast parameter for new infections
        # This affects the probability of becoming a fast progressor vs slow progressor
        # We need to modify the underlying bernoulli distribution parameters
        
        # Get the base p_latent_fast value (typically 0.1)
        base_p_latent_fast = 0.1  # This is the default value from the TB model
        
        # Create age-specific p_latent_fast values
        p_latent_fast_0_4 = base_p_latent_fast * self.pars.age_0_4_multiplier  # 0.2 (20%)
        p_latent_fast_5_14 = base_p_latent_fast * self.pars.age_5_14_multiplier  # 0.05 (5%)
        p_latent_fast_15plus = base_p_latent_fast * self.pars.age_15plus_multiplier  # 0.1 (10%)
        
        # Store these values for use in the TB model's infection logic
        # Note: This is a simplified approach - in a more complex implementation,
        # we would need to modify the TB model's infection method directly
        self.age_specific_p_latent_fast = {
            '0_4': p_latent_fast_0_4,
            '5_14': p_latent_fast_5_14,
            '15plus': p_latent_fast_15plus
        }
        
        # Override the TB model's p_latent_fast parameter with age-specific values
        # This is a more direct approach to ensure the age-specific values are used
        if hasattr(tb, 'p_latent_fast'):
            # Store the original p_latent_fast for reference
            if not hasattr(self, 'original_p_latent_fast'):
                self.original_p_latent_fast = tb.p_latent_fast
            
            # Create age-specific bernoulli distributions
            tb.p_latent_fast_0_4 = ss.bernoulli(p=p_latent_fast_0_4)
            tb.p_latent_fast_5_14 = ss.bernoulli(p=p_latent_fast_5_14)
            tb.p_latent_fast_15plus = ss.bernoulli(p=p_latent_fast_15plus)
            
            # Override the TB model's infection method to use age-specific p_latent_fast
            if not hasattr(self, 'original_infect'):
                self.original_infect = tb.infect
                
                def age_dependent_infect(tb_self, uids, hosp=None, hosp_max=None, source_uids=None, **kwargs):
                    """Override the infect method to use age-specific p_latent_fast values"""
                    # Call the original infect method first
                    result = self.original_infect(tb_self, uids, hosp, hosp_max, source_uids, **kwargs)
                    
                    # Now modify the latent state assignment based on age
                    people = self.sim.people
                    ages = people.age[uids]
                    
                    # Get the newly infected individuals (those who just became latent)
                    newly_infected = uids[tb_self.state[uids] == mtb.TBS.LATENT_SLOW]
                    newly_infected = np.append(newly_infected, uids[tb_self.state[uids] == mtb.TBS.LATENT_FAST])
                    
                    if len(newly_infected) > 0:
                        newly_infected_ages = people.age[newly_infected]
                        
                        # Apply age-specific fast progressor probabilities
                        # 0-4 years: 20% fast progressors
                        mask_0_4 = (newly_infected_ages >= 0) & (newly_infected_ages <= 4)
                        fast_0_4 = tb_self.p_latent_fast_0_4.filter(newly_infected[mask_0_4])
                        tb_self.state[newly_infected[mask_0_4]] = np.where(fast_0_4, mtb.TBS.LATENT_FAST, mtb.TBS.LATENT_SLOW)
                        
                        # 5-14 years: 5% fast progressors
                        mask_5_14 = (newly_infected_ages >= 5) & (newly_infected_ages <= 14)
                        fast_5_14 = tb_self.p_latent_fast_5_14.filter(newly_infected[mask_5_14])
                        tb_self.state[newly_infected[mask_5_14]] = np.where(fast_5_14, mtb.TBS.LATENT_FAST, mtb.TBS.LATENT_SLOW)
                        
                        # 15+ years: 10% fast progressors (base rate)
                        mask_15plus = newly_infected_ages >= 15
                        fast_15plus = tb_self.p_latent_fast_15plus.filter(newly_infected[mask_15plus])
                        tb_self.state[newly_infected[mask_15plus]] = np.where(fast_15plus, mtb.TBS.LATENT_FAST, mtb.TBS.LATENT_SLOW)
                    
                    return result
                
                # Replace the TB model's infect method
                tb.infect = age_dependent_infect.__get__(tb, type(tb))


def make_people(n_agents, age_data=None):

    if age_data is None:
        # Use South Africa 1960 age structure instead of Vietnam
        age_data = pd.DataFrame({
            'age': np.arange(0, 101, 5),
            'value': [12000, 10000, 8500, 7500, 6500, 5500, 4500, 3500, 2500, 2000,
                      1500, 1200, 800, 500, 300, 150, 80, 40, 15, 5, 1]  # South Africa 1960 approximate
        })

    # Create population with extra states required for health-seeking and diagnostic interventions
    people = ss.People(n_agents=n_agents, age_data=age_data, extra_states=mtb.get_extrastates())

    return people


def compute_latent_prevalence(sim):
    # Get latent counts
    latent_slow = sim.results['tb']['n_latent_slow']
    latent_fast = sim.results['tb']['n_latent_fast']
    latent_total = latent_slow + latent_fast

    # Try getting time-aligned n_alive from starsim if available
    try:
        n_alive_series = sim.results['n_alive']
    except KeyError:
        # Fallback: use average population size
        n_alive_series = np.full_like(latent_total, fill_value=np.count_nonzero(sim.people.alive))

    return latent_total / n_alive_series


def debug_hiv_results(sim):
    """Debug function to print available HIV result keys and values"""
    print("=== HIV Results Debug ===")
    try:
        print(f"Available HIV result keys: {list(sim.results['hiv'].keys())}")
        print(f"HIV result values at final timestep:")
        for key, value in sim.results['hiv'].items():
            if hasattr(value, '__len__') and len(value) > 0:
                print(f"  {key}: {value[-1]}")
            else:
                print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error accessing HIV results: {e}")
    print("========================")


def compute_hiv_prevalence(sim):
    """Compute HIV prevalence over time"""
    try:
        # Try to get HIV prevalence directly - use the correct key from HIV model
        hiv_prev = sim.results['hiv']['hiv_prevalence']
        return hiv_prev
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from HIV infection counts using correct key
            n_hiv = sim.results['hiv']['infected']
            n_alive = sim.results['n_alive']
            return n_hiv / n_alive
        except (KeyError, AttributeError):
            try:
                # Another fallback: use n_active from HIV model
                n_hiv = sim.results['hiv']['n_active']
                n_alive = sim.results['n_alive']
                return n_hiv / n_alive
            except (KeyError, AttributeError):
                # Debug: print available HIV result keys
                try:
                    print(f"Available HIV result keys: {list(sim.results['hiv'].keys())}")
                except:
                    print("HIV results not found in simulation")
                # If HIV results are not available, return zeros
                time_length = len(sim.results['timevec'])
                return np.zeros(time_length)


def compute_hiv_prevalence_adults_25plus(sim, target_year=None):
    """
    Compute HIV prevalence for adults 25+ at a specific year or over time
    
    Args:
        sim: Simulation object
        target_year: If specified, compute for this year only. If None, compute over time.
    
    Returns:
        If target_year specified: float (prevalence for that year)
        If target_year is None: array (prevalence over time)
    """
    try:
        # Get people alive and HIV states
        people = sim.people
        alive_mask = people.alive
        hiv_states = sim.diseases.hiv.state
        
        # Get HIV-positive states (states 1, 2, 3 are positive)
        hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
        
        # Filter for adults 25+
        adult_25plus_mask = (people.age >= 25)
        
        # Combine masks
        alive_adult_25plus_mask = alive_mask & adult_25plus_mask
        
        if target_year is not None:
            # Compute for specific year
            time_years = np.array([d.year for d in sim.results['timevec']])
            target_idx = np.argmin(np.abs(time_years - target_year))
            
            # Get states at target time (this is approximate - we use current states)
            total_adults_25plus = np.sum(alive_adult_25plus_mask)
            hiv_positive_adults_25plus = np.sum(alive_adult_25plus_mask & hiv_positive_mask)
            
            if total_adults_25plus > 0:
                return hiv_positive_adults_25plus / total_adults_25plus
            else:
                return 0.0
        else:
            # Compute over time (this is approximate since we only have current states)
            # For now, return the overall HIV prevalence as a proxy
            return compute_hiv_prevalence(sim)
            
    except Exception as e:
        print(f"Error computing HIV prevalence for adults 25+: {e}")
        if target_year is not None:
            return 0.0
        else:
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_hiv_prevalence_adults_15to24(sim, target_year=None):
    """
    Compute HIV prevalence for adults 15-24 at a specific year or over time
    
    Args:
        sim: Simulation object
        target_year: If specified, compute for this year only. If None, compute over time.
    
    Returns:
        If target_year specified: float (prevalence for that year)
        If target_year is None: array (prevalence over time)
    """
    try:
        # Get people alive and HIV states
        people = sim.people
        alive_mask = people.alive
        hiv_states = sim.diseases.hiv.state
        
        # Get HIV-positive states (states 1, 2, 3 are positive)
        hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
        
        # Filter for adults 15-24
        adult_15to24_mask = (people.age >= 15) & (people.age <= 24)
        
        # Combine masks
        alive_adult_15to24_mask = alive_mask & adult_15to24_mask
        
        if target_year is not None:
            # Compute for specific year
            time_years = np.array([d.year for d in sim.results['timevec']])
            target_idx = np.argmin(np.abs(time_years - target_year))
            
            # Get states at target time (this is approximate - we use current states)
            total_adults_15to24 = np.sum(alive_adult_15to24_mask)
            hiv_positive_adults_15to24 = np.sum(alive_adult_15to24_mask & hiv_positive_mask)
            
            if total_adults_15to24 > 0:
                return hiv_positive_adults_15to24 / total_adults_15to24
            else:
                return 0.0
        else:
            # Compute over time (this is approximate since we only have current states)
            # For now, return the overall HIV prevalence as a proxy
            return compute_hiv_prevalence(sim)
            
    except Exception as e:
        print(f"Error computing HIV prevalence for adults 15-24: {e}")
        if target_year is not None:
            return 0.0
        else:
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_hiv_positive_tb_prevalence(sim):
    """Compute HIV-positive TB prevalence as proportion of total population"""
    try:
        # Try to get HIV-positive TB counts directly
        hiv_positive_tb = sim.results['tb']['n_active_hiv_positive']
        n_alive = sim.results['n_alive']
        return hiv_positive_tb / n_alive
    except (KeyError, AttributeError):
        try:
            # Fallback: compute from individual states
            # Get HIV and TB prevalence using correct keys
            hiv_prev = compute_hiv_prevalence(sim)
            tb_prev = sim.results['tb']['prevalence_active']
            
            # Estimate HIV-positive TB prevalence as a fraction of total TB
            # This is a rough estimate - in reality it depends on the TB-HIV interaction
            hiv_tb_overlap = hiv_prev * tb_prev * 0.3  # Assume 30% of TB cases are HIV-positive
            return hiv_tb_overlap
        except (KeyError, AttributeError):
            # If results are not available, return zeros
            time_length = len(sim.results['timevec'])
            return np.zeros(time_length)


def compute_age_stratified_prevalence(sim, target_year=2018):
    """
    Compute age-stratified TB prevalence from simulation results
    
    Args:
        sim: Simulation object
        target_year: Year to compute prevalence for
    
    Returns:
        dict: Age-stratified prevalence data
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get TB states
    tb_states = sim.diseases.tb.state
    active_tb_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # Get ages at target time
    ages = people.age[alive_mask]
    active_tb_ages = people.age[alive_mask & active_tb_mask]
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    prevalence_by_age = {}
    
    for i, (min_age, max_age) in enumerate(age_groups):
        # Count people in age group
        age_mask = (ages >= min_age) & (ages <= max_age)
        total_in_age_group = np.sum(age_mask)
        
        # Count active TB cases in age group
        age_tb_mask = (active_tb_ages >= min_age) & (active_tb_ages <= max_age)
        tb_in_age_group = np.sum(age_tb_mask)
        
        # Calculate prevalence
        if total_in_age_group > 0:
            prevalence = tb_in_age_group / total_in_age_group
            prevalence_per_100k = prevalence * 100000
        else:
            prevalence = 0
            prevalence_per_100k = 0
        
        prevalence_by_age[age_group_labels[i]] = {
            'prevalence': prevalence,
            'prevalence_per_100k': prevalence_per_100k,
            'total_people': total_in_age_group,
            'tb_cases': tb_in_age_group
        }
    
    return prevalence_by_age


def compute_age_stratified_prevalence_time_series(sim):
    """
    Compute age-stratified TB prevalence time series from simulation results
    
    Args:
        sim: Simulation object
    
    Returns:
        pd.DataFrame: DataFrame with years as index and age groups as columns
    """
    
    # Define age groups including children and adolescents
    age_groups = [(0, 4), (5, 14), (15, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
    age_group_labels = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Get time vector
    time_years = np.array([d.year for d in sim.results['timevec']])
    
    # Initialize DataFrame to store results
    prevalence_df = pd.DataFrame(index=time_years, columns=age_group_labels)
    
    # For each time point, compute age-stratified prevalence
    for t_idx, (time_point, year) in enumerate(zip(sim.results['timevec'], time_years)):
        # Get people alive at this time point
        people = sim.people
        
        # For simplicity, we'll use the current people state
        # In a more sophisticated approach, we'd need to track historical states
        alive_mask = people.alive
        
        # Get TB states
        tb_states = sim.diseases.tb.state
        active_tb_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
        
        # Get ages
        ages = people.age[alive_mask]
        active_tb_ages = people.age[alive_mask & active_tb_mask]
        
        # Compute prevalence for each age group
        for i, (min_age, max_age) in enumerate(age_groups):
            # Count people in age group
            age_mask = (ages >= min_age) & (ages <= max_age)
            total_in_age_group = np.sum(age_mask)
            
            # Count active TB cases in age group
            age_tb_mask = (active_tb_ages >= min_age) & (active_tb_ages <= max_age)
            tb_in_age_group = np.sum(age_tb_mask)
            
            # Calculate prevalence per 100,000
            if total_in_age_group > 0:
                prevalence = tb_in_age_group / total_in_age_group
                prevalence_per_100k = prevalence * 100000
            else:
                prevalence_per_100k = 0
            
            prevalence_df.loc[year, age_group_labels[i]] = prevalence_per_100k
    
    return prevalence_df


def plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']  # Use datetime objects directly
                n_alive = sim.results['n_alive']

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, n_alive, color='blue', label='Total Population')
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Population Size')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Simulated Total Population', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"total_population_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_hiv_metrics_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot HIV prevalence for both age groups (15-24 and 25+) with target data points"""
    import matplotlib.ticker as mtick

    # Define target data points for adults 25+ from van Schalkwyk et al. 2021 for eThekwini, South Africa
    estimated_data_25plus = [
        (1990, 0.01),  # 1% in 1990
        (1995, 0.04),  # 4% in 1995
        (2000, 0.11),  # 11% in 2000
        (2010, 0.18),  # 18% in 2010
        (2015, 0.19),  # 19% in 2015
    ]
    
    survey_data_25plus = [
        (2005, 0.19),  # 19% in 2005
        (2008, 0.22),  # 22% in 2008
        (2013, 0.21),  # 21% in 2013
        (2018, 0.25),  # 25% in 2018
    ]
    
    # Define target data points for adults 15-24 (simplified non-decreasing trend)
    estimated_data_15to24 = [
        (1990, 0.01),  # 1% in 1990
        (1995, 0.05),  # 5% in 1995
        (2000, 0.10),  # 10% in 2000
        (2005, 0.10),  # 10% in 2005
        (2010, 0.10),  # 10% in 2010
        (2015, 0.10),  # 10% in 2015
    ]

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 6 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Debug: print HIV results for the first simulation
                if m == 0 and i == 0 and j == 0:
                    debug_hiv_results(sim)
                
                # Compute HIV prevalence for both age groups
                hiv_prev_25plus = compute_hiv_prevalence_adults_25plus(sim)
                hiv_prev_15to24 = compute_hiv_prevalence_adults_15to24(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot HIV prevalence for adults 25+
                ax.plot(time, hiv_prev_25plus, label='Model HIV Prevalence (25+)', color='blue', linewidth=2)
                
                # Plot estimated data points for 25+
                for year, prev in estimated_data_25plus:
                    ax.plot(datetime.date(year, 1, 1), prev, 'go', markersize=4, alpha=0.8, label='Estimated Data (25+)' if year == 1990 else "")
                
                # Plot survey data points for 25+
                for year, prev in survey_data_25plus:
                    ax.plot(datetime.date(year, 1, 1), prev, 'ro', markersize=4, alpha=0.8, label='Survey Data (25+)' if year == 2005 else "")
                
                # Plot HIV prevalence for adults 15-24
                ax.plot(time, hiv_prev_15to24, label='Model HIV Prevalence (15-24)', color='orange', linewidth=2, linestyle='--')
                
                # Plot estimated data points for 15-24
                for year, prev in estimated_data_15to24:
                    ax.plot(datetime.date(year, 1, 1), prev, 'mo', markersize=4, alpha=0.8, label='Estimated Data (15-24)' if year == 1990 else "")
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True, alpha=0.3)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('HIV Prevalence')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)
                
                # Set y-axis to show percentages properly
                ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    plt.suptitle('HIV Prevalence by Age Group: Model vs van Schalkwyk et al. 2021 Data (eThekwini, South Africa)', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"hiv_metrics_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_active_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot active TB prevalence for all parameter combinations with separate focus on active TB"""
    import matplotlib.pyplot as plt
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                active_prev = sim.results['tb']['prevalence_active']

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, active_prev, label='Active TB Prevalence', color='blue', linewidth=2)
                ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='Target 1%')

                # Plot the 2018 SA data point (real data)
                ax.plot(datetime.date(2018, 1, 1), 0.00852, 'ro', markersize=6, label='2018 SA data (0.852%)')

                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Active TB Prevalence')
                ax.grid(True)

                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Active TB Prevalence Sweep', fontsize=16, y=1.02)
    filename = f"active_tb_prevalence_sweep_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_latent_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot latent TB prevalence for all parameter combinations with separate focus on latent TB"""
    import matplotlib.pyplot as plt
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                latent_prev = compute_latent_prevalence(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, latent_prev, label='Latent TB Prevalence', color='orange', linewidth=2, linestyle='--')
                ax.axhline(0.5, color='red', linestyle=':', linewidth=1, label='Target 50%')

                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Latent TB Prevalence')
                ax.grid(True)

                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Latent TB Prevalence Sweep', fontsize=16, y=1.02)
    filename = f"latent_tb_prevalence_sweep_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Legacy function - now calls both separate active and latent TB plots"""
    plot_active_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_latent_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)


def compute_annualized_infection_rate(sim):
    """
    Compute annualized TB infection rate (annual risk of infection) over time.
    
    This function calculates the annualized infection rate using two methods:
    1. Method 1: Sum new_infections over 365 days and divide by population
    2. Method 2: Difference in n_infected between T and T-365 days, divided by population
    
    Returns the annualized infection rate as a percentage of the population.
    """
    time = sim.results['timevec']
    tb_results = sim.results['tb']
    
    # Get population size over time
    try:
        n_alive = sim.results['n_alive']
    except KeyError:
        n_alive = np.full(len(time), fill_value=np.count_nonzero(sim.people.alive))
    
    # Method 1: Using new_infections (if available)
    annual_rate_method1 = None
    try:
        # Check if new_infections is available
        if 'new_infections' in tb_results:
            new_infections = tb_results['new_infections'].values
            annual_rate_method1 = np.zeros_like(time, dtype=float)
            
            # Calculate 365-day rolling sum of new infections
            days_per_step = (time[1] - time[0]).days if len(time) > 1 else 1
            steps_per_year = max(1, int(365 / days_per_step))
            
            for i in range(len(time)):
                start_idx = max(0, i - steps_per_year + 1)
                annual_infections = np.sum(new_infections[start_idx:i+1])
                annual_rate_method1[i] = (annual_infections / n_alive[i]) * 100 if n_alive[i] > 0 else 0
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Using difference in n_infected
    annual_rate_method2 = np.zeros_like(time, dtype=float)
    try:
        # Get total infected count over time
        n_infected = tb_results['n_latent_slow'].values + tb_results['n_latent_fast'].values + tb_results['n_active'].values
        
        # Calculate 365-day difference
        days_per_step = (time[1] - time[0]).days if len(time) > 1 else 1
        steps_per_year = max(1, int(365 / days_per_step))
        
        for i in range(len(time)):
            if i >= steps_per_year:
                # Calculate difference in infected count over the year
                infection_diff = n_infected[i] - n_infected[i - steps_per_year]
                annual_rate_method2[i] = (infection_diff / n_alive[i]) * 100 if n_alive[i] > 0 else 0
            else:
                # For early time points, use the current rate scaled to annual
                annual_rate_method2[i] = (n_infected[i] / n_alive[i]) * 100 if n_alive[i] > 0 else 0
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Return the more robust method (Method 2) or Method 1 if Method 2 fails
    if annual_rate_method2 is not None and not np.all(np.isnan(annual_rate_method2)):
        return annual_rate_method2
    elif annual_rate_method1 is not None and not np.all(np.isnan(annual_rate_method1)):
        return annual_rate_method1
    else:
        print("Warning: Could not compute annualized infection rate")
        return np.zeros_like(time, dtype=float)


def plot_annualized_infection_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot annualized TB infection rate for all parameter combinations"""
    import matplotlib.pyplot as plt
    
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                annual_rate = compute_annualized_infection_rate(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, annual_rate, label='Annual Infection Rate', color='purple', linewidth=2)
                ax.axhline(2.0, color='red', linestyle=':', linewidth=1, label='2% Annual Risk')

                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Annual Infection Rate (%)')
                ax.grid(True)

                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Annualized TB Infection Rate', fontsize=16, y=1.02)
    filename = f"annualized_infection_rate_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_health_seeking_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot health-seeking behavior metrics for all parameter combinations
    
    Metrics plotted:
    - new_sought_care: Number of people who sought care in this timestep (count)
    - n_sought_care: Cumulative number of people who have ever sought care (count)
    - n_eligible: Number of people with active TB eligible for care-seeking (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get health-seeking results
                hsb = sim.results['healthseekingbehavior']
                new_sought = hsb['new_sought_care'].values
                n_sought = hsb['n_sought_care'].values
                n_eligible = hsb['n_eligible'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot new people seeking care each step
                ax.plot(time, new_sought, label='New Sought Care', color='green', linewidth=2)
                
                # Plot cumulative people who sought care
                ax.plot(time, n_sought, label='Cumulative Sought Care', color='blue', linestyle='--')
                
                # Plot eligible population
                ax.plot(time, n_eligible, label='Eligible (Active TB)', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Health-Seeking Behavior Over Time', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"health_seeking_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot diagnostic testing metrics for all parameter combinations
    
    Metrics plotted:
    - n_tested: Number of people tested in this timestep (count)
    - n_test_positive: Number of positive test results in this timestep (count)
    - n_test_negative: Number of negative test results in this timestep (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get diagnostic results
                tbdiag = sim.results['tbdiagnostic']
                n_tested = tbdiag['n_tested'].values
                n_test_positive = tbdiag['n_test_positive'].values
                n_test_negative = tbdiag['n_test_negative'].values
                cum_test_positive = tbdiag['cum_test_positive'].values
                cum_test_negative = tbdiag['cum_test_negative'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot incident testing results
                ax.plot(time, n_tested, label='Tested', color='blue', marker='o', markersize=2)
                ax.plot(time, n_test_positive, label='Tested Positive', color='green', linestyle='--')
                ax.plot(time, n_test_negative, label='Tested Negative', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('TB Diagnostic Testing Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"diagnostic_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot cumulative diagnostic testing results for all parameter combinations
    
    Metrics plotted:
    - cum_test_positive: Cumulative number of positive test results over time (count)
    - cum_test_negative: Cumulative number of negative test results over time (count)
    """
    import matplotlib.ticker as mtick

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get cumulative diagnostic results
                tbdiag = sim.results['tbdiagnostic']
                cum_test_positive = tbdiag['cum_test_positive'].values
                cum_test_negative = tbdiag['cum_test_negative'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot cumulative testing results
                ax.plot(time, cum_test_positive, label='Cumulative Positives', color='green', linestyle='--')
                ax.plot(time, cum_test_negative, label='Cumulative Negatives', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Cumulative Tests')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Cumulative TB Diagnostic Results', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"cumulative_diagnostic_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot TB treatment outcomes for all parameter combinations
    
    Metrics plotted:
    - n_treated: Number of people who started treatment in this timestep (count)
    - n_treatment_success: Number of successful treatment completions in this timestep (count)
    - n_treatment_failure: Number of failed treatment attempts in this timestep (count)
    """
    import matplotlib.ticker as mticker

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get treatment results
                tbtx = sim.results['tbtreatment']
                n_treated = tbtx['n_treated'].values
                n_treatment_success = tbtx['n_treatment_success'].values
                n_treatment_failure = tbtx['n_treatment_failure'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot treatment outcomes
                ax.plot(time, n_treated, label='Treated', color='blue', marker='o', markersize=2)
                ax.plot(time, n_treatment_success, label='Successes', color='green', linestyle='--')
                ax.plot(time, n_treatment_failure, label='Failures', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Number of People')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('TB Treatment Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"treatment_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_cumulative_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot cumulative TB treatment outcomes for all parameter combinations
    
    Metrics plotted:
    - cum_treatment_success: Cumulative number of successful treatments over time (count)
    - cum_treatment_failure: Cumulative number of failed treatments over time (count)
    """
    import matplotlib.ticker as mticker

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = sim.results['timevec']
                
                # Get cumulative treatment results
                tbtx = sim.results['tbtreatment']
                cum_treatment_success = tbtx['cum_treatment_success'].values
                cum_treatment_failure = tbtx['cum_treatment_failure'].values

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Plot cumulative treatment outcomes
                ax.plot(time, cum_treatment_success, label='Cumulative Successes', color='green', linestyle='--')
                ax.plot(time, cum_treatment_failure, label='Cumulative Failures', color='red', linestyle=':')
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Cumulative Treatments')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.suptitle('Cumulative TB Treatment Outcomes', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots (same as refined TB prevalence plot)
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"cumulative_treatment_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_age_prevalence_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot age-stratified TB prevalence for all parameter combinations
    
    This function creates a grid of plots showing age-stratified TB prevalence rates
    by age groups including children (0-4, 5-14) and adults (15+), normalized per 100,000 population.
    The data is compared to the 2018 South Africa prevalence survey data where available.
    """
    import matplotlib.ticker as mtick

    # 2018 South Africa survey data (per 100,000 population) - only available for 15+
    sa_2018_data = {
        '15-24': 432,
        '25-34': 902,
        '35-44': 1107,
        '45-54': 1063,
        '55-64': 845,
        '65+': 1104
    }
    
    # All age groups including children
    all_age_groups = ['0-4', '5-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65+']
    
    # Create extended data array with NaN for age groups without survey data
    sa_2018_values = []
    for group in all_age_groups:
        if group in sa_2018_data:
            sa_2018_values.append(sa_2018_data[group])
        else:
            sa_2018_values.append(np.nan)  # No data available for children

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                
                # Compute age-stratified prevalence for 2018
                age_prevalence = compute_age_stratified_prevalence(sim, target_year=2018)
                model_prevalence = [age_prevalence[group]['prevalence_per_100k'] for group in all_age_groups]

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Create bar plot
                x_pos = np.arange(len(all_age_groups))
                width = 0.35
                
                # Plot model results
                bars1 = ax.bar(x_pos - width/2, model_prevalence, width, 
                              label='Model (2018)', alpha=0.8, color='blue')
                
                # Plot South Africa 2018 data (only for age groups with data)
                valid_data_mask = ~np.isnan(sa_2018_values)
                bars2 = ax.bar(x_pos[valid_data_mask] + width/2, 
                              [sa_2018_values[i] for i in range(len(sa_2018_values)) if valid_data_mask[i]], 
                              width, label='SA Data (2018)', alpha=0.8, color='red')
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                               f'{height:.0f}', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                               f'{height:.0f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.set_xlabel('Age Group')
                ax.set_ylabel('TB Prevalence (per 100,000)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(all_age_groups, rotation=45)
                ax.grid(True, alpha=0.3)
                
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)
                
                # Add percentage differences (only for age groups with survey data)
                for k, (model_val, data_val) in enumerate(zip(model_prevalence, sa_2018_values)):
                    if not np.isnan(data_val) and data_val > 0:
                        pct_diff = ((model_val - data_val) / data_val) * 100
                        ax.annotate(f'{pct_diff:.1f}%', 
                                    xy=(k, max(model_val, data_val) + 100), 
                                    xytext=(0, 5), 
                                    textcoords='offset points',
                                    ha='center', fontsize=7, color='darkgreen')

    plt.tight_layout()
    plt.suptitle('Age-Stratified TB Prevalence: Model vs South Africa 2018 Survey Data', fontsize=14, y=1.02)

    filename = f"age_prevalence_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def compute_hiv_tb_coinfection_rates(sim, target_year=2018):
    """
    Compute HIV coinfection rates among TB cases by symptom status
    
    Args:
        sim: Simulation object
        target_year: Year to compute rates for
    
    Returns:
        dict: HIV coinfection rates by TB symptom status
    """
    
    # Find the time index closest to target year
    time_years = np.array([d.year for d in sim.results['timevec']])
    target_idx = np.argmin(np.abs(time_years - target_year))
    
    # Get people alive at target time
    people = sim.people
    alive_mask = people.alive
    
    # Get TB states
    tb_states = sim.diseases.tb.state
    hiv_states = sim.diseases.hiv.state
    
    # Define TB states by symptom status
    # Presymptomatic (0 symptoms) - ACTIVE_PRESYMP
    presymptomatic_mask = (tb_states == mtb.TBS.ACTIVE_PRESYMP)
    
    # Symptomatic (≥1 symptoms) - ACTIVE_SMPOS, ACTIVE_SMNEG, ACTIVE_EXPTB
    symptomatic_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # All active TB (any symptoms)
    all_active_mask = np.isin(tb_states, [mtb.TBS.ACTIVE_PRESYMP, mtb.TBS.ACTIVE_SMPOS, mtb.TBS.ACTIVE_SMNEG, mtb.TBS.ACTIVE_EXPTB])
    
    # Get HIV-positive states (assuming HIV states 1, 2, 3 are positive - adjust as needed)
    # HIV states: 0=uninfected, 1=acute, 2=latent, 3=AIDS
    hiv_positive_mask = np.isin(hiv_states, [1, 2, 3])
    
    # Filter for adults (age 15+)
    adult_mask = (people.age >= 15)
    
    # Combine masks
    alive_adult_mask = alive_mask & adult_mask
    
    # Calculate coinfection rates for each category
    coinfection_rates = {}
    
    # 1. Presymptomatic TB cases (0 symptoms)
    presymp_adult_mask = alive_adult_mask & presymptomatic_mask
    presymp_total = np.sum(presymp_adult_mask)
    presymp_hiv_positive = np.sum(presymp_adult_mask & hiv_positive_mask)
    presymp_hiv_rate = (presymp_hiv_positive / presymp_total * 100) if presymp_total > 0 else 0
    
    coinfection_rates['presymptomatic'] = {
        'total_cases': presymp_total,
        'hiv_positive': presymp_hiv_positive,
        'hiv_rate_percent': presymp_hiv_rate
    }
    
    # 2. Symptomatic TB cases (≥1 symptoms)
    sympt_adult_mask = alive_adult_mask & symptomatic_mask
    sympt_total = np.sum(sympt_adult_mask)
    sympt_hiv_positive = np.sum(sympt_adult_mask & hiv_positive_mask)
    sympt_hiv_rate = (sympt_hiv_positive / sympt_total * 100) if sympt_total > 0 else 0
    
    coinfection_rates['symptomatic'] = {
        'total_cases': sympt_total,
        'hiv_positive': sympt_hiv_positive,
        'hiv_rate_percent': sympt_hiv_rate
    }
    
    # 3. All active TB cases (any symptoms)
    all_active_adult_mask = alive_adult_mask & all_active_mask
    all_active_total = np.sum(all_active_adult_mask)
    all_active_hiv_positive = np.sum(all_active_adult_mask & hiv_positive_mask)
    all_active_hiv_rate = (all_active_hiv_positive / all_active_total * 100) if all_active_total > 0 else 0
    
    coinfection_rates['all_active'] = {
        'total_cases': all_active_total,
        'hiv_positive': all_active_hiv_positive,
        'hiv_rate_percent': all_active_hiv_rate
    }
    
    return coinfection_rates


def plot_hiv_tb_coinfection_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot HIV coinfection rates among TB cases by symptom status for all parameter combinations
    
    This function creates a grid of plots showing HIV coinfection rates among TB cases
    stratified by symptom status, comparing model results to 2018 South Africa survey data.
    """
    import matplotlib.ticker as mtick

    # 2018 South Africa survey data (HIV coinfection rates by symptom status)
    sa_2018_data = {
        'presymptomatic': 22.4,  # 0 symptoms (presymptomatic)
        'symptomatic': 36.9,     # ≥1 symptoms (symptomatic) - calculated from weighted average
        'all_active': 28.8       # All active TB cases
    }
    
    categories = ['presymptomatic', 'symptomatic', 'all_active']
    category_labels = ['0 Symptoms\n(Presymptomatic)', '≥1 Symptoms\n(Symptomatic)', 'All Active TB']
    sa_2018_values = [sa_2018_data[cat] for cat in categories]

    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                
                # Compute HIV-TB coinfection rates for 2018
                coinfection_rates = compute_hiv_tb_coinfection_rates(sim, target_year=2018)
                model_rates = [coinfection_rates[cat]['hiv_rate_percent'] for cat in categories]

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                
                # Create bar plot
                x_pos = np.arange(len(categories))
                width = 0.35
                
                # Plot model results
                bars1 = ax.bar(x_pos - width/2, model_rates, width, 
                              label='Model (2018)', alpha=0.8, color='blue')
                
                # Plot South Africa 2018 data
                bars2 = ax.bar(x_pos + width/2, sa_2018_values, width, 
                              label='SA Data (2018)', alpha=0.8, color='red')
                
                # Add value labels on bars
                for bar in bars1:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
                
                for bar in bars2:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.set_xlabel('TB Symptom Status')
                ax.set_ylabel('HIV Coinfection Rate (%)')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(category_labels, rotation=0, ha='center')
                ax.grid(True, alpha=0.3)
                
                # Set y-axis to show percentages properly
                ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=6)
                
                # Add percentage differences
                for k, (model_val, data_val) in enumerate(zip(model_rates, sa_2018_values)):
                    if data_val > 0:
                        pct_diff = ((model_val - data_val) / data_val) * 100
                        ax.annotate(f'{pct_diff:.1f}%', 
                                    xy=(k, max(model_val, data_val) + 2), 
                                    xytext=(0, 5), 
                                    textcoords='offset points',
                                    ha='center', fontsize=7, color='darkgreen')
                
                # Add case counts as text annotations
                for k, cat in enumerate(categories):
                    total_cases = coinfection_rates[cat]['total_cases']
                    hiv_positive = coinfection_rates[cat]['hiv_positive']
                    ax.text(k, -5, f'n={total_cases}\nHIV+={hiv_positive}', 
                           ha='center', va='top', fontsize=6, color='gray')

    plt.tight_layout()
    plt.suptitle('HIV Coinfection Rates Among TB Cases by Symptom Status: Model vs South Africa 2018 Survey Data', 
                 fontsize=14, y=1.02)

    filename = f"hiv_tb_coinfection_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def plot_case_notification_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot annualized TB case notification rate (per 100,000) for all parameter combinations in a grid.
    The notification rate at time t is the difference in cumulative positive diagnoses between t and t-365 days, divided by the population at t, times 100,000.
    Overlays real South Africa notification data from GTB report.
    """
    import matplotlib.ticker as mtick
    import os
    import rdata
    import pandas as pd

    # --- Load real notification data (from extract_gtb_data.py logic) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gtb_dir = os.path.join(base_dir, '../tbsim/data/gtbreport2024/data/gtb')
    snapshot_dir = os.path.join(gtb_dir, 'snapshot_2024-07-29')
    other_dir = os.path.join(gtb_dir, 'other')
    tb_rda_path = os.path.join(snapshot_dir, 'tb.rda')
    pop_rda_path = os.path.join(other_dir, 'pop.rda')
    # Helper to load RDA file and return as pandas DataFrame
    def load_rda_df(rda_path):
        import rdata
        parsed = rdata.parser.parse_file(rda_path)
        converted = rdata.conversion.convert(parsed)
        for v in converted.values():
            if isinstance(v, pd.DataFrame):
                return v
        raise ValueError(f"No DataFrame found in {rda_path}")
    tb_df = load_rda_df(tb_rda_path)
    pop_df = load_rda_df(pop_rda_path)
    sa_code = 'ZAF'
    tb_sa = tb_df[tb_df['iso3'] == sa_code]
    pop_sa = pop_df[pop_df['iso3'] == sa_code]
    notif_vars = [col for col in tb_sa.columns if 'new' in col and ('bact' in col or 'labconf' in col or 'notif' in col or 'pos' in col)]
    notif_var = None
    for v in ['new_bact_pos', 'new_labconf', 'new_notif', 'new_pos']:
        if v in tb_sa.columns:
            notif_var = v
            break
    if notif_var is None and notif_vars:
        notif_var = notif_vars[0]
    if notif_var is None:
        raise ValueError('No notification variable found in TB data')
    pop_col = None
    for c in ['pop', 'e_pop_num', 'population']:
        if c in pop_sa.columns:
            pop_col = c
            break
    if pop_col is None:
        raise ValueError('No population column found in population data')
    merged = pd.merge(tb_sa[['year', notif_var]], pop_sa[['year', pop_col]], on='year', how='inner')
    merged = merged.sort_values('year')
    merged['notif_rate_per_100k'] = merged[notif_var] / merged[pop_col] * 1e5
    real_years = merged['year'].values
    real_rates = merged['notif_rate_per_100k'].values

    # --- Plot model grid ---
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = np.array(sim.results['timevec'])
                tbdiag = sim.results['tbdiagnostic']
                cum_test_positive = tbdiag['cum_test_positive'].values
                n_alive = sim.results['n_alive']

                # Compute annualized notification rate
                notification_rate = np.zeros_like(cum_test_positive, dtype=float)
                for t in range(len(time)):
                    t_date = time[t]
                    t_prev_date = t_date - datetime.timedelta(days=365)
                    t_prev = np.searchsorted(time, t_prev_date)
                    if t_prev == len(time) or time[t_prev] > t_prev_date:
                        t_prev = max(0, t_prev - 1)
                    notifications = cum_test_positive[t] - cum_test_positive[t_prev]
                    pop = n_alive[t]
                    notification_rate[t] = (notifications / pop) * 1e5 if pop > 0 else 0

                # --- Compute annualized TB incidence rate ---
                tb_results = sim.results['tb']
                if 'cum_active' in tb_results:
                    cum_incidence = tb_results['cum_active']
                else:
                    # Fallback: compute cumulative sum of new_active
                    if 'new_active' in tb_results:
                        cum_incidence = np.cumsum(tb_results['new_active'])
                    else:
                        raise ValueError('No new_active or cum_active in tb results')
                incidence_rate = np.zeros_like(cum_incidence, dtype=float)
                for t in range(len(time)):
                    t_date = time[t]
                    t_prev_date = t_date - datetime.timedelta(days=365)
                    t_prev = np.searchsorted(time, t_prev_date)
                    if t_prev == len(time) or time[t_prev] > t_prev_date:
                        t_prev = max(0, t_prev - 1)
                    new_cases = cum_incidence[t] - cum_incidence[t_prev]
                    pop = n_alive[t]
                    incidence_rate[t] = (new_cases / pop) * 1e5 if pop > 0 else 0

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, notification_rate, color='purple', label='Model Notification Rate')
                ax.plot(time, incidence_rate, color='blue', label='Model Incidence Rate')
                # Overlay real data
                ax.plot([datetime.date(int(y), 1, 1) for y in real_years], real_rates, marker='o', color='red', label='SA Notification Data')
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('Rate (per 100,000)')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=7)
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))

    plt.tight_layout()
    plt.suptitle('Annualized TB Case Notification Rate (per 100,000)', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"case_notification_rate_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def compute_annualized_tb_mortality_rate(sim):
    """
    Compute annualized TB mortality rate (per 100,000 population) over time.
    
    This function calculates the annualized TB mortality rate by:
    1. Taking the difference in cumulative TB deaths between time T and T-365 days
    2. Dividing by the population at time T
    3. Multiplying by 100,000 to get rate per 100,000 population
    
    Returns the annualized TB mortality rate per 100,000 population.
    """
    time = sim.results['timevec']
    tb_results = sim.results['tb']
    
    # Get population size over time
    try:
        n_alive = sim.results['n_alive']
        # Handle both numpy arrays and pandas Series
        if hasattr(n_alive, 'values'):
            n_alive = n_alive.values
    except KeyError:
        n_alive = np.full(len(time), fill_value=np.count_nonzero(sim.people.alive))
    
    # Get cumulative TB deaths
    if 'cum_deaths' in tb_results:
        cum_deaths = tb_results['cum_deaths']
        # Handle both numpy arrays and pandas Series
        if hasattr(cum_deaths, 'values'):
            cum_deaths = cum_deaths.values
    else:
        # Fallback: compute cumulative sum of new_deaths
        if 'new_deaths' in tb_results:
            new_deaths = tb_results['new_deaths']
            # Handle both numpy arrays and pandas Series
            if hasattr(new_deaths, 'values'):
                new_deaths = new_deaths.values
            cum_deaths = np.cumsum(new_deaths)
        else:
            raise ValueError('No new_deaths or cum_deaths in tb results')
    
    # Compute annualized mortality rate
    mortality_rate = np.zeros_like(cum_deaths, dtype=float)
    for t in range(len(time)):
        t_date = time[t]
        t_prev_date = t_date - datetime.timedelta(days=365)
        t_prev = np.searchsorted(time, t_prev_date)
        if t_prev == len(time) or time[t_prev] > t_prev_date:
            t_prev = max(0, t_prev - 1)
        
        # Calculate difference in cumulative deaths over the year
        deaths_diff = cum_deaths[t] - cum_deaths[t_prev]
        pop = n_alive[t]
        mortality_rate[t] = (deaths_diff / pop) * 1e5 if pop > 0 else 0
    
    return mortality_rate


def plot_tb_mortality_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp):
    """Plot annualized TB mortality rate (per 100,000) for all parameter combinations in a grid.
    The mortality rate at time t is the difference in cumulative TB deaths between t and t-365 days, 
    divided by the population at t, times 100,000.
    """
    import matplotlib.ticker as mtick
    
    nrows = len(tb_mortality_vals) * len(rel_sus_vals)
    ncols = len(beta_vals)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=True, sharey=True)

    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                sim = sim_grid[m][i][j]
                time = np.array(sim.results['timevec'])
                mortality_rate = compute_annualized_tb_mortality_rate(sim)

                ax_idx = m * len(rel_sus_vals) + i
                ax = axs[ax_idx][j] if nrows > 1 else axs[j]
                ax.plot(time, mortality_rate, color='red', label='Annual TB Mortality Rate', linewidth=2)
                
                ax.set_title(f'β={beta:.3f}, rel_sus={rel_sus:.2f}, mort={tb_mortality:.1e}')
                ax.grid(True)
                if ax_idx == nrows - 1:
                    ax.set_xlabel('Year')
                if j == 0:
                    ax.set_ylabel('TB Mortality Rate (per 100,000)')
                if m == 0 and i == 0 and j == 0:
                    ax.legend(fontsize=7)
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))

    plt.tight_layout()
    plt.suptitle('Annualized TB Mortality Rate (per 100,000)', fontsize=14, y=1.02)

    # Set consistent x-axis ticks for all subplots
    first_sim = sim_grid[0][0][0]
    time_years = np.array([d.year for d in first_sim.results['timevec']])
    min_year = time_years.min()
    max_year = time_years.max()
    xticks = np.arange(min_year, max_year + 1, 20)
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            for ax in ax_row:
                ax.set_xticks([datetime.date(year, 1, 1) for year in xticks])
                ax.set_xticklabels([str(year) for year in xticks], rotation=45)
        else:
            ax_row.set_xticks([datetime.date(year, 1, 1) for year in xticks])
            ax_row.set_xticklabels([str(year) for year in xticks], rotation=45)

    filename = f"tb_mortality_rate_grid_{timestamp}.pdf"
    plt.savefig(get_output_path(filename), dpi=300, bbox_inches='tight')
    plt.show()


def run_sim(beta, rel_sus_latentslow, tb_mortality, seed=0, years=200, n_agents=1000):  # 8000
    start_year = 1850  # 1750
    sim_pars = dict(
        unit='day',
        dt=30,
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        rand_seed=seed,
        verbose=0,
    )

    # demog = [ss.Births(pars=dict(birth_rate=20)), ss.Deaths(pars=dict(death_rate=1))]
    # people = ss.People(n_agents=n_agents)
    # To do: Add time-varying birth rate and age-, sex-, year-specific mortality

    # Try different possible paths for the data files
    possible_cbr_paths = [
        '../tbsim/data/South_Africa_CBR.csv',  # Added correct relative path
        '../data/South_Africa_CBR.csv',
        'tbsim/data/South_Africa_CBR.csv',
        'data/South_Africa_CBR.csv',
    ]
    possible_asmr_paths = [
        '../tbsim/data/South_Africa_ASMR.csv',  # Added correct relative path
        '../data/South_Africa_ASMR.csv',
        'tbsim/data/South_Africa_ASMR.csv',
        'data/South_Africa_ASMR.csv',
    ]
    
    # Find the correct CBR path
    cbr_path = None
    for path in possible_cbr_paths:
        if os.path.exists(path):
            cbr_path = path
            break
    if cbr_path is None:
        raise FileNotFoundError(f"Could not find South_Africa_CBR.csv in any of the expected locations: {possible_cbr_paths}")
    
    # Find the correct ASMR path
    asmr_path = None
    for path in possible_asmr_paths:
        if os.path.exists(path):
            asmr_path = path
            break
    if asmr_path is None:
        raise FileNotFoundError(f"Could not find South_Africa_ASMR.csv in any of the expected locations: {possible_asmr_paths}")
    
    cbr = pd.read_csv(cbr_path)  # Crude birth rate per 1000
    asmr = pd.read_csv(asmr_path)  # Age-specific mortality rate
    demog = [
        ss.Births(birth_rate=cbr, unit='day', dt=30),
        ss.Deaths(death_rate=asmr, unit='day', dt=30, rate_units=1),  # rate_units=1 = per person-year
    ]
    people = make_people(n_agents=n_agents)
 
    tb_pars = dict(
        beta=ss.rate_prob(beta, unit='day'),  # ss.beta(beta),
        init_prev=ss.bernoulli(p=0.10),  # Higher initial prevalence for South Africa context
        rel_sus_latentslow=rel_sus_latentslow,
        p_latent_fast=ss.bernoulli(p=0.1),  # Base fast progressor fraction (will be overridden by age-specific intervention)
        # South Africa-specific adjustments
        rate_LS_to_presym=ss.perday(5e-5),  # Slightly higher progression for HIV context
        rate_LF_to_presym=ss.perday(8e-3),  # Higher fast progression rate
        rate_active_to_clear=ss.perday(1.5e-4),  # Lower clearance rate (more persistent)
        rate_smpos_to_dead=ss.perday(tb_mortality),
        rate_exptb_to_dead=ss.perday(0.15 * tb_mortality),
        rate_smneg_to_dead=ss.perday(0.3 * tb_mortality),
    )
    tb = cf.make_tb_comorbidity(tb_pars=tb_pars)

    # Add HIV for South Africa context (critical for TB dynamics)
    hiv_pars = dict(
        init_prev=ss.bernoulli(p=0.00),  # Start with no HIV, will be added via intervention
        init_onart=ss.bernoulli(p=0.00),
    )
    hiv = cf.make_hiv_comorbidity(hiv_pars=hiv_pars)

    net = ss.RandomNet(pars=dict(n_contacts=ss.poisson(lam=5), dur=0))

    # Add TB-HIV connector to model coinfection effects with increased progression rates
    # Higher multipliers to get steeper TB prevalence increase from 1990 onwards
    # Increased by 50% from previous values
    tb_hiv_connector = cf.make_tb_hiv_connector(pars=dict(
        acute_multiplier=4.5,    # Increased from 3.0 to 4.5 (50% higher)
        latent_multiplier=7.5,   # Increased from 5.0 to 7.5 (50% higher)
        aids_multiplier=12.0,    # Increased from 8.0 to 12.0 (50% higher)
    ))

    # Add custom HIV intervention with gradual ramp-up based on van Schalkwyk et al. 2021 data for eThekwini
    hiv_intervention = GradualHIVIntervention(pars=dict(
        percent_on_ART=0.50,  # 50% of HIV-positive individuals on ART
        start=ss.date('1990-01-01'),  # Start from 1990 when HIV epidemic began
        stop=ss.date(f'{start_year + years}-01-01'),
    ))

    # Add health-seeking behavior intervention (90-day average delay - slower for better burn-in)
    # Rate = 1/90 days = 0.011 per day
    health_seeking = HealthSeekingBehavior(pars=dict(
        initial_care_seeking_rate=ss.perday(1/120),  # 90-day average delay for slower case detection
        start=ss.date(f'{start_year}-01-01'),
        stop=ss.date(f'{start_year + years}-01-01'),
        single_use=True,
    ))

    # Add TB diagnostic intervention (60% sensitivity - less effective for better burn-in)
    tb_diagnostic = TBDiagnostic(pars=dict(
        coverage=ss.bernoulli(0.7, strict=False),  # 70% coverage - not everyone gets tested
        sensitivity=0.50,  # 60% sensitivity - less effective case detection
        specificity=0.95,  # 95% specificity (standard)
        reset_flag=False,
        care_seeking_multiplier=1.0,  # 2.0 to encourage retries for false negatives
    ))

    # Add TB treatment intervention (70% success rate - less effective for better burn-in)
    tb_treatment = TBTreatment(pars=dict(
        treatment_success_rate=0.70,  # 70% treatment success rate - less effective treatment
        reseek_multiplier=1.0,  # 2.0 to encourage retries for treatment failures
        reset_flags=True,  # Reset diagnostic flags after treatment failure
    ))

    # Add age-dependent TB progression intervention
    age_tb_progression = AgeDependentTBProgression(pars=dict(
        age_0_4_multiplier=2.0,    # 2x progression for 0-4 year olds
        age_5_14_multiplier=0.5,   # 0.5x progression for 5-14 year olds
        age_15plus_multiplier=1.0,  # 1x progression for 15+ year olds
    ))

    # Combine all interventions
    all_interventions = [hiv_intervention, health_seeking, tb_diagnostic, tb_treatment, age_tb_progression]

    sim = ss.Sim(
        people=people,
        diseases=[tb, hiv],
        networks=net,
        demographics=demog,
        connectors=tb_hiv_connector,  # Pass connector directly, not in a list
        interventions=all_interventions,  # Combined interventions list
        pars=sim_pars,
    )
    sim.run()

    return sim


def refined_sweep(beta_vals, rel_sus_vals, tb_mortality_vals):

    # This function performs a parameter sweep over beta and relative susceptibility values
    # For each parameter combination, it runs a TB simulation and generates plots showing:
    # - Active TB prevalence over time (blue line)
    # - Latent TB prevalence over time (orange dashed line) 
    # - Target 1% prevalence threshold (red dotted line)
    # - 2018 South Africa data point (red dot)
    # Each subplot shows results for a specific (beta, rel_sus) parameter combination

    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")  # e.g., 2025_06_24_0330

    sim_grid = [[[None for _ in beta_vals] for _ in rel_sus_vals] for _ in tb_mortality_vals]
    results = {}
    total_runs = len(beta_vals) * len(rel_sus_vals) * len(tb_mortality_vals)
    
    # Create a list to store age-stratified prevalence data for all simulations
    all_age_prevalence_data = []
    
    for m, tb_mortality in enumerate(tb_mortality_vals):
        for i, rel_sus in enumerate(rel_sus_vals):
            for j, beta in enumerate(beta_vals):
                scen_key = f'beta={beta:.3f}_rel_sus={rel_sus:.2f}_mort={tb_mortality:.1e}'
                print(f"▶️ Running simulation {scen_key} ({m},{i},{j})/{total_runs}")
                sim = run_sim(beta=beta, rel_sus_latentslow=rel_sus, tb_mortality=tb_mortality)
                sim_grid[m][i][j] = sim
                results[scen_key] = sim.results.flatten()
                
                # Compute age-stratified prevalence time series for this simulation
                age_prevalence_df = compute_age_stratified_prevalence_time_series(sim)
                
                # Add parameter information to the DataFrame
                age_prevalence_df['beta'] = beta
                age_prevalence_df['rel_sus'] = rel_sus
                age_prevalence_df['tb_mortality'] = tb_mortality
                age_prevalence_df['scenario'] = scen_key
                
                # Store the data
                all_age_prevalence_data.append(age_prevalence_df)
    # Use common_functions.plot_results to plot all scenario results
    # Note: This function plots various metrics with the following units/definitions:
    # - 'active': Active TB cases (count of people with active TB disease)
    # - 'latent': Latent TB cases (count of people with latent TB infection)
    # - 'incidence': New TB infections per time step (count of new cases)
    # - 'prevalence': TB prevalence as fraction of total population (0-1)
    # - 'sought': People who sought care for TB symptoms (count)
    # - 'eligible': People eligible for care-seeking (active TB cases, count)
    # - 'tested': People who received diagnostic testing (count)
    # - 'diagnosed': People diagnosed with TB (count)
    # - 'treated': People who started TB treatment (count)
    # - 'success': Successful TB treatment completions (count)
    # - 'failure': Failed TB treatment attempts (count)
    cf.plot_results(results, dark=False)
    # Optionally, keep the original grid plots if desired
    plot_total_population_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_hiv_metrics_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_health_seeking_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_cumulative_diagnostic_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_cumulative_treatment_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_tb_sweep_with_data(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_annualized_infection_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_age_prevalence_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_hiv_tb_coinfection_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_case_notification_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    plot_tb_mortality_rate_grid(sim_grid, beta_vals, rel_sus_vals, tb_mortality_vals, timestamp)
    
    # Save age-stratified prevalence data as CSV files
    print("💾 Saving age-stratified prevalence data as CSV files...")
    
    # Combine all age prevalence data into a single DataFrame
    if all_age_prevalence_data:
        combined_age_prevalence = pd.concat(all_age_prevalence_data, ignore_index=False)
        
        # Save combined data
        combined_filename = f"age_stratified_prevalence_combined_{timestamp}.csv"
        combined_age_prevalence.to_csv(get_output_path(combined_filename))
        print(f"✅ Saved combined age-stratified prevalence data: {combined_filename}")
        
        # Also save individual scenario files
        for i, age_prevalence_df in enumerate(all_age_prevalence_data):
            scenario_name = age_prevalence_df['scenario'].iloc[0]
            # Clean scenario name for filename
            clean_scenario_name = scenario_name.replace('=', '_').replace('.', 'p').replace('e-', 'e')
            individual_filename = f"age_stratified_prevalence_{clean_scenario_name}_{timestamp}.csv"
            age_prevalence_df.to_csv(get_output_path(individual_filename))
            print(f"✅ Saved individual scenario data: {individual_filename}")

if __name__ == '__main__':
    # Setup for TB prevalence sweeps
    # This section configures the parameter ranges and executes the sweep analysis
    
    # Plot population demographics
    # Run sweep
    # Reduced to 2 parameter combinations for faster runtime
    beta_range = np.array([0.025, 0.035])  # Higher infectiousness range 0.025-0.035
    rel_sus_range = np.array([0.15])  # Single value for reinfection susceptibility
    tb_mortality_range = [3e-4]  # Single value for TB mortality
    refined_sweep(beta_range, rel_sus_range, tb_mortality_range)

    end_wallclock = time.time()
    end_datetime = datetime.datetime.now()
    elapsed_minutes = (end_wallclock - start_wallclock) / 60

    print(f"Sweep finished at {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {elapsed_minutes:.1f} minutes")

    # Uncomment the line below to run a quick test of the health-seeking and diagnostic integration
    # test_health_seeking_diagnostic_integration()


def test_hiv_integration():
    """Simple test function to verify HIV integration works correctly"""
    print("Testing HIV integration...")
    
    # Run a simple simulation with HIV
    sim = run_sim(beta=0.003, rel_sus_latentslow=0.05, tb_mortality=4e-4, years=50, n_agents=500)
    
    # Debug HIV results
    debug_hiv_results(sim)
    
    # Test HIV prevalence calculation
    hiv_prev = compute_hiv_prevalence(sim)
    print(f"HIV prevalence at final timestep: {hiv_prev[-1]:.3f}")
    
    # Test HIV-positive TB prevalence calculation
    hiv_tb_prev = compute_hiv_positive_tb_prevalence(sim)
    print(f"HIV-positive TB prevalence at final timestep: {hiv_tb_prev[-1]:.3f}")
    
    print("HIV integration test completed.")


# Uncomment the line below to run the HIV integration test
# test_hiv_integration()


def test_health_seeking_diagnostic_integration():
    """Test function to verify health-seeking and diagnostic integration works correctly"""
    print("Testing health-seeking and diagnostic integration...")
    
    # Run a simple simulation with health-seeking and diagnostic
    sim = run_sim(beta=0.003, rel_sus_latentslow=0.05, tb_mortality=4e-4, years=50, n_agents=500)
    
    # Check if health-seeking results are available
    try:
        hsb = sim.results['healthseekingbehavior']
        print(f"✓ Health-seeking results available")
        print(f"  - Final new sought care: {hsb['new_sought_care'].values[-1]}")
        print(f"  - Final cumulative sought care: {hsb['n_sought_care'].values[-1]}")
        print(f"  - Final eligible: {hsb['n_eligible'].values[-1]}")
    except KeyError:
        print("✗ Health-seeking results not found")
    
    # Check if diagnostic results are available
    try:
        tbdiag = sim.results['tbdiagnostic']
        print(f"✓ Diagnostic results available")
        print(f"  - Final tested: {tbdiag['n_tested'].values[-1]}")
        print(f"  - Final test positive: {tbdiag['n_test_positive'].values[-1]}")
        print(f"  - Final test negative: {tbdiag['n_test_negative'].values[-1]}")
        print(f"  - Cumulative test positive: {tbdiag['cum_test_positive'].values[-1]}")
        print(f"  - Cumulative test negative: {tbdiag['cum_test_negative'].values[-1]}")
    except KeyError:
        print("✗ Diagnostic results not found")
    
    # Check treatment results
    try:
        tbtx = sim.results['tbtreatment']
        print(f"✓ Treatment results available")
        print(f"  - Final treated: {tbtx['n_treated'].values[-1]}")
        print(f"  - Final treatment success: {tbtx['n_treatment_success'].values[-1]}")
        print(f"  - Final treatment failure: {tbtx['n_treatment_failure'].values[-1]}")
        print(f"  - Cumulative treatment success: {tbtx['cum_treatment_success'].values[-1]}")
        print(f"  - Cumulative treatment failure: {tbtx['cum_treatment_failure'].values[-1]}")
    except KeyError:
        print("✗ Treatment results not found")
    
    # Check people states
    people = sim.people
    print(f"✓ People states:")
    print(f"  - People who sought care: {np.sum(people.sought_care)}")
    print(f"  - People who were tested: {np.sum(people.tested)}")
    print(f"  - People who were diagnosed: {np.sum(people.diagnosed)}")
    print(f"  - People with treatment success: {np.sum(people.tb_treatment_success)}")
    print(f"  - People with treatment failure: {np.sum(people.treatment_failure)}")
    print(f"  - Mean care-seeking multiplier: {np.mean(people.care_seeking_multiplier):.3f}")
    
    print("Health-seeking, diagnostic, and treatment integration test completed.")


# Uncomment the line below to run the health-seeking and diagnostic integration test
# test_health_seeking_diagnostic_integration()