# Bacille Calmette-Guérin (BCG) vaccination intervention
import numpy as np
import starsim as ss
import logging
from scipy import stats
from tbsim.tb import TBS

__all__ = ['BCGProtection']
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class BCGProtection(ss.Intervention):
    #region Documentation
    """
    BCG vaccination intervention for tuberculosis prevention.
    
    Vaccinates eligible individuals (within `age_range`, not yet vaccinated) and applies protection effects
    that modify TB disease risk. Age eligibility is checked dynamically each timestep.
    
    For detailed documentation, see: `tbsim/interventions/intervention notes/BCG_DOCSTRING.md`
    
    **Key Features:**
    - Dynamic age eligibility: individuals age into/out of eligibility over time
    - Immediate or distributed vaccination timing (via `vaccination_timing` distribution)
    - Protection modifies TB risk: reduces activation (0.5-0.65x), increases clearance (1.3-1.5x), reduces death (0.05-0.15x)
    - Fixed protection duration (`immunity_period`, default: 10 years)
    - Gradual waning over time using Starsim distribution survival function (default: exponential)
    
    **Requirements:**
    - TB disease model (`sim.diseases.tb`) with `rr_activation`, `rr_clearance`, `rr_death` arrays
    - Population with age information (`sim.people.age`)
    
    **Parameters:**
    - `coverage` (float): Fraction vaccinated per timestep (default: 0.5)
    - `age_range` (tuple): (min_age, max_age) for eligibility (default: (0, 5))
    - `efficacy` (float): Vaccine response probability (default: 0.8)
    - `immunity_period` (ss.years): Protection duration (default: ss.years(10))
    - `vaccination_timing` (ss.Dist or None): Distribution for vaccination timing in years from start (default: None = immediate)
    - `waning` (ss.Dist): Waning distribution using survival function at normalized time (default: ss.expon(scale=1.0))
    - `activation_modifier` (ss.uniform): Activation risk modifier (default: uniform(0.5, 0.65))
    - `clearance_modifier` (ss.uniform): Clearance modifier (default: uniform(1.3, 1.5))
    - `death_modifier` (ss.uniform): Death risk modifier (default: uniform(0.05, 0.15))
    
    **Examples:**
    
        # Newborn vaccination
        bcg = BCGProtection(pars={'coverage': 0.9, 'age_range': (0, 1)})
        
        # Distributed rollout over 5 years
        bcg = BCGProtection(pars={
            'age_range': (0, 18),
            'vaccination_timing': ss.uniform(0, 5)
        })
        
        # Custom waning (Weibull)
        bcg = BCGProtection(pars={'waning': ss.weibull(scale=1.0, c=2.0)})
    """
    #endregion Documentation
    
    def __init__(self, pars={}, **kwargs):
        """Initialize a BCGProtection intervention instance."""
        super().__init__(**kwargs)
        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
            age_range=[0, 5],
            efficacy=ss.bernoulli(p=0.8),  # Default 80% efficacy (will be converted to dist after update_pars)
            
            coverage=ss.bernoulli(p=pars.get('coverage', 0.5)),  # Default 50% coverage
            vaccination_timing=None,  # Distribution for vaccination timing (years from start), None = immediate (e.g. for vaccination campaigns)
            immunity_period=ss.years(10),  # Default 10 years
            waning=ss.expon(scale=1.0),  # Waning distribution (default: exponential)
            
            # Default modifiers
            activation_modifier=ss.uniform(0.5, 0.65),  # Reduces activation risk
            clearance_modifier=ss.uniform(1.3, 1.5),    # Increases clearance
            death_modifier=ss.uniform(0.05, 0.15),      # Reduces death risk
        )
        self.update_pars(pars)
        # Convert efficacy to distribution if it's a float (needs to be in pars before init_pre for linking)
        if not isinstance(self.pars.efficacy, ss.Dist):
            self.pars.efficacy = ss.bernoulli(p=self.pars.efficacy)
            
        self.min_age = self.pars.age_range[0]
        self.max_age = self.pars.age_range[1]
        self.n_eligible = 0
        self.eligible = []
        self.define_states(
            ss.BoolArr('is_bcg_vaccinated', default=False),
            ss.FloatArr('ti_bcg_vaccinated'),
            ss.FloatArr('ti_bcg_protection_expires'),
            ss.FloatArr('ti_bcg_scheduled'),  # Scheduled vaccination time (timesteps from start)
            ss.FloatArr('bcg_activation_modifier_applied'),
            ss.FloatArr('bcg_clearance_modifier_applied'),
            ss.FloatArr('bcg_death_modifier_applied'),
        )
        
    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        self._vaccination_timing_dist = None
        self._start_ti = None  # Will be set in init_post when sim.ti is available
        if self.pars.vaccination_timing is not None and isinstance(self.pars.vaccination_timing, ss.Dist):
            # Store distribution reference - will be initialized in init_post if needed
            self._vaccination_timing_dist = self.pars.vaccination_timing
    
    def init_post(self):
        """Initialize the intervention after the simulation starts."""
        super().init_post()
        # Store the start timestep for vaccination timing calculations
        self._start_ti = self.sim.ti
        # Initialize vaccination timing distribution with sim context if needed
        # (distributions are stored once, not recreated each step - just sampled from)
        if self._vaccination_timing_dist is not None and hasattr(self._vaccination_timing_dist, 'init'):
            self._vaccination_timing_dist.init(self.sim)
        
        # Initialize waning distribution
        if isinstance(self.pars.waning, ss.Dist) and hasattr(self.pars.waning, 'init'):
            self.pars.waning.init(self.sim)
        
        self.init_results()
    
    
    def step(self):
        """Execute BCG intervention step: handle expiration, vaccination, and protection effects."""
        # Check time window
        if self.sim.now < self.pars.start or self.sim.now > self.pars.stop:
            return
        
        current_time = self.sim.ti
        all_vaccinated = self.is_bcg_vaccinated.uids
        
        # Handle expiration and maintain protection
        if len(all_vaccinated) > 0:
            protected_mask = self.is_bcg_vaccinated[all_vaccinated] & (current_time <= self.ti_bcg_protection_expires[all_vaccinated]) & ~np.isnan(self.ti_bcg_protection_expires[all_vaccinated])
            expired_uids = all_vaccinated[~protected_mask]
            protected_uids = all_vaccinated[protected_mask]
            
            if len(expired_uids) > 0:
                self._remove_protection(expired_uids)
            if len(protected_uids) > 0:
                self._apply_protection_effects(protected_uids)
        
        # Handle vaccination timing: schedule eligible individuals if using distribution
        eligible = ((self.sim.people.age >= self.min_age) & (self.sim.people.age <= self.max_age) & ~self.is_bcg_vaccinated).uids
        self.n_eligible = len(eligible)
        
        # If using vaccination timing distribution, schedule eligible individuals
        if self._vaccination_timing_dist is not None and len(eligible) > 0:
            # Find eligible individuals who haven't been scheduled yet
            not_scheduled = eligible[np.isnan(self.ti_bcg_scheduled[eligible])]
            if len(not_scheduled) > 0:
                # Sample vaccination timing from distribution (years from start)
                vaccination_delays_years = self._vaccination_timing_dist.rvs(not_scheduled)
                # Convert years to timesteps and add to start time (Starsim handles time conversions)
                vaccination_delays_timesteps = (vaccination_delays_years * 365.25) / self.sim.dt.days
                self.ti_bcg_scheduled[not_scheduled] = self._start_ti + vaccination_delays_timesteps
        
        # Vaccinate individuals whose scheduled time has arrived (or immediately if no timing distribution)
        if self._vaccination_timing_dist is not None:
            ready_to_vaccinate = eligible[(self.ti_bcg_scheduled[eligible] <= current_time) & 
                                          (~np.isnan(self.ti_bcg_scheduled[eligible]))]
            ready_to_vaccinate = self.pars.coverage.filter(ready_to_vaccinate)
        else:
            ready_to_vaccinate = self.pars.coverage.filter(eligible)
        
        if len(ready_to_vaccinate) == 0:
            return
        
        self.is_bcg_vaccinated[ready_to_vaccinate] = True
        self.ti_bcg_vaccinated[ready_to_vaccinate] = current_time
        
        vaccine_responders = self.pars.efficacy.filter(ready_to_vaccinate)
        if len(vaccine_responders) > 0:
            immunity_period = self.pars.immunity_period
            # Convert immunity_period (always in years) to timesteps
            dt_days = self.sim.dt.days if hasattr(self.sim.dt, 'days') else self.sim.dt.value
            immunity_period_ts = immunity_period.days / dt_days
            
            # Set expiration times for all responders (same period for all)
            self.ti_bcg_protection_expires[vaccine_responders] = current_time + immunity_period_ts
            self._apply_protection_effects(vaccine_responders)

    def _calculate_waning_factor(self, vaccination_times, current_time):
        """Calculate waning factor using the waning distribution's survival function."""
        time_since_vaccination = np.asarray(current_time - vaccination_times, dtype=float)
        
        # Get dt_days from sim (Starsim handles time conversions)
        dt_days = self.sim.dt.days if hasattr(self.sim.dt, 'days') else self.sim.dt.value
        
        # Convert immunity_period (always in years) to timesteps
        immunity_period = self.pars.immunity_period
        immunity_period_timesteps = immunity_period.days / dt_days
        
        time_since_vaccination = np.maximum(time_since_vaccination, 0.0)
        normalized_time = time_since_vaccination / immunity_period_timesteps
        try:
            if hasattr(self.pars.waning, 'dist') and hasattr(self.pars.waning.dist, 'sf'):
                dist_pars = self.pars.waning.pars
                waning_factor = self.pars.waning.dist.sf(normalized_time, **dist_pars)
            else:
                waning_factor = np.exp(-normalized_time)
        except Exception:
            waning_factor = np.exp(-normalized_time)
        
        # Ensure waning_factor is an array and clip to [0, 1]
        if not isinstance(waning_factor, np.ndarray):
            waning_factor = np.full(len(time_since_vaccination), waning_factor)
        waning_factor = np.clip(waning_factor, 0.0, 1.0)
        
        return waning_factor
    
    def _apply_protection_effects(self, protected_uids):
        """Apply BCG protection effects to TB risk modifiers, incorporating waning."""
        if len(protected_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        tb.state[protected_uids] = TBS.PROTECTED
        current_time = self.sim.ti
        
        # Check which individuals already have modifiers (re-apply with waning) vs new (apply first time)
        has_modifiers = ~np.isnan(self.bcg_activation_modifier_applied[protected_uids])
        existing_uids = protected_uids[has_modifiers]
        new_uids = protected_uids[~has_modifiers]
        
        # Re-apply stored modifiers with updated waning for existing protected individuals
        if len(existing_uids) > 0:
            activation_mods_base = self.bcg_activation_modifier_applied[existing_uids]
            clearance_mods_base = self.bcg_clearance_modifier_applied[existing_uids]
            death_mods_base = self.bcg_death_modifier_applied[existing_uids]
            
            # Remove old modifiers (divide them out to restore original values)
            tb.rr_activation[existing_uids] /= activation_mods_base
            tb.rr_clearance[existing_uids] /= clearance_mods_base
            tb.rr_death[existing_uids] /= death_mods_base
            
            # Calculate waning factors
            vaccination_times = self.ti_bcg_vaccinated[existing_uids]
            waning_factors = self._calculate_waning_factor(vaccination_times, current_time)
            
            # Apply waned modifiers: interpolate between base_modifier (factor=1) and 1.0 (factor=0)
            activation_mods_waned = activation_mods_base + (1.0 - waning_factors) * (1.0 - activation_mods_base)
            clearance_mods_waned = clearance_mods_base + (1.0 - waning_factors) * (1.0 - clearance_mods_base)
            death_mods_waned = death_mods_base + (1.0 - waning_factors) * (1.0 - death_mods_base)
            
            # Apply waned modifiers
            tb.rr_activation[existing_uids] *= activation_mods_waned
            tb.rr_clearance[existing_uids] *= clearance_mods_waned
            tb.rr_death[existing_uids] *= death_mods_waned
        
        # Apply to newly protected individuals (those without stored modifiers)
        if len(new_uids) > 0:
            # Sample, store base modifiers (full protection values)
            activation_mods_base = self.pars.activation_modifier.rvs(new_uids)
            clearance_mods_base = self.pars.clearance_modifier.rvs(new_uids)
            death_mods_base = self.pars.death_modifier.rvs(new_uids)
            
            # Store base modifiers (these represent full protection)
            self.bcg_activation_modifier_applied[new_uids] = activation_mods_base
            self.bcg_clearance_modifier_applied[new_uids] = clearance_mods_base
            self.bcg_death_modifier_applied[new_uids] = death_mods_base
            
            # Calculate waning factors (should be 1.0 for newly vaccinated, but calculate for consistency)
            vaccination_times = self.ti_bcg_vaccinated[new_uids]
            waning_factors = self._calculate_waning_factor(vaccination_times, current_time)
            
            # Apply waned modifiers: interpolate between base_modifier (factor=1) and 1.0 (factor=0)
            activation_mods_waned = activation_mods_base + (1.0 - waning_factors) * (1.0 - activation_mods_base)
            clearance_mods_waned = clearance_mods_base + (1.0 - waning_factors) * (1.0 - clearance_mods_base)
            death_mods_waned = death_mods_base + (1.0 - waning_factors) * (1.0 - death_mods_base)
            
            # Apply waned modifiers
            tb.rr_activation[new_uids] *= activation_mods_waned
            tb.rr_clearance[new_uids] *= clearance_mods_waned
            tb.rr_death[new_uids] *= death_mods_waned

    def _remove_protection(self, expired_uids):
        """Remove BCG protection effects when protection expires."""
        if len(expired_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        tb.state[expired_uids] = TBS.NONE
        
        # Get stored modifiers and reverse them
        activation_modifiers = self.bcg_activation_modifier_applied[expired_uids]
        clearance_modifiers = self.bcg_clearance_modifier_applied[expired_uids]
        death_modifiers = self.bcg_death_modifier_applied[expired_uids]
        
        valid_mask = ~np.isnan(activation_modifiers)
        if not np.any(valid_mask):
            return
        
        valid_uids = expired_uids[valid_mask]
        tb.rr_activation[valid_uids] /= activation_modifiers[valid_mask]
        tb.rr_clearance[valid_uids] /= clearance_modifiers[valid_mask]
        tb.rr_death[valid_uids] /= death_modifiers[valid_mask]
        
        # Clear modifiers (expiration times preserved for historical tracking)
        self.bcg_activation_modifier_applied[expired_uids] = np.nan
        self.bcg_clearance_modifier_applied[expired_uids] = np.nan
        self.bcg_death_modifier_applied[expired_uids] = np.nan
            
    def init_results(self):
        """Define simulation result metrics for the BCG intervention."""
        super().init_results()
        if hasattr(self, 'results') and 'n_vaccinated' in self.results:
            return
            
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),           # Total vaccinated individuals
            ss.Result('n_eligible', dtype=int),             # Eligible individuals this timestep
            ss.Result('n_newly_vaccinated', dtype=int),     # Newly vaccinated this timestep
            ss.Result('n_vaccine_responders', dtype=int),   # Individuals who responded to vaccine
            ss.Result('n_protected', dtype=int),            # Currently protected individuals
            ss.Result('n_protection_expired', dtype=int),   # Individuals whose protection expired
            ss.Result('vaccination_coverage', dtype=float), # Coverage rate (vaccinated/eligible)
            ss.Result('protection_coverage', dtype=float),  # Protection rate (protected/total_pop)
            ss.Result('vaccine_effectiveness', dtype=float), # Response rate (responders/vaccinated)
            ss.Result('cumulative_vaccinated', dtype=int),  # Total ever vaccinated
            ss.Result('cumulative_responders', dtype=int),  # Total ever responded
            ss.Result('cumulative_expired', dtype=int),     # Total ever expired protection
            ss.Result('avg_protection_duration', dtype=float), # Average protection immunity_period
            
            
        )

    def update_results(self):
        """Update all result metrics for the current timestep."""
        # Use self.ti (intervention's timestep counter) instead of self.sim.ti
        # This ensures we index into results arrays that are sized based on npts
        ti = self.ti  # Intervention's timestep counter (0-indexed, within bounds)
        sim_ti = self.sim.ti  # Simulation's absolute timestep (may exceed npts)
        
        total_vaccinated = self.is_bcg_vaccinated.sum()
        all_vaccinated_uids = self.is_bcg_vaccinated.uids
        
        # Calculate protected count (use sim_ti for comparison with expiration times)
        if len(all_vaccinated_uids) > 0:
            expires_array = np.array(self.ti_bcg_protection_expires[all_vaccinated_uids])
            protected_mask = (sim_ti <= self.ti_bcg_protection_expires[all_vaccinated_uids]) & ~np.isnan(expires_array)
            n_protected = protected_mask.sum()
        else:
            n_protected = 0
        
        newly_vaccinated = ((self.ti_bcg_vaccinated == sim_ti) & self.is_bcg_vaccinated).sum()
        protection_expires_array = np.array(self.ti_bcg_protection_expires)
        vaccine_responders = np.sum(~np.isnan(protection_expires_array))
        total_pop = len(self.sim.people)
        
        # Update all results using ti (intervention's timestep, guaranteed in bounds)
        self.results['n_vaccinated'][ti] = total_vaccinated
        self.results['n_eligible'][ti] = self.n_eligible
        self.results['n_newly_vaccinated'][ti] = newly_vaccinated
        self.results['n_vaccine_responders'][ti] = vaccine_responders
        self.results['n_protected'][ti] = n_protected
        self.results['n_protection_expired'][ti] = len(all_vaccinated_uids) - n_protected
        self.results['vaccination_coverage'][ti] = newly_vaccinated / self.n_eligible if self.n_eligible > 0 else 0.0
        self.results['protection_coverage'][ti] = n_protected / total_pop if total_pop > 0 else 0.0
        self.results['vaccine_effectiveness'][ti] = vaccine_responders / total_vaccinated if total_vaccinated > 0 else 0.0
        self.results['cumulative_vaccinated'][ti] = total_vaccinated
        self.results['cumulative_responders'][ti] = vaccine_responders
        self.results['cumulative_expired'][ti] = len(all_vaccinated_uids) - n_protected
        
        # Calculate average protection duration
        immunity_period = self.pars.immunity_period
        if hasattr(immunity_period, 'value'):
            avg_period_years = immunity_period.value
        else:
            avg_period_years = float(immunity_period)
        self.results['avg_protection_duration'][ti] = avg_period_years


