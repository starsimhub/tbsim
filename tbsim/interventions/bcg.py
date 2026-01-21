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
                self._update_protection_effects(expired_uids, apply=False)
            if len(protected_uids) > 0:
                self._update_protection_effects(protected_uids, apply=True)
        
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
            self._update_protection_effects(vaccine_responders, apply=True)

    def _calculate_waning_factor(self, vaccination_times, current_time):
        """
        Calculate waning factor using the waning distribution's survival function.
        
        The waning factor represents the remaining protection strength:
        - 1.0 = full protection (at vaccination time)
        - 0.0 = no protection (at or after immunity_period)
        """
        if len(vaccination_times) == 0:
            return np.array([], dtype=float)
        
        # Calculate time since vaccination (vectorized)
        time_since_vaccination = np.maximum(current_time - np.asarray(vaccination_times, dtype=float), 0.0)
        
        # Cache dt_days and immunity_period_timesteps (compute once per call)
        dt_days = self.sim.dt.days if hasattr(self.sim.dt, 'days') else float(self.sim.dt)
        immunity_period_timesteps = self.pars.immunity_period.days / dt_days
        
        # Normalize time: 0.0 at vaccination, 1.0 at immunity_period
        # Clip to [0, 1] to handle cases where protection has expired (shouldn't happen, but safe)
        normalized_time = np.clip(time_since_vaccination / immunity_period_timesteps, 0.0, 1.0)
        
        # Calculate waning factor using survival function
        # For exponential: sf(x) = exp(-x), but we want it to reach ~0 at x=1.0
        # So we scale: use exp(-scale * normalized_time) where scale ensures exp(-scale) ≈ 0
        try:
            if hasattr(self.pars.waning, 'dist') and hasattr(self.pars.waning.dist, 'sf'):
                dist_pars = getattr(self.pars.waning, 'pars', {})
                # For exponential with scale=1.0, sf(1.0) = exp(-1.0) ≈ 0.368
                # To get sf(1.0) ≈ 0.01, we'd need scale ≈ 4.6
                # But we'll use the distribution's scale parameter if available
                waning_factor = self.pars.waning.dist.sf(normalized_time, **dist_pars)
            else:
                # Default exponential: scale by a factor to ensure stronger waning
                # Using scale=3.0 gives exp(-3.0) ≈ 0.05 at normalized_time=1.0
                waning_factor = np.exp(-3.0 * normalized_time)
        except Exception:
            # Fallback: use scaled exponential for stronger waning
            waning_factor = np.exp(-3.0 * normalized_time)
        
        # Ensure waning_factor is an array and clip to [0, 1]
        if not isinstance(waning_factor, np.ndarray):
            waning_factor = np.full(len(time_since_vaccination), float(waning_factor))
        
        # At normalized_time=1.0 (end of immunity period), force waning_factor to 0
        # This ensures protection fully wanes by expiration
        waning_factor = np.where(normalized_time >= 1.0, 0.0, waning_factor)
        
        return np.clip(waning_factor, 0.0, 1.0)
    
    def _update_protection_effects(self, uids, apply=True):
        """
        Apply or remove BCG protection effects to TB risk modifiers.
        
        Args:
            uids: Array of individual IDs to update
            apply: If True, apply protection effects (with waning). If False, remove protection effects.
        """
        if len(uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        
        if apply:
            # Apply protection effects with waning
            current_time = self.sim.ti
            
            # Separate individuals who already have base modifiers vs new (need to sample)
            has_modifiers = ~np.isnan(self.bcg_activation_modifier_applied[uids])
            new_uids = uids[~has_modifiers]
            
            # For new individuals: sample and store base modifiers
            if len(new_uids) > 0:
                self.bcg_activation_modifier_applied[new_uids] = self.pars.activation_modifier.rvs(new_uids)
                self.bcg_clearance_modifier_applied[new_uids] = self.pars.clearance_modifier.rvs(new_uids)
                self.bcg_death_modifier_applied[new_uids] = self.pars.death_modifier.rvs(new_uids)
            
            # Get base modifiers for all individuals
            activation_mods_base = self.bcg_activation_modifier_applied[uids]
            clearance_mods_base = self.bcg_clearance_modifier_applied[uids]
            death_mods_base = self.bcg_death_modifier_applied[uids]
            
            # Calculate waning factors for all individuals
            vaccination_times = self.ti_bcg_vaccinated[uids]
            waning_factors = self._calculate_waning_factor(vaccination_times, current_time)
            
            # Apply waned modifiers: interpolate between base_modifier (waning=1.0) and 1.0 (waning=0.0)
            # Formula: waned = base * waning_factor + 1 * (1 - waning_factor)
            activation_mods_waned = activation_mods_base * waning_factors + (1.0 - waning_factors)
            clearance_mods_waned = clearance_mods_base * waning_factors + (1.0 - waning_factors)
            death_mods_waned = death_mods_base * waning_factors + (1.0 - waning_factors)
            
            # Apply modifiers directly (TB resets to 1.0 at end of each step)
            tb.rr_activation[uids] = activation_mods_waned
            tb.rr_clearance[uids] = clearance_mods_waned
            tb.rr_death[uids] = death_mods_waned
        else:
            # Remove protection effects
            # Reset modifiers to 1.0 (baseline) - only for individuals who actually had modifiers applied
            has_modifiers = ~np.isnan(self.bcg_activation_modifier_applied[uids])
            if np.any(has_modifiers):
                valid_uids = uids[has_modifiers]
                tb.rr_activation[valid_uids] = 1.0
                tb.rr_clearance[valid_uids] = 1.0
                tb.rr_death[valid_uids] = 1.0
            
            # Clear stored modifiers (expiration times preserved for historical tracking)
            self.bcg_activation_modifier_applied[uids] = np.nan
            self.bcg_clearance_modifier_applied[uids] = np.nan
            self.bcg_death_modifier_applied[uids] = np.nan
            
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


