# Bacille Calmette-Guérin (BCG) vaccination intervention
import numpy as np
import starsim as ss
import logging
import datetime as dt
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
    - Distribution-based vaccination delivery timing
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
    - `delivery` (ss.Dist): Vaccination delivery timing distribution (required)
        - Distribution specifies when individuals are vaccinated (in years from intervention start)
        - For immediate vaccination, use `ss.constant(0)`
        - For distributed rollout, use a distribution (e.g., `ss.uniform(0, 5)` for rollout over 5 years)
    - `waning` (ss.Dist): Waning distribution using survival function at normalized time (default: ss.expon(scale=1.0))
    - `activation_modifier` (ss.uniform): Activation risk modifier (default: uniform(0.5, 0.65))
    - `clearance_modifier` (ss.uniform): Clearance modifier (default: uniform(1.3, 1.5))
    - `death_modifier` (ss.uniform): Death risk modifier (default: uniform(0.05, 0.15))
    
    **Examples:**
    
        # Immediate vaccination (all at once)
        bcg = BCGProtection(pars={
            'coverage': 0.9, 
            'age_range': (0, 1),
            'delivery': ss.constant(0)  # Immediate vaccination
        })
        
        # Distributed rollout over 5 years
        bcg = BCGProtection(pars={
            'age_range': (0, 18),
            'delivery': ss.uniform(0, 5)  # Distributed over 5 years
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
            coverage=ss.bernoulli(p=pars.get('coverage', 0.5) if isinstance(pars, dict) else 0.5),  # Default 50% coverage
            delivery=ss.weibull(scale=1.0, c=2.0),
            immunity_period=ss.years(10),  # Default 10 years
            waning=ss.expon(scale=1.0),  # Waning distribution (default: exponential)
            
            # Default modifiers
            activation_modifier=ss.uniform(0.5, 0.65),  # Reduces activation risk
            clearance_modifier=ss.uniform(1.3, 1.5),    # Increases clearance
            death_modifier=ss.uniform(0.05, 0.15),      # Reduces death risk
        )
        self.update_pars(pars)
        
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
        self._start_ti = None  # Will be set in init_post when sim.ti is available
    
    def init_post(self):
        """Initialize the intervention after the simulation starts."""
        super().init_post()
        # Store the start timestep for vaccination timing calculations
        self._start_ti = self.sim.ti
        self.pars.delivery.init(self.sim)
        self.pars.waning.init(self.sim)
        self.init_results()
    
    
    def step(self):
        """Execute BCG intervention step: handle expiration, vaccination, and protection effects."""
        # Check time window
        if self.sim.now < self.pars.start or self.sim.now > self.pars.stop:
            return
        
        current_time = self.sim.ti
        all_vaccinated = self.is_bcg_vaccinated.uids
        
        # CRITICAL: Re-apply protection effects at the start of each timestep
        # The TB model resets rr_activation, rr_clearance, rr_death to 1.0 at the end of its step,
        # so we MUST re-apply protection effects here BEFORE TB uses them in this timestep.
        # This ensures protection is active when TB calculates transition probabilities.
        if len(all_vaccinated) > 0:
            # Convert to numpy array for proper boolean operations
            expires_array = np.array(self.ti_bcg_protection_expires[all_vaccinated])
            # Find protected individuals: vaccinated AND not expired AND has expiration time set
            protected_mask = (
                self.is_bcg_vaccinated[all_vaccinated] & 
                (current_time <= expires_array) & 
                ~np.isnan(expires_array)
            )
            expired_uids = all_vaccinated[~protected_mask]
            protected_uids = all_vaccinated[protected_mask]
            
            # Remove protection from expired individuals
            if len(expired_uids) > 0:
                self._update_protection_effects(expired_uids, apply=False)
            
            # Re-apply protection effects for all currently protected individuals
            # This MUST happen every timestep because TB resets the modifiers to 1.0 at end of its step
            # We apply this FIRST, before handling new vaccinations, to ensure protection is active
            # when TB step() runs and uses the modifiers
            if len(protected_uids) > 0:
                self._update_protection_effects(protected_uids, apply=True)
        
        # Handle vaccination: always use distribution-based scheduling
        eligible = ((self.sim.people.age >= self.min_age) & (self.sim.people.age <= self.max_age) & ~self.is_bcg_vaccinated).uids
        self.n_eligible = len(eligible)
        
        # Schedule eligible individuals using the delivery distribution
        if len(eligible) > 0:
            # Find eligible individuals who haven't been scheduled yet
            not_scheduled = eligible[np.isnan(self.ti_bcg_scheduled[eligible])]
            if len(not_scheduled) > 0:
                # Ensure _start_ti is set (should be set in init_post, but check for safety)
                if self._start_ti is None:
                    self._start_ti = self.sim.ti
                
                # Sample vaccination timing from distribution (years from start)
                vaccination_delays_years = self.pars.delivery.rvs(not_scheduled)
                # Convert years to timesteps and add to start time
                vaccination_delays_timesteps = (vaccination_delays_years * 365.25) / self.sim.dt.days
                self.ti_bcg_scheduled[not_scheduled] = self._start_ti + vaccination_delays_timesteps
        
        # Vaccinate individuals whose scheduled time has arrived
        ready_to_vaccinate = eligible[(self.ti_bcg_scheduled[eligible] <= current_time) & 
                                      (~np.isnan(self.ti_bcg_scheduled[eligible]))]
        ready_to_vaccinate = self.pars.coverage.filter(ready_to_vaccinate)
        
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
            
            # Apply protection effects immediately to newly vaccinated responders
            # This ensures protection is active for the current timestep
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
        
        This method calculates waning factors and base modifiers, then calls the TB model's
        listener method to apply the protection effects. The TB model handles the actual
        application of waning to the modifiers.
        
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
            
            # Call TB model's listener method to apply protection effects
            # The TB model handles applying waning to the modifiers internally
            tb.apply_vaccine_protection(
                uids=uids,
                waning_factors=waning_factors,
                activation_modifier_base=activation_mods_base,
                clearance_modifier_base=clearance_mods_base,
                death_modifier_base=death_mods_base,
                apply=True
            )
        else:
            # Remove protection effects - call TB listener to reset modifiers
            has_modifiers = ~np.isnan(self.bcg_activation_modifier_applied[uids])
            if np.any(has_modifiers):
                valid_uids = uids[has_modifiers]
                # Pass dummy waning factors and modifiers (not used when apply=False)
                dummy_waning = np.ones(len(valid_uids))
                dummy_mods = np.ones(len(valid_uids))
                tb.apply_vaccine_protection(
                    uids=valid_uids,
                    waning_factors=dummy_waning,
                    activation_modifier_base=dummy_mods,
                    clearance_modifier_base=dummy_mods,
                    death_modifier_base=dummy_mods,
                    apply=False
                )
            
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
        # CRITICAL: Re-apply protection effects after TB step (which resets modifiers to 1.0)
        # This ensures protection is active for the NEXT timestep when TB uses the modifiers
        # This is essential because TB resets rr_activation, rr_clearance, rr_death to 1.0
        # at the end of its step(), so we must re-apply protection here
        #
        # BUG FIX (Issue #3): Exclude newly vaccinated this timestep from re-application.
        # They already had protection applied in step() before TB ran. Including them here
        # would double-apply protection (squaring efficacy: 0.5 -> 0.25) in the same timestep.
        all_vaccinated = self.is_bcg_vaccinated.uids
        if len(all_vaccinated) > 0:
            current_time = self.sim.ti
            expires_array = np.array(self.ti_bcg_protection_expires[all_vaccinated])
            protected_mask = (
                self.is_bcg_vaccinated[all_vaccinated] & 
                (current_time <= expires_array) & 
                ~np.isnan(expires_array)
            )
            protected_uids = all_vaccinated[protected_mask]
            # Exclude newly vaccinated this timestep - they were already applied in step()
            # (avoids double-application bug: squaring efficacy 0.5 -> 0.25 in first timestep)
            newly_vaccinated_mask = np.asarray(self.ti_bcg_vaccinated[protected_uids]) == current_time
            previously_protected = protected_uids[~newly_vaccinated_mask]
            if len(previously_protected) > 0:
                # Re-apply protection to ensure it's active for next timestep
                # This happens AFTER TB has reset modifiers, so protection will be active
                # when BCG step() runs at the start of the next timestep
                self._update_protection_effects(previously_protected, apply=True)
        
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
