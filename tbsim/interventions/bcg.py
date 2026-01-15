# Bacille Calmette-Guérin (BCG) vaccination intervention
import numpy as np
import starsim as ss
import logging
from tbsim.tb import TBS

__all__ = ['BCGProtection']
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class BCGProtection(ss.Intervention):
    """
    Simulates BCG-like vaccination for tuberculosis prevention in individuals within a specified age range.
    
    This intervention identifies individuals within a configurable age range who have not yet been vaccinated. At each timestep, a proportion of these eligible individuals is selected based on the `coverage` parameter to receive simulated BCG protection. Once vaccinated, individuals are considered protected for a fixed number of years (`immunity_period`). While protected, their TB-related risk modifiers — activation, clearance, and death — are adjusted using scaled and sampled values from BCG-specific probability distributions.

    
    Required Dependencies
    ----------------------
        - TB Disease Model: The simulation must include a TB disease model with the following attributes:
          - `rr_activation`: Risk ratio for TB activation (modified by BCG)
          - `rr_clearance`: Risk ratio for bacterial clearance (modified by BCG)  
          - `rr_death`: Risk ratio for TB mortality (modified by BCG)
          - `state`: TB disease states (set to TBS.PROTECTED for vaccinated individuals)
        - Population: Must have age information accessible via `sim.people.age`
        - Starsim Framework: Requires starsim probability distributions and date handling

    Required TB States (TBS enum)
    ------------------------------
        - `TBS.PROTECTED` (100): Special state for BCG-protected individuals
        - `TBS.NONE` (-1): Default state for unprotected individuals
        - Standard TB states: `LATENT_SLOW`, `LATENT_FAST`, `ACTIVE_PRESYMP`, `ACTIVE_SMPOS`, `ACTIVE_SMNEG`, `ACTIVE_EXPTB`, `DEAD`


    Parameters
    ----------
    
    Core Parameters
        - `coverage` (float): Fraction of eligible individuals vaccinated per timestep (default: 0.5)
        - `start` (str/datetime.date): Start date for the intervention (default: '1900-01-01')
        - `stop` (str/datetime.date): Stop date for the intervention (default: '2100-12-31')
        - `efficacy` (float): Probability of effective vaccine response (default: 0.8)
        - `immunity_period` (ss.years): Immunity period as Starsim time unit (default: ss.years(10))
        - `age_range` (tuple): Age range (min_age, max_age) for eligibility (default: (0, 5))

    Risk Modifier Distributions
        - `activation_modifier` (ss.uniform): TB activation risk modifier (default: uniform(0.5, 0.65))
        - `clearance_modifier` (ss.uniform): Bacterial clearance modifier (default: uniform(1.3, 1.5))
        - `death_modifier` (ss.uniform): TB mortality modifier (default: uniform(0.05, 0.15))

    Examples
    --------

        # Standard newborn BCG vaccination
        bcg = BCGProtection(pars={
            'coverage': 0.9,
            'age_range': (0, 1),
            'efficacy': 0.8,
            'immunity_period': ss.years(10)
        })

        # School-based vaccination program
        bcg = BCGProtection(pars={
            'coverage': 0.7,
            'start': '2020-01-01',
            'stop': '2025-12-31',
            'age_range': (5, 15),
            'efficacy': 0.75,
            'immunity_period': ss.years(8)
        })
    
        # Research scenario with custom risk modifiers
        bcg = BCGProtection(pars={
            'activation_modifier': ss.uniform(0.4, 0.6),
            'clearance_modifier': ss.uniform(1.2, 1.6),
            'death_modifier': ss.uniform(0.03, 0.12)
        })

    Attributes
    ----------
    
    Core Attributes
        - `pars.coverage` (ss.bernoulli): Probability distribution for vaccination coverage
        - `pars.efficacy` (ss.bernoulli): Probability distribution for vaccine response
        - `eligible` (list): List of currently eligible individual UIDs
        - `n_eligible` (int): Number of individuals eligible for vaccination in current step
        - `pars.start` (datetime.date): Start date for the intervention
        - `pars.stop` (datetime.date): Stop date for the intervention
        - `min_age` (float): Minimum age for vaccination eligibility
        - `max_age` (float): Maximum age for vaccination eligibility

    Individual States
        - `is_bcg_vaccinated` (bool): Whether individual has received BCG vaccine
        - `ti_bcg_vaccinated` (float): Timestep when individual was vaccinated
        - `ti_bcg_protection_expires` (float): Timestep when protection expires (preserved for historical tracking even after expiration)
        - `bcg_activation_modifier_applied` (float): Activation risk modifier applied
        - `bcg_clearance_modifier_applied` (float): Clearance modifier applied
        - `bcg_death_modifier_applied` (float): Death risk modifier applied


    Notes
    -----

    Requires a TB disease model (`sim.diseases.tb`) with `rr_activation`, `rr_clearance`, and `rr_death` arrays.
    Each timestep, eligible individuals (within `age_range`, not yet vaccinated) are selected based on `coverage` 
    probability. Vaccine responders (determined by `efficacy`) receive protection for `immunity_period` years.
    
    During protection, TB risk modifiers are multiplied by sampled values: activation (0.5-0.65x), clearance (1.3-1.5x), 
    death (0.05-0.15x). Protection expires after `immunity_period`, at which point modifiers are reversed. 
    Expiration times are preserved in `ti_bcg_protection_expires` for tracking responders even after protection ends.
    """
    
    def __init__(self, pars={}, **kwargs):
        """
        Initialize a BCGProtection intervention instance.
        Parameters are: coverage, start, stop, efficacy, immunity_period, age_range, activation_modifier, 
        clearance_modifier, death_modifier
        
        """
        super().__init__(**kwargs)
        self.define_pars(
            coverage=ss.bernoulli(p=pars.get('coverage', 0.5)),  # Default 50% coverage
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
            efficacy=0.8,  # Default 80% efficacy (will be converted to dist after update_pars)
            immunity_period=ss.years(10),  # Default 10 years (Starsim v3.0 time unit)
            age_range=[0, 5],
            # Default modifiers
            activation_modifier=ss.uniform(0.5, 0.65),  # Reduces activation risk
            clearance_modifier=ss.uniform(1.3, 1.5),    # Increases clearance
            death_modifier=ss.uniform(0.05, 0.15),      # Reduces death risk
        )
        self.update_pars(pars)
        # Convert efficacy to distribution if it's a float (needs to be in pars before init_pre for linking)
        if not isinstance(self.pars.efficacy, ss.Dist):
            efficacy_val = self.pars.efficacy
            self.pars.efficacy = ss.bernoulli(p=efficacy_val)
        self.min_age = self.pars.age_range[0]
        self.max_age = self.pars.age_range[1]
        self.n_eligible = 0
        self.eligible = []
        self.define_states(
            ss.BoolArr('is_bcg_vaccinated', default=False),
            ss.FloatArr('ti_bcg_vaccinated'),
            ss.FloatArr('ti_bcg_protection_expires'),
            ss.FloatArr('bcg_activation_modifier_applied'),
            ss.FloatArr('bcg_clearance_modifier_applied'),
            ss.FloatArr('bcg_death_modifier_applied'),
        )
        
    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        # Convert immunity_period from years to timesteps
        if hasattr(self.pars.immunity_period, 'value'):
            self.pars.immunity_period = (self.pars.immunity_period.value * 365.25) / sim.dt.value
    
    def init_post(self):
        """Initialize the intervention after the simulation starts."""
        super().init_post()
        self.init_results()
    
    
    def select_for_vaccination(self):
        """Identify and randomly select eligible individuals for BCG vaccination."""
        eligible = ((self.sim.people.age >= self.min_age) & (self.sim.people.age <= self.max_age) & ~self.is_bcg_vaccinated).uids
        self.n_eligible = len(eligible)
        return self.pars.coverage.filter(eligible)
        
    def is_protected(self, uids, current_time):
        """Return boolean array indicating protection status."""
        uids = ss.uids(uids)
        expires = np.array(self.ti_bcg_protection_expires[uids])
        return self.is_bcg_vaccinated[uids] & (current_time <= self.ti_bcg_protection_expires[uids]) & ~np.isnan(expires)
    
    def _is_within_time_window(self):
        """Check if current time is within intervention window."""
        now = self.sim.now.date() if hasattr(self.sim.now, 'date') else self.sim.now
        return self.pars.start.date() <= now <= self.pars.stop.date()
    
    def step(self):
        """
        This method implements a targeted Bacille Calmette-Guérin (BCG) immunization strategy
        for individuals within a specified age range. It models age-filtered eligibility, 
        stochastic coverage, and vaccine-induced protection with time-limited efficacy.
        
        The method handles:
        1. Checking if current time is within the intervention window
        2. Removing expired protection and re-applying ongoing protection
        3. Vaccinating new eligible individuals based on coverage
        4. Filtering vaccine responders based on efficacy
        5. Applying protection effects to responders
        """
        # Check time window
        if not self._is_within_time_window():
            return
        
        current_time = self.sim.ti
        all_vaccinated = self.is_bcg_vaccinated.uids
        
        # Handle expiration and maintain protection
        if len(all_vaccinated) > 0:
            expires_array = np.array(self.ti_bcg_protection_expires[all_vaccinated])
            protected_mask = (current_time <= self.ti_bcg_protection_expires[all_vaccinated]) & ~np.isnan(expires_array)
            expired_uids = all_vaccinated[~protected_mask]
            protected_uids = all_vaccinated[protected_mask]
            
            if len(expired_uids) > 0:
                self._remove_protection(expired_uids)
            if len(protected_uids) > 0:
                self._apply_protection_effects(protected_uids, force=True)
        
        # Vaccinate new eligible individuals
        eligible = self.select_for_vaccination()
        if len(eligible) == 0:
            return
        
        self.is_bcg_vaccinated[eligible] = True
        self.ti_bcg_vaccinated[eligible] = current_time
        
        # Filter responders and apply protection
        vaccine_responders = self.pars.efficacy.filter(eligible)
        if len(vaccine_responders) > 0:
            self.ti_bcg_protection_expires[vaccine_responders] = current_time + self.pars.immunity_period
            self._apply_protection_effects(vaccine_responders, force=False)

    def _apply_protection_effects(self, protected_uids, force=False):
        """Apply BCG protection effects to TB risk modifiers."""
        if len(protected_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        tb.state[protected_uids] = TBS.PROTECTED
        
        if force:
            # Re-apply stored modifiers
            activation_mods = self.bcg_activation_modifier_applied[protected_uids]
            clearance_mods = self.bcg_clearance_modifier_applied[protected_uids]
            death_mods = self.bcg_death_modifier_applied[protected_uids]
            valid_mask = ~np.isnan(activation_mods)
            if not np.any(valid_mask):
                return
            valid_uids = protected_uids[valid_mask]
            tb.rr_activation[valid_uids] *= activation_mods[valid_mask]
            tb.rr_clearance[valid_uids] *= clearance_mods[valid_mask]
            tb.rr_death[valid_uids] *= death_mods[valid_mask]
        else:
            # Apply to newly protected only (those without stored modifiers)
            new_uids = protected_uids[np.isnan(self.bcg_activation_modifier_applied[protected_uids])]
            if len(new_uids) == 0:
                return
            
            # Sample, store, and apply modifiers
            activation_mods = self.pars.activation_modifier.rvs(new_uids)
            clearance_mods = self.pars.clearance_modifier.rvs(new_uids)
            death_mods = self.pars.death_modifier.rvs(new_uids)
            
            self.bcg_activation_modifier_applied[new_uids] = activation_mods
            self.bcg_clearance_modifier_applied[new_uids] = clearance_mods
            self.bcg_death_modifier_applied[new_uids] = death_mods
            
            tb.rr_activation[new_uids] *= activation_mods
            tb.rr_clearance[new_uids] *= clearance_mods
            tb.rr_death[new_uids] *= death_mods

    def _remove_protection(self, expired_uids):
        """Remove BCG protection effects when protection expires. Preserves expiration times for historical tracking."""
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
            ss.Result('avg_age_at_vaccination', dtype=float), # Average age when vaccinated
            ss.Result('avg_protection_duration', dtype=float), # Average protection immunity_period
            
            
        )

    def update_results(self):
        """Update all result metrics for the current timestep."""
        ti = self.sim.ti
        total_vaccinated = self.is_bcg_vaccinated.sum()
        all_vaccinated_uids = self.is_bcg_vaccinated.uids
        
        # Calculate protected count
        if len(all_vaccinated_uids) > 0:
            expires_array = np.array(self.ti_bcg_protection_expires[all_vaccinated_uids])
            protected_mask = (ti <= self.ti_bcg_protection_expires[all_vaccinated_uids]) & ~np.isnan(expires_array)
            n_protected = protected_mask.sum()
        else:
            n_protected = 0
        
        newly_vaccinated = ((self.ti_bcg_vaccinated == ti) & self.is_bcg_vaccinated).sum()
        protection_expires_array = np.array(self.ti_bcg_protection_expires)
        vaccine_responders = np.sum(~np.isnan(protection_expires_array))
        total_pop = len(self.sim.people)
        
        # Update all results
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
        
        # Age at vaccination
        if total_vaccinated > 0:
            dt_in_years = self.sim.dt.value / 365.25
            ages_at_vaccination = self.sim.people.age[all_vaccinated_uids] - (ti - self.ti_bcg_vaccinated[all_vaccinated_uids]) * dt_in_years
            self.results['avg_age_at_vaccination'][ti] = np.mean(ages_at_vaccination)
        else:
            self.results['avg_age_at_vaccination'][ti] = 0.0
        
        self.results['avg_protection_duration'][ti] = self.pars.immunity_period



    def get_summary_stats(self):
        """Get summary statistics for the intervention."""
        total_vaccinated = np.count_nonzero(self.is_bcg_vaccinated)
        # Convert to numpy array first to avoid starsim BooleanOperationError
        protection_expires_array = np.array(self.ti_bcg_protection_expires)
        total_responders = np.sum(~np.isnan(protection_expires_array))
        
        # Calculate final coverage
        total_pop = len(self.sim.people)
        final_coverage = total_vaccinated / total_pop if total_pop > 0 else 0.0
        
        # Calculate effectiveness
        effectiveness = total_responders / total_vaccinated if total_vaccinated > 0 else 0.0
        
        return {
            'total_vaccinated': total_vaccinated,
            'total_responders': total_responders,
            'final_coverage': final_coverage,
            'vaccine_effectiveness': effectiveness,
            'total_population': total_pop
        }
