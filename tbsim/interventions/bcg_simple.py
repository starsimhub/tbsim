"""
Simplified BCG intervention that avoids the problematic array definitions
that cause recursion errors in the Starsim library.
"""

import starsim as ss
import numpy as np
import sciris as sc
import logging

logger = logging.getLogger(__name__)

class BCGProtectionSimple(ss.Intervention):
    """Simplified BCG intervention that avoids problematic array definitions."""
    
    def __init__(self, pars=None, **kwargs):
        """Initialize the BCG intervention with simplified parameters."""
        super().__init__(**kwargs)
        
        # Set default parameters
        default_pars = {
            'start': ss.date('2000-01-01'),
            'stop': ss.date('2010-12-31'),
            'age_range': [0, 5],
            'coverage': 0.8,
            'efficacy': 0.8,
            'immunity_period': 10,  # years
            'activation_modifier': ss.constant(v=0.5),
            'clearance_modifier': ss.constant(v=1.0),
            'death_modifier': ss.constant(v=0.8),
        }
        
        # Update with provided parameters
        if pars is not None:
            default_pars.update(pars)
        
        # Set parameters using setattribute
        self.setattribute('pars', sc.objdict(default_pars))
        
        # Initialize simple tracking variables
        self.vaccinated_uids = set()
        self.vaccination_times = {}
        self.vaccination_ages = {}
        
        logger.info(f"Simplified BCG Intervention configured")
    
    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        return
    
    def check_eligibility(self):
        """Identify eligible individuals for BCG vaccination."""
        # Get current age and vaccination status
        current_age = self.sim.people.age.values  # Get numpy array
        min_age, max_age = self.pars.age_range
        
        # Find individuals within age range who haven't been vaccinated
        age_mask = (current_age >= min_age) & (current_age <= max_age)
        
        # Convert to numpy arrays for comparison
        all_uids = np.array(self.sim.people.uid)
        vaccinated_uids_array = np.array(list(self.vaccinated_uids))
        
        # Find those not vaccinated
        if len(vaccinated_uids_array) > 0:
            not_vaccinated_mask = ~np.isin(all_uids, vaccinated_uids_array)
        else:
            not_vaccinated_mask = np.ones(len(all_uids), dtype=bool)
        
        # Combine masks
        eligible_mask = age_mask & not_vaccinated_mask
        eligible_uids = all_uids[eligible_mask]
        
        # Apply coverage probability
        if len(eligible_uids) > 0:
            # Simple random selection based on coverage
            n_eligible = len(eligible_uids)
            n_to_select = int(n_eligible * self.pars.coverage)
            if n_to_select > 0:
                selected_indices = np.random.choice(n_eligible, size=n_to_select, replace=False)
                selected_uids = eligible_uids[selected_indices]
                return selected_uids
            else:
                return np.array([], dtype=int)
        
        return np.array([], dtype=int)
        
        return np.array([], dtype=int)
    
    def is_protected(self, uids, current_time):
        """Check if individuals are still protected."""
        protected = []
        
        for uid in uids:
            if uid in self.vaccinated_uids:
                vaccination_time = self.vaccination_times[uid]
                time_since_vaccination = current_time - vaccination_time
                protected.append(time_since_vaccination <= self.pars.immunity_period)
            else:
                protected.append(False)
        
        return np.array(protected)
    
    def step(self):
        """Execute BCG vaccination during the current simulation timestep."""
        # Check temporal eligibility
        now = self.sim.now
        if hasattr(now, 'date'):
            now_date = now.date()
        else:
            now_date = now
        
        # Convert dates for comparison
        start_date = self.pars.start
        stop_date = self.pars.stop
        if hasattr(start_date, 'date'):
            start_date = start_date.date()
        if hasattr(stop_date, 'date'):
            stop_date = stop_date.date()
            
        if now_date < start_date or now_date > stop_date:
            return
        
        current_time = self.ti
        
        # Get eligible individuals
        eligible = self.check_eligibility()
        if len(eligible) == 0:
            return
        
        # Apply vaccine efficacy
        n_eligible = len(eligible)
        n_responders = int(n_eligible * self.pars.efficacy)
        if n_responders > 0:
            responder_indices = np.random.choice(n_eligible, size=n_responders, replace=False)
            responders = eligible[responder_indices]
        else:
            responders = np.array([], dtype=int)
        
        if len(responders) == 0:
            return
        
        # Record vaccinations
        for uid in responders:
            self.vaccinated_uids.add(uid)
            self.vaccination_times[uid] = current_time
            self.vaccination_ages[uid] = self.sim.people.age[self.sim.people.uid == uid][0]
        
        # Apply protection effects
        self._apply_protection_effects(responders)
        
        logger.info(f"BCG vaccinated {len(responders)} individuals at timestep {current_time}")
    
    def _apply_protection_effects(self, protected_uids):
        """Apply BCG protection effects to TB risk modifiers."""
        if len(protected_uids) == 0:
            return
        
        tb = self.sim.diseases.tb
        
        # Apply modifiers
        activation_modifiers = self.pars.activation_modifier.rvs(protected_uids)
        clearance_modifiers = self.pars.clearance_modifier.rvs(protected_uids)
        death_modifiers = self.pars.death_modifier.rvs(protected_uids)
        
        # Apply to TB risk rates
        tb.rr_activation[protected_uids] *= activation_modifiers
        tb.rr_clearance[protected_uids] *= clearance_modifiers
        tb.rr_death[protected_uids] *= death_modifiers
    
    def begin_step(self):
        """Called at the beginning of each timestep."""
        super().begin_step()
        
        # Re-apply protection effects for all currently protected individuals
        current_time = self.ti
        all_vaccinated = list(self.vaccinated_uids)
        
        if len(all_vaccinated) > 0:
            all_vaccinated = np.array(all_vaccinated)
            still_protected = self.is_protected(all_vaccinated, current_time)
            protected_uids = all_vaccinated[still_protected]
            
            if len(protected_uids) > 0:
                self._apply_protection_effects(protected_uids)
    
    def init_results(self):
        """Initialize results tracking."""
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_protected', dtype=int),
            ss.Result('cum_vaccinated', dtype=int),
        )
    
    def update_results(self):
        """Update results for the current timestep."""
        current_time = self.ti
        
        # Count currently protected individuals
        all_vaccinated = list(self.vaccinated_uids)
        if len(all_vaccinated) > 0:
            all_vaccinated = np.array(all_vaccinated)
            still_protected = self.is_protected(all_vaccinated, current_time)
            n_protected = np.sum(still_protected)
        else:
            n_protected = 0
        
        self.results['n_vaccinated'][self.ti] = len(self.vaccinated_uids)
        self.results['n_protected'][self.ti] = n_protected
        
        if self.ti > 0:
            self.results['cum_vaccinated'][self.ti] = self.results['cum_vaccinated'][self.ti - 1] + len(self.vaccinated_uids)
        else:
            self.results['cum_vaccinated'][self.ti] = len(self.vaccinated_uids)
