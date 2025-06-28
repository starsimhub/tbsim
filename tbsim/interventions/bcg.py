import numpy as np
import starsim as ss
import sciris as sc
import logging
import datetime

__all__ = ['BCGProtection']

logger = logging.getLogger(__name__)

class BCGProtection(ss.Intervention):
    """
    Simulates BCG-like vaccination for tuberculosis prevention in individuals under a specified age.

    This intervention identifies individuals below a configurable age limit who have not yet 
    been vaccinated. At each timestep, a proportion of these eligible individuals is 
    selected based on the `coverage` parameter to receive simulated BCG protection.

    Once vaccinated, individuals are considered protected for a fixed number of years 
    (`duration`). While protected, their TB-related risk modifiers — activation, clearance, 
    and death — are adjusted using scaled and sampled values from BCG-specific probability 
    distributions.

    Parameters:
        pars (dict, optional): Dictionary of parameters. Supported keys:
            - 'coverage'            (float): Fraction of eligible individuals vaccinated per timestep (default: 0.6).
            - 'start'               (str/datetime.date): Start date for the intervention (default: '1900-01-01').
            - 'stop'                (str/datetime.date): Stop date for the intervention (default: '2100-12-31').
            - 'efficacy'            (float): Probability of effective vaccine response (default: 0.8).
            - 'duration'            (int): Duration (in years) for which BCG protection remains effective (default: 10).
            - 'age_limit'           (int): Maximum age (in years) to be considered eligible for vaccination (default: 5).
            - 'activation_modifier' (ss.uniform): Distribution for TB activation risk modifier (default: uniform(0.5, 0.65)).
            - 'clearance_modifier'  (ss.uniform): Distribution for bacterial clearance modifier (default: uniform(1.3, 1.5)).
            - 'death_modifier'      (ss.uniform): Distribution for TB mortality modifier (default: uniform(0.05, 0.15)).

    Usage Examples:
        # Basic usage with default parameters (targets individuals ≤5 years)
        bcg = BCGProtection()
        
        # Custom parameters using dictionary (targets individuals ≤7 years)
        bcg = BCGProtection(pars={
            'coverage': 0.8,
            'start': '2020-01-01',
            'stop': '2030-12-31',
            'efficacy': 0.9,
            'duration': 15,
            'age_limit': 7
        })
        

        # Custom probability distributions
        bcg = BCGProtection(pars={
            'activation_modifier': ss.uniform(0.4, 0.6),
            'clearance_modifier': ss.uniform(1.2, 1.6),
            'death_modifier': ss.uniform(0.03, 0.12)
        })

    Attributes:
        coverage_dist (ss.bernoulli): Probability distribution for vaccination coverage.
        eligible      (np.ndarray): Boolean mask of currently eligible individuals.
        n_eligible    (int): Number of individuals eligible for vaccination in the current step.
        p_vaccine_response (ss.bernoulli): Probability distribution for vaccine response.
        start         (datetime.date): Start date for the intervention.
        stop          (datetime.date): Stop date for the intervention.

    States:
        is_bcg_vaccinated (bool): Indicates whether an individual has received the BCG vaccine.
        ti_bcg_vaccinated (float): Timestep at which the individual was vaccinated.
        ti_bcg_protection_expires (float): Timestep when protection expires.

    Methods:
        check_eligibility(): Identify and randomly select eligible individuals for vaccination.
        check_eligibility_short(): Simplified eligibility check using starsim indexing.
        is_protected(uids, current_time): Return boolean mask indicating protected individuals.
        step(): Apply BCG protection and adjust TB risk modifiers accordingly.
        _apply_protection_effects(protected_uids): Apply BCG protection effects to TB risk modifiers.
        _remove_protection(expired_uids): Remove BCG protection effects when protection expires.
        _maintain_ongoing_protection(current_time): Maintain protection effects for all currently protected individuals.
        init_results(): Define simulation result metrics.
        update_results(): Record the number of vaccinated and eligible individuals each timestep.

    Notes:
        This intervention assumes the presence of a TB disease model attached to the simulation 
        and modifies its rr_activation, rr_clearance, and rr_death arrays. The intervention
        uses starsim probability distributions for stochastic modeling and proper date handling
        for temporal eligibility checks. The age_limit parameter allows targeting of any age group,
        making this intervention suitable for both pediatric and adult vaccination strategies.
    """

    def __init__(self, pars={}, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            coverage=0.6,  # Fraction of eligible individuals vaccinated per timestep
            start=sc.date('1900-01-01'),  # Year when the intervention starts
            stop=sc.date('2100-12-31'),  # Year when the intervention stops
            efficacy=0.8,  # Probability of protection
            duration=10,  # Duration (in years) for which BCG protection remains effective
            age_limit=5,  # Maximum age (in years) to be considered eligible for vaccination
            
            # BCG-specific probability distributions for TB risk modifiers
            activation_modifier=ss.uniform(0.5, 0.65),  # Modifier for TB activation risk
            clearance_modifier=ss.uniform(1.3, 1.5),    # Modifier for bacterial clearance probability
            death_modifier=ss.uniform(0.05, 0.15),      # Modifier for TB-related mortality risk
        )
        self.update_pars(pars)
        
        # Define parameters
        self.coverage = self.pars.coverage
        self.coverage_dist = ss.bernoulli(p=self.pars.coverage)
        self.start = self.pars.start
        self.stop = self.pars.stop
        self.efficacy = self.pars.efficacy
        self.duration = self.pars.duration
        self.age_limit = self.pars.age_limit
        self.n_eligible = 0
        self.eligible = []
        
        # Create probability objects for vaccine response 
        self.p_vaccine_response = ss.bernoulli(p=self.pars.efficacy)
        
        # Define state arrays with clearer names
        self.define_states(
            ss.BoolArr('is_bcg_vaccinated', default=False), # Indicates if vaccinated
            ss.FloatArr('ti_bcg_vaccinated'),               # Timestep of vaccination
            ss.Arr('ti_bcg_protection_expires')             # Timestep when protection expires
        )
        logger.debug(self.pars)

    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        # The probability objects will be automatically initialized by ss.link_dists()
        # which is called in the parent init_pre method
        return

    def check_eligibility(self):
        ages = self.sim.people.age
        under_age = ages <= self.age_limit

        eligible = under_age & ~self.is_bcg_vaccinated
        eligible_uids = np.where(eligible)[0]
        n_to_vaccinate = int(len(eligible_uids) * self.coverage)
        if n_to_vaccinate > 0:
            chosen = self.coverage_dist.filter(eligible_uids)
            self.eligible = np.zeros_like(eligible)
            self.eligible[chosen] = True
        else:
            chosen = np.array([], dtype=int)
            self.eligible = np.zeros_like(eligible)
        self.n_eligible = len(chosen)
        return ss.uids(chosen)
    
    
    def check_eligibility_short(self):
        eli = ((self.sim.people.age <= self.age_limit) & ~self.is_bcg_vaccinated).uids
        chos = self.coverage_dist.filter(eli)
        return chos
        
    def is_protected(self, uids, current_time):
        """Return boolean array: True if still protected (within duration), else False."""
        # Convert uids to ss.uids if it's a numpy array
        if isinstance(uids, np.ndarray):
            uids = ss.uids(uids)
        return (self.is_bcg_vaccinated[uids]) & ((current_time - self.ti_bcg_vaccinated[uids]) <= self.duration)

    def step(self):
        """
        Executes BCG vaccination during the current simulation timestep.

        This method implements a targeted Bacille Calmette-Guérin (BCG) immunization strategy
        for individuals below a specified age threshold. It models age-filtered eligibility, 
        stochastic coverage, and vaccine-induced protection with time-limited efficacy.

        Notes
        -----
        This intervention performs the following operations:

        1. **Temporal eligibility check**:
           Verifies the current simulation time falls within the intervention window (start/stop dates).

        2. **Protection expiration management**:
           Checks for previously vaccinated individuals whose protection has expired and removes
           their protection effects from TB risk modifiers.

        3. **Eligibility determination**:
           Identifies individuals meeting age criteria (≤ age_limit) who have not been vaccinated.

        4. **Vaccination assignment**:
           Randomly selects eligible individuals based on coverage probability and marks them
           as vaccinated, recording the vaccination timestep.

        5. **Vaccine response modeling**:
           Simulates individual vaccine response using efficacy probability. Only responders
           receive protection benefits.

        6. **Protection duration assignment**:
           Sets expiration timestep for vaccine responders based on protection duration.

        7. **TB risk modification**:
           Applies BCG-specific modifiers to TB activation, clearance, and death rates
           using starsim probability distributions for individual heterogeneity.

        8. **Ongoing protection maintenance**:
           Ensures protection effects persist for all currently protected individuals.

        The method uses starsim indexing and probability distributions for efficient
        population-level operations and stochastic modeling.
        """
        
        # 1. Assess temporal eligibility: verify if the current simulation time falls within the intervention window
        now = self.sim.now
        if hasattr(now, 'date'):
            now_date = now.date()  # Convert pd.Timestamp to datetime.date
        else:
            now_date = now
            
        if now_date < self.start or now_date > self.stop:
            return
        
        current_time = self.ti  # Current timestep
        
        # 2. Evaluate ongoing protection: check if individuals remain within the effective protection period
        # Get all previously vaccinated individuals using starsim indexing
        all_vaccinated = self.is_bcg_vaccinated.uids
        if len(all_vaccinated) > 0:
            # Check which vaccinated individuals are still protected
            still_protected = self.is_protected(all_vaccinated, current_time)
            # Remove protection from those who have expired
            expired_uids = all_vaccinated[~still_protected]
            if len(expired_uids) > 0:
                self._remove_protection(expired_uids)
        
        # 3. Determine immunization eligibility: identify individuals meeting age criteria who have not been vaccinated
        eligible = self.check_eligibility()
        
        if len(eligible) == 0:
            return
            
        # 4. Randomly select individuals for vaccination based on coverage probability
        # (This is already handled in check_eligibility())
        
        # 5. Update vaccination status: mark individuals as vaccinated and record the vaccination time
        self.is_bcg_vaccinated[eligible] = True
        self.ti_bcg_vaccinated[eligible] = current_time
        
        # 6. Consider vaccine efficacy: account for the probability of immunological response post-vaccination
        # Model host response heterogeneity: incorporate inter-individual variability in vaccine uptake and effectiveness
        vaccine_responders = self.p_vaccine_response.filter(eligible)
        
        if len(vaccine_responders) == 0:
            return
            
        # 7. Calculate immunological protection duration: set the timepoint when vaccine-induced immunity wanes
        protection_expires = current_time + self.duration
        self.ti_bcg_protection_expires[vaccine_responders] = protection_expires
        
        # 8. Adjust TB risk modifiers: apply BCG-specific modifiers to TB activation, clearance, and death rates
        self._apply_protection_effects(vaccine_responders)
        
        # 9. Maintain ongoing protection effects for all currently protected individuals
        self._maintain_ongoing_protection(current_time)

    def _apply_protection_effects(self, protected_uids):
        """
        Apply BCG protection effects to TB risk modifiers.
        
        Parameters:
            protected_uids: Array of UIDs who are currently protected
        """
        if len(protected_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        
        # Sample individual protection levels using starsim probability distributions
        activation_modifiers = self.pars.activation_modifier.rvs(protected_uids)
        clearance_modifiers = self.pars.clearance_modifier.rvs(protected_uids)
        death_modifiers = self.pars.death_modifier.rvs(protected_uids)
        
        # Apply modifiers to TB risk rates
        tb.rr_activation[protected_uids] *= activation_modifiers
        tb.rr_clearance[protected_uids] *= clearance_modifiers
        tb.rr_death[protected_uids] *= death_modifiers
    
    def _remove_protection(self, expired_uids):
        """
        Remove BCG protection effects when protection expires.
        
        Parameters:
            expired_uids: Array of UIDs whose protection has expired
        """
        if len(expired_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        
        # Reset TB risk modifiers to baseline (divide by the applied modifiers)
        # Note: This is a simplified approach. In practice, you might want to track
        # the original values and restore them exactly
        activation_modifiers = self.pars.activation_modifier.rvs(expired_uids)
        clearance_modifiers = self.pars.clearance_modifier.rvs(expired_uids)
        death_modifiers = self.pars.death_modifier.rvs(expired_uids)
        
        tb.rr_activation[expired_uids] /= activation_modifiers
        tb.rr_clearance[expired_uids] /= clearance_modifiers
        tb.rr_death[expired_uids] /= death_modifiers
        
        # Clear protection expiration time
        self.ti_bcg_protection_expires[expired_uids] = np.nan

    def _maintain_ongoing_protection(self, current_time):
        """
        Maintain protection effects for all currently protected individuals.
        
        This ensures that protection effects are continuously applied to all
        vaccinated individuals who are still within their protection period.
        
        Parameters:
            current_time: Current simulation timestep
        """
        # Get all vaccinated individuals using starsim indexing
        all_vaccinated = self.is_bcg_vaccinated.uids
        if len(all_vaccinated) == 0:
            return
            
        # Find those who are currently protected
        currently_protected = self.is_protected(all_vaccinated, current_time)
        protected_uids = all_vaccinated[currently_protected]
        
        if len(protected_uids) > 0:
            # Re-apply protection effects to ensure they persist
            self._apply_protection_effects(protected_uids)

    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        self.results['n_vaccinated'][self.ti] = np.count_nonzero(self.is_bcg_vaccinated)
        self.results['n_eligible'][self.ti] = self.n_eligible
