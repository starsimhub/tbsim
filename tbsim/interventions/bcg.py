import numpy as np
import starsim as ss
import sciris as sc
import logging
import datetime
from tbsim.wrappers import Agents

__all__ = ['BCGProtection']
logger = logging.getLogger(__name__)
class BCGProtection(ss.Intervention):
    """
    Simulates BCG-like vaccination for tuberculosis prevention in individuals within a specified age range.
    This intervention identifies individuals within a configurable age range who have not yet 
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
            - 'age_range'           (tuple): Age range (min_age, max_age) for vaccination eligibility (default: (0, 5)).
            - 'activation_modifier' (ss.uniform): Distribution for TB activation risk modifier (default: uniform(0.5, 0.65)).
            - 'clearance_modifier'  (ss.uniform): Distribution for bacterial clearance modifier (default: uniform(1.3, 1.5)).
            - 'death_modifier'      (ss.uniform): Distribution for TB mortality modifier (default: uniform(0.05, 0.15)).
    Usage Examples:
        # Basic usage with default parameters (targets individuals 0-5 years)
        bcg = BCGProtection()
        
        # Custom parameters using dictionary (targets individuals 1-7 years)
        bcg = BCGProtection(pars={
            'coverage': 0.8,
            'start': '2020-01-01',
            'stop': '2030-12-31',
            'efficacy': 0.9,
            'duration': 15,
            'age_range': (1, 7)
        })
        
        # Adult vaccination example (targets individuals 18-65 years)
        bcg = BCGProtection(pars={
            'age_range': (18, 65),
            'coverage': 0.4,
            'efficacy': 0.7,
            'duration': 8
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
        min_age       (float): Minimum age for vaccination eligibility.
        max_age       (float): Maximum age for vaccination eligibility.
    States:
        is_bcg_vaccinated (bool): Indicates whether an individual has received the BCG vaccine.
        ti_bcg_vaccinated (float): Timestep at which the individual was vaccinated.
        ti_bcg_protection_expires (float): Timestep when protection expires.
        age_at_vaccination (float): Age when the individual was vaccinated.
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
        calculate_tb_impact(tb_disease): Calculate estimated TB cases and deaths averted by BCG vaccination.
        get_summary_stats(): Get summary statistics for the intervention.
        debug_population(): Debug method to check population demographics and vaccination status.
    Notes:
        This intervention assumes the presence of a TB disease model attached to the simulation 
        and modifies its rr_activation, rr_clearance, and rr_death arrays. The intervention
        uses starsim probability distributions for stochastic modeling and proper date handling
        for temporal eligibility checks. The age_range parameter allows targeting of any age group,
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
            age_range=(0, 5),  # Age range (min_age, max_age) for vaccination eligibility
            
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
        self.age_range = self.pars.age_range
        self.min_age = self.age_range[0]
        self.max_age = self.age_range[1]
        self.n_eligible = 0
        self.eligible = []
        
        # Create probability objects for vaccine response 
        self.p_vaccine_response = ss.bernoulli(p=self.pars.efficacy)
        
        # Define state arrays with clearer names
        self.define_states(
            ss.BoolArr('is_bcg_vaccinated', default=False), # Indicates if vaccinated
            ss.FloatArr('ti_bcg_vaccinated'),               # Timestep of vaccination
            ss.Arr('ti_bcg_protection_expires'),            # Timestep when protection expires
            ss.FloatArr('age_at_vaccination'),              # Age when vaccinated
        )
        
        # Log intervention configuration
        logger.info(f"BCG Intervention configured: start={self.start}, stop={self.stop}, age_range={self.age_range}, coverage={self.coverage}, efficacy={self.efficacy}")
        logger.debug(self.pars)
    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        # The probability objects will be automatically initialized by ss.link_dists()
        # which is called in the parent init_pre method
        return
    def check_eligibility(self):
        """
        Identify and randomly select eligible individuals for vaccination.
        
        This method checks for individuals within the configured age range who have not
        been vaccinated yet, and applies the coverage probability to select which
        eligible individuals will be vaccinated.
        
        Returns:
            ss.uids: Array of UIDs selected for vaccination
        """
        ages = self.sim.people.age
        under_age = (ages >= self.min_age) & (ages <= self.max_age)
        eligible = under_age & ~self.is_bcg_vaccinated
        eligible_uids = np.where(eligible)[0]
        
        # Debug logging
        logger.debug(f"BCG Eligibility Check: age_range={self.age_range}, total_pop={len(ages)}, under_age={np.sum(under_age)}, already_vaccinated={np.sum(self.is_bcg_vaccinated)}, eligible={len(eligible_uids)}")
        
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
        """
        Simplified eligibility check using starsim indexing.
        
        This is a more concise version of check_eligibility() that uses
        starsim's indexing methods for better performance.
        
        Returns:
            ss.uids: Array of UIDs selected for vaccination
        """
        eli = ((self.sim.people.age >= self.min_age) & (self.sim.people.age <= self.max_age) & ~self.is_bcg_vaccinated).uids
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
        for individuals within a specified age range. It models age-filtered eligibility, 
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
           Identifies individuals meeting age criteria (within age_range) who have not been vaccinated.
        4. **Vaccination assignment**:
           Randomly selects eligible individuals based on coverage probability and marks them
           as vaccinated, recording the vaccination timestep and age at vaccination.
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
            
        # Debug temporal eligibility
        if self.ti % 5 == 0:  # Every 5 timesteps
            logger.info(f"Temporal Eligibility Check - timestep {self.ti}: now_date={now_date}, start={self.start}, stop={self.stop}, within_window={self.start <= now_date <= self.stop}")
            
        if now_date < self.start or now_date > self.stop:
            if self.ti % 10 == 0:  # Log less frequently for skipped timesteps
                logger.info(f"BCG intervention skipped - outside time window: {now_date} not in [{self.start}, {self.stop}]")
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
        
        # Debug: Check population demographics periodically
        if self.ti % 10 == 0:  # Every 10 timesteps
            self.debug_population()
        
        if len(eligible) == 0:
            return
            
        # 4. Randomly select individuals for vaccination based on coverage probability
        # (This is already handled in check_eligibility())
        
        # 5. Update vaccination status: mark individuals as vaccinated and record the vaccination time
        self.is_bcg_vaccinated[eligible] = True
        self.ti_bcg_vaccinated[eligible] = current_time
        self.age_at_vaccination[eligible] = self.sim.people.age[eligible]
        
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
        """
        Define simulation result metrics for the BCG intervention.
        
        This method sets up all the result tracking arrays including:
        - Basic vaccination metrics (total, eligible, newly vaccinated, responders)
        - Protection status metrics (protected, expired)
        - Coverage and effectiveness metrics
        - Age-specific metrics for 1-5, 5-15, and 15+ years
        - Cumulative metrics
        - Intervention timing metrics (average age at vaccination, protection duration)
        - TB impact metrics (cases and deaths averted)
        """
        self.define_results(
            # Basic vaccination metrics
            ss.Result('n_vaccinated', dtype=int),           # Total vaccinated individuals
            ss.Result('n_eligible', dtype=int),             # Eligible individuals this timestep
            ss.Result('n_newly_vaccinated', dtype=int),     # Newly vaccinated this timestep
            ss.Result('n_vaccine_responders', dtype=int),   # Individuals who responded to vaccine
            
            # Protection status metrics
            ss.Result('n_protected', dtype=int),            # Currently protected individuals
            ss.Result('n_protection_expired', dtype=int),   # Individuals whose protection expired
            
            # Coverage and effectiveness metrics
            ss.Result('vaccination_coverage', dtype=float), # Coverage rate (vaccinated/eligible)
            ss.Result('protection_coverage', dtype=float),  # Protection rate (protected/total_pop)
            ss.Result('vaccine_effectiveness', dtype=float), # Response rate (responders/vaccinated)
            
            # Age-specific metrics (manual calculation)
            ss.Result('n_eligible_1_5', dtype=int),         # Eligible 1-5 years
            ss.Result('n_vaccinated_1_5', dtype=int),       # Vaccinated 1-5 years
            ss.Result('coverage_1_5', dtype=float),         # Coverage 1-5 years
            ss.Result('n_eligible_5_15', dtype=int),        # Eligible 5-15 years
            ss.Result('n_vaccinated_5_15', dtype=int),      # Vaccinated 5-15 years
            ss.Result('coverage_5_15', dtype=float),        # Coverage 5-15 years
            ss.Result('n_eligible_15_plus', dtype=int),     # Eligible 15+ years
            ss.Result('n_vaccinated_15_plus', dtype=int),   # Vaccinated 15+ years
            ss.Result('coverage_15_plus', dtype=float),     # Coverage 15+ years
            
            # Cumulative metrics
            ss.Result('cumulative_vaccinated', dtype=int),  # Total ever vaccinated
            ss.Result('cumulative_responders', dtype=int),  # Total ever responded
            ss.Result('cumulative_expired', dtype=int),     # Total ever expired protection
            
            # Intervention timing metrics
            ss.Result('avg_age_at_vaccination', dtype=float), # Average age when vaccinated
            ss.Result('avg_protection_duration', dtype=float), # Average protection duration
            
            # TB impact metrics (if TB disease is present)
            ss.Result('tb_cases_averted', dtype=int),       # TB cases prevented (estimated)
            ss.Result('tb_deaths_averted', dtype=int),      # TB deaths prevented (estimated)
        )
    def update_results(self):
        """
        Update all result metrics for the current timestep.
        
        This method calculates and stores all the intervention metrics including:
        - Basic vaccination counts and rates
        - Protection status for vaccinated individuals
        - Age-specific coverage metrics for 1-5, 5-15, and 15+ years
        - Cumulative totals
        - Average age at vaccination and protection duration
        - TB impact estimates (placeholder values)
        """
        current_time = self.ti
        
        # Basic vaccination metrics
        self.results['n_vaccinated'][self.ti] = np.count_nonzero(self.is_bcg_vaccinated)
        self.results['n_eligible'][self.ti] = self.n_eligible
        
        # Calculate newly vaccinated this timestep
        newly_vaccinated = np.sum((self.ti_bcg_vaccinated == current_time) & self.is_bcg_vaccinated)
        self.results['n_newly_vaccinated'][self.ti] = newly_vaccinated
        
        # Calculate vaccine responders (those with protection expiration set)
        # Convert to numpy array first to avoid starsim BooleanOperationError
        protection_expires_array = np.array(self.ti_bcg_protection_expires)
        vaccine_responders = np.sum(~np.isnan(protection_expires_array))
        self.results['n_vaccine_responders'][self.ti] = vaccine_responders
        
        # Protection status metrics
        all_vaccinated = self.is_bcg_vaccinated.uids
        if len(all_vaccinated) > 0:
            currently_protected = self.is_protected(all_vaccinated, current_time)
            n_protected = np.sum(currently_protected)
            n_expired = len(all_vaccinated) - n_protected
        else:
            n_protected = 0
            n_expired = 0
            
        self.results['n_protected'][self.ti] = n_protected
        self.results['n_protection_expired'][self.ti] = n_expired
        
        # Coverage and effectiveness metrics
        total_pop = len(self.sim.people)
        if self.n_eligible > 0:
            vaccination_coverage = newly_vaccinated / self.n_eligible
        else:
            vaccination_coverage = 0.0
            
        protection_coverage = n_protected / total_pop if total_pop > 0 else 0.0
        
        total_vaccinated = np.count_nonzero(self.is_bcg_vaccinated)
        if total_vaccinated > 0:
            vaccine_effectiveness = vaccine_responders / total_vaccinated
        else:
            vaccine_effectiveness = 0.0
            
        self.results['vaccination_coverage'][self.ti] = vaccination_coverage
        self.results['protection_coverage'][self.ti] = protection_coverage
        self.results['vaccine_effectiveness'][self.ti] = vaccine_effectiveness
        
        # Age-specific metrics (manual calculation)
        ages = self.sim.people.age
        
        # Calculate eligible and vaccinated by age groups (only for age groups within our target range)
        eligible_1_5 = np.sum((ages > 1) & (ages <= 5) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        vaccinated_1_5 = np.sum((ages > 1) & (ages <= 5) & self.is_bcg_vaccinated)
        
        eligible_5_15 = np.sum((ages > 5) & (ages <= 15) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        vaccinated_5_15 = np.sum((ages > 5) & (ages <= 15) & self.is_bcg_vaccinated)
        
        eligible_15_plus = np.sum((ages > 15) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        vaccinated_15_plus = np.sum((ages > 15) & self.is_bcg_vaccinated)
        
        self.results['n_eligible_1_5'][self.ti] = eligible_1_5
        self.results['n_vaccinated_1_5'][self.ti] = vaccinated_1_5
        self.results['coverage_1_5'][self.ti] = vaccinated_1_5 / eligible_1_5 if eligible_1_5 > 0 else 0.0
        
        self.results['n_eligible_5_15'][self.ti] = eligible_5_15
        self.results['n_vaccinated_5_15'][self.ti] = vaccinated_5_15
        self.results['coverage_5_15'][self.ti] = vaccinated_5_15 / eligible_5_15 if eligible_5_15 > 0 else 0.0
        
        self.results['n_eligible_15_plus'][self.ti] = eligible_15_plus
        self.results['n_vaccinated_15_plus'][self.ti] = vaccinated_15_plus
        self.results['coverage_15_plus'][self.ti] = vaccinated_15_plus / eligible_15_plus if eligible_15_plus > 0 else 0.0
        
        # Cumulative metrics
        self.results['cumulative_vaccinated'][self.ti] = total_vaccinated
        self.results['cumulative_responders'][self.ti] = vaccine_responders
        self.results['cumulative_expired'][self.ti] = n_expired
        
        # Intervention timing metrics
        # Calculate average age at vaccination using stored age at vaccination
        if total_vaccinated > 0:
            ages_at_vaccination = self.age_at_vaccination[self.is_bcg_vaccinated]
            avg_age_at_vaccination = np.mean(ages_at_vaccination)
        else:
            avg_age_at_vaccination = 0.0
            
        # Average protection duration is the configured duration (since it's fixed for all individuals)
        avg_protection_duration = self.duration
            
        self.results['avg_age_at_vaccination'][self.ti] = avg_age_at_vaccination
        self.results['avg_protection_duration'][self.ti] = avg_protection_duration
        
        # TB impact metrics (simplified estimates)
        # These would need to be calculated based on TB disease state and risk reduction
        # For now, we'll set placeholder values that could be updated by the TB disease model
        self.results['tb_cases_averted'][self.ti] = 0  # Placeholder
        self.results['tb_deaths_averted'][self.ti] = 0  # Placeholder
    def calculate_tb_impact(self, tb_disease):
        """
        Calculate estimated TB cases and deaths averted by BCG vaccination.
        
        This method should be called by the TB disease model to estimate the impact
        of BCG vaccination on TB outcomes.
        
        Parameters:
            tb_disease: The TB disease model instance
            
        Returns:
            dict: Dictionary containing estimated cases and deaths averted
        """
        if not hasattr(self.sim, 'diseases') or not hasattr(self.sim.diseases, 'tb'):
            return {'cases_averted': 0, 'deaths_averted': 0}
            
        # Get currently protected individuals
        all_vaccinated = self.is_bcg_vaccinated.uids
        if len(all_vaccinated) == 0:
            return {'cases_averted': 0, 'deaths_averted': 0}
            
        currently_protected = self.is_protected(all_vaccinated, self.ti)
        protected_uids = all_vaccinated[currently_protected]
        
        if len(protected_uids) == 0:
            return {'cases_averted': 0, 'deaths_averted': 0}
        
        # Calculate risk reduction for protected individuals
        # This is a simplified calculation - in practice, you'd want to track
        # the actual risk reduction applied to each individual
        
        # Estimate cases averted based on activation risk reduction
        activation_reduction = 1.0 - np.mean(self.pars.activation_modifier.rvs(protected_uids))
        potential_cases = len(protected_uids) * 0.01  # Assume 1% baseline activation rate
        cases_averted = int(potential_cases * activation_reduction)
        
        # Estimate deaths averted based on death risk reduction
        death_reduction = 1.0 - np.mean(self.pars.death_modifier.rvs(protected_uids))
        potential_deaths = len(protected_uids) * 0.001  # Assume 0.1% baseline death rate
        deaths_averted = int(potential_deaths * death_reduction)
        
        # Update results
        self.results['tb_cases_averted'][self.ti] = cases_averted
        self.results['tb_deaths_averted'][self.ti] = deaths_averted
        
        return {
            'cases_averted': cases_averted,
            'deaths_averted': deaths_averted,
            'protected_individuals': len(protected_uids)
        }
    def get_summary_stats(self):
        """
        Get summary statistics for the intervention.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
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
    def debug_population(self):
        """
        Debug method to check population demographics and vaccination status.
        
        This method provides detailed information about the population age distribution,
        vaccination status by age group, and eligibility counts for debugging purposes.
        
        Returns:
            dict: Dictionary containing debug information about population demographics
        """
        ages = self.sim.people.age
        total_pop = len(ages)
        
        # Age distribution
        age_1_5 = np.sum((ages > 1) & (ages <= 5))
        age_5_15 = np.sum((ages > 5) & (ages <= 15))
        age_15_plus = np.sum(ages > 15)
        
        # Vaccination status by age
        vaccinated_1_5 = np.sum((ages > 1) & (ages <= 5) & self.is_bcg_vaccinated)
        vaccinated_5_15 = np.sum((ages > 5) & (ages <= 15) & self.is_bcg_vaccinated)
        vaccinated_15_plus = np.sum((ages > 15) & self.is_bcg_vaccinated)
        
        # Eligible by age (considering age_range)
        eligible_1_5 = np.sum((ages > 1) & (ages <= 5) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        eligible_5_15 = np.sum((ages > 5) & (ages <= 15) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        eligible_15_plus = np.sum((ages > 15) & (ages >= self.min_age) & (ages <= self.max_age) & ~self.is_bcg_vaccinated)
        
        debug_info = {
            'total_population': total_pop,
            'age_range': self.age_range,
            'age_1_5': age_1_5,
            'age_5_15': age_5_15,
            'age_15_plus': age_15_plus,
            'vaccinated_1_5': vaccinated_1_5,
            'vaccinated_5_15': vaccinated_5_15,
            'vaccinated_15_plus': vaccinated_15_plus,
            'eligible_1_5': eligible_1_5,
            'eligible_5_15': eligible_5_15,
            'eligible_15_plus': eligible_15_plus,
        }
        
        logger.info(f"BCG Population Debug: {debug_info}")
        return debug_info
