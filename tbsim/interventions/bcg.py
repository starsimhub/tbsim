import numpy as np
import starsim as ss
import sciris as sc
import logging
import datetime
from tbsim.wrappers import Agents
from tbsim.tb import TBS

__all__ = ['BCGProtection']
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class BCGProtection(ss.Intervention):
    """
    Simulates BCG-like vaccination for tuberculosis prevention in individuals within a specified age range.
    
    This intervention identifies individuals within a configurable age range who have not yet been vaccinated. At each timestep, a proportion of these eligible individuals is selected based on the `coverage` parameter to receive simulated BCG protection. Once vaccinated, individuals are considered protected for a fixed number of years (`immunity_period`). While protected, their TB-related risk modifiers — activation, clearance, and death — are adjusted using scaled and sampled values from BCG-specific probability distributions.

    PARAMETERS:
        pars (`dict`, optional): Dictionary of parameters. Supported keys:
        - `coverage` (`float`): Fraction of eligible individuals vaccinated per timestep (default: 0.5).
        - `start` (`str`/`datetime.date`): Start date for the intervention (default: '1900-01-01').
        - `stop` (`str`/`datetime.date`): Stop date for the intervention (default: '2100-12-31').
        - `efficacy` (`float`): Probability of effective vaccine response (default: 0.8).
        - `immunity_period` (`int`): Immunity period (in years) for which BCG protection remains effective (default: 10).
        - `age_range` (`tuple`): Age range (`min_age`, `max_age`) for vaccination eligibility (default: (0, 5)).
        - `activation_modifier` (`ss.uniform`): Distribution for TB activation risk modifier (default: `uniform(0.5, 0.65)`) - reduces activation risk.
        - `clearance_modifier` (`ss.uniform`): Distribution for bacterial clearance modifier (default: `uniform(1.3, 1.5)`) - increases clearance probability.
        - `death_modifier` (`ss.uniform`): Distribution for TB mortality modifier (default: `uniform(0.05, 0.15)`) - reduces mortality risk (lower values = better protection).

    USAGE EXAMPLES:
        ```python
        # Basic usage with default parameters (targets individuals 0-5 years)
        bcg = BCGProtection()
        
        # Custom parameters using dictionary (targets individuals 1-7 years)
        bcg = BCGProtection(pars={
            'coverage': 0.8,
            'start': '2020-01-01',
            'stop': '2030-12-31',
            'efficacy': 0.9,
            'immunity_period': 15,
            'age_range': (1, 7)
        })
        
        # Adult vaccination example (targets individuals 18-65 years)
        bcg = BCGProtection(pars={
            'age_range': (18, 65),
            'coverage': 0.4,
            'efficacy': 0.7,
            'immunity_period': 8
        })
        
        # Custom probability distributions
        bcg = BCGProtection(pars={
            'activation_modifier': ss.uniform(0.4, 0.6),
            'clearance_modifier': ss.uniform(1.2, 1.6),
            'death_modifier': ss.uniform(0.03, 0.12)
        })
        ```

    ATTRIBUTES:
        coverage_dist (ss.bernoulli): Probability distribution for vaccination coverage.
        eligible      (np.ndarray): Boolean mask of currently eligible individuals.
        n_eligible    (int): Number of individuals eligible for vaccination in the current step.
        p_vaccine_response (ss.bernoulli): Probability distribution for vaccine response.
        start         (datetime.date): Start date for the intervention.
        stop          (datetime.date): Stop date for the intervention.
        min_age       (float): Minimum age for vaccination eligibility.
        max_age       (float): Maximum age for vaccination eligibility.
        
    STATES:
        is_bcg_vaccinated (bool): Indicates whether an individual has received the BCG vaccine.
        ti_bcg_vaccinated (float): Timestep at which the individual was vaccinated.
        ti_bcg_protection_expires (float): Timestep when protection expires.
        age_at_vaccination (float): Age when the individual was vaccinated.
        bcg_activation_modifier_applied (float): Activation risk modifier applied to each individual.
        bcg_clearance_modifier_applied (float): Clearance modifier applied to each individual.
        bcg_death_modifier_applied (float): Death risk modifier applied to each individual.
    Methods:
        check_eligibility(): Identify and randomly select eligible individuals for vaccination.
        is_protected(uids, current_time): Return boolean mask indicating protected individuals.
        step(): Apply BCG protection and adjust TB risk modifiers accordingly.
        _apply_protection_effects(protected_uids): Apply BCG protection effects to TB risk modifiers.
        _remove_protection(expired_uids): Remove BCG protection effects when protection expires.
        _maintain_ongoing_protection(current_time): Maintain protection effects for all currently protected individuals.
        init_results(): Define simulation result metrics.
        update_results(): Record the number of vaccinated and eligible individuals each timestep.
        get_summary_stats(): Get summary statistics for the intervention.
        debug_population(): Debug method to check population demographics and vaccination status.
        calculate_tb_impact(tb_model): Calculate the impact of BCG vaccination on TB outcomes.
    Notes:
        This intervention assumes the presence of a TB disease model attached to the simulation and modifies its rr_activation, rr_clearance, and rr_death arrays. The intervention uses starsim probability distributions for stochastic modeling and proper date handling for temporal eligibility checks. The age_range parameter allows targeting of any age group, making this intervention suitable for both pediatric and adult vaccination strategies.
    """
    def __init__(self, pars={}, **kwargs):
        """
        Initialize a BCGProtection intervention instance.

        This constructor sets up the BCG vaccination intervention with user-specified or default parameters. It defines the probability distributions for coverage, efficacy, and risk modifiers, as well as the age range and timing for eligibility. It also initializes the state arrays that track vaccination and protection status for each individual in the simulation.

        Parameters:
            pars (`dict`, optional): Dictionary of intervention parameters. See class docstring for details.
            **kwargs: Additional keyword arguments passed to the parent Intervention class.
        """
        super().__init__(**kwargs)
        # Handle duration as alias for immunity_period
           
        self.define_pars(
            coverage=ss.bernoulli(p=pars.get('coverage', 0.5)),  # Default 50% coverage
            start=sc.date('1900-01-01'),
            stop=sc.date('2100-12-31'),
            efficacy= 0.8,  # Default 80% efficacy
            immunity_period=ss.years(10),  # Default 10 years
            age_range=(0, 5),
            # Default modifiers
            activation_modifier= ss.uniform(0.5, 0.65),  # Reduces activation risk
            clearance_modifier=ss.uniform(1.3, 1.5),    # Increases clearance
            death_modifier=ss.uniform(0.05, 0.15),          # Reduces death risk
        )
        self.update_pars(pars)
        self.min_age = self.pars.age_range[0]
        self.max_age = self.pars.age_range[1]
        self.n_eligible = 0
        self.eligible = []
        self.p_vaccine_response = ss.bernoulli(p=self.pars.efficacy)
        self.define_states(
            ss.BoolArr('is_bcg_vaccinated', default=False),
            ss.FloatArr('ti_bcg_vaccinated'),
            ss.Arr('ti_bcg_protection_expires'),
            ss.FloatArr('age_at_vaccination'),
            ss.FloatArr('bcg_activation_modifier_applied'),
            ss.FloatArr('bcg_clearance_modifier_applied'),
            ss.FloatArr('bcg_death_modifier_applied'),
        )
        logger.info(f"BCG Intervention configured: start={self.pars.start}, stop={self.pars.stop}, age_range={self.pars.age_range}, coverage={self.pars.coverage}, efficacy={self.pars.efficacy}")
        logger.debug(self.pars)
        
    def init_pre(self, sim):
        """Initialize the intervention before the simulation starts."""
        super().init_pre(sim)
        # The probability objects will be automatically initialized by ss.link_dists()
        # which is called in the parent init_pre method
        return
    
    
    def check_eligibility(self):
        """
        Identify and randomly select eligible individuals for BCG vaccination in the current timestep.

        This method uses starsim's efficient indexing to find individuals who are within the configured age range and have not yet been vaccinated. It then applies the coverage probability to randomly select a subset of these eligible individuals for vaccination.

        Returns:
            ss.uids: Array of UIDs selected for vaccination in this timestep.
        """
        eli = ((self.sim.people.age >= self.min_age) & (self.sim.people.age <= self.max_age) & ~self.is_bcg_vaccinated).uids
        chos = self.pars.coverage.filter(eli)
        return chos
        
    def is_protected(self, uids, current_time):
        """Return boolean array: True if still protected (within immunity_period), else False."""
        # Convert uids to ss.uids if it's a numpy array
        if isinstance(uids, np.ndarray):
            uids = ss.uids(uids)
        
        # Check if individuals are vaccinated and still within protection immunity_period
        vaccinated = self.is_bcg_vaccinated[uids]
        within_duration = (current_time - self.ti_bcg_vaccinated[uids]) <= self.pars.immunity_period
        
        # Also check if protection expiration time is set (for vaccine responders)
        has_protection_expiry = ~np.isnan(self.ti_bcg_protection_expires[uids])
        
        return vaccinated & within_duration & has_protection_expiry
    
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
        6. **Protection immunity_period assignment**:
           Sets expiration timestep for vaccine responders based on protection immunity_period.
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
            logger.info(f"Temporal Eligibility Check - timestep {self.ti}: now_date={now_date}, start={self.pars.start}, stop={self.pars.stop}, within_window={self.pars.start <= now_date <= self.pars.stop}")
            
        if now_date < self.pars.start or now_date > self.pars.stop:
            if self.ti % 10 == 0:  # Log less frequently for skipped timesteps
                logger.info(f"BCG intervention skipped - outside time window: {now_date} not in [{self.pars.start}, {self.pars.stop}]")
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
        self.age_at_vaccination[eligible] = self.sim.people.age[eligible]
        
        # 6. Consider vaccine efficacy: account for the probability of immunological response post-vaccination
        # Model host response heterogeneity: incorporate inter-individual variability in vaccine uptake and effectiveness
        vaccine_responders = self.p_vaccine_response.filter(eligible)
        
        if len(vaccine_responders) == 0:
            return
            
        # 7. Calculate immunological protection immunity_period: set the timepoint when vaccine-induced immunity wanes
        protection_expires = current_time + self.pars.immunity_period
        self.ti_bcg_protection_expires[vaccine_responders] = protection_expires
        
        # 8. Adjust TB risk modifiers: apply BCG-specific modifiers to TB activation, clearance, and death rates
        self._apply_protection_effects(vaccine_responders)
        
        # 9. Maintain protection effects for all currently protected individuals
        # This is necessary because the TB model resets risk modifiers every timestep
        self._maintain_ongoing_protection(current_time)
        
    def begin_step(self):
        """
        Called at the very start of each simulation timestep, before any disease model steps.
        Ensures that BCG protection effects are applied to all currently protected individuals
        before the TB model uses the risk modifiers.
        """
        current_time = self.sim.ti
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
        tb.state[protected_uids] = TBS.PROTECTED      # TB Protected status - 100 -> BCG
        # Check if modifiers have already been applied to avoid multiple applications
        
        already_protected = ~np.isnan(self.bcg_activation_modifier_applied[protected_uids])
        if np.any(already_protected):
            logger.warning(f"BCG protection effects already applied to {np.sum(already_protected)} individuals. Skipping re-application.")
            return
        
        # Sample individual protection levels using starsim probability distributions
        activation_modifiers = self.pars.activation_modifier.rvs(protected_uids)
        clearance_modifiers = self.pars.clearance_modifier.rvs(protected_uids)
        death_modifiers = self.pars.death_modifier.rvs(protected_uids)
        
        # Store the modifiers applied to each individual
        self.bcg_activation_modifier_applied[protected_uids] = activation_modifiers
        self.bcg_clearance_modifier_applied[protected_uids] = clearance_modifiers
        self.bcg_death_modifier_applied[protected_uids] = death_modifiers
        
        # Apply modifiers to TB risk rates
        tb.rr_activation[protected_uids] *= activation_modifiers
        tb.rr_clearance[protected_uids] *= clearance_modifiers
        tb.rr_death[protected_uids] *= death_modifiers
    
    def _apply_protection_effects_force(self, protected_uids):
        """
        Apply BCG protection effects to TB risk modifiers, bypassing the "already applied" check.
        
        This method is used to re-apply protection effects every timestep since the TB model
        resets risk modifiers at the end of each timestep.
        
        Parameters:
            protected_uids: Array of UIDs who are currently protected
        """
        if len(protected_uids) == 0:
            return
            
        tb = self.sim.diseases.tb

        # Get the stored modifiers that were originally applied to each individual
        activation_modifiers = self.bcg_activation_modifier_applied[protected_uids]
        clearance_modifiers = self.bcg_clearance_modifier_applied[protected_uids]
        death_modifiers = self.bcg_death_modifier_applied[protected_uids]
        
        # Check if modifiers were actually applied (not NaN)
        valid_modifiers = ~np.isnan(activation_modifiers)
        if not np.any(valid_modifiers):
            logger.warning("No valid BCG modifiers found for protected individuals. Skipping re-application.")
            return
            
        # Only process individuals with valid modifiers
        valid_uids = protected_uids[valid_modifiers]
        tb.rr_activation[valid_uids] *= activation_modifiers[valid_modifiers]
        tb.rr_clearance[valid_uids] *= clearance_modifiers[valid_modifiers]
        tb.rr_death[valid_uids] *= death_modifiers[valid_modifiers]

    def _remove_protection(self, expired_uids):
        """
        Remove BCG protection effects when protection expires.
        
        Parameters:
            expired_uids: Array of UIDs whose protection has expired
        """
        if len(expired_uids) == 0:
            return
            
        tb = self.sim.diseases.tb
        tb.state[expired_uids] = TBS.NONE
        # Get the stored modifiers that were originally applied
        activation_modifiers = self.bcg_activation_modifier_applied[expired_uids]
        clearance_modifiers = self.bcg_clearance_modifier_applied[expired_uids]
        death_modifiers = self.bcg_death_modifier_applied[expired_uids]
        
        # Check if modifiers were actually applied (not NaN)
        valid_modifiers = ~np.isnan(activation_modifiers)
        if not np.any(valid_modifiers):
            logger.warning("No valid BCG modifiers found for expired individuals. Skipping removal.")
            return
            
        # Only process individuals with valid modifiers
        valid_uids = expired_uids[valid_modifiers]
        tb.rr_activation[valid_uids] /= activation_modifiers[valid_modifiers]
        tb.rr_clearance[valid_uids] /= clearance_modifiers[valid_modifiers]
        tb.rr_death[valid_uids] /= death_modifiers[valid_modifiers]
        self.bcg_activation_modifier_applied[expired_uids] = np.nan
        self.bcg_clearance_modifier_applied[expired_uids] = np.nan
        self.bcg_death_modifier_applied[expired_uids] = np.nan
        
        # Clear protection expiration time
        self.ti_bcg_protection_expires[expired_uids] = np.nan
        
        # Reset vaccination status to False when protection expires
        self.is_bcg_vaccinated[expired_uids] = False
        self.ti_bcg_vaccinated[expired_uids] = np.nan
        self.age_at_vaccination[expired_uids] = np.nan
        
    def _maintain_ongoing_protection(self, current_time):
        """
        Maintain protection effects for all currently protected individuals.
        
        This ensures that protection effects are continuously applied to all
        vaccinated individuals who are still within their protection period.
        This is necessary because the TB model resets risk modifiers every timestep.
        
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
            # We need to bypass the "already applied" check since TB model resets modifiers
            self._apply_protection_effects_force(protected_uids)
            
    def init_results(self):
        """
        Define simulation result metrics for the BCG intervention.
        
        This method sets up all the result tracking arrays including:
        - Basic vaccination metrics (total, eligible, newly vaccinated, responders)
        - Protection status metrics (protected, expired)
        - Coverage and effectiveness metrics
        - Age-specific metrics for 1-5, 5-15, and 15+ years
        - Cumulative metrics
        - Intervention timing metrics (average age at vaccination, protection immunity_period)
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
            
            # Cumulative metrics
            ss.Result('cumulative_vaccinated', dtype=int),  # Total ever vaccinated
            ss.Result('cumulative_responders', dtype=int),  # Total ever responded
            ss.Result('cumulative_expired', dtype=int),     # Total ever expired protection
            
            # Intervention timing metrics
            ss.Result('avg_age_at_vaccination', dtype=float), # Average age when vaccinated
            ss.Result('avg_protection_duration', dtype=float), # Average protection immunity_period
            
            
        )

    def update_results(self):
        """
        Update all result metrics for the current timestep.
        This method calculates and stores all the intervention metrics including:
        - Basic vaccination counts and rates
        - Protection status for vaccinated individuals
        - Age-specific coverage metrics for 1-5, 5-15, and 15+ years
        - Cumulative totals
        - Average age at vaccination and protection immunity_period
        - TB impact estimates (now calculated, not placeholder)
        """
        current_time = self.ti
        self.results['n_vaccinated'][self.ti] = np.count_nonzero(self.is_bcg_vaccinated)
        self.results['n_eligible'][self.ti] = self.n_eligible
        newly_vaccinated = np.sum((self.ti_bcg_vaccinated == current_time) & self.is_bcg_vaccinated)
        self.results['n_newly_vaccinated'][self.ti] = newly_vaccinated
        protection_expires_array = np.array(self.ti_bcg_protection_expires)
        vaccine_responders = np.sum(~np.isnan(protection_expires_array))
        self.results['n_vaccine_responders'][self.ti] = vaccine_responders
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
        self.results['cumulative_vaccinated'][self.ti] = total_vaccinated
        self.results['cumulative_responders'][self.ti] = vaccine_responders
        self.results['cumulative_expired'][self.ti] = n_expired
        if total_vaccinated > 0:
            ages_at_vaccination = self.age_at_vaccination[self.is_bcg_vaccinated]
            avg_age_at_vaccination = np.mean(ages_at_vaccination)
        else:
            avg_age_at_vaccination = 0.0
        avg_protection_duration = self.pars.immunity_period
        self.results['avg_age_at_vaccination'][self.ti] = avg_age_at_vaccination
        self.results['avg_protection_duration'][self.ti] = avg_protection_duration



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
        
        # Count individuals in the configured age range
        min_age, max_age = self.pars.age_range
        in_age_range = np.sum((ages >= min_age) & (ages <= max_age))
        vaccinated_in_range = np.sum((ages >= min_age) & (ages <= max_age) & self.is_bcg_vaccinated)
        eligible_in_range = np.sum((ages >= min_age) & (ages <= max_age) & ~self.is_bcg_vaccinated)
        
        debug_info = {
            'total_population': total_pop,
            'age_range': self.pars.age_range,
            f'age_{min_age}_{max_age}': in_age_range,
            f'vaccinated_{min_age}_{max_age}': vaccinated_in_range,
            f'eligible_{min_age}_{max_age}': eligible_in_range,
        }
        
        logger.info(f"BCG Population Debug: {debug_info}")
        return debug_info
    
    def calculate_tb_impact(self, tb_model):
        """
        Calculate the impact of BCG vaccination on TB outcomes.
        
        Parameters:
            tb_model: The TB disease model
            
        Returns:
            dict: Dictionary containing TB impact metrics
        """
        if not hasattr(self, 'sim') or self.sim is None:
            return {'error': 'Simulation not initialized'}
            
        # Get vaccinated and unvaccinated individuals
        vaccinated_uids = self.is_bcg_vaccinated.uids
        unvaccinated_uids = (~self.is_bcg_vaccinated).uids
        
        # Calculate TB outcomes by vaccination status
        tb_states = tb_model.state.raw
        
        # Check if arrays are initialized and have the right size
        if len(tb_states) == 0:
            return {
                'vaccinated_tb_deaths': 0,
                'unvaccinated_tb_deaths': 0,
                'vaccinated_tb_cases': 0,
                'unvaccinated_tb_cases': 0,
                'vaccinated_death_modifier': 1.0,
                'unvaccinated_death_modifier': 1.0,
                'deaths_averted': 0,
                'cases_averted': 0,
                'error': 'TB model not initialized'
            }
        
        # Check bounds before indexing
        max_uid = max(np.max(vaccinated_uids) if len(vaccinated_uids) > 0 else 0,
                     np.max(unvaccinated_uids) if len(unvaccinated_uids) > 0 else 0)
        
        if max_uid >= len(tb_states):
            return {
                'vaccinated_tb_deaths': 0,
                'unvaccinated_tb_deaths': 0,
                'vaccinated_tb_cases': 0,
                'unvaccinated_tb_cases': 0,
                'vaccinated_death_modifier': 1.0,
                'unvaccinated_death_modifier': 1.0,
                'deaths_averted': 0,
                'cases_averted': 0,
                'error': 'UIDs out of bounds for TB state array'
            }
        
        vaccinated_tb_deaths = np.sum(tb_states[vaccinated_uids] == 4) if len(vaccinated_uids) > 0 else 0
        unvaccinated_tb_deaths = np.sum(tb_states[unvaccinated_uids] == 4) if len(unvaccinated_uids) > 0 else 0
        
        vaccinated_tb_cases = np.sum(tb_states[vaccinated_uids] == 2) if len(vaccinated_uids) > 0 else 0
        unvaccinated_tb_cases = np.sum(tb_states[unvaccinated_uids] == 2) if len(unvaccinated_uids) > 0 else 0
        
        # Calculate risk modifiers
        vaccinated_death_modifier = np.mean(tb_model.rr_death.raw[vaccinated_uids]) if len(vaccinated_uids) > 0 else 1.0
        unvaccinated_death_modifier = np.mean(tb_model.rr_death.raw[unvaccinated_uids]) if len(unvaccinated_uids) > 0 else 1.0
        
        return {
            'vaccinated_tb_deaths': vaccinated_tb_deaths,
            'unvaccinated_tb_deaths': unvaccinated_tb_deaths,
            'vaccinated_tb_cases': vaccinated_tb_cases,
            'unvaccinated_tb_cases': unvaccinated_tb_cases,
            'vaccinated_death_modifier': vaccinated_death_modifier,
            'unvaccinated_death_modifier': unvaccinated_death_modifier,
            'deaths_averted': unvaccinated_tb_deaths - vaccinated_tb_deaths,
            'cases_averted': unvaccinated_tb_cases - vaccinated_tb_cases,
        }


