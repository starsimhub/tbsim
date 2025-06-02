import numpy as np
import starsim as ss
from tbsim.utils import Agents


 
__all__ = ['BCGProtection']

class BCGProtection(ss.Intervention):
    """
    Applies BCG-like protection against tuberculosis in children under age 5.

    This intervention identifies children under 5 years old who are not yet vaccinated and
    randomly selects a proportion of them (based on the `coverage` parameter) to receive simulated
    protection against TB. Once vaccinated, individuals experience a probabilistic reduction in
    their activation, death, and clearance risk modifiers within the TB disease model.

    Attributes:
        coverage (float): Proportion of under-5 children to vaccinate each timestep (default 0.9).
        year (int): Reference year for the intervention (default 1900).
        vaccinated (UIDs): Set of UIDs representing individuals who have been vaccinated.
        n_eligible (int): Number of individuals eligible for vaccination at the current timestep.
        eligible (np.ndarray): Boolean array indicating eligibility status in the current timestep.

    Defined States:
        vaccinated (bool): State flag indicating whether an individual has received the BCG vaccine.

    Methods:
        check_eligibility(): Identifies unvaccinated children under age 5, samples based on
            coverage, and returns their UIDs.
        step(): Applies vaccination and updates the TB disease risk modifiers for newly vaccinated individuals.
        init_results(): Initializes the results tracking for number vaccinated and number eligible.
        update_results(): Records the current timestep's vaccination and eligibility counts.
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.coverage = pars.get('coverage', 0.9)   # Default coverage if not specified
        self.year = pars.get('year', 1900)          # Default year if not specified
        # self.target_age = pars.get('target_age', 5) # Default target age if not specified
        
        
        print(self.pars)
        self.vaccinated = ss.uids()
        self.n_eligible = 0
        self.eligible = []
        # Define states for individuals who 
        self.define_states(
            ss.State('vaccinated', default=False)
        )
        

    def check_eligibility(self):
        self.eligible = np.zeros(self.sim.people.n_uids, dtype=bool) #Resets eligibility for all individuals
        
        under5 = Agents.under_5(self.sim.people)    # Get UIDs of individuals under 5 years old
        eligible = under5 & ~self.vaccinated        # Filter out those who are already vaccinated
        eligible_coverage = int(len(eligible) * self.coverage)  # Calculate number of individuals to vaccinate based on coverage
        eligible_subset = np.random.choice(eligible, size=eligible_coverage, replace=False) # Randomly select individuals to vaccinate based on coverage
        
        if len(eligible_subset)>0: 
            self.eligible[eligible_subset] = True  # Mark these individuals as eligible for vaccination
        
        self.n_eligible = np.count_nonzero(self.eligible)  # Count the number of eligible individuals

        return ss.uids(eligible_subset)
    
    
    def step(self):
        eligible = self.check_eligibility()

        if len(eligible) == 0: return  # If no eligible individuals, exit early
        
        self.vaccinated[eligible] = True  # Mark eligible individuals as vaccinated
        tb = self.sim.diseases.tb   # Access the TB disease model   
        tb.rr_activation[eligible] *= Probability.activation()   # Apply random activation rate to eligible individuals
        tb.rr_clearance[eligible] *= Probability.clearance()     # Apply random clearance rate to eligible individuals
        tb.rr_death[eligible] *= Probability.death()             # Apply random death rate to eligible individuals  


    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        self.results['n_vaccinated'][self.ti] =  np.count_nonzero(self.vaccinated)
        self.results['n_eligible'][self.ti] =  self.n_eligible    # this count gets reset every step, so it only counts the current step's eligible individuals


class Probability:
    """
    Static class providing random scaling factors for modifying TB risks post-vaccination.

    These factors simulate the biological variability in how BCG vaccination impacts:
    - Activation risk (from latent to active TB)
    - Clearance probability (spontaneous recovery)
    - Mortality risk (death from TB)

    The sampled values are multiplicative factors applied to individual risk modifiers.

    Methods:
        activation(): Returns a sampled multiplier (0.50–0.65) reducing TB activation risk.
        clearance(): Returns a sampled multiplier (1.3–1.5) increasing TB clearance probability.
        death(): Returns a sampled multiplier (0.05–0.15) reducing TB-related mortality.
    """
    @staticmethod
    def _sample_uniform(low: float, high: float) -> float:
        """ Sample a value from a uniform distribution in the given range. """
        return np.random.uniform(low, high)

    @staticmethod
    def activation() -> float:
        """ Probability of activation (e.g. latent to active disease). """
        return Probability._sample_uniform(0.50, 0.65)

    @staticmethod
    def clearance() -> float:
        """ Probability of clearing an infection (natural recovery)."""
        return Probability._sample_uniform(1.3, 1.5)

    @staticmethod
    def death() -> float:
        """ Probability of death due to the disease."""
        return Probability._sample_uniform(0.05, 0.15)