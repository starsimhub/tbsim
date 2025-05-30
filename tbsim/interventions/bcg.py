import numpy as np
import starsim as ss
from tbsim.utils import Agents


 
__all__ = ['BCGProtection']

class BCGProtection(ss.Intervention):
    """
    Simulates the effect of BCG vaccination on tuberculosis outcomes in children under a target age.

    This intervention applies BCG vaccine protection to a subset of children under a specified age 
    threshold (default 5 years) based on a coverage probability. The vaccine modifies individual-level
    tuberculosis risk by reducing activation, clearance, and death rates for those vaccinated.

    Attributes:
        coverage (float): Proportion of eligible individuals to vaccinate (default 0.9).
        year (int): Reference year for intervention (default 1900).
        target_age (int): Maximum age for eligibility in years (default 5). (NOT USED FOR NOW).
        vaccinated (UIDs): Set of vaccinated individuals.
        n_eligible (int): Number of individuals eligible in the current time step.
        eligible (np.ndarray): Boolean array marking eligible individuals at each step.

    States:
        vaccinated (bool): Whether an individual has been vaccinated.

    Methods:
        prob_activation(): Draws a random factor to reduce TB activation risk post-vaccination.
        prob_clearance(): Draws a random factor to modify TB clearance probability post-vaccination.
        prob_death(): Draws a random factor to reduce TB death probability post-vaccination.
        check_eligibility(): Identifies eligible individuals under age threshold who are not yet vaccinated,
                            samples them based on coverage, and returns selected UIDs.
        step(): Executes the intervention for the current timestep: applies vaccination and modifies
                TB risk factors for newly vaccinated individuals.
        init_results(): Defines result metrics for number vaccinated and number eligible.
        update_results(): Updates the time series results for number vaccinated and number eligible.
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
        
    @staticmethod
    def prob_activation():
        min = .50
        max = .65
        return np.random.uniform(min, max)
            
    @staticmethod
    def prob_clearance():
        min = 1.3
        max = 1.5                   
        return np.random.uniform(min, max)
        
    @staticmethod
    def prob_death():
        min = .05
        max = .15
        return np.random.uniform(min, max)
        
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
        tb.rr_activation[eligible] *= self.prob_activation()   # Apply random activation rate to eligible individuals
        tb.rr_clearance[eligible] *= self.prob_clearance()     # Apply random clearance rate to eligible individuals
        tb.rr_death[eligible] *= self.prob_death()             # Apply random death rate to eligible individuals  


    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        self.results['n_vaccinated'][self.ti] =  np.count_nonzero(self.vaccinated)
        self.results['n_eligible'][self.ti] =  self.n_eligible    # this count gets reset every step, so it only counts the current step's eligible individuals
