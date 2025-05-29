import numpy as np
import starsim as ss

__all__ = ['BCGProtection']

class BCGProtection(ss.Intervention):
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            year=[1900],         # Placeholder; can be used for scheduling logic
            coverage=0.95,
            target_age=5,        # Age cutoff in years
        )
        self.update_pars(pars=pars, **kwargs)
        self.vaccinated = ss.uids()
        self.n_eligible = 0
        # Define states for individuals who 
        self.define_states(
            ss.State('vaccinated', default=False),
            ss.State('eligible', default=False)
        )
    @staticmethod
    def prob_activation(self):
        min = .80
        max = .95
        return np.random.uniform(min, max)
            
    @staticmethod
    def prob_clearance():
        min = .10
        max = .30                   
        return np.random.uniform(min, max)
        
    @staticmethod
    def prob_death():
        min = .05
        max = .15
        return np.random.uniform(min, max)
        
    def check_eligibility(self):
        # Call the superclass method
        super().check_eligibility()

        # Get the uids of alive individuals
        uids = self.sim.people.auids

        # Identify eligible individuals: alive and age <= 5
        el = uids[self.sim.people.age[uids] <= 5]

        # Mark eligible individuals
        self.eligible[el] = True

        # Determine how many individuals are eligible
        eligible_uids = el
        self.n_eligible = len(eligible_uids)

        # Sample who gets vaccinated based on coverage
        vaccinated_mask = np.random.binomial(1, self.pars.coverage, self.n_eligible).astype(bool)
        
        # Apply vaccination to selected eligible individuals
        self.vaccinated = eligible_uids[vaccinated_mask]
        self.sim.people.vaccination_year[self.vaccinated] = self.pars.year[0]


    def step(self):
        self.check_eligibility()
        if self.vaccinated is None or len(self.vaccinated) == 0:
            return

        tb = self.sim.diseases.tb
        ppl = self.sim.people

        # Only apply effects to still-alive vaccinated individuals
        alive_mask = ppl.alive[self.vaccinated]
        uids = self.vaccinated[alive_mask]

        # Apply protection - these are SAMPLE values and effects
        # TODO: Please provide the actual values and effects of the BCG vaccine
        
        tb.rr_activation[uids] *= self.prob_activation(self)
        tb.rr_clearance[uids] *= self.prob_clearance()
        tb.rr_death[uids] *= self.prob_death()


    def init_results(self):
        self.define_results(
            ss.Result('n_vaccinated', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        self.results['n_vaccinated'][self.ti] = len(self.vaccinated)
        self.results['n_eligible'][self.ti] = self.n_eligible
