"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc

__all__ = ['Nutrition']

class Nutrition(ss.Disease):
    """
    This class implements a basic Nutrition model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having a condition, or accelerate a condition 
    (e.g., TB, HIV) and associated mortality.
    """
    # Possible references:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    # https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf 
    
    def __init__(self, pars=None):
        # According to https://www.who.int/news-room/questions-and-answers/item/malnutrition
        # Actual Disease class initialization.
        pars = ss.omergeleft(pars,
            init_risk = ss.bernoulli(p=0.3),     # undernourished initial prevalence of risk factors
            years_from_risk_to_un = ss.expon(scale=10),
        )
        super().__init__(pars=pars)

        # Additional Nutrition States
        self.add_states(
            ss.State('at_risk', bool, False),        # Normal- at_risk nutrition state (no malnutrition or overnutrition
            ss.State('undernourished', bool, False),    # malnutrition, this includes stunting, wasting, underweight 
        )
        self.add_states(
            ss.State('ti_risk', int, ss.INT_NAN),           # Time at which risk began
            ss.State('ti_undernourished', int, ss.INT_NAN), # Time at which undernourished was diagnosed
        )
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
        alive_uids = ss.true(sim.people.alive)
        initial_risk = self.pars['init_risk'].filter(alive_uids)
        self.at_risk[initial_risk] = True
        self.ti_risk = sim.ti
        self.ti_undernourished[initial_risk] = sim.ti + self.pars['years_from_risk_to_un'].rvs(initial_risk) / sim.dt
        return initial_risk

    def update_pre(self, sim):
        # Make all the updates from the NCD model 
        # Add nutrition dynamics here
        return
    
    def make_new_cases(self, sim):
        new_cases = ss.true(self.ti_undernourished == sim.ti)
        if len(new_cases) > 0:
            self.at_risk[new_cases] = False
            self.undernourished[new_cases] = True
       
        super().set_prognoses(sim, new_cases) # Logging
        return new_cases

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
        ]
        return

    def update_results(self, sim):
        super().update_results(sim)
        ti = sim.ti
        alive = sim.people.alive
        self.results.prevalence[ti]    = np.count_nonzero(self.undernourished & alive)/alive.count()
        return
