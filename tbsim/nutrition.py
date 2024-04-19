"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
from enum import IntEnum, auto

__all__ = ['Nutrition', 'MacroNutrients', 'MicroNutrients']

class MacroNutrients(IntEnum):
    STANDARD_OR_ABOVE = auto()
    SLIGHTLY_BELOW_STANDARD = auto()
    MARGINAL = auto()
    UNSATISFACTORY = auto()

class MicroNutrients(IntEnum):
    NORMAL = auto()
    DEFICIENT = auto()

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
        #pars = ss.omergeleft(pars,
        #    init_risk = ss.bernoulli(p=0.3),     # undernourished initial prevalence of risk factors
        #    years_from_risk_to_un = ss.expon(scale=10),
        #)
        super().__init__(pars=pars)

        # Additional Nutrition States
        self.add_states(
            ss.State('macro', int, MacroNutrients.STANDARD_OR_ABOVE), # Values set elsewhere
            ss.State('micro', int, MicroNutrients.NORMAL), # Values set elsewhere
        )
        self.add_states(
            ss.State('ti_macro', int, ss.INT_NAN),          # Time index of change in macronutrition
            ss.State('new_macro_state', int, ss.INT_NAN),   # New macro nutrition state

            ss.State('ti_micro', int, ss.INT_NAN), # Time at which undernourished was diagnosed
            ss.State('new_micro_state', int, ss.INT_NAN),   # New micro nutrition state
        )
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
        return

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
        new_macro = ss.true(self.ti_macro == sim.ti)
        if len(new_macro) > 0:
            self.macro[new_macro] = self.new_macro_state
        super().set_prognoses(sim, new_macro) # Logging

        new_micro = ss.true(self.ti_micro == sim.ti)
        if len(new_micro) > 0:
            self.micro[new_micro] = self.new_micro_state
        super().set_prognoses(sim, new_micro) # Logging
       
        return new_macro, new_micro

    def init_results(self, sim):
        """
        Initialize results
        """
        super().init_results(sim)
        self.results += [
            ss.Result(self.name, 'prev_macro_standard_or_above', sim.npts, dtype=float),
            ss.Result(self.name, 'prev_macro_slightly_below', sim.npts, dtype=float),
            ss.Result(self.name, 'prev_macro_marginal', sim.npts, dtype=float),
            ss.Result(self.name, 'prev_macro_unsatisfactory', sim.npts, dtype=float),

            ss.Result(self.name, 'prev_micro_normal', sim.npts, dtype=float),
            ss.Result(self.name, 'prev_micro_deficient', sim.npts, dtype=float),
        ]
        return

    def update_results(self, sim):
        super().update_results(sim)
        ti = sim.ti
        alive = sim.people.alive
        self.results.prev_macro_standard_or_above[ti] = np.count_nonzero((self.macro==MacroNutrients.STANDARD_OR_ABOVE) & alive)/alive.count()
        self.results.prev_macro_slightly_below[ti] = np.count_nonzero((self.macro==MacroNutrients.SLIGHTLY_BELOW_STANDARD) & alive)/alive.count()
        self.results.prev_macro_marginal[ti] = np.count_nonzero((self.macro==MacroNutrients.MARGINAL) & alive)/alive.count()
        self.results.prev_macro_unsatisfactory[ti] = np.count_nonzero((self.macro==MacroNutrients.UNSATISFACTORY) & alive)/alive.count()

        self.results.prev_micro_normal[ti] = np.count_nonzero((self.micro==MicroNutrients.NORMAL) & alive)/alive.count()
        self.results.prev_micro_deficient[ti] = np.count_nonzero((self.micro==MicroNutrients.DEFICIENT) & alive)/alive.count()
        return
