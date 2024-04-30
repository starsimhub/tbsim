"""
Define non-communicable disease (Malnutrition) model
"""

import numpy as np
import starsim as ss
from enum import IntEnum, auto


__all__ = ['Malnutrition', 'MacroNutrients', 'MicroNutrients']

class MacroNutrients(IntEnum):
    STANDARD_OR_ABOVE = auto()
    SLIGHTLY_BELOW_STANDARD = auto()
    MARGINAL = auto()
    UNSATISFACTORY = auto()

class MicroNutrients(IntEnum):
    NORMAL = auto()
    DEFICIENT = auto()

class Malnutrition(ss.Disease):         
    """
    This class implements a basic Malnutrition model. It inherits from the startim Disease class.
    """
    # Possible references:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    # https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf 
    
    def __init__(self, pars=None):
        # According to https://www.who.int/news-room/questions-and-answers/item/malnutrition
        super().__init__(pars=pars)

        self.add_states(
            ss.State('macro_state', int, MacroNutrients.STANDARD_OR_ABOVE), # To keep track of the macronutrients state
            ss.State('micro_state', int, MicroNutrients.NORMAL),            # To keep track of the micronutrients state
        )
        self.add_states(
            ss.State('ti_macro', int, ss.INT_NAN),                          # Time index of change in macronutrition
            ss.State('new_macro_state', int, ss.INT_NAN),                   # New macro nutrition state

            ss.State('ti_micro', int, ss.INT_NAN),                          # Time at which undernourished was diagnosed
            ss.State('new_micro_state', int, ss.INT_NAN),                   # New micro nutrition state
        )
        
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
        return
    
    def update_pre(self, sim):
        new_macro = ss.true(self.ti_macro == sim.ti)
        if len(new_macro) > 0:
            self.macro_state[new_macro] = self.new_macro_state[new_macro]
        #super().set_prognoses(sim, new_macro) # Logging

        new_micro = ss.true(self.ti_micro == sim.ti)
        if len(new_micro) > 0:
            self.micro_state[new_micro] = self.new_micro_state[new_micro]
        #super().set_prognoses(sim, new_micro) # Logging
       
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
            ss.Result(self.name, 'people_alive', sim.npts, dtype=float),
        ]
        return

    def update_results(self, sim):
        super().update_results(sim)
        ti = sim.ti                 # Current time index (step)
        alive = sim.people.alive    # People alive at current time index
        
        self.results.prev_macro_standard_or_above[ti] = np.count_nonzero((self.macro_state==MacroNutrients.STANDARD_OR_ABOVE) & alive)/alive.count()
        self.results.prev_macro_slightly_below[ti] = np.count_nonzero((self.macro_state==MacroNutrients.SLIGHTLY_BELOW_STANDARD) & alive)/alive.count()
        self.results.prev_macro_marginal[ti] = np.count_nonzero((self.macro_state==MacroNutrients.MARGINAL) & alive)/alive.count()
        self.results.prev_macro_unsatisfactory[ti] = np.count_nonzero((self.macro_state==MacroNutrients.UNSATISFACTORY) & alive)/alive.count()

        self.results.prev_micro_normal[ti] = np.count_nonzero((self.micro_state==MicroNutrients.NORMAL) & alive)/alive.count()
        self.results.prev_micro_deficient[ti] = np.count_nonzero((self.micro_state==MicroNutrients.DEFICIENT) & alive)/alive.count()
        self.results.people_alive[ti] = alive.count()/sim.pars['n_agents']
        return
