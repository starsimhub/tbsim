"""
Define non-communicable disease (Malnutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc
from enum import IntEnum, auto
import tbsim.nutritionenums as ne

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
    
    def __init__(self, pars=None, **kwargs):
        # According to https://www.who.int/news-room/questions-and-answers/item/malnutrition

        super().__init__(**kwargs)
        self.default_pars(
            beta = 1.0,         # Transmission rate  - TODO: Check if there is one
            init_prev = 0.001,  # Initial prevalence 
        )
        self.update_pars(pars, **kwargs)

        self.add_states(
            ss.FloatArr('macro_state', default= MacroNutrients.STANDARD_OR_ABOVE), # To keep track of the macronutrients state
            ss.FloatArr('micro_state', default=MicroNutrients.NORMAL),            # To keep track of the micronutrients state
            ss.FloatArr('bmi_state', default=0.0),                                # To keep track of the BMI state
        )
        self.add_states(
            ss.FloatArr('ti_macro'),                          # Time index of change in macronutrition
            ss.FloatArr('new_macro_state'),                   # New macro nutrition state

            ss.FloatArr('ti_micro'),                          # Time index of change in micronutrition
            ss.FloatArr('new_micro_state'),                   # New micro nutrition state
            
            ss.FloatArr('ti_bmi'),                          # Time index of change in micronutrition
            ss.FloatArr('new_bmi_state'),                   # New micro nutrition state

        )
        
        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
        return
    
    def update_pre(self):
        ti = self.sim.ti
        new_macro = (self.ti_macro == ti).uids
        if len(new_macro) > 0:
            self.macro_state[new_macro] = self.new_macro_state[new_macro]

        new_micro = (self.ti_micro == ti).uids
        if len(new_micro) > 0:
            self.micro_state[new_micro] = self.new_micro_state[new_micro]

        new_bmi = (self.ti_bmi == ti).uids
        if len(new_bmi) > 0:
            self.bmi_state[new_bmi] = self.new_bmi_state[new_bmi]
       
       
        return new_macro, new_micro, new_bmi

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        npts = self.sim.npts
        self.results += [
            ss.Result(self.name, 'prev_macro_standard_or_above', npts, dtype=float),
            ss.Result(self.name, 'prev_macro_slightly_below', npts, dtype=float),
            ss.Result(self.name, 'prev_macro_marginal', npts, dtype=float),
            ss.Result(self.name, 'prev_macro_unsatisfactory', npts, dtype=float),

            ss.Result(self.name, 'prev_micro_normal', npts, dtype=float),
            ss.Result(self.name, 'prev_micro_deficient', npts, dtype=float),
            ss.Result(self.name, 'people_alive', npts, dtype=float),
            
            ss.Result(self.name, 'prev_severe_thinness', npts, dtype=float),
            ss.Result(self.name, 'prev_moderate_thinness', npts, dtype=float),
            ss.Result(self.name, 'prev_mild_thinness', npts, dtype=float),
            ss.Result(self.name, 'prev_normal_weight', npts, dtype=float),
            # ss.Result(self.name, 'prev_overweight', npts, dtype=float),
        ]
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti            # Current time index (step)
        alive = self.sim.people.alive    # People alive at current time index
        n_agents = self.sim.pars['n_agents']
        n_alive = alive.count()
        
        self.results.prev_macro_standard_or_above[ti] = np.count_nonzero((self.macro_state==MacroNutrients.STANDARD_OR_ABOVE) & alive)/n_alive
        self.results.prev_macro_slightly_below[ti] = np.count_nonzero((self.macro_state==MacroNutrients.SLIGHTLY_BELOW_STANDARD) & alive)/n_alive
        self.results.prev_macro_marginal[ti] = np.count_nonzero((self.macro_state==MacroNutrients.MARGINAL) & alive)/n_alive
        self.results.prev_macro_unsatisfactory[ti] = np.count_nonzero((self.macro_state==MacroNutrients.UNSATISFACTORY) & alive)/n_alive

        self.results.prev_micro_normal[ti] = np.count_nonzero((self.micro_state==MicroNutrients.NORMAL) & alive)/n_alive
        self.results.prev_micro_deficient[ti] = np.count_nonzero((self.micro_state==MicroNutrients.DEFICIENT) & alive)/n_alive
        self.results.people_alive[ti] = alive.count()/n_agents
        
        
        self.results.prev_severe_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.SEVERE_THINNESS) & alive) / n_alive
        self.results.prev_moderate_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.MODERATE_THINNESS) & alive) / n_alive
        self.results.prev_mild_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.MILD_THINNESS) & alive) / n_alive
        self.results.prev_normal_weight[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.NORMAL_WEIGHT) & alive) / n_alive
        # self.results.prev_overweight[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.OVERWEIGHT) & alive) / n_alive

        
        return
