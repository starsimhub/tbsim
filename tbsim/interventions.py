"""
Define Malnutrition intervention
"""

import numpy as np
import starsim as ss
from tbsim import Malnutrition, eMicroNutrients, eMacroNutrients, eBmiStatus
import sciris as sc
from enum import IntEnum, auto

__all__ = ['MicroNutrientsSupply', 'MacroNutrientsSupply', 'BmiNormalizationIntervention']

class StudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()
    
def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == eMacroNutrients.UNSATISFACTORY)] = 0

    return p

class MicroNutrientsSupply(ss.Intervention):
    def __init__(self, year: np.array, rate: np.array, p_micro_recovery_func=None, ration=1, **kwargs):
        self.requires = Malnutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.p_micro_recovery_func = p_micro_recovery_default if p_micro_recovery_func is None else p_micro_recovery_func
        self.ration = ration   # Ration of Vitatims supply

        super().__init__(**kwargs)

        self.p_micro_recovery = ss.bernoulli(p=self.p_micro_recovery_func)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_recovered', self.sim.npts, dtype=int)
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['malnutrition']
        micro_deficient_uids = (
            (sim.people.arm != StudyArm.CONTROL) & 
            (nut.micro_state == eMicroNutrients.DEFICIENT)
        ).uids
        recover_uids = self.p_micro_recovery.filter(micro_deficient_uids)

        nut.ti_micro[recover_uids] = sim.ti + 1 # Next time step
        nut.new_micro_state[recover_uids] = eMicroNutrients.NORMAL

        self.results['n_recovered'][sim.ti] = len(recover_uids)

        return len(recover_uids)


class MacroNutrientsSupply(ss.Intervention):

    def __init__(self, year, rate, from_state, to_state, new_micro_state=None, p_new_micro=0, arm=None, ration=1, **kwargs):
        self.requires = Malnutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.from_state = from_state
        self.to_state = to_state
        self.new_micro_state = new_micro_state
        self.p_new_micro = p_new_micro
        self.arm = None
        self.name = f'Nutrition change from {self.from_state} to {self.to_state}'
        self.ration = ration    # Ration of food supply
        super().__init__(**kwargs)

        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        self.p_micro = ss.bernoulli(p=self.p_new_micro) # Prob of changing micro when changing macro
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        #self.results += ss.Result(self.name, 'n_recovered', self.sim.npts, dtype=int)
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['malnutrition']
        ppl = sim.people
        eligible = (nut.macro_state == self.from_state) & ppl.alive
        if self.arm is not None:
            eligible &= ppl.arm == self.arm
        eligible_uids = eligible.uids

        change_uids = self.p.filter(eligible_uids)

        nut.ti_macro[change_uids] = sim.ti + 1 # Next time step
        nut.new_macro_state[change_uids] = self.to_state

        if (self.p_new_micro > 0) and (self.new_micro_state is not None):
            change_micro_uids = self.p_micro.filter(change_uids)
            nut.ti_micro[change_micro_uids] = sim.ti + 1 # Next time step
            nut.new_micro_state[change_micro_uids] = self.new_micro_state

        return len(change_uids)




def p_MicroRecoveryBmiBased_func(self, sim, uids):
    # it performs linear interpolation to get the probability It interpolates the value
    # of self.sim.year based on self.year and self.rate multiplied by self.sim.dt
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)

    # p is an array of length equal to the number of uids, filled with the value of prob
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == eMacroNutrients.UNSATISFACTORY)] = 0

    return p

def bmiToMacroConversion(bmi):
    if bmi == eBmiStatus.SEVERE_THINNESS:
        return eMacroNutrients.UNSATISFACTORY
    elif bmi == eBmiStatus.MODERATE_THINNESS:
        return eMacroNutrients.MARGINAL
    elif bmi == eBmiStatus.MILD_THINNESS:
        return eMacroNutrients.SLIGHTLY_BELOW_STANDARD
    elif bmi == eBmiStatus.NORMAL_WEIGHT:
        return eMacroNutrients.STANDARD_OR_ABOVE
    else:
        return eMacroNutrients.STANDARD_OR_ABOVE


class BmiNormalizationIntervention(ss.Intervention):
    """
    An intervention class for normalizing BMI (Body Mass Index) values.

    Parameters:
    - year_arr (array-like):        Array of years when the intervention is delivered.
    - rate_arr (array-like):        Array of rates at which the intervention is delivered.
    - from_bmi_state (float or array-like): Starting BMI state(s) to be normalized.
    - to_bmi_state (float or array-like): Target BMI state(s) after normalization.
    - p_MicroRecoveryBmiBased_func (function, optional): Function to calculate the probability of micro recovery based on BMI. Default is None.
    - new_micro_state (float or array-like, optional): New micro state(s) after normalization. Default is None.
    - p_new_micro (float, optional): Probability of changing micro state when changing macro state. Default is 0.
    - ration (float, optional):     Percentage of food supply based on default ration value. Default is 1.
    - arm (int, optional):          Arm identifier. Default is None.
    - **kwargs:                     Additional keyword arguments.

    Attributes:
    - requires: The required intervention for this class (Malnutrition).
    - year: Array of years when the intervention is delivered.
    - rate: Array of rates at which the intervention is delivered.
    - from_macrostate: Starting macro state(s) converted from BMI.
    - to_macrostate: Target macro state(s) converted from BMI.
    - new_micro_state: New micro state(s) after normalization.
    - p_new_micro: Probability of changing micro state when changing macro state.
    - arm: Arm identifier.
    - name: Name of the intervention.
    - ration: Percentage of food supply based on default ration value.
    - p: Bernoulli distribution for the probability of intervention delivery.
    - p_micro: Bernoulli distribution for the probability of changing micro state.

    Methods:
    - init_pre(sim): Pre-initialization method.
    - apply(sim): Apply the intervention.

    """

    def __init__(self, year_arr, rate_arr, from_bmi_state, to_bmi_state, p_MicroRecoveryBmiBased_func=None, new_micro_state=None, p_new_micro=0, ration=1, arm=None, **kwargs):
        self.requires = Malnutrition
        self.year = sc.promotetoarray(year_arr)
        self.rate = sc.promotetoarray(rate_arr)
        self.to_bmi_state = to_bmi_state
        self.from_macrostate = bmiToMacroConversion(bmi=from_bmi_state)
        self.to_macrostate = bmiToMacroConversion(bmi=to_bmi_state)
        self.new_micro_state = new_micro_state
        self.p_new_micro = p_new_micro
        self.arm = arm
        self.name = f'Nutrition change from {self.from_macrostate} to {self.to_macrostate}'
        self.ration = ration
        super().__init__(**kwargs)
        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate * sim.dt))
        self.p_micro = ss.bernoulli(p=self.p_new_micro)
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['malnutrition']
        ppl = sim.people

        eligible_macro = (nut.macro_state == self.from_macrostate) & ppl.alive
        if self.arm is not None:
            eligible_macro &= ppl.arm == self.arm
        eligible_uids = eligible_macro.uids

        change_uids = self.p.filter(eligible_uids)

        nut.ti_macro[change_uids] = sim.ti + 1
        nut.new_macro_state[change_uids] = self.to_macrostate

        nut.ti_bmi[change_uids] = sim.ti + 1
        nut.new_bmi_state[change_uids] = self.to_bmi_state

        if (self.p_new_micro > 0) and (self.new_micro_state is not None):
            change_micro_uids = self.p_micro.filter(change_uids)
            nut.ti_micro[change_micro_uids] = sim.ti + 1
            nut.new_micro_state[change_micro_uids] = self.new_micro_state

        return len(change_uids)

