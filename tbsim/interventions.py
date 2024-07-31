"""
Define Malnutrition intervention
"""

import numpy as np
import starsim as ss
from tbsim import Malnutrition, eMicroNutrients, eMacroNutrients
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

class BmiNormalizationIntervention(ss.Intervention):
    """
    

    Args:
        ss (_type_): _description_
    """
    
    def __init__(self, year_arr, rate_arr, from_state, to_state, new_macro_state=None, p_new_macro=0, new_micro_state=None, p_new_micro=0, p_micro_recovery_func=None, ration=1, arm=None, **kwargs):
        
        
        self.requires = Malnutrition
        self.year = sc.promotetoarray(year_arr)
        self.rate = sc.promotetoarray(rate_arr)
        
        self.from_state = from_state
        self.to_state = to_state
        
        self.new_macro_state = new_macro_state
        self.p_new_macro = p_new_macro
        
        self.new_micro_state = new_micro_state
        self.p_new_micro = p_new_micro
        
        self.p_micro_recovery_func = p_micro_recovery_default if p_micro_recovery_func is None else p_micro_recovery_func
        self.ration = ration  # Percentage of food supply based on default ration value (for instance, for index agents it is 1, but for household contacts it is 0.9 or so)
        
        self.arm = arm
        self.name = f'Nutrition change from {self.from_state} to {self.to_state}'
        
        super().__init__(**kwargs)
        # Probability of changing macro nutrients
        # TODO: CHECK WITH DAN
        
        self.p_macro = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))    # Probability of changing macro nutrients
        self.p_micro_recovery = ss.bernoulli(p=self.p_micro_recovery_func)                            # Probability of recovery from micro nutrients deficiency
        self.p_micro = ss.bernoulli(p=self.p_new_micro)                               # Probability of changing micro nutrients
        return

    def init_pre(self, sim):
        super().init_pre(sim)
        self.results += ss.Result(self.name, 'n_recovered_macro', self.sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_recovered_micro', self.sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_recovered_bmi', self.sim.npts, dtype=int)
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['malnutrition']
        ppl = sim.people

        # Macro intervention
        eligible_macro = (nut.macro_state == self.from_state) & ppl.alive
        if self.arm is not None:
            eligible_macro &= ppl.arm == self.arm
        eligible_macro_uids = eligible_macro.uids

        change_macro_uids = self.p_macro.filter(eligible_macro_uids)

        nut.ti_macro[change_macro_uids] = sim.ti + 1  # Next time step
        nut.new_macro_state[change_macro_uids] = self.to_state

        if (self.p_new_macro > 0) and (self.new_macro_state is not None):
            change_new_macro_uids = self.p_micro.filter(change_macro_uids)
            nut.ti_macro[change_new_macro_uids] = sim.ti + 1  # Next time step
            nut.new_macro_state[change_new_macro_uids] = self.new_macro_state

        # Micro intervention
        micro_deficient_uids = (
            (sim.people.arm != StudyArm.CONTROL) & 
            (nut.micro_state == eMicroNutrients.DEFICIENT)
        ).uids
        recover_micro_uids = self.p_micro_recovery.filter(micro_deficient_uids)

        nut.ti_micro[recover_micro_uids] = sim.ti + 1  # Next time step
        nut.new_micro_state[recover_micro_uids] = eMicroNutrients.NORMAL



        # Update results
        # self.results['n_recovered_macro'][sim.ti] = len(change_macro_uids)
        # self.results['n_recovered_micro'][sim.ti] = len(recover_micro_uids)
     #   self.results['n_recovered_bmi'][sim.ti] = len(recover_micro_uids)
        
        return len(change_macro_uids) + len(recover_micro_uids)

