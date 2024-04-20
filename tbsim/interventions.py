"""
Define Nutrition intervention
"""

# WORK IN PROGRESS, CODE NOT FUNCTIONAL YET

import numpy as np
import starsim as ss
from tbsim import TB, Nutrition, MicroNutrients, MacroNutrients, StudyArm
import sciris as sc

__all__ = ['VitaminSupplementation', 'LargeScaleFoodFortification']

class VitaminSupplementation(ss.Intervention):

    def __init__(self, year: np.array, rate: np.array, **kwargs):
        self.requires = Nutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)

        super().__init__(**kwargs)

        self.p_micro_recovery = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_recovered', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['nutrition']
        micro_deficient_uids = ss.true(
            (sim.people.arm!=StudyArm.CONTROL) & 
            (nut.micro == MicroNutrients.DEFICIENT) & 
            (nut.macro != MacroNutrients.UNSATISFACTORY)
        )
        recover_uids = self.p_micro_recovery.filter(micro_deficient_uids)

        nut.ti_micro[recover_uids] = sim.ti + 1 # Next time step
        nut.new_micro_state[recover_uids] = MicroNutrients.NORMAL

        self.results['n_recovered'][sim.ti] = len(recover_uids)

        return len(recover_uids)


class LargeScaleFoodFortification(ss.Intervention):

    def __init__(self, year, rate, from_state, to_state, arm=None, **kwargs):
        self.requires = Nutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.from_state = from_state
        self.to_state = to_state
        self.arm = None
        self.name = f'LSFF from {self.from_state} to {self.to_state}'

        super().__init__(**kwargs)

        self.p = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.rate*sim.dt))
        return

    def initialize(self, sim):
        super().initialize(sim)
        #self.results += ss.Result(self.name, 'n_recovered', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        nut = sim.diseases['nutrition']
        ppl = sim.people
        eligible = (nut.macro == self.from_state) & ppl.alive
        if self.arm is not None:
            eligible &= ppl.arm == self.arm
        eligible_uids = ss.true(eligible)

        recover_uids = self.p.filter(eligible_uids)

        nut.ti_macro[recover_uids] = sim.ti + 1 # Next time step
        nut.new_macro_state[recover_uids] = self.to_state

        return len(recover_uids)
