"""
Define Malnutrition intervention
"""

import numpy as np
import starsim as ss
import tbsim as mtb
import sciris as sc

__all__ = ['VitaminSupplementation', 'NutritionChange']


def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == mtb.MacroNutrients.UNSATISFACTORY)] = 0

    return p

class VitaminSupplementation(ss.Intervention):
    def __init__(self, year: np.array, rate: np.array, p_micro_recovery_func=None, **kwargs):
        self.requires = mtb.Malnutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.p_micro_recovery_func = p_micro_recovery_default if p_micro_recovery_func is None else p_micro_recovery_func

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
            (sim.people.arm != mtb.StudyArm.CONTROL) & 
            (nut.micro_state == mtb.MicroNutrients.DEFICIENT)
        ).uids
        recover_uids = self.p_micro_recovery.filter(micro_deficient_uids)

        nut.ti_micro[recover_uids] = sim.ti + 1 # Next time step
        nut.new_micro_state[recover_uids] = mtb.MicroNutrients.NORMAL

        self.results['n_recovered'][sim.ti] = len(recover_uids)

        return len(recover_uids)


class NutritionChange(ss.Intervention):

    def __init__(self, year, rate, from_state, to_state, new_micro_state=None, p_new_micro=0, arm=None, **kwargs):
        self.requires = mtb.Malnutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(rate)
        self.from_state = from_state
        self.to_state = to_state
        self.new_micro_state = new_micro_state
        self.p_new_micro = p_new_micro
        self.arm = None
        self.name = f'Nutrition change from {self.from_state} to {self.to_state}'

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

