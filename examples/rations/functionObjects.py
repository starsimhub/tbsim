import numpy as np
from tbsim.nutritionenums import eMacroNutrients, eMicroNutrients

def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == eMacroNutrients.UNSATISFACTORY)] = 0

    return p

