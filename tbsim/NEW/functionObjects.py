import numpy as np
import tbsim as mtb
from tbsim.NEW.tbsimenums import MacroNutrients

def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == MacroNutrients.UNSATISFACTORY)] = 0

    return p

def compute_rel_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    #ret[micro == mtb.MicroNutrients.DEFICIENT] = 5
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE)         & (micro == mtb.MicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD)   & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == mtb.MacroNutrients.MARGINAL)                  & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY)            & (micro == mtb.MicroNutrients.DEFICIENT)] = 3.0
    return ret