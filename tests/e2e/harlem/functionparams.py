import numpy as np
import tbsim as mtb

def compute_rel_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE) & (micro == mtb.MicroNutrients.DEFICIENT)] = 1.5
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD) & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.0
    ret[(macro == mtb.MacroNutrients.MARGINAL) & (micro == mtb.MicroNutrients.DEFICIENT)] = 2.5
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY) & (micro == mtb.MicroNutrients.DEFICIENT)] = 3.0
    return ret

def compute_rel_prog_alternate(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == mtb.MacroNutrients.STANDARD_OR_ABOVE) & (micro == mtb.MicroNutrients.DEFICIENT)] = 2
    ret[(macro == mtb.MacroNutrients.SLIGHTLY_BELOW_STANDARD) & (micro == mtb.MicroNutrients.DEFICIENT)] = 5
    ret[(macro == mtb.MacroNutrients.MARGINAL) & (micro == mtb.MicroNutrients.DEFICIENT)] = 10
    ret[(macro == mtb.MacroNutrients.UNSATISFACTORY) & (micro == mtb.MicroNutrients.DEFICIENT)] = 20
    return ret

def run_scen(scen, filter):
    if filter is None:
        return True
    return scen in filter


def p_micro_recovery_default(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']
    p[(nut.macro_state[uids] == mtb.MacroNutrients.UNSATISFACTORY)] = 0

    return p

def p_micro_recovery_alt(self, sim, uids):
    prob = np.interp(self.sim.year, self.year, self.rate*self.sim.dt)
    p = np.full(len(uids), prob)

    return p

def p_cure_func(self, sim, uids):
    rate = np.zeros(len(uids))

    # No recovery for those with unsatisfactory macro nutrients
    nut = sim.diseases['malnutrition']

    # Clearance rate (units are per-year)
    rate[(nut.micro_state[uids] == mtb.MicroNutrients.NORMAL)] = 2

    p = 1 - np.exp(-rate * sim.dt) # Linear conversion might be sufficient

    return p
