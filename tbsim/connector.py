"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
from tbsim import TB, Nutrition, TBS, MacroNutrients, MicroNutrients

__all__ = ['TB_Nutrition_Connector']

def compute_rel_LS_prog(macro, micro):
    assert len(macro) == len(micro), 'Length of macro and micro must match.'
    ret = np.ones_like(macro)
    ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.NORMAL)] = 1.5
    ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.DEFICIENT)] = 2.0
    ret[macro == MacroNutrients.UNSATISFACTORY] = 3.0
    return ret

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Nutrition """

    def __init__(self, pars=None):
        super().__init__(pars=pars, label='TB-Nutrition', diseases=[TB, Nutrition])
        self.pars = ss.omerge({
            'rel_LS_prog_func': compute_rel_LS_prog,
        }, self.pars)
        return

    def initialize(self, sim):
        tb = sim.diseases['tb']
        nut = sim.diseases['nutrition']
        uids = ss.true(sim.people.alive)
        tb.rel_LS_prog[uids] = self.pars.rel_LS_prog_func(nut.macro[uids], nut.micro[uids])
        return

    def update(self, sim):
        """ Specify how nutrition increases the latent slow progression risk """
        # Newly undernourished
        nut = sim.diseases['nutrition']
        tb = sim.diseases['tb']
        change_uids = ss.true( (nut.ti_macro == sim.ti) | (nut.ti_micro == sim.ti) )
        if len(change_uids) > 0:
            #sim.people.tb.rel_LS_prog[~sim.people.nutrition.undernourished] = 1.0

            k_old = tb.rel_LS_prog_risk[change_uids] #self.pars.rel_LS_prog_risk(nut.macro[change_uids], nut.micro[change_uids])
            k_new = self.pars.rel_LS_prog_func(nut.new_macro_state[change_uids], nut.new_micro_state[change_uids])
            diff = k_old != k_new
            diff_uids = change_uids[diff]

            # Because undernourished while in latent slow
            slow_uids = ss.true(sim.people.tb.state[diff_uids] == TBS.LATENT_SLOW)
            if len(slow_uids) > 0:
                # New time of switching from LS to presymp:
                R = tb.ppf_LS_to_presymp[slow_uids]
                #k = self.pars.rel_LS_prog_risk
                r = tb.pars.rate_LS_to_presym * 365 # Converting days to years
                t_latent = tb.ti_latent[slow_uids]*sim.dt
                t_now = sim.ti*sim.dt # Time of switching from health to undernourished

                tb.ti_presymp[slow_uids] = -1/(k_new[diff]*r) * np.log( np.exp(-k_old[diff]*r*t_latent) - np.exp(-k_old[diff]*r*t_now) + np.exp(-k_new[diff]*r*t_now) - R) / sim.dt

        return
