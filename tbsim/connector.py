"""
Define a connector between TB and Malnutrition
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition, TBS, MacroNutrients, MicroNutrients

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Malnutrition """

    def __init__(self, pars=None):
        super().__init__(pars=pars, label='TB-Malnutrition', diseases=[TB, Malnutrition])
        self.requires = [TB, Malnutrition]
        self.pars = ss.dictmergeleft({
            'rel_LS_prog_func': self.compute_rel_LS_prog,
            'relsus_microdeficient': 5,
        }, self.pars)
        return

    def initialize(self, sim):
        tb = sim.diseases['tb']
        nut = sim.diseases['malnutrition']
        uids = sim.people.alive.uids
        tb.rel_LS_prog[uids] = self.pars.rel_LS_prog_func(nut.macro_state[uids], nut.micro_state[uids])
        return

    @staticmethod
    def compute_rel_LS_prog(macro, micro):
        assert len(macro) == len(micro), 'Length of macro and micro must match.'
        ret = np.ones_like(macro)
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.NORMAL)] = 1.25
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.DEFICIENT)] = 2.5
        ret[macro == MacroNutrients.UNSATISFACTORY] = 4.0
        return ret

    def update(self, sim):
        """ Specify how malnutrition and TB interact """
        nut = sim.diseases['malnutrition']
        tb = sim.diseases['tb']

        # Let's set rel_sus!
        tb.rel_sus[nut.micro_state == MicroNutrients.DEFICIENT] = self.pars.relsus_microdeficient
        tb.rel_sus[nut.micro_state == MicroNutrients.NORMAL] = 1

        change_macro_uids = (nut.ti_macro == sim.ti).uids
        change_micro_uids = (nut.ti_micro == sim.ti).uids
        if len(change_macro_uids) > 0 or len(change_micro_uids) > 0:
            change_uids = np.unique(np.concatenate([change_macro_uids, change_micro_uids]))
            k_old = tb.rel_LS_prog[change_uids] #self.pars.rel_LS_prog_risk(nut.macro_state[change_uids], nut.micro_state[change_uids])

            mac = nut.macro_state[change_uids]
            mac[change_macro_uids] = nut.new_macro_state[change_macro_uids]

            mic = nut.micro_state[change_uids]
            mic[change_micro_uids] = nut.new_micro_state[change_micro_uids]

            k_new = self.pars.rel_LS_prog_func(mac, mic)
            diff = k_old != k_new

            if not diff.any():
                return

            tb.rel_LS_prog[change_uids] = k_new # Update rel_LS_prog

            # Check for rate change while in latent slow
            slow_change = diff & (sim.people.tb.state[change_uids] == TBS.LATENT_SLOW)
            slow_change_uids = ss.true(slow_change)
            if len(slow_change_uids) > 0:
                # New time of switching from LS to presymp:
                R = tb.ppf_LS_to_presymp[slow_change_uids]
                r = tb.pars.rate_LS_to_presym * 365 # Converting days to years
                t_latent = tb.ti_latent[slow_change_uids]*sim.dt
                t_now = sim.ti*sim.dt # Time of switching from health to undernourished

                tb.ti_presymp[slow_change_uids] = -1/(k_new[slow_change]*r) * np.log( np.exp(-k_old[slow_change]*r*t_latent) - np.exp(-k_old[slow_change]*r*t_now) + np.exp(-k_new[slow_change]*r*t_now) - R) / sim.dt

        return
