"""
Define a connector between TB and Malnutrition
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition, TBS, MacroNutrients, MicroNutrients

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Malnutrition """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-Malnutrition', requires=[TB, Malnutrition])

        self.default_pars(
            rel_LS_prog_func = self.compute_rel_LS_prog,
            relsus_microdeficient = 5,
        )
        self.update_pars(pars, **kwargs)
        return

    def init_vals(self):
        super().init_vals()
        tb = self.sim.diseases['tb']
        nut = self.sim.diseases['malnutrition']
        tb.rel_LS_prog[self.sim.people.uid] = self.pars.rel_LS_prog_func(nut.macro_state, nut.micro_state)
        return

    @staticmethod
    def compute_rel_LS_prog(macro, micro):
        assert len(macro) == len(micro), 'Length of macro and micro must match.'
        ret = np.ones_like(macro)
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.NORMAL)] = 1.25
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.DEFICIENT)] = 2.5
        ret[macro == MacroNutrients.UNSATISFACTORY] = 4.0
        return ret

    def update(self):
        """ Specify how malnutrition and TB interact """
        nut = self.sim.diseases['malnutrition']
        tb = self.sim.diseases['tb']
        ti = self.sim.ti
        dt = self.sim.dt

        # Let's set rel_sus!
        tb.rel_sus[nut.micro_state == MicroNutrients.DEFICIENT] = self.pars.relsus_microdeficient
        tb.rel_sus[nut.micro_state == MicroNutrients.NORMAL] = 1

        change_macro_uids = (nut.ti_macro == ti).uids
        change_micro_uids = (nut.ti_micro == ti).uids
        if len(change_macro_uids) > 0 or len(change_micro_uids) > 0:
            change_uids = ss.uids(np.unique(np.concatenate([change_macro_uids, change_micro_uids])))
            k_old = tb.rel_LS_prog[change_uids]

            nut.macro_state[change_macro_uids] = nut.new_macro_state[change_macro_uids]
            nut.micro_state[change_micro_uids] = nut.new_micro_state[change_micro_uids]

            k_new = self.pars.rel_LS_prog_func(nut.macro_state[change_uids], nut.micro_state[change_uids])
            diff = k_old != k_new

            if not diff.any():
                return

            tb.rel_LS_prog[change_uids] = k_new # Update rel_LS_prog

            # Check for rate change while in latent slow
            slow_change = diff & (self.sim.people.tb.state[change_uids] == TBS.LATENT_SLOW)
            slow_change_uids = change_uids[slow_change]
            if len(slow_change_uids) > 0:
                # New time of switching from LS to presymp:
                R = tb.ppf_LS_to_presymp[slow_change_uids]
                r = tb.pars.rate_LS_to_presym * 365 # Converting days to years
                t_latent = tb.ti_latent[slow_change_uids]*dt
                t_now = ti*dt # Time of switching from health to undernourished

                C = np.exp(-k_old[slow_change]*r*t_latent) - np.exp(-k_old[slow_change]*r*t_now)

                time_from_C_to_R = -np.log(1-R)/ (k_new[slow_change]*r) - -np.log(1-C)/ (k_new[slow_change]*r)
                tb.ti_presymp[slow_change_uids] = np.ceil(ti + time_from_C_to_R/dt)

        return
