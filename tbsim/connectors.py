"""
Define a connector between TB and Malnutrition
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition, TBS#, MacroNutrients, MicroNutrients

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Malnutrition """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-Malnutrition', requires=[TB, Malnutrition])

        self.default_pars(
            rr_activation_func = self.ones_rr, #self.supplementation_rr, self.longroth_bmi_rr,
            rr_clearance_func = self.ones_rr,
            relsus_func = self.compute_relsus,
        )
        self.update_pars(pars, **kwargs)

        return

    def init_vals(self):
        super().init_vals()
        self.update() # needed?
        return

    @staticmethod
    def supplementation_rr(tb, mn, uids, rate_ratio=0.5):
        rr = np.ones_like(uids)
        rr[mn.receiving_macro[uids] & mn.receiving_micro[uids]] = rate_ratio
        return rr


    @staticmethod
    def longroth_bmi_rr(tb, mn, uids):
        rr = np.ones_like(uids)
        bmi = mn.weight[uids] / mn.height[uids]**2
        rr[bmi < 18] = 2
        return rr

    @staticmethod
    def ones_rr(tb, mn, uids):
        rr = np.ones_like(uids)
        return rr

    @staticmethod
    def compute_relsus(tb, mn, uids):
        rel_sus = np.ones_like(uids)
        rel_sus[mn.micro[uids]<0.2] = 2 # Double the susceptibility if micro is low???
        return rel_sus

    def update(self):
        """ Specify how malnutrition and TB interact """
        tb = self.sim.diseases['tb']
        mn = self.sim.diseases['malnutrition']

        uids = tb.infected.uids
        # Relative rates start at 1 each time step
        tb.rr_activation[uids] *= self.pars.rr_activation_func(tb, mn, uids)
        tb.rr_clearance[uids] *= self.pars.rr_clearance_func(tb, mn, uids)

        uids = (~tb.infected).uids
        tb.rel_sus[uids] = self.pars.relsus_func(tb, mn, uids)

        return
