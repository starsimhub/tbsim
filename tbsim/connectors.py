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
            rr_activation_func = self.compute_rr_activation,
            rr_clearance_func = self.compute_rr_clearance,
            relsus_func = self.compute_relsus,
        )
        self.update_pars(pars, **kwargs)

        return

    def init_vals(self):
        super().init_vals()
        self.update() # needed?
        return

    @staticmethod
    def compute_rr_activation(mn, uids):
        rr = np.ones_like(uids)
        bmi = mn.weight[uids] / mn.height[uids]**2
        rr[bmi < 18] = 2
        return rr

    @staticmethod
    def compute_rr_clearance(mn, uids):
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
        tb.rr_activation[uids] = self.pars.rr_activation_func(mn, uids)
        tb.rr_clearance[uids] = self.pars.rr_clearance_func(mn, uids)

        uids = (~tb.infected).uids
        tb.rel_sus[uids] = self.pars.relsus_func(tb, mn, uids)

        return
