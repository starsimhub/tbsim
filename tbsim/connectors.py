"""
Define a connector between TB and Malnutrition
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Malnutrition """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-Malnutrition')

        self.define_pars(
            rr_activation_func = self.ones_rr, #self.supplementation_rr, self.lonnroth_bmi_rr,
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
    def lonnroth_bmi_rr(tb, mn, uids, scale=2, slope=3, bmi50=25):
        bmi = 10_000 * mn.weight(uids) / mn.height(uids)**2
        #tb_incidence_per_100k_year = 10**(-0.05*(bmi-15) + 2) # incidence rate of 100 at BMI of 15
        # How to go from incidence rate to relative risk?
        # --> How about a sigmoid?
        x = -0.05*(bmi-15) + 2 # Log linear relationship from lonnroth et al.
        x0 = -0.05*(bmi50-15) + 2 # Center on 25
        rr = scale / (1+10**(-slope * (x-x0) ))

        '''
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(bmi, rr)
        '''

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
