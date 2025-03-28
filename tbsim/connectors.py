"""
Define a connector between TB and Malnutrition
"""

import numpy as np
import starsim as ss
from tbsim import TB, Malnutrition, HIV, HIVState

__all__ = ['TB_Nutrition_Connector', 'TB_HIV_Connector']

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

    def step(self):
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


class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV.
    
    This connector uses the HIV state from the simplified HIV model to adjust TB parameters.
    
    For TB‐infected individuals, the TB activation relative risk is multiplied by:
      - ATRISK: 1.0 (baseline)
      - HIV:    1.5
      - LATENT: 2.0
      - AIDS:   3.0
      - DEAD:   0.0 (if dead from HIV)
    
    For TB‐noninfected individuals, the relative susceptibility is set accordingly.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_activation_rr_func = self.compute_tb_activation_rr,
            tb_clearance_rr_func  = self.ones_rr,
            tb_rel_sus_func       = self.compute_tb_rel_sus,
        )
        self.update_pars(pars, **kwargs)
    
    @staticmethod
    def ones_rr(tb, hiv, uids):
        return np.ones_like(uids, dtype=float)
    
    @staticmethod
    def compute_tb_activation_rr(tb, hiv, uids, base_factor=1.0):
        rr = np.ones_like(uids, dtype=float)
        states = hiv.state[uids]
        for i, s in enumerate(states):
            if s == HIVState.ACUTE:
                rr[i] = 1.5
            elif s == HIVState.LATENT:
                rr[i] = 2.0
            elif s == HIVState.AIDS:
                rr[i] = 3.0
            elif s == HIVState.DEAD:
                rr[i] = 0.0
            else:  # ATRISK
                rr[i] = 1.0
        return rr * base_factor
    
    @staticmethod
    def compute_tb_rel_sus(tb, hiv, uids, baseline=1.0):
        rel_sus = np.ones_like(uids, dtype=float) * baseline
        states = hiv.state[uids]
        for i, s in enumerate(states):
            if s == HIVState.ACUTE:
                rel_sus[i] = baseline * 1.5
            elif s == HIVState.LATENT:
                rel_sus[i] = baseline * 2.0
            elif s == HIVState.AIDS:
                rel_sus[i] = baseline * 3.0
            elif s == HIVState.DEAD:
                rel_sus[i] = 0.0
            else:
                rel_sus[i] = baseline
        return rel_sus
    
    def step(self):
        tb = self.sim.diseases['tb']
        hiv = self.sim.diseases['hiv']
        
        uids_tb = tb.infected.uids
        tb.rr_activation[uids_tb] *= self.pars.tb_activation_rr_func(tb, hiv, uids_tb)
        tb.rr_clearance[uids_tb] *= self.pars.tb_clearance_rr_func(tb, hiv, uids_tb)
        
        uids_no_tb = (~tb.infected).uids
        tb.rel_sus[uids_no_tb] = self.pars.tb_rel_sus_func(tb, hiv, uids_no_tb)
        return
