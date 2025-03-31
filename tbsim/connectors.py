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
    
    This connector uses the HIV state (ATRISK, HIV, LATENT, AIDS, DEAD) and the on_ART flag
    from the HIV disease model to modify TB parameters.
    
    For TB‐infected individuals, the TB activation relative risk (rr_activation) is multiplied by:
      - ATRISK: 1.0 (baseline)
      - ACUTE:    1.5 (or 1.5 * art_tb_multiplier if on ART)
      - LATENT: 2.0 (or 2.0 * art_tb_multiplier if on ART)
      - AIDS:   3.0 (or 3.0 * art_tb_multiplier if on ART)
      - DEAD:   0.0
      
    Similarly, for TB‐noninfected individuals, their TB relative susceptibility is set accordingly.
    
    The art_tb_multiplier parameter (default 0.8) reduces the TB risk for individuals on ART.
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_activation_rr_func = self.compute_tb_activation_rr,
            tb_clearance_rr_func  = self.ones_rr,  # No clearance modification.
            tb_rel_sus_func       = self.compute_tb_rel_sus,
            art_tb_multiplier     = 0.8,
        )
        self.update_pars(pars, **kwargs)
    
    @staticmethod
    def ones_rr(tb, hiv, uids):
        return np.ones_like(uids, dtype=float)
    
    @staticmethod
    def compute_tb_activation_rr(tb, hiv, uids, base_factor=1.0):
        rr = np.ones_like(uids, dtype=float)
        art_multiplier = tb.pars.get('art_tb_multiplier', 0.8)
        states = hiv.state[uids]
        for i, s in enumerate(states):
            if s == HIVState.ACUTE:
                multiplier = 1.5
            elif s == HIVState.LATENT:
                multiplier = 2.0
            elif s == HIVState.AIDS:
                multiplier = 3.0
            elif s == HIVState.DEAD:
                multiplier = 0.0
            else:  # ATRISK
                multiplier = 1.0
            # If the individual is on ART, reduce the multiplier.
            if hiv.on_ART[uids[i]]:
                multiplier *= art_multiplier
            rr[i] = multiplier
        return rr * base_factor
    
    @staticmethod
    def compute_tb_rel_sus(tb, hiv, uids, baseline=1.0):
        rel_sus = np.ones_like(uids, dtype=float) * baseline
        art_multiplier = tb.pars.get('art_tb_multiplier', 0.8)
        states = hiv.state[uids]
        for i, s in enumerate(states):
            if s == HIVState.ACUTE:
                multiplier = 1.5
            elif s == HIVState.LATENT:
                multiplier = 2.0
            elif s == HIVState.AIDS:
                multiplier = 3.0
            elif s == HIVState.DEAD:
                multiplier = 0.0
            else:
                multiplier = 1.0
            if hiv.on_ART[uids[i]]:
                multiplier *= art_multiplier
            rel_sus[i] = baseline * multiplier
        return rel_sus
    
    def step(self):
        tb = self.sim.diseases['tb']
        hiv = self.sim.diseases['hiv']
        
        # Adjust parameters for TB-infected individuals.
        uids_tb = tb.infected.uids
        tb.rr_activation[uids_tb] *= self.pars.tb_activation_rr_func(tb, hiv, uids_tb)
        tb.rr_clearance[uids_tb] *= self.pars.tb_clearance_rr_func(tb, hiv, uids_tb)
        
        # Adjust susceptibility for TB-noninfected individuals.
        uids_no_tb = (~tb.infected).uids
        tb.rel_sus[uids_no_tb] = self.pars.tb_rel_sus_func(tb, hiv, uids_no_tb)
        return
