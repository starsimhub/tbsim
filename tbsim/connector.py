"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
from tbsim import TB, Nutrition, TBS
import sciris as sc

__all__ = ['TB_Nutrition_Connector']

class TB_Nutrition_Connector(ss.Connector):
    """ Connect TB to Nutrition """

    def __init__(self, pars=None):
        super().__init__(pars=pars, label='TB-Nutrition', diseases=[TB, Nutrition])
        self.pars = ss.omerge({
            'rel_LS_prog_risk': 2.0, # 2x the rate for those experiencing undernutrition
        }, self.pars)
        return

    def update(self, sim):
        """ Specify how nutrition increases the latent slow progression risk """
        # Newly undernourished
        new_cases = ss.true(sim.diseases['nutrition'].ti_undernourished == sim.ti)
        if len(new_cases) > 0:
            #sim.people.tb.rel_LS_prog[~sim.people.nutrition.undernourished] = 1.0
            sim.people.tb.rel_LS_prog[new_cases] = self.pars.rel_LS_prog_risk

            # Because undernourished while in latent slow
            slow_uids = ss.true(sim.people.tb.state[new_cases] == TBS.LATENT_SLOW)
            if len(slow_uids) > 0:
                sim.diseases['tb'].set_ti(sim, slow_uids) # Recalculate with modified LS rate
        return
