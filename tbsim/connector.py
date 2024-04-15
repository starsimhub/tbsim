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
                tb = sim.diseases['tb']
                # New time of switching from LS to presymp:
                R = tb.ppf_LS_to_presymp[slow_uids]
                k = self.pars.rel_LS_prog_risk
                r = tb.pars.rate_LS_to_presym * 365 # Converting days to years
                t_latent = tb.ti_latent[slow_uids]*sim.dt
                t_now = sim.ti*sim.dt # Time of switching from health to undernourished

                tb.ti_presymp[slow_uids] = -1/(k*r) * np.log( np.exp(-r*t_latent) - np.exp(-r*t_now) + np.exp(-k*r*t_now) - R) / sim.dt

                tb.set_ti(sim, slow_uids)
        return
