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
            rel_LF_prog_func = self.compute_rel_LF_prog,
            relsus_microdeficient = 5,
        )
        self.update_pars(pars, **kwargs)
        return

    def init_vals(self):
        super().init_vals()
        tb = self.sim.diseases['tb']
        nut = self.sim.diseases['malnutrition']
        tb.rel_LS_prog[self.sim.people.uid] = self.pars.rel_LS_prog_func(nut.macro_state, nut.micro_state)
        tb.rel_LF_prog[self.sim.people.uid] = self.pars.rel_LF_prog_func(nut.macro_state, nut.micro_state)
        return
    
    @staticmethod    
    def compute_rel_prog(macro, micro, normal_factor=0, deficient_factor=0, unsatisfactory_factor=0):
        assert len(macro) == len(micro), 'Length of macro and micro must match.'
        ret = np.ones_like(macro)
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.NORMAL)] = normal_factor
        ret[(macro == MacroNutrients.MARGINAL) & (micro == MicroNutrients.DEFICIENT)] = deficient_factor
        ret[macro == MacroNutrients.UNSATISFACTORY] = unsatisfactory_factor
        return ret
    
    def compute_rel_LS_prog(macro, micro):
        ret = TB_Nutrition_Connector.compute_rel_prog(macro, micro, normal_factor=1.25, deficient_factor=2.5, unsatisfactory_factor=4.0)
        return ret
    
    def compute_rel_LF_prog(macro, micro):
        ret =  TB_Nutrition_Connector.compute_rel_prog(macro, micro, normal_factor=2.5, deficient_factor=5.0, unsatisfactory_factor=6.0)
        return ret
    
    def update_rel_prog(self, tb, change_uids, k_old, k_new, state, rate, ti, dt):
        """Update the rel_prog values and calculate the new time of switching from latent to presymp"""
        diff = k_old != k_new
        if not diff.any():
            return

        if state==TBS.LATENT_SLOW: 
            tb.rel_LS_prog[change_uids] = k_new # Update rel_prog
        if state==TBS.LATENT_FAST: 
            tb.rel_LF_prog[change_uids] = k_new # Update rel_prog
            
        # Check for rate change while in latent state
        state_change = diff & (self.sim.people.tb.state[change_uids] == state)
        state_change_uids = change_uids[state_change]
        if len(state_change_uids) > 0:
            # New time of switching from latent to presymp:
            if state==TBS.LATENT_SLOW: 
                R = tb.ppf_LS_to_presymp[state_change_uids]
            elif state==TBS.LATENT_FAST:
                R = tb.ppf_LF_to_presymp[state_change_uids]
            else:
                raise ValueError(f"Invalid state: {state}")
            
            r = rate * 365 # Converting days to years
            t_latent = tb.ti_latent[state_change_uids]*self.sim.dt
            t_now = self.sim.ti*self.sim.dt

            C = np.exp(-k_old[state_change]*r*t_latent) - np.exp(-k_old[state_change]*r*t_now)
            time_from_C_to_R = -np.log(1-R)/ (k_new[state_change]*r) - -np.log(1-C)/ (k_new[state_change]*r)
            tb.ti_presymp[state_change_uids] = np.ceil(ti + time_from_C_to_R/dt)
            
        return
            
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
            k_old_ls = tb.rel_LS_prog[change_uids]
            k_old_lf = tb.rel_LF_prog[change_uids]

            nut.macro_state[change_macro_uids] = nut.new_macro_state[change_macro_uids]
            nut.micro_state[change_micro_uids] = nut.new_micro_state[change_micro_uids]

            k_new_ls = self.pars.rel_LS_prog_func(nut.macro_state[change_uids], nut.micro_state[change_uids])
            k_new_lf = self.pars.rel_LF_prog_func(nut.macro_state[change_uids], nut.micro_state[change_uids])
            diff_ls = k_old_ls != k_new_ls
            diff_lf = k_old_lf != k_new_lf

            if not diff_ls.any() and not diff_lf.any():
                return

            tb.rel_LS_prog[change_uids] = k_new_ls # Update rel_LS_prog
            tb.rel_LF_prog[change_uids] = k_new_lf # Update rel_LF_prog
            
            self.update_rel_prog(tb, change_uids, k_old_ls, k_new_ls, TBS.LATENT_SLOW, tb.pars.rate_LS_to_presym, ti, dt)   # Update rel_LS_prog
            #self.update_rel_prog(tb, change_uids, k_old_lf, k_new_lf, TBS.LATENT_FAST, tb.pars.rate_LF_to_presym, ti, dt)   # Update rel_LF_prog

        return
