import numpy as np
import starsim as ss
from tbsim import TB, HIV, HIVState

__all__ = ['TB_HIV_Connector']

class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV.
    
    This connector uses the HIV state (ATRISK, HIV, LATENT, AIDS, DEAD) and the on_ART flag
    from the HIV disease model to modify TB parameters.
    
    For TB‐infected individuals, the TB activation relative risk (rr_activation) is multiplied by:
      - ATRISK: 1.0 (baseline)
      - ACUTE:  1.5 (or 1.5 * art_tb_multiplier if on ART)
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
        rel_sus = np.full_like(uids, fill_value=baseline, dtype=float)
        art_multiplier = tb.pars.get('art_tb_multiplier', 0.8)
        states = hiv.state[uids]
        on_art = hiv.on_ART[uids]

        # Define multipliers for each state
        state_multipliers = {
            HIVState.ACUTE: 1.5,
            HIVState.LATENT: 2.0,
            HIVState.AIDS: 3.0,
            HIVState.DEAD: 0.0,
        }

        # Get multiplier for each state, default to 1.0
        vectorized_multipliers = np.vectorize(lambda s: state_multipliers.get(s, 1.0))(states).astype(float)

        # Apply ART multiplier
        vectorized_multipliers *= np.where(on_art, art_multiplier, 1.0)

        # Compute final susceptibility
        rel_sus = baseline * vectorized_multipliers
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
