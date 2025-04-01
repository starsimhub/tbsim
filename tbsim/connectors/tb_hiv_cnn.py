import numpy as np
import starsim as ss
from tbsim import TB, HIV, HIVState

__all__ = ['TB_HIV_Connector']

class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV.

    This connector uses the HIV state (ATRISK, ACUTE, LATENT, AIDS, DEAD) and ART status
    from the HIV disease model to modify TB progression parameters.

    Adjustments:
      - TB-infected individuals have increased:
          - Risk of progression from latent TB to presymptomatic TB (via `rr_activation`)
          - Risk of death due to active TB (via `rr_death`)
      - TB-noninfected individuals have increased susceptibility to TB infection (`rel_sus`)
      - ART reduces these risks by a multiplicative factor `art_tb_multiplier` (default: 0.8)

    State multipliers:
      - ATRISK: 1.0
      - ACUTE:  1.5
      - LATENT: 2.0
      - AIDS:   3.0
      - DEAD:   0.0 (no progression or susceptibility)
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_hiv_rr_func      = self.compute_tb_hiv_risk_rr,
            tb_clearance_rr_func= self.ones_rr,  # No modification to clearance
            tb_rel_sus_func     = self.compute_tb_rel_sus,
            art_tb_multiplier   = 0.75,
        )
        self.update_pars(pars, **kwargs)
        self.state_multipliers = {
            HIVState.ACUTE: 1.5,
            HIVState.LATENT: 2.0,
            HIVState.AIDS: 3.0,
            HIVState.DEAD: 0.0,
        }

    @staticmethod
    def ones_rr(tb, hiv, uids):
        return np.ones_like(uids, dtype=float)

    @staticmethod
    def compute_tb_hiv_risk_rr(self, tb, hiv, uids, base_factor=1.0):
        """
        Computes the relative risk (RR) multiplier for TB progression and mortality 
        based on HIV state and ART (antiretroviral therapy) status.
        Parameters:
            tb (object): The TB model object (not directly used in this function).
            hiv (object): The HIV model object containing state and ART status information.
            uids (array-like): Array of unique identifiers for individuals.
            base_factor (float, optional): A base multiplier applied to the computed RR. 
                                           Defaults to 1.0.
        Returns:
            numpy.ndarray: An array of relative risk multipliers for the given individuals.
                           
        Notes:
            - The function initializes the RR multipliers to 1.0 for all individuals.
            - It applies state-specific multipliers based on the individual's HIV state.
            - If an individual is on ART and not in the DEAD state, an additional ART 
              multiplier is applied to their RR.
            - The final RR is scaled by the `base_factor` parameter.
        """
        art_multiplier = self.pars.art_tb_multiplier
        states = hiv.state[uids]
        on_art = hiv.on_ART[uids]
        # Initialize multipliers with 1.0
        rr = np.ones_like(uids, dtype=float)
        # Define risk multipliers by HIV state
        state_multipliers = self.state_multipliers
        # Apply state-specific risk multipliers
        for state, mult in state_multipliers.items():
            rr[states == state] = mult
        # Apply ART multiplier (except if DEAD)
        rr[(on_art) & (states != HIVState.DEAD)] *= art_multiplier

        return rr * base_factor

    @staticmethod

    def compute_tb_rel_sus(self, tb, hiv, uids, baseline=1.0):    
        """
        Computes the relative susceptibility to tuberculosis (TB) for uninfected individuals.

        This method calculates the relative susceptibility to TB based on the HIV state,
        ART (antiretroviral therapy) status, and predefined multipliers for different states.

        Args:
            self: The instance of the class containing parameters and state multipliers.
            tb: The TB model (not used in this method).
            hiv: The HIV model containing state and ART status information.
            uids (array-like): The unique identifiers for individuals.
            baseline (float, optional): The baseline susceptibility value. Defaults to 1.0.

        Returns:
            numpy.ndarray: An array of relative susceptibility values for the given individuals.
        """

        art_multiplier = self.pars.art_tb_multiplier
        states = hiv.state[uids]
        on_art = hiv.on_ART[uids]
        rel_sus = np.full_like(uids, fill_value=baseline, dtype=float)
        state_multipliers = self.state_multipliers
        for state, mult in state_multipliers.items():
            rel_sus[states == state] = baseline * mult
        rel_sus[(on_art) & (states != HIVState.DEAD)] *= art_multiplier

        return rel_sus

    def step(self):
        tb = self.sim.diseases['tb']
        hiv = self.sim.diseases['hiv']

        # Adjust TB progression and death risks for infected individuals
        uids_tb = tb.infected.uids
        rr = self.pars.tb_hiv_rr_func(self, tb, hiv, uids_tb)

        tb.rr_activation[uids_tb] *= rr
        tb.rr_death[uids_tb]      *= rr
        tb.rr_clearance[uids_tb]  *= self.pars.tb_clearance_rr_func(tb, hiv, uids_tb)  # Defaults to 1.0

        # Adjust TB susceptibility for uninfected individuals
        uids_no_tb = (~tb.infected).uids
        tb.rel_sus[uids_no_tb] = self.pars.tb_rel_sus_func(self, tb, hiv, uids_no_tb)

        return
