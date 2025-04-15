import numpy as np
import starsim as ss
from tbsim import TB, HIV, HIVState

__all__ = ['TB_HIV_Connector']

class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV.

    This connector uses the HIV state ( ACUTE, LATENT, AIDS)
    from the HIV disease model to modify TB progression parameters.

    Adjustments:
      - TB-infected individuals have increased:
          - Risk of progression from latent TB to presymptomatic TB (via `rr_activation`)

    State multipliers:
      - ACUTE:  1.5
      - LATENT: 2.0
      - AIDS:   3.0
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_hiv_rr_func      = self.compute_tb_hiv_risk_rr,
            acute_multiplier     = 1.5,
            latent_multiplier    = 2.0,
            aids_multiplier      = 3.0,
        )
        self.update_pars(pars, **kwargs)
        self.state_multipliers = {
            HIVState.ACUTE: self.pars.acute_multiplier,
            HIVState.LATENT: self.pars.latent_multiplier,
            HIVState.AIDS:   self.pars.aids_multiplier,

        }

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
            - The final RR is scaled by the `base_factor` parameter.
        """
        states = hiv.state[uids]
        
        # Initialize multipliers with 1.0
        rr = np.ones_like(uids, dtype=float)
        
        # Define risk multipliers by HIV state
        state_multipliers = self.state_multipliers
        
        # Apply state-specific risk multipliers
        for state, mult in state_multipliers.items():
            rr[states == state] = mult
            
        return rr * base_factor

    def step(self):
        """
        This is where the actual modification of TB parameters occurs.
        """
        tb = self.sim.diseases['tb']
        hiv = self.sim.diseases['hiv']

        # Adjust TB progression and death risks for infected individuals
        uids_tb = tb.infected.uids
        rr = self.pars.tb_hiv_rr_func(self, tb, hiv, uids_tb)

        tb.rr_activation[uids_tb] *= rr

        return
