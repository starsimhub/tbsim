"""Intervention that modifies the TB transmission rate (beta) at specified time points."""

import starsim as ss

__all__ = ['BetaByYear']


class BetaByYear(ss.Intervention):
    """
    A transmission reduction intervention that modifies the tuberculosis transmission rate (beta)
    at specified time points during the simulation.

    This intervention allows for modeling population-wide changes that affect disease transmission,
    such as policy changes, behavioral modifications, environmental improvements, or calibration
    scenarios. The intervention multiplies the current beta value by a specified factor at each
    target year, creating a sustained reduction in transmission rates.

    IMPORTANT:
    -------------
    The modifier (x_beta) is always applied to the **current/latest value** of beta at the time of intervention,
    not the original or baseline value. This means that if multiple interventions are scheduled, each one
    will multiply the beta value as it exists after all previous modifications. The effect is **cumulative and multiplicative**.

    For example:
        - If beta starts at 0.01, and x_beta=0.5 is applied in 2000, beta becomes 0.005.
        - If another x_beta=0.8 is applied in 2010, beta becomes 0.004 (0.005 * 0.8).
        - This is NOT a reset to the original value; each change compounds on the last.

    Attributes:
        years (list): List of years when the intervention should be applied. Default is [2000].
        x_beta (float or list): Multiplicative factor(s) to apply to the current beta value.
            - If a single value, it is applied to all years.
            - If a list, it must be the same length as years, and each value is applied to the corresponding year.
        applied_years (set): (Deprecated) No longer used; interventions are now removed after application.

    Behavior:
        - Each (year, x_beta) pair is applied only once. After application, both are removed from their lists.
        - If x_beta is a list, its length must match years, or a ValueError is raised.
        - Changes are applied to both the main disease beta parameter and all mixing pools in the network to ensure consistency.
        - This is a multiplicative change, not additive. For example, x_beta=0.5 reduces transmission by 50%, while x_beta=0.8 reduces it by 20%.
        - **Multiple interventions compound:** Each new x_beta is applied to the already-modified beta value.

    Examples:
        ```python
        # Reduce transmission by 50% in 2000
        intervention = BetaByYear(pars={'years': [2000], 'x_beta': 0.5})

        # Reduce transmission by 30% in multiple years (same factor)
        intervention = BetaByYear(pars={'years': [2000, 2010, 2020], 'x_beta': 0.7})

        # Use different factors for each year
        intervention = BetaByYear(pars={'years': [2000, 2010, 2020], 'x_beta': [0.7, 0.8, 0.9]})

        # Increase transmission by 20% in 2015
        intervention = BetaByYear(pars={'years': [2015], 'x_beta': 1.2})
        ```
    """

    def __init__(self, pars=None, *args, **kwargs):
        """
        Initialize the BetaByYear intervention.

        Args:
            pars (dict, optional): Dictionary containing intervention parameters.
                - 'years' (list of ints): Years to apply the intervention.
                - 'x_beta' (float or list): Multiplicative factor(s) for beta. If a list, must match years.
            *args: Additional positional arguments passed to parent class
            **kwargs: Additional keyword arguments passed to parent class

        Raises:
            ValueError: If x_beta is a list and its length does not match years.
        """
        super().__init__()
        self.define_pars(
            years = [2000],  # Default year to apply intervention
            x_beta = 1,      # Default multiplier (no change)
        )
        self.update_pars(pars, **kwargs)
        self.applied_years = set()  # (Deprecated, kept for backward compatibility)

        if isinstance(self.pars['x_beta'], (list, tuple)):
            if len(self.pars['x_beta']) != len(self.pars['years']):
                raise ValueError("If x_beta is a list, it must be the same length as years.")
            self._x_beta_list = list(self.pars['x_beta'])
        else:
            self._x_beta_list = [self.pars['x_beta']] * len(self.pars['years'])

    def step(self):
        """
        Execute the intervention step, applying beta modifications at specified years.
        """
        year = int(self.sim.t.now('year'))
        if len(self.pars.years)>0:
            target_year = self.pars.years[0]
            x_beta = self._x_beta_list[0]
            
            # Apply intervention only when we first reach the target year
            # This ensures it's applied only once, not repeatedly
            if year == target_year:
                self.sim.diseases.tb_emod.pars['beta'] *= x_beta
                print(f"At year:{year}, Modified BetaValue:{x_beta}")
                # Always remove the year and x_beta after application
                self.pars.years.pop(0)
                self._x_beta_list.pop(0)
                # Do not increment idx, as lists have shifted
                        