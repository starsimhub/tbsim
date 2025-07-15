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
        idx = 0
        if len(self.pars.years)>0:
            target_year = self.pars.years[idx]
            x_beta = self._x_beta_list[idx]
            if (year >= target_year) and (year < target_year + self.t.dt_year):
                self.sim.logger.info(f"NEW BETA: At year:{year}, Value:{x_beta}")
                self.sim.diseases.tb.pars['beta'] *= x_beta
                for net in self.sim.networks.values():
                    if isinstance(net, ss.MixingPools):
                        for pool_idx in range(len(net.pools)):
                            net.pools[pool_idx].pars.beta = self.sim.diseases.tb.pars['beta']
                    elif isinstance(net, ss.MixingPool):
                        net.pars.beta = self.sim.diseases.tb.pars['beta']
                # Always remove the year and x_beta after application
                self.pars.years.pop(idx)
                self._x_beta_list.pop(idx)
                # Do not increment idx, as lists have shifted
            else:
                idx += 1
                        