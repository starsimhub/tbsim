import starsim as ss 

__all__ = ['BetaIntervention']
class BetaIntervention(ss.Intervention):
    """
    A transmission reduction intervention that modifies the tuberculosis transmission rate (beta) 
    at specified time points during the simulation.
    
    This intervention allows for modeling population-wide changes that affect disease transmission,
    such as policy changes, behavioral modifications, environmental improvements, or calibration
    scenarios. The intervention multiplies the current beta value by a specified factor at each
    target year, creating a sustained reduction in transmission rates.
    
    Attributes:
        years (list): List of years when the intervention should be applied. Default is [2000].
        x_beta (float): Multiplicative factor to apply to the current beta value. 
                       Values < 1 reduce transmission, values > 1 increase transmission.
                       Default is 1 (no change).
        applied_years (set): Internal tracking of which years the intervention has already been applied.
    
    Examples:
        ```python
        # Reduce transmission by 50% in 2000
        intervention = BetaIntervention(pars={'years': [2000], 'x_beta': 0.5})
        
        # Reduce transmission by 30% in multiple years
        intervention = BetaIntervention(pars={'years': [2000, 2010, 2020], 'x_beta': 0.7})
        
        # Increase transmission by 20% in 2015
        intervention = BetaIntervention(pars={'years': [2015], 'x_beta': 1.2})
        ```
    
    Note:
        - The intervention is applied only once per specified year, even if the simulation
          steps through the same year multiple times.
        - Changes are applied to both the main disease beta parameter and all mixing pools
          in the network to ensure consistency.
        - This is a multiplicative change, not additive. For example, x_beta=0.5 reduces
          transmission by 50%, while x_beta=0.8 reduces it by 20%.
    """

    def __init__(self, pars=None, *args, **kwargs):
        """
        Initialize the BetaIntervention.
        
        Args:
            pars (dict, optional): Dictionary containing intervention parameters.
                                 Keys: 'years' (list of ints), 'x_beta' (float)
            *args: Additional positional arguments passed to parent class
            **kwargs: Additional keyword arguments passed to parent class
        """
        super().__init__()
        self.define_pars(
            years = [2000],  # Default year to apply intervention
            x_beta = 1,      # Default multiplier (no change)
        )
        self.update_pars(pars, **kwargs)
        self.applied_years = set()  # Track which years intervention has been applied

    def step(self):
        """
        Execute the intervention step, applying beta modifications at specified years.
        
        This method is called at each simulation timestep. It checks if the current year
        matches any of the target years for intervention application. If a match is found
        and the intervention hasn't been applied to that year yet, it:
        
        1. Multiplies the tuberculosis disease beta parameter by the specified factor
        2. Updates all mixing pools in the network to maintain consistency
        3. Marks the year as applied to prevent duplicate applications
        
        The intervention is applied within the time window [target_year, target_year + dt_year)
        to ensure it happens exactly once per target year, even with variable timesteps.
        
        Returns:
            None
            
        Note:
            - Changes are applied to both the main disease parameters and all network mixing pools
            - The intervention only applies once per target year, tracked via self.applied_years
            - This ensures consistency across all transmission pathways in the model
        """
        year = self.sim.t.now('year')
        
        # Check if current year is in the list of years to apply the intervention
        for target_year in self.pars.years:
            # Apply intervention if we're in the target year and haven't applied it yet
            if (year >= target_year) & (year < target_year + self.t.dt_year) and target_year not in self.applied_years:
                # Apply multiplicative change to the main disease beta parameter
                self.sim.diseases.tb.pars['beta'] *= self.pars.x_beta
                self.applied_years.add(target_year)

                # Update all mixing pools to maintain consistency across the network
                for net in self.sim.networks.values():
                    if isinstance(net, ss.MixingPools): # A collection of pools
                        for pool_idx in range(len(net.pools)):
                            net.pools[pool_idx].pars.beta = self.sim.diseases.tb.pars['beta']
                    elif isinstance(net, ss.MixingPool): # A single pool
                            net.pars.beta = self.sim.diseases.tb.pars['beta']
                        