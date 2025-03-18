"""
Define non-communicable disease (Malnutrition) model.
"""

import os
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import norm
from tbsim import DATADIR

__all__ = ["Malnutrition"]


class Malnutrition(ss.Disease):
    """
    A basic Malnutrition model that extends the starsim Disease class.
    
    References:
      - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
      - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
      - https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf
    """

    def dweight_loc(self, sim, uids):
        """
        Compute the location (mean) for weight change.
        
        For individuals receiving macro supplementation, the percentile drifts upward.
        """
        mu = np.zeros(len(uids))
        # Upwards drift in percentile for those receiving macro supplementation
        mu[self.receiving_macro] = 1.0 * self.ti
        return mu

    def dweight_scale(self, sim, uids):
        """
        Compute the scale (standard deviation) for weight change.
        """
        std = np.full(len(uids), fill_value=0.01 * self.ti)
        return std

    def weight(self, uids=None):
        """
        Calculate weight using the LMS method based on weight percentile.
        """
        return self.lms(self.weight_percentile, uids, metric='Weight')

    def height(self, uids=None):
        """
        Calculate height using the LMS method based on height percentile.
        """
        return self.lms(self.height_percentile, uids, metric='Height')

    def lms(self, percentile, uids=None, metric='Weight'):
        """
        Convert a given percentile to the corresponding metric (Weight, Height, Length, or BMI)
        using Cole's lambda, mu, and sigma (LMS) method.
        
        Parameters:
            percentile: array-like percentiles.
            uids: optional array of user ids; if None, use all people.
            metric: one of ['Weight', 'Height', 'Length', 'BMI'].
        """
        assert metric in ['Weight', 'Height', 'Length', 'BMI']

        if uids is None:
            uids = self.sim.people.auids

        ret = np.zeros(len(uids))
        ppl = self.sim.people
        female = ppl.female[uids]

        # Process for both sexes: 'Female' and 'Male'
        for sex, fem in zip(['Female', 'Male'], [female, ~female]):
            u = uids[fem]
            age = ppl.age[u] * 12  # convert age to months

            age_bins = self.LMS_data.loc[sex]['Age']
            lam = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_L'])
            mu = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_M'])
            sigma = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_S'])

            # Convert percentile to z-score using the normal distribution.
            p = percentile[u]
            Z = norm().ppf(p)
            # If lam == 0, an exponential form could be used: weight = mu * np.exp(sigma * Z)
            ret[fem] = mu * (lam * sigma * Z + 1) ** (1 / lam)

        return ret

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the Malnutrition model.
        """
        super().__init__(**kwargs)
        self.define_pars(
            beta=1.0,         # Transmission rate (to be verified)
            init_prev=0.001,  # Initial prevalence 
        )
        self.update_pars(pars, **kwargs)

        # Load anthropometry data for LMS calculations.
        anthro_path = os.path.join(DATADIR, 'anthropometry.csv')
        self.LMS_data = pd.read_csv(anthro_path).set_index('Sex')

        # Define states related to malnutrition.
        self.define_states(
            # RATIONS trial hooks
            ss.BoolArr('receiving_macro', default=False),  # Determines weight trend
            ss.BoolArr('receiving_micro', default=False),  # Determines micronutrient trend

            # Internal states
            ss.FloatArr('height_percentile', default=ss.uniform(name='height_percentile')),
            ss.FloatArr('weight_percentile', default=ss.uniform(name='weight_percentile')),
            ss.FloatArr('micro', default=ss.uniform(name='micro')),
        )

        # Define the weight change process using a normal distribution.
        self.dweight = ss.normal(loc=self.dweight_loc, scale=self.dweight_scale)

    def set_initial_states(self, sim):
        """
        Set initial state values for the simulation.
        """
        # Future implementation: correlate weight and height via a Gaussian distribution.
        pass

    def step(self):
        """
        Update the weight percentile for all alive individuals.
        """
        uids = self.sim.people.auids  # All alive user ids

        # Update weight percentile using a random walk.
        self.weight_percentile[uids] += self.dweight(uids)
        self.weight_percentile[uids] = np.clip(self.weight_percentile[uids], 0.025, 0.975)
        return

    def init_results(self):
        """
        Initialize results tracking.
        """
        super().init_results()
        self.define_results(
            ss.Result(name='people_alive', dtype=float, label='People alive'),
        )

    def update_results(self):
        """
        Update results at each simulation step.
        """
        super().update_results()
        ti = self.sim.ti  # Current time index
        alive = self.sim.people.alive  # Alive individuals at current time step
        n_agents = self.sim.pars['n_agents']
        self.results.people_alive[ti] = alive.count() / n_agents
