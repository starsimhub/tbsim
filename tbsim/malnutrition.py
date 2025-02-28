"""
Define non-communicable disease (Malnutrition) model
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
    This class implements a basic Malnutrition model. It inherits from the startim Disease class.
    """
    # Possible references:
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    # https://www.espen.org/files/ESPEN-guidelines-on-definitions-and-terminology-of-clinical-nutrition.pdf 

    @staticmethod
    def dweight_loc(self, sim, uids):
        mu = np.zeros(len(uids))
        mu[self.receiving_macro] = 1.0*self.ti # Upwards drift in percentile for those receiving macro supplementation
        return mu

    @staticmethod
    def dweight_scale(self, sim, uids):
        std = np.full(len(uids), fill_value=0.01*self.ti)
        return std

    def weight(self, uids=None):
        weight = self.lms(self.weight_percentile, uids, 'Weight')
        return weight

    def height(self, uids=None):
        height = self.lms(self.height_percentile, uids, 'Height')
        return height

    def lms(self, percentile, uids=None, metric='Weight'):
        # Return weight given a percentile using Cole's lambda, mu, and sigma (LMS) method 

        assert metric in ['Weight', 'Height', 'Length', 'BMI']

        if uids is None:
            uids = self.sim.people.auids

        ret = np.zeros(len(uids))

        ppl = self.sim.people
        female = ppl.female[uids]

        for sex, fem in zip(['Female', 'Male'], [female, ~female]):
            u = uids[fem]
            age = ppl.age[u] * 12 # in months

            age_bins = self.LMS_data.loc[sex]['Age']
            lam = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_L'])
            mu = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_M'])
            sigma = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_S'])

            # https://indianpediatrics.net/jan2014/jan-37-43.htm
            #Z = 1/(sigma*lam) * ((WEIGHT/mu)**lam - 1) 

            p = percentile[u]
            Z = norm().ppf(p) # Convert percentile to z-score
            ret[fem] = mu * (lam*sigma*Z + 1)**(1/lam) # if lam=0, w = mu * np.exp(sigma * Z)

            # https://iris.who.int/bitstream/handle/10665/44026/9789241547635_eng.pdf?sequence=1

        return ret

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            beta = 1.0,         # Transmission rate  - TODO: Check if there is one
            init_prev = 0.001,  # Initial prevalence 
        )
        self.update_pars(pars, **kwargs)

        anthro_path = os.path.join(DATADIR, 'anthropometry.csv')
        self.LMS_data = pd.read_csv(anthro_path).set_index('Sex')

        # Adding Malnutrition states to handle the Individual Properties related to this disease 
        self.define_states(
            # Hooks to the RATIONS trial
            ss.BoolArr('receiving_macro', default=False), # Determines weight trend
            ss.BoolArr('receiving_micro', default=False), # Determines micro trend

            # Internal state
            # PROBLEM: Correlation between weight and height
            ss.FloatArr('height_percentile', default=ss.uniform(name='height_percentile')), # Percentile, stays fixed
            ss.FloatArr('weight_percentile', default=ss.uniform(name='weight_percentile')), # Percentile, increases when receiving micro, then declines?
            ss.FloatArr('micro', default=ss.uniform(name='micro')), # Continuous? Normal distribution around zero. Z-score, sigmoid thing. Half-life.
        )
        self.dweight = ss.normal(loc=self.dweight_loc, scale=self.dweight_scale)

        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
        # Could correlate weight and height here, via gaussian along the diagonal with corner correction?
        return

    '''
    def set_macro_supplement(self, uids, kcal):
        self.macro_drift[uids] = kcal
        pass

    def set_micro_supplement(self, uids, stop):
        self.micro_drift[uids[~stop]] = True
        self.micro_drift[uids[stop]] = 0
        pass
    '''

    def step(self):
        uids = self.sim.people.auids # All alive uids

        # Random walks
        self.weight_percentile[uids] += self.dweight(uids)
        self.weight_percentile[uids] = np.clip(self.weight_percentile[uids], 0.025, 0.975) # needed?

        '''
        new_macro = (self.ti_macro == ti).uids
        if len(new_macro) > 0:
            self.macro_state[new_macro] = self.new_macro_state[new_macro]

        new_micro = (self.ti_micro == ti).uids
        if len(new_micro) > 0:
            self.micro_state[new_micro] = self.new_micro_state[new_micro]

        return new_macro, new_micro
        '''
        return

    def init_results(self):
        """
        Initialize results
        """
        super().init_results()
        self.define_results(
            ss.Result(name='people_alive', dtype=float, label='People alive'),
        )
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti            # Current time index (step)
        alive = self.sim.people.alive    # People alive at current time index
        n_agents = self.sim.pars['n_agents']
        self.results.people_alive[ti] = alive.count()/n_agents
        return
