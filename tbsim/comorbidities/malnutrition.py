"""
Malnutrition disease model and TB-Malnutrition connector for TB simulation.
"""

import os
import numpy as np
import pandas as pd
import starsim as ss
from scipy.stats import norm
from tbsim import DATADIR

__all__ = ["Malnutrition", "TB_Nutrition_Connector"]


class Malnutrition(ss.Disease):
    """
    Malnutrition disease model for tuberculosis simulation studies.

    Tracks anthropometric measurements (weight, height) using the LMS
    (Lambda-Mu-Sigma) method and simulates effects of nutritional interventions
    on growth and development.

    Uses Cole's LMS method for growth reference curves and implements random
    walk processes for weight percentile evolution.

    Args:
        pars (dict, optional): Parameter overrides ('beta', 'init_prev')
        **kwargs: Additional keyword arguments passed to ss.Disease

    States:
        receiving_macro (bool): Whether individual receives macronutrient supplementation
        receiving_micro (bool): Whether individual receives micronutrient supplementation
        height_percentile (float): Height percentile (0-1), assumed constant
        weight_percentile (float): Weight percentile (0-1), evolves over time
        micro (float): Micronutrient status z-score, evolves over time

    Example
    -------
    ::

        import starsim as ss
        import tbsim
        from tbsim.comorbidities.malnutrition import Malnutrition, TB_Nutrition_Connector

        tb   = tbsim.TB_LSHTM(name='tb')
        mn   = Malnutrition(name='malnutrition')
        conn = TB_Nutrition_Connector()
        sim  = ss.Sim(diseases=[tb, mn], connectors=conn,
                      pars=dict(start='2000', stop='2020'))
        sim.run()

    **References:**
        - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10876842/
        - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9971264/
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with default malnutrition parameters; override via ``pars``."""
        super().__init__(**kwargs)
        self.define_pars(
            beta = 1.0,         # Transmission rate  - TODO: Check if there is one
            init_prev = 0.001,  # Initial prevalence
        )
        self.update_pars(pars, **kwargs)

        anthro_path = os.path.join(DATADIR, 'anthropometry.csv')
        self.LMS_data = pd.read_csv(anthro_path).set_index('Sex')

        self.define_states(
            ss.BoolArr('receiving_macro', default=False),
            ss.BoolArr('receiving_micro', default=False),
            ss.FloatArr('height_percentile', default=ss.uniform(0.0, 1.0)),
            ss.FloatArr('weight_percentile', default=ss.uniform(0.0, 1.0)),
            ss.FloatArr('micro', default=ss.uniform(0.0, 1.0)),
        )
        self.dweight = ss.normal(loc=0, scale=self.dweight_scale)

        return

    @staticmethod
    def dweight_scale(self, sim, uids):
        """Scale parameter for weight change distribution: 0.01 * time_index."""
        std = np.full(len(uids), fill_value=0.01*self.ti)
        return std

    def weight(self, uids=None):
        """Calculate weight (kg) from weight percentiles using LMS method."""
        return self.lms(self.weight_percentile, uids, 'Weight')

    def height(self, uids=None):
        """Calculate height (cm) from height percentiles using LMS method."""
        return self.lms(self.height_percentile, uids, 'Height')

    def lms(self, percentile, uids=None, metric='Weight'):
        """
        Calculate anthropometric measurements using the LMS method.

        Converts percentiles to actual measurements (weight/height) using
        age- and sex-specific LMS parameters interpolated from reference data.

        Args:
            percentile (np.ndarray): Percentile values (0-1)
            uids (np.ndarray, optional): Individual IDs (default: all alive)
            metric (str): 'Weight', 'Height', 'Length', or 'BMI'

        Returns:
            np.ndarray: Measurements in native units (kg, cm, or kg/m^2)
        """
        assert metric in ['Weight', 'Height', 'Length', 'BMI']

        if uids is None:
            uids = self.sim.people.auids

        ret = np.zeros(len(uids))
        ppl = self.sim.people
        female = ppl.female[uids]

        for sex, fem in zip(['Female', 'Male'], [female, ~female]):
            u = uids[fem]
            age = ppl.age[u] * 12  # in months

            age_bins = self.LMS_data.loc[sex]['Age']
            lam = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_L'])
            mu = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_M'])
            sigma = np.interp(age, age_bins, self.LMS_data.loc[sex][f'{metric}_S'])

            p = percentile[u]
            Z = norm().ppf(p)
            ret[fem] = mu * (lam*sigma*Z + 1)**(1/lam)

        return ret

    def set_initial_states(self, sim):
        """Set initial state values (placeholder for future correlated weight/height)."""
        return

    def step(self):
        """Update weight percentiles via random walk each time step."""
        uids = self.sim.people.auids
        self.weight_percentile[uids] += self.dweight(uids)
        self.weight_percentile[uids] = np.clip(self.weight_percentile[uids], 0.025, 0.975)

        return

    def init_results(self):
        """Initialize results tracking."""
        super().init_results()
        self.define_results(
            ss.Result(name='people_alive', dtype=float, label='People alive'),
        )

    def update_results(self):
        """Record proportion of individuals alive at this time step."""
        super().update_results()
        ti = self.sim.ti
        alive = self.sim.people.alive
        n_agents = self.sim.pars['n_agents']
        self.results.people_alive[ti] = np.count_nonzero(alive)/n_agents

        return


class TB_Nutrition_Connector(ss.Connector):
    """
    Connector between TB and Malnutrition disease models.

    Modifies TB transition rates based on nutritional status:
    - Activation risk ratio: how malnutrition affects latent-to-active TB
    - Clearance risk ratio: how malnutrition affects TB recovery
    - Relative susceptibility: how malnutrition affects new TB infection risk

    Args:
        pars (dict, optional): Override functions for 'rr_activation_func',
            'rr_clearance_func', 'relsus_func'
        **kwargs: Additional keyword arguments passed to ss.Connector
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with pluggable risk-ratio and relative-susceptibility functions."""
        super().__init__(label='TB-Malnutrition')
        self.define_pars(
            rr_activation_func = self.ones_rr,
            rr_clearance_func = self.ones_rr,
            relsus_func = self.compute_relsus,
        )
        self.update_pars(pars, **kwargs)

        return

    @staticmethod
    def supplementation_rr(tb, mn, uids, rate_ratio=0.5):
        """Risk ratio based on supplementation: rate_ratio for those receiving both macro+micro, else 1."""
        rr = np.ones_like(uids)
        rr[mn.receiving_macro[uids] & mn.receiving_micro[uids]] = rate_ratio
        return rr

    @staticmethod
    def lonnroth_bmi_rr(tb, mn, uids, scale=2, slope=3, bmi50=25):
        """Sigmoid BMI-based risk ratio following Lonnroth et al. log-linear relationship."""
        bmi = 10_000 * mn.weight(uids) / mn.height(uids)**2
        x = -0.05*(bmi-15) + 2
        x0 = -0.05*(bmi50-15) + 2
        rr = scale / (1+10**(-slope * (x-x0)))
        return rr

    @staticmethod
    def ones_rr(tb, mn, uids):
        """Neutral risk ratios (all ones, no effect)."""
        return np.ones_like(uids)

    @staticmethod
    def compute_relsus(tb, mn, uids):
        """Relative susceptibility: 2x if micronutrient status < 0.2, else 1x."""
        rel_sus = np.ones_like(uids)
        rel_sus[mn.micro[uids] < 0.2] = 2
        return rel_sus

    def step(self):
        """Apply nutritional effects to TB transition rates each time step."""
        tb = self.sim.diseases['tb_emod']
        mn = self.sim.diseases['malnutrition']

        uids = tb.infected.uids
        tb.rr_activation[uids] *= self.pars.rr_activation_func(tb, mn, uids)
        tb.rr_clearance[uids] *= self.pars.rr_clearance_func(tb, mn, uids)

        uids = (~tb.infected).uids
        tb.rel_sus[uids] = self.pars.relsus_func(tb, mn, uids)

        return
