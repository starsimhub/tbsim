"""
Define non-communicable disease (Malnutrition) model
"""

import numpy as np
import starsim as ss
from scipy.stats import norm

__all__ = ['Malnutrition']

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
        mu[self.receiving_macro] = 0.1*sim.dt # Upwards drift in percentile for those receiving macro supplementation
        return mu

    @staticmethod
    def dweight_scale(self, sim, uids):
        std = np.full(len(uids), fill_value=0.01*sim.dt)
        return std

    @property
    def weight(self):
        # Return weight given a percentile using Cole's lambda, mu, and sigma (LMS) method 

        # Parameters should be by age and sex, guessing
        mu = 65 #13.6
        sigma = 0.18 #0.147
        lam = 0.85 #0.3

        # https://indianpediatrics.net/jan2014/jan-37-43.htm
        #Z = 1/(sigma*lam) * ((WEIGHT/mu)**lam - 1) 

        p = self.weight_percentile
        Z = norm().ppf(p) # Convert percentile to z-score
        weight = mu * (lam*sigma*Z + 1)**(1/lam)

        return weight

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.default_pars(
            beta = 1.0,         # Transmission rate  - TODO: Check if there is one
            init_prev = 0.001,  # Initial prevalence 
        )
        self.update_pars(pars, **kwargs)

        # Adding Malnutrition states to handle the Individual Properties related to this disease 
        self.add_states(
            # Hooks to the RATIONS trial
            ss.BoolArr('receiving_macro', default=False), # Determines weight trend
            ss.BoolArr('receiving_micro', default=False), # Determines micro trend

            # Internal state
            # PROBLEM: Correlation between weight and height
            ss.FloatArr('height_percentile', default=ss.uniform()), # Percentile
            ss.FloatArr('weight_percentile', default=ss.uniform()), # Percentile, increases when receiving micro, then declines
            ss.FloatArr('micro', default=ss.uniform()), # Continuous? Normal distribution around zero. Z-score, sigmoid thing. Half-life.

            # With downstream implications via the connector to:
            # * LS progression rate
            # * LF progression rate
            # * Susceptibility
            # * Other TB stuff?

            # via functions that look like...
            # * Longroth: BMI --> (rel_sus, progression)
            # * weight_trend --> (rel_sus, progression)
            # * weight_trend + micro --> (rel_sus, progression)

            # Dose response mapping (continuous instead of discrete states)
            # Time in exposure/risk state, more precisely with continuous state
            # Recent change vs long term
        )

        self.dweight = ss.normal(loc=self.dweight_loc, scale=self.dweight_scale)

        return

    def set_initial_states(self, sim):
        """
        Set initial values for states.
        """
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

    def update_pre(self):
        ti = self.sim.ti
        dt = self.sim.dt

        uids = self.sim.people.auids # All alive uids

        # Random walks
        self.weight_percentile[uids] += self.dweight(uids)
        self.weight_percentile[uids] = np.clip(self.weight_percentile[uids], 0.025, 0.975)

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
        npts = self.sim.npts
        self.results += [
            # ss.Result(self.name, 'prev_macro_standard_or_above', npts, dtype=float),
            # ss.Result(self.name, 'prev_macro_slightly_below', npts, dtype=float),
            # ss.Result(self.name, 'prev_macro_marginal', npts, dtype=float),
            # ss.Result(self.name, 'prev_macro_unsatisfactory', npts, dtype=float),

            # ss.Result(self.name, 'prev_micro_normal', npts, dtype=float),
            # ss.Result(self.name, 'prev_micro_deficient', npts, dtype=float),
            
            # ss.Result(self.name, 'prev_normal_weight', npts, dtype=float),
            # ss.Result(self.name, 'prev_mild_thinness', npts, dtype=float),
            # ss.Result(self.name, 'prev_moderate_thinness', npts, dtype=float),
            # ss.Result(self.name, 'prev_severe_thinness', npts, dtype=float),

            ss.Result(self.name, 'people_alive', npts, dtype=float),
        ]
        return

    def update_results(self):
        super().update_results()
        ti = self.sim.ti            # Current time index (step)
        alive = self.sim.people.alive    # People alive at current time index
        n_agents = self.sim.pars['n_agents']
        n_alive = alive.count()
        
        ##self.results.prev_macro_standard_or_above[ti] = np.count_nonzero((self.macro_state==MacroNutrients.STANDARD_OR_ABOVE) & alive)/n_alive
        #self.results.prev_macro_slightly_below[ti] = np.count_nonzero((self.macro_state==MacroNutrients.SLIGHTLY_BELOW_STANDARD) & alive)/n_alive
        #self.results.prev_macro_marginal[ti] = np.count_nonzero((self.macro_state==MacroNutrients.MARGINAL) & alive)/n_alive
        #self.results.prev_macro_unsatisfactory[ti] = np.count_nonzero((self.macro_state==MacroNutrients.UNSATISFACTORY) & alive)/n_alive

        #self.results.prev_micro_normal[ti] = np.count_nonzero((self.micro_state==MicroNutrients.NORMAL) & alive)/n_alive
        #self.results.prev_micro_deficient[ti] = np.count_nonzero((self.micro_state==MicroNutrients.DEFICIENT) & alive)/n_alive
        self.results.people_alive[ti] = alive.count()/n_agents


        #self.results.prev_normal_weight[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.NORMAL_WEIGHT) & alive) / n_alive
        #self.results.prev_mild_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.MILD_THINNESS) & alive) / n_alive
        #self.results.prev_moderate_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.MODERATE_THINNESS) & alive) / n_alive
        #self.results.prev_severe_thinness[ti] = np.count_nonzero((self.bmi_state == ne.eBmiStatus.SEVERE_THINNESS) & alive) / n_alive
        return
