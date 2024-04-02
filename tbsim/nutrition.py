"""
Define non-communicable disease (Nutrition) model
"""

import numpy as np
import starsim as ss
import sciris as sc

__all__ = ['Nutrition']

class Nutrition(ss.Disease):
    """
    Example non-communicable disease

    This class implements a basic Nutrition model with risk of developing a condition
    (e.g., hypertension, diabetes), a state for having the condition, and associated
    mortality.
    """
    def __init__(self, pars=None):
        default_pars = dict(
            initial_risk = ss.bernoulli(p=0.3), # Initial prevalence of risk factors
            #'affection_rate': ss.rate(p=0.1), # Instantaneous rate of acquisition applied to those at risk (units are acquisitions / year)
            dur_risk = ss.expon(scale=10),
            prognosis = ss.weibull_min(c=2, scale=5), # Time in years between first becoming under_nutried and death
        )

        super().__init__(ss.omerge(default_pars, pars))
        self.add_states(
            ss.State('normal', bool, True),
            ss.State('under_nutried', bool, False),
            ss.State('overweigth', bool, False),
            ss.State('ti_under_nutried', int, ss.INT_NAN),
        )
        return

    @property
    def not_normal(self):
        return ~self.normal

    def set_initial_states(self, sim):
        """
        Setting initial nutrition states
        """
        alive_uids = ss.true(sim.people.alive)
        initial_risk = self.pars['initial_risk'].filter(alive_uids)
        self.normal[initial_risk] = True
        self.ti_under_nutried[initial_risk] = sim.ti + sc.randround(self.pars['dur_risk'].rvs(initial_risk) / sim.dt)
        return initial_risk

    def make_new_cases(self, sim):
        new_cases = ss.true(self.ti_under_nutried == sim.ti)
        self.under_nutried[new_cases] = True
        prog_years = self.pars['prognosis'].rvs(new_cases)
        self.ti_dead[new_cases] = sim.ti + sc.randround(prog_years / sim.dt)
        super().set_prognoses(sim, new_cases)
        return new_cases

    # def init_results(self, sim):
    #     """
    #     Initialize results
    #     """
    #     super().init_results(sim)
    #     self.results += [
    #         ss.Result(self.name, 'n_not_normal', sim.npts, dtype=int),
    #         ss.Result(self.name, 'prevalence', sim.npts, dtype=float),
    #         ss.Result(self.name, 'new_deaths', sim.npts, dtype=int),
    #     ]
    #     return

    # def update_results(self, sim):
    #     super().update_results(sim)
    #     ti = sim.ti
    #     alive = sim.people.alive
    #     self.results.n_not_normal[ti] = np.count_nonzero(self.not_normal & alive)
    #     self.results.prevalence[ti]    = np.count_nonzero(self.under_nutried & alive)/alive.count()
    #     self.results.new_deaths[ti]    = np.count_nonzero(self.ti_dead == ti)
    #     return
