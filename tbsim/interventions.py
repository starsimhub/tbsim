"""
Define Nutrition intervention
"""

# WORK IN PROGRESS, CODE NOT FUNCTIONAL YET

import numpy as np
import starsim as ss
from tbsim import TB, Nutrition, TBS
import sciris as sc

__all__ = ['VitaminSupplementation']

class VitaminSupplementation(ss.Intervention):

    def __init__(self, year: np.array, coverage: np.array, **kwargs):
        self.requires = Nutrition
        self.year = sc.promotetoarray(year)
        self.rate = sc.promotetoarray(coverage)

        super().__init__(**kwargs)

        self.p_recovery = ss.bernoulli(p=lambda self, sim, uids: np.interp(sim.year, self.year, self.coverage))
        self.eff_pphintv = ss.bernoulli(p=PPH_INTV_EFFICACY)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.results += ss.Result(self.name, 'n_pphintv', sim.npts, dtype=int)
        self.results += ss.Result(self.name, 'n_mothers_saved', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        if sim.year < self.year[0]:
            return

        pph = sim.demographics['pph']
        maternal_deaths = ss.true(pph.ti_dead <= sim.ti)
        receive_pphintv = self.p_pphintv.filter(maternal_deaths)
        pph_deaths_averted = self.eff_pphintv.filter(receive_pphintv)
        pph.ti_dead[pph_deaths_averted] = ss.INT_NAN

        # Add results
        self.results['n_pphintv'][sim.ti] = len(receive_pphintv)
        self.results['n_mothers_saved'][sim.ti] = len(pph_deaths_averted)

        return len(pph_deaths_averted)