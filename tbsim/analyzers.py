"""
Define Malnutrition analyzers
"""

import numpy as np
import starsim as ss
from tbsim import TB, TBS, Malnutrition, MicroNutrients, MacroNutrients, StudyArm
import sciris as sc

import pandas as pd

__all__ = ['HarlemAnalyzer']

class HarlemAnalyzer(ss.Analyzer):

    def __init__(self, **kwargs):
        self.requires = [TB, Malnutrition]
        self.data = []
        self.df = None # Created on finalize

        super().__init__(**kwargs)
        return

    def initialize(self, sim):
        super().initialize(sim)
        #self.results += ss.Result(self.name, 'n_recovered', sim.npts, dtype=int)
        self.initialized = True
        return

    def apply(self, sim):
        super().apply(sim)

        tb = self.sim.diseases['tb']
        nut = self.sim.diseases['malnutrition']
        ti = self.sim.ti
        for arm in [StudyArm.CONTROL, StudyArm.VITAMIN]:
            ppl = (self.sim.people.arm==arm) & (self.sim.people.alive)

            n_people = np.count_nonzero(ppl)
            new_infections = np.count_nonzero(tb.ti_infected[ppl] == ti)
            n_infected = np.count_nonzero(tb.infected[ppl])
            n_died = np.count_nonzero( (tb.ti_dead[(self.sim.people.arm==arm)] == ti) )
            n_latent_slow = np.count_nonzero(tb.state[ppl] == TBS.LATENT_SLOW)
            n_latent_fast = np.count_nonzero(tb.state[ppl] == TBS.LATENT_FAST)
            n_deficient = np.count_nonzero(nut.micro_state[ppl] == MicroNutrients.DEFICIENT)
            rel_LS_mean = tb.rel_LS_prog[ppl & tb.infected].mean()
            rel_LF_mean = tb.rel_LF_prog[ppl & tb.infected].mean()
            self.data.append([self.sim.year, arm.name, n_people, new_infections, n_infected, n_died,  n_latent_fast, n_latent_slow, n_deficient, rel_LF_mean, rel_LS_mean])
        return

    def finalize(self):
        super().finalize()
        self.df = pd.DataFrame(self.data, columns = ['year', 'arm', 'n_people', 'new_infections', 'n_infected', 'n_died', 'n_latent_fast', 'n_latent_slow', 'n_deficient', 'rel_LF_mean', 'rel_LS_mean'])

        self.df['cum_infections'] = self.df.groupby(['arm'])['new_infections'].cumsum()
        self.df.drop('new_infections', axis=1, inplace=True)

        self.df['cum_died'] = self.df.groupby(['arm'])['n_died'].cumsum()
        self.df.drop('n_died', axis=1, inplace=True)
        return

    def plot(self):
        import seaborn as sns

        d = pd.melt(self.df, id_vars=['year', 'arm'], var_name='channel', value_name='Value')
        g = sns.relplot(data=d, kind='line', x='year', hue='arm', col='channel', y='Value', palette='Set1', facet_kws={'sharey':False})

        return g.figure