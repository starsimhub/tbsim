"""
Define Malnutrition analyzers
"""

import numpy as np
import starsim as ss
from tbsim import TB, TBS, Malnutrition, eMicroNutrients, eMacroNutrients, eBmiStatus
import networkx as nx
import pandas as pd
from enum import IntEnum, auto


__all__ = ['RationsAnalyzer', 'GenHHAnalyzer', 'GenNutritionAnalyzer']
class StudyArm(IntEnum):
    CONTROL = auto()
    VITAMIN = auto()
    
class RationsAnalyzer(ss.Analyzer):

    def __init__(self, **kwargs):
        self.requires = [TB, Malnutrition]
        self.data = []
        self.df = None # Created on finalize

        super().__init__(**kwargs)
        return

    def init_results(self):
        super().init_results()
        #self.results += ss.Result(self.name, 'n_recovered', sim.npts, dtype=int)
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
            new_active_infections = np.count_nonzero(tb.ti_active[ppl] == ti)
            n_infected = np.count_nonzero(tb.infected[ppl])
            n_died = np.count_nonzero( (tb.ti_dead[(self.sim.people.arm==arm)] == ti) )
            n_latent_slow = np.count_nonzero(tb.state[ppl] == TBS.LATENT_SLOW)
            n_latent_fast = np.count_nonzero(tb.state[ppl] == TBS.LATENT_FAST)
            n_micro_deficient = np.count_nonzero(nut.micro_state[ppl] == eMicroNutrients.DEFICIENT)
            n_macro_deficient = np.count_nonzero( (nut.macro_state[ppl] == eMacroNutrients.UNSATISFACTORY) | (nut.macro_state[ppl] == eMacroNutrients.MARGINAL) )
            n_bmi_deficient = np.count_nonzero( (nut.bmi_state[ppl] < eBmiStatus.NORMAL_WEIGHT))
            infected = ppl & tb.infected
            if not infected.any():
                rel_LS_mean = np.nan
                rel_LF_mean = np.nan
            else:
                rel_LS_mean = tb.rel_LS_prog[infected].mean()
                rel_LF_mean = tb.rel_LF_prog[ppl & tb.infected].mean()

            self.data.append([self.sim.year, 
                              arm.name, 
                              n_people, 
                              new_infections, 
                              new_active_infections, 
                              n_infected, 
                              n_died, 
                              n_latent_slow, 
                              n_latent_fast, 
                              n_micro_deficient, 
                              n_macro_deficient, 
                              n_bmi_deficient,
                              rel_LS_mean, 
                              rel_LF_mean])
        return

    def finalize(self):
        super().finalize()
        self.df = pd.DataFrame(
                        self.data, 
                        columns=[
                            'year', 
                            'arm', 
                            'n_people', 
                            'new_infections', 
                            'new_active_infections', 
                            'n_infected', 
                            'n_died', 
                            'n_latent_slow', 
                            'n_latent_fast',
                            'n_micro_deficient',            # Number of people with micro nutrient deficiency
                            'n_macro_deficient',            # Number of people with macro nutrient deficiency
                            'n_bmi_deficient',              # Number of people with BMI deficiency
                            'rel_LS_mean', 
                            'rel_LF_mean'
                        ]
                    )

        self.df['cum_infections'] = self.df.groupby(['arm'])['new_infections'].cumsum()
        self.df.drop('new_infections', axis=1, inplace=True)

        self.df['cum_active_infections'] = self.df.groupby(['arm'])['new_active_infections'].cumsum()
        self.df.drop('new_active_infections', axis=1, inplace=True)

        self.df['cum_died'] = self.df.groupby(['arm'])['n_died'].cumsum()
        self.df.drop('n_died', axis=1, inplace=True)
        return

    def plot(self):
        import seaborn as sns

        d = pd.melt(self.df, id_vars=['year', 'arm'], var_name='channel', value_name='Value')
        g = sns.relplot(data=d, kind='line', x='year', hue='arm', col='channel', y='Value', palette='Set1', facet_kws={'sharey':False})

        return g.figure

class GenHHAnalyzer(ss.Analyzer):

    def __init__(self, **kwargs):
        self.requires = [TB, Malnutrition]
        self.data = []
        self.df = None # Created on finalize

        super().__init__(**kwargs)
        return

    def init_results(self):
        super().init_results()
        return

    def apply(self, sim, snap_years = [2017, 2021]):
        super().apply(sim)

        year = self.sim.year
        dt = self.sim.dt

        snap = False
        for sy in snap_years:
            if year >= sy and year < sy+dt:
                snap = True
                break
        
        if not snap:
            return

        hhid, hh_sizes = np.unique(sim.people.hhid, return_counts=True)
        cnt, hh_size = np.histogram(hh_sizes, bins=range(1, 11))

        #hhn = self.sim.networks['Rationsnet']
        #el = [(p1, p2) for p1,p2 in zip(hhn.edges['p1'], hhn.edges['p2'])]
        #G = nx.from_edgelist(el)
        #hh_sizes = np.array([len(c) for c in nx.connected_components(G)])
        #cnt, hh_size = np.histogram(hh_sizes, bins=range(20))

        df = pd.DataFrame({sy:cnt}, index=pd.Index(hh_size[:-1], name='HH Size'))
        self.data.append(df)
        return

    def finalize(self):
        super().finalize()
        self.df = pd.concat(self.data, axis=1)
        return

class GenNutritionAnalyzer(ss.Analyzer):

    def __init__(self, **kwargs):
        self.requires = [TB, Malnutrition]
        self.data = []
        self.df = None # Created on finalize

        super().__init__(**kwargs)
        return

    def apply(self, sim, snap_years = [2017, 2021]):
        super().apply(sim)

        year = self.sim.year
        dt = self.sim.dt

        snap = False
        for sy in snap_years:
            if year >= sy and year < sy+dt:
                snap = True
                break
        
        if not snap:
            return

        macro_lookup = {eMacroNutrients[name].value: name for name in eMacroNutrients._member_names_}
        micro_lookup = {eMicroNutrients[name].value: name for name in eMicroNutrients._member_names_}
        arm_lookup = {StudyArm[name].value: name for name in StudyArm._member_names_}

        nut = self.sim.diseases['malnutrition']
        ppl = self.sim.people
        df = pd.DataFrame({
            'Macro': [macro_lookup[v] for v in nut.macro_state.values],
            'Micro': [micro_lookup[v] for v in nut.micro_state.values],
            'Arm': [arm_lookup[v] for v in ppl.arm.values],
        }, index=pd.Index(ppl.uid))

        sz = df.groupby(['Arm', 'Macro', 'Micro']).size()
        sz.name = str(sy)
        self.data.append(sz)
        return

    def finalize(self):
        super().finalize()
        self.df = pd.concat(self.data, axis=1)
        return