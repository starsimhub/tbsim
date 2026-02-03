import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


#%% Analyzer to track age specific infections 
class AgeInfect(ss.Analyzer):

    def init_pre(self, sim):
        super().init_pre(sim)
        self.define_results(
            ss.Result('ninf_5', dtype=int, label='[0,5) Newly Infected'),
            ss.Result('ninf_5_6', dtype=int, label='[5,6) Newly Infected'),
            ss.Result('ninf_6_15', dtype=int, label='[6,15) Newly Infected'),
            ss.Result('ninf_15+', dtype=int, label='>=15 Newly Infected'),
            ss.Result('einf_5', dtype=int, label='[0,5) Ever Infected'),
            ss.Result('einf_5_6', dtype=int, label='[5,6) Ever Infected'),
            ss.Result('einf_6_15', dtype=int, label='[6,15) Ever Infected'),
            ss.Result('einf_15+', dtype=int, label='>=15 Ever Infected'),
            ss.Result('pop_5', dtype=int, label='[0,5) Alive'),
            ss.Result('pop_5_6', dtype=int, label='[5,6) Alive'),
            ss.Result('pop_6_15', dtype=int, label='[6,15) Alive'),
            ss.Result('pop_15+', dtype=int, label='>=15 Alive'),
        )
        return

    def step(self):
        ti = self.t.ti
        res = self.results
        ever_infected = self.sim.diseases.tb.ever_infected
        newly_infected = self.sim.diseases.tb.infected
        age = self.sim.people.age

        # Record age specific new infections
        res['ninf_5'][ti]  = np.count_nonzero(newly_infected[(age<5)])
        res['ninf_5_6'][ti]  = np.count_nonzero(newly_infected[(age>=5) & (age<6)])
        res['ninf_6_15'][ti] = np.count_nonzero(newly_infected[(age>=6) & (age<15)])
        res['ninf_15+'][ti] = np.count_nonzero(newly_infected[(age>=15)])
        # Record age specific ever infected
        res['einf_5'][ti]  = np.count_nonzero(ever_infected[(age<5)])
        res['einf_5_6'][ti]  = np.count_nonzero(ever_infected[(age>=5) & (age<6)])
        res['einf_6_15'][ti] = np.count_nonzero(ever_infected[(age>=6) & (age<15)])
        res['einf_15+'][ti]   = np.count_nonzero(ever_infected[(age>=15)])
        # Record age specific population
        res['pop_5'][ti]  = np.count_nonzero((age<5))
        res['pop_5_6'][ti]  = np.count_nonzero((age>=5) & (age<6))
        res['pop_6_15'][ti] = np.count_nonzero((age>=6) & (age<15))
        res['pop_15+'][ti]   = np.count_nonzero((age>=15))
        return



class prev_by_age(ss.Analyzer):
    """ Record prevalence by age (ever, infected, active) at one point in time (year) """
    def __init__(self, year, **kwargs):
        self.year = year
        super().__init__(**kwargs)
        return
    
    def step(self):
        if self.year >= self.t.now('year') and  self.year < self.t.now('year')+self.t.dt_year:
            self.age_bins = np.arange(0, 101, 5)
            self.n, _        = np.histogram(self.sim.people.age, bins=self.age_bins)
            self.ever, _     = np.histogram(self.sim.people.age[self.sim.diseases.tb.ever_infected], bins=self.age_bins)
            self.infected, _ = np.histogram(self.sim.people.age[self.sim.diseases.tb.infected], bins=self.age_bins)
            self.active, _   = np.histogram(self.sim.people.age[self.sim.diseases.tb.infectious], bins=self.age_bins)
        return


class state_transition(ss.Analyzer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cur_state = None
        self.prev_ti = None
        return

    def init_post(self):
        super().init_post()
        self.cur_state = pd.Series(self.sim.diseases.tb.state.values, index=self.sim.people.auids.to_numpy())
        self.prev_t = self.t.now()
        self.transitions = pd.DataFrame({
            'uid': self.sim.people.auids,
            'to_state': self.sim.diseases.tb.state.values,
        }).set_index('uid')
        self.transitions['t'] = self.prev_t
        self.transitions['t_enter'] = self.prev_t
        self.transitions['from_state'] = 'INIT'
        self.transitions['dwell'] = np.nan

    def step(self):
        auids = self.sim.people.auids
        died_uids = np.setdiff1d(self.cur_state.index, auids)

        dfs = []
        if len(died_uids):
            t_enter = self.transitions.loc[died_uids].groupby('uid')['t'].last()
            df = pd.DataFrame({
                'uid': died_uids,
                'from_state': self.cur_state.loc[died_uids],
                't_enter': t_enter,
                'dwell': [self.t.now() - t for t in t_enter.values]
            }).set_index('uid')
            df['t'] = self.t.now()
            df['to_state'] = 'DEAD'
            dfs.append(df)

        born_uids = np.setdiff1d(auids, self.cur_state.index)
        if len(born_uids):
            df = pd.DataFrame({
                'uid': born_uids,
                'to_state': self.sim.diseases.tb.state[born_uids],
            }).set_index('uid')
            df['t'] = self.t.now()
            df['t_enter'] = self.t.now()
            df['dwell'] = np.nan
            df['from_state'] = 'BORN'
            dfs.append(df)

        match_uids = ss.uids(np.intersect1d(auids, self.cur_state.index))
        change_uids = match_uids[self.sim.diseases.tb.state[match_uids] != self.cur_state.loc[match_uids]]
        if len(change_uids):
            # Find the last ti for each uid
            t_enter = self.transitions.loc[change_uids].groupby('uid')['t'].last()
            df = pd.DataFrame({
                'uid': change_uids,
                'from_state': self.cur_state.loc[change_uids],
                'to_state': self.sim.diseases.tb.state[change_uids],
                't_enter': t_enter,
                'dwell': [self.t.now() - t for t in t_enter.values],
            }).set_index('uid')
            df['t'] = self.t.now()
            dfs.append(df)

        self.transitions = pd.concat([self.transitions] + dfs, ignore_index=False)
        self.cur_state = pd.Series(self.sim.diseases.tb.state.values, index=self.sim.people.auids.to_numpy())

        return

    def finalize(self):
        t_enter = self.transitions.loc[self.cur_state.index.values].groupby('uid')['t'].last()
        df = pd.DataFrame({
            'uid': self.cur_state.index,
            'from_state': self.cur_state,
            't_enter': t_enter,
            'dwell': [self.t.now() - t for t in t_enter.values],
        }).set_index('uid')
        df['to_state']: 'END'
        df['t'] = self.t.now()

        self.transitions = pd.concat([self.transitions, df], ignore_index=False)

        return
