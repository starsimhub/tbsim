
import numpy as np
import starsim as ss
import tbsim 

__all__ = ['HealthSeekingBehavior']

class HealthSeekingBehavior(ss.Intervention):
    """
    Eligible agents (symptomatic in LSHTM; smear+/smear-/EPTB in legacy TB) seek care
    at a rate per timestep. If the people object has a sought_care attribute, it is set
    to True for agents who seek care (for use by downstream interventions).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            initial_care_seeking_rate = ss.perday(0.1),
            care_seeking_dist         = ss.bernoulli(p=0),
            care_retry_steps          = None, 
            start                     = None,  # if not provided will take the same value as the simulation start date
            stop                      = None,  # if not provided will take the same value as the simulation stop date
            custom_states             = None, 
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(
            ss.IntArr('n_care_sought',       default=0),    # times sought in current episode; resets when leaving eligible states
            ss.IntArr('n_care_sought_total', default=0),    # lifetime count; never resets
            ss.FloatArr('ti_last_sought',    default=-np.inf),
        )
        self.care_seeking_dist = ss.bernoulli(p=self.pars.initial_care_seeking_rate.to_prob())  
    
    @property
    def tbsl(self):
        return tbsim.TBSL
    
    def init_post(self):
        super().init_post()
        tb = getattr(self.sim.diseases, 'tb_lshtm', None)
        if tb is None:
            raise ValueError("HealthSeekingBehavior requires a TB_LSHTM disease module named 'tb_lshtm'.")
        self._tb = tb
        
        if self.pars.custom_states is not None:
            self._states = np.asarray(self.pars.custom_states) 
            if not np.isin(self._states,    self.tbsl.care_seeking_eligible()).all():
                raise ValueError("Custom states must be a subset of the eligible states.")
        else:
            self._states = self.tbsl.care_seeking_eligible()
        
        self._new_seekers_count = 0
        if self.pars.start is None: self.pars.start = self.sim.t.start
        if self.pars.stop  is None: self.pars.stop  = self.sim.t.stop

    def step(self):
        sim = self.sim
        ppl = sim.people
        t = sim.now.date() 

        if t < self.pars.start.date() or t > self.pars.stop.date():
            self._new_seekers_count = 0
            return

        active = np.isin(self._tb.state, self._states) & ppl.alive
        # Reset episode counter when leaving eligible states so future episodes can seek care again.
        self.n_care_sought[~active] = 0
        if self.pars.care_retry_steps is not None and int(self.pars.care_retry_steps) > 0:
            can_retry = (self.ti - self.ti_last_sought) >= int(self.pars.care_retry_steps)
            eligible_for_seek = active & ((self.n_care_sought == 0) | can_retry)
        else:
            eligible_for_seek = active & (self.n_care_sought == 0)
        not_yet_sought = np.flatnonzero(eligible_for_seek)
        self._new_seekers_count = 0

        if len(not_yet_sought) == 0:
            return

        rate = self.pars.initial_care_seeking_rate
        self.pars.care_seeking_dist.set(p=rate.to_prob())
        seeking_uids = self.pars.care_seeking_dist.filter(ss.uids(not_yet_sought))

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        self.n_care_sought[seeking_uids] += 1
        self.n_care_sought_total[seeking_uids] += 1
        self.ti_last_sought[seeking_uids] = self.ti
        if hasattr(ppl, 'sought_care'):
            ppl.sought_care[seeking_uids] = True

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_sought_care',    dtype=int),
            ss.Result('n_sought_care',      dtype=int),
            ss.Result('n_ever_sought_care', dtype=int),
            ss.Result('n_eligible',         dtype=int),
        )

    def update_results(self):
        t = self.sim.now.date() 
        if t < self.pars.start.date() or t > self.pars.stop.date():
            return
        ppl = self.sim.people
        self.results['n_sought_care'][self.ti] = np.count_nonzero(self.n_care_sought > 0)
        self.results['n_ever_sought_care'][self.ti] = np.count_nonzero(self.n_care_sought_total > 0)
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        active = np.isin(self._tb.state, self._states) & ppl.alive
        self.results['n_eligible'][self.ti] = np.count_nonzero(active & (self.n_care_sought == 0))