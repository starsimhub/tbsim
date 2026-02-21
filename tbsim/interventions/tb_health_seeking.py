
import numpy as np
import starsim as ss
from tbsim import TBS, TB
from tbsim.tb_lshtm import TB_LSHTM, TB_LSHTM_Acute, TBSL

__all__ = ['HealthSeekingBehavior']

_DISEASE_ENUM = {
    TB_LSHTM_Acute: TBSL,
    TB_LSHTM:       TBSL,
    TB:             TBS,
}

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
            care_retry_steps          = None,   # the number of timesteps after which an agent can seek care again (e.g. 1 month)
            start                     = None, # if not provided will take the same value as the simulation start date
            stop                      = None,  # if not provided will take the same value as the simulation stop date
            custom_states             = None,  # The researcher can provide a different set of desired states (e.g. for ACF during the first year of the intervention)
        )
        self.update_pars(pars=pars, **kwargs)

        self.define_states(
            ss.IntArr('n_care_sought',       default=0),    # times sought in current episode; resets when leaving eligible states
            ss.IntArr('n_care_sought_total', default=0),    # lifetime count; never resets
            ss.FloatArr('ti_last_sought',    default=-np.inf),
        )
        self.care_seeking_dist = ss.bernoulli(p=0)  # Placeholder; auto-initialized by the framework

    def init_post(self):
        super().init_post()
        tb = getattr(self.sim.diseases, 'tb', None) or getattr(self.sim.diseases, 'tb_lshtm', None) or self.sim.diseases[0]
        if self.pars.custom_states is not None:
            self._states = np.asarray(self.pars.custom_states)
        else:
            enum = next((_DISEASE_ENUM[cls] for cls in type(tb).__mro__ if cls in _DISEASE_ENUM), None)
            if enum is None:
                raise ValueError("Could not infer care-seeking states. Provide `custom_states`.")
            self._states = enum.care_seeking_eligible()
            
        if self.pars.start is None: self.pars.start = self.sim.pars.start
        if self.pars.stop is None: self.pars.stop = self.sim.pars.stop
        self._tb = tb
        self._new_seekers_count = 0
        self._has_triggered_seek = False

    def step(self):
        sim = self.sim
        ppl = sim.people
        t   = sim.now

        if t < self.pars.start or t > self.pars.stop:
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
        self.care_seeking_dist.set(p=rate.to_prob())
        seeking_uids = self.care_seeking_dist.filter(ss.uids(not_yet_sought))

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        self._has_triggered_seek = True
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
        ppl = self.sim.people
        self.results['n_sought_care'][self.ti] = np.count_nonzero(self.n_care_sought > 0)
        self.results['n_ever_sought_care'][self.ti] = np.count_nonzero(self.n_care_sought_total > 0)
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        active = np.isin(self._tb.state, self._states) & ppl.alive
        self.results['n_eligible'][self.ti] = np.count_nonzero(active & (self.n_care_sought == 0))