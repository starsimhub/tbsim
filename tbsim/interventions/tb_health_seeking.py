"""HealthSeekingBehavior: eligible active TB cases seek care and start treatment."""

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
    at a rate per timestep and start treatment. Requires TBPeople (sought_care state).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            initial_care_seeking_rate=ss.perday(0.1),
            single_use=True,
            start=None,
            stop=None,
        )
        self.update_pars(pars=pars, **kwargs)
        self._new_seekers_count = 0

    def init_post(self):
        super().init_post()
        tb = getattr(self.sim.diseases, 'tb', None) or getattr(self.sim.diseases, 'tb_lshtm', None) or self.sim.diseases[0]
        for cls in type(tb).__mro__:
            if cls in _DISEASE_ENUM:
                self._states = getattr(_DISEASE_ENUM[cls], 'care_seeking_eligible')()
                break
        self._tb = tb

    def step(self):
        sim = self.sim
        ppl = sim.people
        t = sim.now
        if self.pars.start is not None and t < self.pars.start:
            self._new_seekers_count = 0
            return
        if self.pars.stop is not None and t > self.pars.stop:
            self._new_seekers_count = 0
            return

        active = np.isin(self._tb.state, self._states) & ppl.alive
        not_yet_sought = np.flatnonzero(active & ~ppl.sought_care)
        self._new_seekers_count = 0

        if len(not_yet_sought) == 0:
            return

        rate = self.pars.initial_care_seeking_rate
        p = rate.to_prob()
        dist = ss.bernoulli(p=p)
        dist.init(trace=self.name or 'HealthSeekingBehavior.care_seeking', sim=sim, module=self)
        seeking_uids = dist.filter(ss.uids(not_yet_sought))

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        ppl.sought_care[seeking_uids] = True
        self._tb.start_treatment(seeking_uids)
        if self.pars.single_use:
            self.expired = True

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_sought_care', dtype=int),
            ss.Result('n_sought_care', dtype=int),
            ss.Result('n_eligible', dtype=int),
        )

    def update_results(self):
        ppl = self.sim.people
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        self.results['n_sought_care'][self.ti] = np.count_nonzero(ppl.sought_care)
        active = np.isin(self._tb.state, self._states) & ppl.alive
        self.results['n_eligible'][self.ti] = np.count_nonzero(active & ~ppl.sought_care)
