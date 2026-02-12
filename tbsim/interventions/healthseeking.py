import numpy as np
import starsim as ss
from tbsim import TBS, TBSL, TB_LSHTM, TB_LSHTM_Acute

__all__ = ['HealthSeekingBehavior']


def _get_tb(sim):
    """Return the TB disease module (sim.diseases.tb or first disease when only one)."""
    return getattr(sim.diseases, 'tb', None) or (sim.diseases[0] if len(sim.diseases) else None)


def _active_tb_eligible(tb):
    """Return boolean state array of agents with active TB eligible for care-seeking (and start_treatment)."""
    if isinstance(tb, (TB_LSHTM, TB_LSHTM_Acute)):
        return (
            (tb.state == TBSL.UNCONFIRMED)
            | (tb.state == TBSL.ASYMPTOMATIC)
            | (tb.state == TBSL.SYMPTOMATIC)
            | (tb.state == TBSL.ACUTE)
        )
    return (
        (tb.state == TBS.ACTIVE_SMPOS)
        | (tb.state == TBS.ACTIVE_SMNEG)
        | (tb.state == TBS.ACTIVE_EXPTB)
    )


class HealthSeekingBehavior(ss.Intervention):
    """
    Care-seeking for active TB; calls tb.start_treatment on those who seek care.
    Works with TB (TBS) and LSHTM (TB_LSHTM / TB_LSHTM_Acute, TBSL).

    Parameters:
        prob (float or ss.Dist): Probability per unit time. Ignored if initial_care_seeking_rate is set.
        initial_care_seeking_rate: Optional rate (e.g. ss.perday(0.1)). If set, overrides prob.
        single_use (bool): Expire after first care-seeking.
        start, stop: Optional time window.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            prob=0.1,
            initial_care_seeking_rate=None,
            single_use=True,
            start=None,
            stop=None,
        )
        self.update_pars(pars=pars, **kwargs)
        self.define_states(ss.BoolArr('sought_care', default=False))
        self._new_seekers_count = 0

    def step(self):
        sim = self.sim
        t = sim.now
        tb = _get_tb(sim)
        if tb is None:
            return
        if self.pars.start is not None and t < self.pars.start:
            return
        if self.pars.stop is not None and t > self.pars.stop:
            return

        active_tb = _active_tb_eligible(tb)
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self._new_seekers_count = 0

        if len(not_yet_sought) == 0:
            return

        if self.pars.initial_care_seeking_rate is not None:
            p = self.pars.initial_care_seeking_rate.to_prob()
            probs = np.full(len(not_yet_sought), p) if np.isscalar(p) else np.asarray(p)
            seeking_uids = not_yet_sought[np.random.rand(len(not_yet_sought)) < probs]
        else:
            if isinstance(self.pars.prob, ss.Dist):
                dist = self.pars.prob
            else:
                dist = ss.bernoulli(p=self.pars.prob)
            dist.init(sim)
            seeking_uids = dist.filter(not_yet_sought)

        if len(seeking_uids) == 0:
            return
        self._new_seekers_count = len(seeking_uids)
        self.sought_care[seeking_uids] = True
        tb.start_treatment(seeking_uids)
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
        sim = self.sim
        tb = _get_tb(sim)
        active_tb = _active_tb_eligible(tb)
        active_uids = active_tb.uids
        not_yet_sought = active_uids[~self.sought_care[active_uids]]
        self.results['new_sought_care'][self.ti] = self._new_seekers_count
        self.results['n_sought_care'][self.ti] = np.count_nonzero(self.sought_care)
        self.results['n_eligible'][self.ti] = len(not_yet_sought)

