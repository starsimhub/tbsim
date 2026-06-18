"""Resistance analyzers and per-strain result tracking."""

import numpy as np
import starsim as ss
import tbsim

__all__ = ['StrainResults', 'DuplicateStrainAnalyzer']


class StrainResults(ss.Analyzer):
    """
    Track per-strain prevalence and incidence channels.

    For each strain ``<uid>`` in the registry, records:

    - ``n_carriers_<uid>``: number of agents currently carrying the strain.
    - ``n_active_<uid>``: carriers in any active TB state.
    - ``new_carriers_<uid>``: agents that gained the strain this step.

    Args:
        disease (str): TB disease module key. Default ``'tb'``.
    """

    def __init__(self, disease='tb', **kwargs):
        super().__init__(**kwargs)
        self.disease = disease
        self._prev_carriers = None
        return

    def init_pre(self, sim):
        # Resolve strain registry uids before super().init_pre() triggers
        # init_results() which depends on self._uids.
        tb = sim.diseases[self.disease]
        if tb.strain_profile is None:
            raise RuntimeError(
                'StrainResults requires the TB disease module to have a strain overlay '
                'configured (TB(strains=[...])).'
            )
        self._uids = list(tb.strain_profile.registry.uids)
        super().init_pre(sim)
        return

    def init_results(self):
        super().init_results()
        results = []
        for uid in self._uids:
            results.append(ss.Result(f'n_carriers_{uid}', dtype=int))
            results.append(ss.Result(f'n_active_{uid}', dtype=int))
            results.append(ss.Result(f'new_carriers_{uid}', dtype=int))
        self.define_results(*results)
        self._prev_carriers = {uid: ss.uids() for uid in self._uids}
        return

    def step(self):
        tb = self.sim.diseases[self.disease]
        from tbsim import TBS
        # Active-TB UIDs via Starsim UID operators (union of three BoolArr-ish queries)
        active_uids = ((tb.state == TBS.NON_INFECTIOUS) | (tb.state == TBS.ASYMPTOMATIC)
                       | (tb.state == TBS.SYMPTOMATIC)).uids
        ti = self.sim.ti
        for s_idx, uid in enumerate(self._uids):
            arr = getattr(tb, tb.strain_profile.names[s_idx])
            carrier_uids = arr.uids
            n_carry = len(carrier_uids)
            n_active = len(carrier_uids.intersect(active_uids))
            prev = self._prev_carriers[uid]
            new_carry = len(carrier_uids.remove(prev))
            self.results[f'n_carriers_{uid}'][ti] = n_carry
            self.results[f'n_active_{uid}'][ti] = n_active
            self.results[f'new_carriers_{uid}'][ti] = new_carry
            self._prev_carriers[uid] = ss.uids(carrier_uids)
        return


class DuplicateStrainAnalyzer(ss.Analyzer):
    """
    Count duplicate-strain superinfection events (Decision 3 in the findings).

    Reads the precise per-step counter
    ``tb._n_duplicate_blocked_this_step`` that ``TB`` increments inside
    ``_assign_transmitted_strains`` each time a transmitted strain is
    silently dropped because the recipient already carries it.

    Use this to decide whether duplicate-strain blocking introduces material
    bias toward rare resistant strains. If ``n_duplicate_blocked`` is a
    significant fraction of ``new_active`` over a calibration window, the
    team should consider switching to a count-based representation rather
    than the current presence/absence model.
    """

    def __init__(self, disease='tb', **kwargs):
        super().__init__(**kwargs)
        self.disease = disease
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_duplicate_blocked', dtype=int),
            ss.Result('cum_duplicate_blocked', dtype=int),
        )
        return

    def step(self):
        tb = self.sim.diseases[self.disease]
        if tb.strain_profile is None:
            return
        ti = self.sim.ti
        n_blocked = int(getattr(tb, '_n_duplicate_blocked_this_step', 0))
        self.results['n_duplicate_blocked'][ti] = n_blocked
        return

    def finalize_results(self):
        super().finalize_results()
        self.results['cum_duplicate_blocked'][:] = np.cumsum(
            self.results['n_duplicate_blocked'][:]
        )
        return
