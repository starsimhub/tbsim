"""Resistance analyzers and per-strain result tracking.

Analyzers expect a :class:`~tbsim.resistance.multistrain_tb.MultiStrainTB`
disease module with ``agent_strains`` configured.
"""

import numpy as np
import sciris as sc
import starsim as ss

__all__ = ['StrainResults', 'DuplicateStrainAnalyzer', 'ResistanceStats']


class StrainResults(ss.Analyzer):
    """
    Track per-strain prevalence and incidence channels.

    For each strain ``<uid>`` in the catalog, records:

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
        # Resolve strain catalog uids before super().init_pre() triggers
        # init_results() which depends on self._uids.
        tb = sim.diseases[self.disease]
        if getattr(tb, 'agent_strains', None) is None:
            raise RuntimeError(
                'StrainResults requires the TB disease module to have a strain overlay '
                'configured (MultiStrainTB(strains=[...])).'
            )
        self._uids = list(tb.agent_strains.catalog.uids)
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
        from ..tb import TBS
        tb = self.sim.diseases[self.disease]
        # Active-TB UIDs via Starsim UID operators (union of three BoolArr-ish queries)
        active_uids = ((tb.state == TBS.NON_INFECTIOUS) | (tb.state == TBS.ASYMPTOMATIC)
                       | (tb.state == TBS.SYMPTOMATIC)).uids
        ti = self.sim.ti
        for s_idx, uid in enumerate(self._uids):
            arr = getattr(tb, tb.agent_strains.names[s_idx])
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
    ``MultiStrainTB._n_duplicate_blocked_this_step`` that
    :class:`~tbsim.resistance.multistrain_tb.MultiStrainTB` increments inside
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
        if tb.agent_strains is None:
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


class ResistanceStats(ss.Analyzer):
    """
    Record ODE-facing aggregate resistance observables.

    This analyzer mirrors the compact output shape used by the two-strain ODE
    validation prototype while reading from the production ``MultiStrainTB``
    representation:

    - ``frac_resist``: fraction of active TB carrying any resistant strain.
    - ``frac_super``: fraction of active TB carrying two or more strains.
    - ``flux_denovo``: new resistance from random/de-novo acquisition.
    - ``flux_txacq``: new resistance from treatment or TPT-driven acquisition.
    - ``flux_transmitted``: new resistant strains acquired via transmission.

    Args:
        disease (str): TB disease module key. Default ``'tb'``.
    """

    def __init__(self, disease='tb', **kwargs):
        super().__init__(**kwargs)
        self.disease = disease
        return

    def init_pre(self, sim):
        tb = sim.diseases[self.disease]
        if getattr(tb, 'agent_strains', None) is None:
            raise RuntimeError(
                'ResistanceStats requires the TB disease module to have a strain overlay '
                'configured (MultiStrainTB(strains=[...])).'
            )
        super().init_pre(sim)
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('frac_resist', dtype=float, scale=False, label='Resistant fraction of active TB'),
            ss.Result('frac_super', dtype=float, scale=False, label='Superinfected fraction of active TB'),
            ss.Result('flux_denovo', dtype=int, label='New resistance: de-novo mutation'),
            ss.Result('flux_txacq', dtype=int, label='New resistance: treatment/TPT-acquired'),
            ss.Result('flux_transmitted', dtype=int, label='New resistance: transmitted'),
        )
        return

    def step(self):
        from ..tb import TBS
        tb = self.sim.diseases[self.disease]
        profile = tb.agent_strains
        active = ((tb.state == TBS.NON_INFECTIOUS) | (tb.state == TBS.ASYMPTOMATIC)
                  | (tb.state == TBS.SYMPTOMATIC)).uids
        ti = self.sim.ti

        if len(active):
            n_strains = profile.n_strains_per_agent(active)
            resistant = np.zeros(len(active), dtype=bool)
            for s_idx, name in enumerate(profile.names):
                if not profile.catalog.resistance[s_idx].any():
                    continue
                resistant |= np.asarray(getattr(tb, name)[active], dtype=bool)
            self.results.frac_resist[ti] = float(resistant.mean())
            self.results.frac_super[ti] = float((n_strains >= 2).mean())

        self.results.flux_denovo[ti]      = int(getattr(tb, '_n_denovo_resistance_this_step', 0))
        self.results.flux_txacq[ti]       = int(getattr(tb, '_n_txacq_resistance_this_step', 0))
        self.results.flux_transmitted[ti] = int(getattr(tb, '_n_transmitted_resistance_this_step', 0))
        return

    def to_df(self, sim):
        """Return recorded observables as a dataframe indexed by sim time."""
        res = sim.results[self.name]
        return sc.dataframe(
            time=sim.results.timevec,
            frac_resist=res.frac_resist,
            frac_super=res.frac_super,
            flux_denovo=res.flux_denovo,
            flux_txacq=res.flux_txacq,
            flux_transmitted=res.flux_transmitted,
        )
