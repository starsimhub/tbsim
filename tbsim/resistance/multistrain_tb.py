"""Multi-strain TB disease module for the drug-resistance overlay.

:class:`MultiStrainTB` subclasses base :class:`~tbsim.tb.TB` and adds per-agent
strain state, superinfection, progression/clearance hooks, and transmission
assignment. Use this instead of ``TB(strains=...)`` — base ``TB`` rejects the
``strains`` keyword.

Typical wiring::

    from tbsim.resistance import MultiStrainTB, StrainSpec, ResistanceConnector

    sim = ss.Sim(
        diseases=MultiStrainTB(strains=[...]),
        connectors=ResistanceConnector(),
    )
"""

import numpy as np
import starsim as ss

from ..tb import TB, TBS
from .resolvers import AcquisitionResolver, ProgressionResolver
from .strains import AgentStrains, StrainCatalog

__all__ = ['MultiStrainTB']


class MultiStrainTB(TB):
    """TB natural history with a multi-strain drug-resistance overlay.

    Extends :class:`~tbsim.tb.TB` with :attr:`agent_strains`, strain-aware
    :meth:`set_prognoses`, :meth:`transition`, :meth:`infect`, and
    :meth:`seed_strains`. Agent-level ``TBS`` states are unchanged; strains ride
    alongside as ``carries_<uid>`` :class:`starsim.BoolArr` arrays.

    Requires ``ResistanceConnector`` in ``sim.connectors`` for fitness-weighted
    transmission (a warning is emitted at :meth:`init_post` if missing).
    """

    _strain_kw = ('progression_mode', 'p_multi', 'p_random_acquisition',
                  'alpha_super', 'alpha_act')

    def __init__(self, pars=None, strains=None, **kwargs):
        """Initialize the strain-aware TB model.

        Args:
            pars (dict): Natural history parameter overrides.
            strains (list/StrainCatalog): Strain configuration.
            progression_mode (str): Progression bottleneck mode. Default ``'bottleneck'``.
            p_multi (float): Multi-strain retention probability at activation. Default 1.0.
            p_random_acquisition (dict/None): Per-drug random acquisition probabilities.
            alpha_super (float/None): Superinfection susceptibility in ``INFECTION``.
            alpha_act (dict/None): Per-state superinfection factors for active disease.
        """
        if strains is None:
            raise ValueError('MultiStrainTB requires a strain configuration.')
        strain_kwargs = {}
        for key in self._strain_kw:
            if key in kwargs:
                strain_kwargs[key] = kwargs.pop(key)
        kwargs.setdefault('name', 'tb')
        super().__init__(pars=pars, **kwargs)
        self._init_strains(strains, **strain_kwargs)
        return

    def _init_strains(self, strains, **kwargs):
        """Initialize strain catalog, per-agent strain states, and resolvers."""
        if isinstance(strains, StrainCatalog):
            catalog = strains
        else:
            catalog = StrainCatalog(list(strains))
        self._strain_catalog = catalog
        self.agent_strains = AgentStrains(catalog)
        self._rng_strain_pick = ss.random(name='tb_rng_strain_pick')
        self._rng_strain_init = ss.random(name='tb_rng_strain_init')
        self._n_duplicate_blocked_this_step = 0
        self._progression_resolver = ProgressionResolver(
            mode=kwargs.pop('progression_mode', 'bottleneck'),
            p_multi=kwargs.pop('p_multi', 1.0),
        )
        self._acquisition_resolver = AcquisitionResolver(
            p_random=kwargs.pop('p_random_acquisition', None),
        )
        self._alpha_super_input = kwargs.pop('alpha_super', None)
        self._alpha_act_input = dict(kwargs.pop('alpha_act', {}) or {})
        self._finalize_alpha_defaults()
        self.define_states(*self.agent_strains.state_defs())
        self.agent_strains.attach(self)
        return

    def init_post(self):
        """Seed initial infections and warn if the resistance connector is missing."""
        out = super().init_post()
        self._warn_if_missing_resistance_connector()
        return out

    def _warn_if_missing_resistance_connector(self):
        """Warn when strain fitness will not be applied to transmission."""
        sim = self.sim
        if sim is None:
            return
        for conn in sim.connectors.values():
            if conn.__class__.__name__ == 'ResistanceConnector':
                return
        ss.warn(
            'MultiStrainTB is configured but no ResistanceConnector was found '
            'in sim.connectors; strain fitness will not modify rel_trans.'
        )
        return

    def seed_strains(self, uids, sources=None):
        """Assign strain identity to ``uids`` using transmission/fallback rules.

        Used for immigrants and other non-network seed events where no
        infectious source is available.
        """
        uids = ss.uids(uids)
        if len(uids) == 0:
            return
        self._assign_transmitted_strains(uids, sources)
        return

    def _finalize_alpha_defaults(self):
        """Set ``self._alpha_super`` and ``self._alpha_act`` from stored inputs."""
        rr_rec = float(self.pars.rr_reinfection_rec)
        self._alpha_super = float(
            self._alpha_super_input if self._alpha_super_input is not None else rr_rec
        )
        self._alpha_act = dict(self._alpha_act_input)
        self._alpha_act.setdefault('non_infectious', self._alpha_super)
        self._alpha_act.setdefault('asymptomatic',   0.0)
        self._alpha_act.setdefault('symptomatic',    0.0)
        return

    def infect(self):
        """Run transmission with alpha-scaled superinfection targets."""
        per_state_alpha = (
            (TBS.INFECTION,      self._alpha_super),
            (TBS.NON_INFECTIOUS, self._alpha_act['non_infectious']),
            (TBS.ASYMPTOMATIC,   self._alpha_act['asymptomatic']),
            (TBS.SYMPTOMATIC,    self._alpha_act['symptomatic']),
        )
        chunks_uids = []
        chunks_alpha = []
        for tbs_val, alpha in per_state_alpha:
            if alpha <= 0:
                continue
            state_uids = (self.state == tbs_val).uids
            if len(state_uids) == 0:
                continue
            chunks_uids.append(state_uids)
            chunks_alpha.append(np.full(len(state_uids), float(alpha)))

        if not chunks_uids:
            return super().infect()

        elig_uids = ss.uids.concatenate(chunks_uids)
        alphas    = np.concatenate(chunks_alpha)
        orig_sus     = np.asarray(self.susceptible[elig_uids], dtype=bool).copy()
        orig_rel_sus = np.asarray(self.rel_sus[elig_uids], dtype=float).copy()
        try:
            self.susceptible[elig_uids] = True
            self.rel_sus[elig_uids] = orig_rel_sus * alphas
            return super().infect()
        finally:
            self.susceptible[elig_uids] = orig_sus
            self.rel_sus[elig_uids] = orig_rel_sus

    def set_prognoses(self, uids, sources=None):
        """Set prognoses for new infections and assign transmitted strains."""
        super(TB, self).set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        susceptible_uids = self.susceptible.uids
        cleared_uids     = (self.state == TBS.CLEARED).uids
        new_uids = uids.intersect(susceptible_uids.union(cleared_uids))

        if len(new_uids):
            self.susceptible[new_uids] = False
            self.infected[new_uids] = True
            self.ever_infected[new_uids] = True
            self.ti_infected[new_uids] = self.ti
            self.state[new_uids] = TBS.INFECTION

        self._assign_transmitted_strains(uids, sources)
        return

    def transition(self, uids, to, rng):
        """Apply TB transition, then apply strain progression hooks."""
        if len(uids) == 0:
            return super().transition(uids, to, rng)

        from_state = int(self.state[uids[0]])
        out = super().transition(uids, to, rng)

        if from_state == int(TBS.INFECTION):
            newly_cleared = uids[self.state[uids] == TBS.CLEARED]
            if len(newly_cleared):
                self.agent_strains.clear_all(newly_cleared)
            progressing = uids[np.isin(self.state[uids], [TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC])]
            self._apply_strain_progression(progressing, activating_only=False)
        elif from_state == int(TBS.NON_INFECTIOUS):
            newly_cleared = uids[self.state[uids] == TBS.CLEARED]
            if len(newly_cleared):
                self.agent_strains.clear_all(newly_cleared)
            progressing = uids[self.state[uids] == TBS.ASYMPTOMATIC]
            self._apply_strain_progression(progressing, activating_only=True)

        return out

    def step(self):
        """Reset strain counters, then advance the TB state machine."""
        self._n_duplicate_blocked_this_step = 0
        return super().step()

    def step_die(self, uids):
        """Apply TB death handling and clear carried strains."""
        out = super().step_die(uids)
        if len(uids):
            self.agent_strains.clear_all(uids)
        return out

    def _apply_strain_progression(self, uids, activating_only):
        """Apply strain progression rules after latent/active transitions.

        Args:
            uids            (ss.uids): Agents that just transitioned out of
                ``INFECTION`` or ``NON_INFECTIOUS``.
            activating_only (bool): If ``True``, run the activation bottleneck
                on all ``uids`` and skip random acquisition. If ``False``,
                run random acquisition on all ``uids`` and run the bottleneck
                only on agents now in ``ASYMPTOMATIC``.
        """
        if len(uids) == 0:
            return
        if activating_only:
            activating = uids
        else:
            activating = uids[self.state[uids] == TBS.ASYMPTOMATIC]
        if self._progression_resolver is not None and len(activating):
            self._progression_resolver.resolve(self.agent_strains, activating)
        if self._acquisition_resolver is not None and not activating_only:
            self._acquisition_resolver.random_acquisition(self.agent_strains, uids)
        return

    def _assign_transmitted_strains(self, uids, sources):
        """Assign strain identity to newly infected ``uids``."""
        profile = self.agent_strains
        catalog = self._strain_catalog
        n = len(uids)
        picks = np.full(n, -1, dtype=int)

        if sources is not None and not np.isscalar(sources):
            try:
                src_arr = np.asarray(sources, dtype=int)
            except (TypeError, ValueError):
                src_arr = None
            if src_arr is not None and src_arr.ndim == 1 and src_arr.size == n:
                real = src_arr >= 0
                if real.any():
                    src_uids = ss.uids(src_arr[real])
                    src_picks = profile.sample_transmitted_strain(
                        src_uids, self._rng_strain_pick,
                    )
                    picks[real] = src_picks

        need_fallback = picks < 0
        if need_fallback.any():
            init_prev = catalog.init_prev
            if init_prev.sum() > 0:
                p = init_prev / init_prev.sum()
                u = np.asarray(
                    self._rng_strain_init.rvs(int(need_fallback.sum())), dtype=float
                )
                cdf = np.cumsum(p)
                fallback_picks = (u[:, None] < cdf).argmax(axis=1)
                picks[need_fallback] = fallback_picks
            else:
                ss.warn(
                    'No StrainSpec init_prev weights configured; agents infected '
                    'without a transmission source will carry no strain until '
                    'acquisition or superinfection occurs.'
                )

        n_blocked = 0
        for idx in range(catalog.n):
            sel = picks == idx
            if not sel.any():
                continue
            target = uids[sel]
            arr = getattr(self, profile.names[idx])
            already = np.asarray(arr[target], dtype=bool)
            n_blocked += int(already.sum())
            new = target[~already]
            if len(new):
                arr[new] = True
        self._n_duplicate_blocked_this_step += n_blocked
        return
