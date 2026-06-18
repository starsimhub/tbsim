"""
Strain-level resolvers used by :class:`tbsim.TB` and strain-aware interventions.

These are lightweight helper classes that operate on a TB module's
:class:`StrainProfile` to apply the spec's progression and acquisition rules.
They do not hold state of their own and are safe to instantiate per call.
"""

import numpy as np
import starsim as ss

__all__ = ['ProgressionResolver', 'AcquisitionResolver']


class ProgressionResolver:
    """
    Decide which strain(s) progress when an agent activates (``INFECTION`` or
    ``NON_INFECTIOUS`` → ``ASYMPTOMATIC``).

    Modes:

    - ``'all'``: all carried strains remain; no bottleneck. Default if
      ``p_multi == 1``.
    - ``'bottleneck'``: with probability ``1 - p_multi`` the agent retains a
      single strain at activation, sampled **uniformly at random** among
      the carried strains. With probability ``p_multi`` all strains remain.
      Per the spec we *do not* apply fitness costs on progression — only
      a transmission bottleneck.

    The resolver is a pure helper; it mutates the supplied
    :class:`StrainProfile` in place.
    """

    MODES = ('all', 'bottleneck')

    def __init__(self, mode='bottleneck', p_multi=1.0, rng=None):
        if mode not in self.MODES:
            raise ValueError(f'ProgressionResolver mode must be in {self.MODES}; got {mode!r}')
        if not (0.0 <= float(p_multi) <= 1.0):
            raise ValueError(f'ProgressionResolver p_multi must be in [0, 1]; got {p_multi!r}')
        self.mode = mode
        self.p_multi = float(p_multi)
        self._rng = rng if rng is not None else ss.random(
            name='resmod_rng_progression', strict=False,
        )
        if not self._rng.initialized:
            self._rng.init()
        return

    def resolve(self, profile, uids):
        """
        Apply the bottleneck (in-place) to *uids* that just activated.

        Args:
            profile (StrainProfile): The TB strain profile.
            uids (ss.uids): UIDs that just activated to ``ASYMPTOMATIC``.

        Returns:
            np.ndarray of int: For each uid, the (single) strain index that
            remained after the bottleneck, or -1 if the agent ended with no
            strain (only possible for ``'all'`` mode with a stray no-strain agent).
        """
        n = len(uids)
        out = np.full(n, -1, dtype=int)
        if n == 0 or self.mode == 'all' or self.p_multi >= 1.0:
            return out

        # Count strains per agent: only those carrying >1 are bottleneck candidates
        counts = profile.n_strains_per_agent(uids)
        multi = counts > 1
        if not multi.any():
            return out

        # Bernoulli(1 - p_multi) per multi-strain agent decides who funnels
        u = np.asarray(self._rng.rvs(n), dtype=float)
        funnel_mask = multi & (u >= self.p_multi)
        if not funnel_mask.any():
            return out

        funnel_uids = uids[funnel_mask]

        # Spec: equal probability across carried strains (no fitness cost on
        # progression). Build per-funnel carrier mask (no fitness weighting).
        registry = profile.registry
        ns = registry.n
        weights = np.zeros((len(funnel_uids), ns), dtype=float)
        for idx, name in enumerate(profile.names):
            carriers = np.asarray(getattr(profile._tb, name)[funnel_uids], dtype=bool)
            weights[:, idx] = carriers.astype(float)
        totals = weights.sum(axis=1)

        # Sample via CDF
        u2 = np.asarray(self._rng.rvs(len(funnel_uids)), dtype=float)
        cdf = np.cumsum(weights / totals[:, None], axis=1)
        picks = (u2[:, None] < cdf).argmax(axis=1)

        # Apply: clear all strains, then restore the pick
        profile.clear_all(funnel_uids)
        for idx in range(ns):
            sel = picks == idx
            if sel.any():
                target = funnel_uids[sel]
                getattr(profile._tb, profile.names[idx])[target] = True
        out[funnel_mask] = picks
        return out


class AcquisitionResolver:
    """
    Resistance acquisition events for the TB resistance overlay.

    Two pathways:

    - :meth:`random_acquisition`: low-rate background mutation, evaluated at
      transition from ``INFECTION`` to ``NON_INFECTIOUS`` or ``ASYMPTOMATIC``.
    - :meth:`selective_acquisition`: high-rate acquisition during failed
      treatment or TPT, evaluated per surviving susceptible strain at
      treatment-outcome resolution.

    Acquisition replaces a susceptible strain with the corresponding
    resistant strain whose phenotype differs by exactly one bit (the acquired
    drug). If no such target strain exists in the registry the acquisition
    event is silently dropped (resistant variant is not configured).
    """

    def __init__(self, p_random=None, p_selective=None, rng=None):
        """
        Args:
            p_random (dict): Per-drug random acquisition probability per
                activation transition. Default {} (off).
            p_selective (dict): Per-drug acquisition probability per
                treatment-failure episode. Default {} (off).
            rng: Optional ``ss.random`` to use; otherwise one is created.
        """
        self.p_random = dict(p_random or {})
        self.p_selective = dict(p_selective or {})
        self._rng = rng if rng is not None else ss.random(
            name='resmod_rng_acquisition', strict=False,
        )
        if not self._rng.initialized:
            self._rng.init()
        return

    def _resistant_target_idx(self, registry, source_idx, drug):
        """Find the strain index that equals source's phenotype + resistance to *drug*.

        Returns ``None`` if no such strain is configured.
        """
        if drug not in registry.drugs:
            return None
        drug_col = registry.drugs.index(drug)
        src_phen = registry.resistance[source_idx]
        # Already resistant to this drug -> no-op
        if src_phen[drug_col] == 1:
            return None
        target_phen = src_phen.copy()
        target_phen[drug_col] = 1
        # Find a strain whose phenotype equals target_phen
        match = np.all(registry.resistance == target_phen, axis=1)
        idxs = np.where(match)[0]
        if len(idxs) == 0:
            return None
        return int(idxs[0])

    def random_acquisition(self, profile, uids):
        """Per-spec background acquisition at progression.

        Spec semantics:

        - Evaluated **once per (agent, carried strain, drug)** at transition
          from ``INFECTION`` to ``NON_INFECTIOUS`` or ``ASYMPTOMATIC``.
        - A susceptible carried strain that hits its draw acquires a new
          resistant variant; **the original strain is retained** (i.e.,
          the agent ends multi-strain rather than the resistant variant
          replacing the source strain).
        - A strain already resistant to drug *d* gets no trial for *d*.

        Args:
            profile (StrainProfile): The TB strain profile.
            uids (ss.uids): Agents that just progressed.
        """
        if len(uids) == 0 or not self.p_random:
            return
        registry = profile.registry
        for drug, p in self.p_random.items():
            if p <= 0 or drug not in registry.drugs:
                continue
            drug_col = registry.drugs.index(drug)
            # For each strain susceptible to this drug, roll one Bernoulli per
            # carrier and ADD the resistant variant to those who hit.
            for s_idx in range(registry.n):
                if registry.resistance[s_idx, drug_col] == 1:
                    continue  # already resistant -> skip per spec
                target_idx = self._resistant_target_idx(registry, s_idx, drug)
                if target_idx is None:
                    continue
                carriers_mask = np.asarray(
                    getattr(profile._tb, profile.names[s_idx])[uids], dtype=bool,
                )
                if not carriers_mask.any():
                    continue
                carrier_uids = uids[carriers_mask]
                u = np.asarray(self._rng.rvs(len(carrier_uids)), dtype=float)
                hit = carrier_uids[u < float(p)]
                if len(hit) == 0:
                    continue
                # ADD the resistant variant (does not remove the original).
                profile.add_strain(hit, target_idx)
        return

    def selective_acquisition(self, profile, uids, drugs_used):
        """Acquire resistance on treatment failure for surviving susceptible strains.

        Args:
            profile (StrainProfile): Strain profile to mutate.
            uids (ss.uids): Agents who failed treatment.
            drugs_used (list[str]): Drugs in the failed regimen.
        """
        if len(uids) == 0 or not self.p_selective:
            return
        for drug in drugs_used:
            p = self.p_selective.get(drug, 0.0)
            if p <= 0:
                continue
            u = np.asarray(self._rng.rvs(len(uids)), dtype=float)
            hit = uids[u < float(p)]
            if len(hit) == 0:
                continue
            self._apply_acquisition(profile, hit, drug)
        return

    def _apply_acquisition(self, profile, uids, drug):
        """For each *uid* carrying a strain susceptible to *drug*, replace one such
        strain with its resistant counterpart. Multi-strain agents pick the
        first susceptible carrier by registry order.
        """
        registry = profile.registry
        if drug not in registry.drugs:
            return
        drug_col = registry.drugs.index(drug)

        # Identify, per uid, the first susceptible carried strain.
        # We iterate strain index in order — first hit wins.
        applied = np.zeros(len(uids), dtype=bool)
        for s_idx in range(registry.n):
            if registry.resistance[s_idx, drug_col] == 1:
                continue  # already resistant
            target_idx = self._resistant_target_idx(registry, s_idx, drug)
            if target_idx is None:
                continue
            carriers = np.asarray(
                getattr(profile._tb, profile.names[s_idx])[uids], dtype=bool
            )
            do = carriers & ~applied
            if not do.any():
                continue
            sub = uids[do]
            profile.replace_strain(sub, old=s_idx, new=target_idx)
            applied |= do
            if applied.all():
                break
        return
