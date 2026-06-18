"""Per-agent strain presence state for the TB resistance overlay."""

import numpy as np
import starsim as ss

__all__ = ['StrainProfile']


class StrainProfile:
    """
    Per-agent strain presence state attached to a :class:`tbsim.TB` instance.

    For each strain in the registry, allocates one :class:`ss.BoolArr` named
    ``carries_<uid>``. The arrays are defined on TB via
    ``tb.define_states(...)`` so Starsim handles agent growth automatically.

    All agent indexing goes through Starsim's ``ss.uids`` /
    ``BoolArr.uids`` / ``intersect()`` APIs rather than raw NumPy.

    Args:
        registry (StrainRegistry): The strain registry.

    Example:
        ::

            profile = StrainProfile(registry)
            profile.attach(tb)
            profile.add_strain(uids, 'pan')
    """

    PREFIX = 'carries_'

    def __init__(self, registry):
        self.registry = registry
        self.names = [f'{self.PREFIX}{uid}' for uid in registry.uids]
        self._tb = None
        return

    def state_defs(self):
        """Return ``ss.BoolArr`` definitions to be added to TB.define_states()."""
        return [ss.BoolArr(name, default=False) for name in self.names]

    def attach(self, tb):
        """Attach to a TB module after its states have been defined."""
        self._tb = tb
        for name in self.names:
            if not hasattr(tb, name):
                raise RuntimeError(
                    f'StrainProfile state {name!r} not found on TB; '
                    f'did you forget to include profile.state_defs() in define_states()?'
                )
        return

    # ---- access helpers ----

    def _arr(self, strain):
        """Return the ``ss.BoolArr`` for *strain* (by uid or idx)."""
        if self._tb is None:
            raise RuntimeError('StrainProfile is not attached to a TB module.')
        return getattr(self._tb, self.names[self._to_idx(strain)])

    def _to_idx(self, strain):
        if isinstance(strain, str):
            return self.registry.index(strain)
        return int(strain)

    def carries(self, strain, uids=None):
        """
        Return boolean mask of agents carrying *strain*.

        Args:
            strain (str/int): Strain uid or index.
            uids (ss.uids/None): If given, restrict to these UIDs (UID-indexed).

        Returns:
            np.ndarray of bool indexed positionally by *uids* (when provided)
            or by active-agent position (when ``uids is None``).
        """
        arr = self._arr(strain)
        if uids is None:
            return np.asarray(arr.values, dtype=bool)
        return np.asarray(arr[uids], dtype=bool)

    def carriers(self, strain):
        """Return ``ss.uids`` of currently active agents carrying *strain*."""
        return self._arr(strain).uids

    def n_strains_per_agent(self, uids=None):
        """Return per-agent count of strains currently carried.

        When ``uids is None`` the result is positional over active agents;
        when ``uids`` is supplied the result is in the same order as ``uids``.
        """
        if self._tb is None:
            raise RuntimeError('StrainProfile is not attached to a TB module.')
        if uids is None:
            n_alive = len(self._tb.sim.people.auids)
            total = np.zeros(n_alive, dtype=np.int32)
            for name in self.names:
                total += np.asarray(getattr(self._tb, name).values, dtype=np.int32)
            return total
        total = np.zeros(len(uids), dtype=np.int32)
        for name in self.names:
            total += np.asarray(getattr(self._tb, name)[uids], dtype=np.int32)
        return total

    # ---- mutation helpers ----

    def add_strain(self, uids, strain):
        """Mark *uids* as carrying *strain*; returns the UIDs that newly gained it."""
        if len(uids) == 0:
            return ss.uids()
        arr = self._arr(strain)
        already = np.asarray(arr[uids], dtype=bool)
        new = uids[~already]
        if len(new):
            arr[new] = True
        return new

    def remove_strain(self, uids, strain):
        """Mark *uids* as no longer carrying *strain*."""
        if len(uids) == 0:
            return
        self._arr(strain)[uids] = False
        return

    def clear_all(self, uids):
        """Clear all strains for *uids* (e.g. on natural clearance or death)."""
        if len(uids) == 0:
            return
        for name in self.names:
            getattr(self._tb, name)[uids] = False
        return

    def replace_strain(self, uids, old, new):
        """Replace *old* with *new* for *uids* (selective acquisition)."""
        if len(uids) == 0:
            return
        self.remove_strain(uids, old)
        self.add_strain(uids, new)
        return

    # ---- transmission helpers ----

    def effective_rel_trans(self, uids):
        """
        Effective per-agent multiplicative fitness for transmission.

        Implements the "fittest strain" model: per-agent effective relative
        transmissibility is the maximum fitness across carried strains (or
        zero if none).
        """
        if self._tb is None:
            raise RuntimeError('StrainProfile is not attached to a TB module.')
        if len(uids) == 0:
            return np.zeros(0, dtype=float)
        fit = self.registry.fitness
        out = np.zeros(len(uids), dtype=float)
        for idx, name in enumerate(self.names):
            carrier_mask = np.asarray(getattr(self._tb, name)[uids], dtype=bool)
            if carrier_mask.any():
                out[carrier_mask] = np.maximum(out[carrier_mask], fit[idx])
        return out

    def sample_transmitted_strain(self, source_uids, rng):
        """
        Choose one strain to transmit per source UID, weighted by fitness.

        Sources carrying no strain return -1.
        """
        if self._tb is None:
            raise RuntimeError('StrainProfile is not attached to a TB module.')
        n = len(source_uids)
        if n == 0:
            return np.zeros(0, dtype=int)

        ns = self.registry.n
        weights = np.zeros((n, ns), dtype=float)
        fit = self.registry.fitness
        for idx, name in enumerate(self.names):
            carrier_mask = np.asarray(getattr(self._tb, name)[source_uids], dtype=bool)
            weights[:, idx] = carrier_mask * fit[idx]

        total = weights.sum(axis=1)
        out = np.full(n, -1, dtype=int)
        active = total > 0
        if not active.any():
            return out

        u = np.asarray(rng.rvs(n), dtype=float)
        cdf = np.cumsum(weights[active] / total[active, None], axis=1)
        u_active = u[active]
        picks = (u_active[:, None] < cdf).argmax(axis=1)
        out[active] = picks
        return out
