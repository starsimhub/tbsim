"""Strain specifications, catalog, and per-agent profile for the TB resistance overlay."""

import numpy as np
import starsim as ss

__all__ = ['StrainSpec', 'StrainCatalog', 'AgentStrains']


class StrainSpec:
    """
    Declarative specification for a single TB strain.

    A StrainSpec is pure configuration: it does not own any simulation state.
    Per-agent state is stored in :class:`AgentStrains`. Per-step behavior is
    applied by :class:`ResistanceConnector` and (later) by progression,
    clearance, and acquisition resolvers.

    Args:
        uid          (str):   Unique identifier for the strain (e.g. 'pan_sus', 'rif_r').
        resistance   (dict):  Mapping of drug/class name to {0, 1}; 1 means resistant.
        fitness      (float): Multiplicative reduction on relative transmissibility (0-1).
            Default 1.0 (no fitness cost).
        init_prev    (float): Initial population prevalence of this strain at sim start.
            Default 0.0. Used only when AgentStrains seeds initial infections.
        acquisition  (dict):  Optional per-drug random acquisition probabilities for this strain.
            Reserved for Phase 2. Default None.
        label        (str):   Optional human-readable label for plots. Defaults to uid.

    Example:
        ::

            pan = StrainSpec(uid='pan_sus', resistance={'INH': 0, 'RIF': 0, 'BDQ': 0})
            inh = StrainSpec(uid='inh_r',   resistance={'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95)
    """

    def __init__(self, uid, resistance, fitness=1.0, init_prev=0.0,
                 acquisition=None, label=None):
        if not isinstance(uid, str) or not uid:
            raise ValueError(f'StrainSpec uid must be a non-empty string; got {uid!r}')
        if not isinstance(resistance, dict) or not resistance:
            raise ValueError(f'StrainSpec resistance must be a non-empty dict; got {resistance!r}')
        for drug, val in resistance.items():
            if val not in (0, 1, True, False):
                raise ValueError(
                    f'StrainSpec resistance values must be 0 or 1; got {drug}={val!r}'
                )
        if not (0.0 <= float(fitness) <= 1.0):
            raise ValueError(f'StrainSpec fitness must be in [0, 1]; got {fitness!r}')
        if not (0.0 <= float(init_prev) <= 1.0):
            raise ValueError(f'StrainSpec init_prev must be in [0, 1]; got {init_prev!r}')

        self.uid = uid
        self.resistance = {drug: int(bool(v)) for drug, v in resistance.items()}
        self.fitness = float(fitness)
        self.init_prev = float(init_prev)
        self.acquisition = dict(acquisition) if acquisition else {}
        self.label = label if label is not None else uid
        return

    @property
    def drugs(self):
        """Tuple of drug/class names this strain declares resistance for."""
        return tuple(self.resistance.keys())

    def is_resistant_to(self, drug):
        """True if this strain is phenotypically resistant to *drug*."""
        return bool(self.resistance.get(drug, 0))

    def __repr__(self):
        bits = ''.join(str(self.resistance[d]) for d in self.resistance)
        return f"StrainSpec(uid={self.uid!r}, resistance={bits}, fitness={self.fitness:g})"


class StrainCatalog:
    """
    Catalog of strains used by a simulation.

    The catalog expands a list of :class:`StrainSpec` into ordered numpy
    arrays for fast indexing during transmission and progression.

    Args:
        strains (list[StrainSpec]): The strains to register.
        drugs   (list[str]):        Optional ordered list of drug/class names.
            If None, drugs are inferred from the union of all strain resistance keys.

    Attributes:
        drugs        (list[str]):           Ordered drug/class names.
        uids         (list[str]):           Strain uids in registration order.
        n            (int):                 Number of strains.
        resistance   (np.ndarray):          Shape (n_strains, n_drugs), 0/1.
        fitness      (np.ndarray):          Shape (n_strains,), float.
        init_prev    (np.ndarray):          Shape (n_strains,), float.

    Example:
        ::

            reg = StrainCatalog([
                StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}),
                StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95),
            ])
            reg.index('inh_r')   # -> 1
    """

    def __init__(self, strains, drugs=None):
        if not strains:
            raise ValueError('StrainCatalog requires at least one StrainSpec.')

        # Validate types
        strains = list(strains)
        for s in strains:
            if not isinstance(s, StrainSpec):
                raise TypeError(f'Expected StrainSpec, got {type(s).__name__}')

        # Determine drug ordering
        if drugs is None:
            drug_set = []
            seen = set()
            for s in strains:
                for d in s.drugs:
                    if d not in seen:
                        seen.add(d)
                        drug_set.append(d)
            drugs = drug_set
        drugs = list(drugs)

        # Check for duplicate uids
        uids = [s.uid for s in strains]
        if len(set(uids)) != len(uids):
            raise ValueError(f'Duplicate strain uids in catalog: {uids}')

        # Build arrays
        n = len(strains)
        nd = len(drugs)
        resistance = np.zeros((n, nd), dtype=np.uint8)
        fitness = np.ones(n, dtype=float)
        init_prev = np.zeros(n, dtype=float)

        for i, s in enumerate(strains):
            for j, d in enumerate(drugs):
                resistance[i, j] = s.resistance.get(d, 0)
            fitness[i] = s.fitness
            init_prev[i] = s.init_prev

        # Warn-ish: enforce init_prev sum <= 1 (allowed to be < 1: not everyone is seeded)
        if init_prev.sum() > 1.0 + 1e-9:
            raise ValueError(
                f'StrainCatalog init_prev sums to {init_prev.sum():g} > 1; '
                f'each agent can only be seeded with at most one initial strain.'
            )

        self.drugs = drugs
        self.uids = uids
        self.n = n
        self.resistance = resistance
        self.fitness = fitness
        self.init_prev = init_prev
        self._specs = strains
        self._uid_to_idx = {uid: i for i, uid in enumerate(uids)}
        return

    def index(self, uid):
        """Return the integer index for *uid*.

        Args:
            uid (str): Strain identifier.

        Returns:
            int: Strain index.
        """
        try:
            return self._uid_to_idx[uid]
        except KeyError:
            raise KeyError(f'Unknown strain uid {uid!r}; known: {self.uids}')

    def spec(self, uid_or_idx):
        """Return the :class:`StrainSpec` for the given uid or index."""
        if isinstance(uid_or_idx, str):
            return self._specs[self.index(uid_or_idx)]
        return self._specs[int(uid_or_idx)]

    def phenotype_bits(self, idx):
        """Return resistance vector for strain *idx* as a numpy array of 0/1."""
        return self.resistance[int(idx)].copy()

    def __iter__(self):
        return iter(self._specs)

    def __len__(self):
        return self.n

    def __repr__(self):
        return (f"StrainCatalog(n={self.n}, drugs={self.drugs}, "
                f"uids={self.uids})")


class AgentStrains:
    """
    Per-agent strain presence state attached to a :class:`tbsim.TB` instance.

    For each strain in the catalog, allocates one :class:`ss.BoolArr` named
    ``carries_<uid>``. The arrays are defined on TB via
    ``tb.define_states(...)`` so Starsim handles agent growth automatically.

    All agent indexing goes through Starsim's ``ss.uids`` /
    ``BoolArr.uids`` / ``intersect()`` APIs rather than raw NumPy.

    Args:
        catalog (StrainCatalog): The strain catalog.

    Example:
        ::

            agent_strains = AgentStrains(catalog)
            agent_strains.attach(tb)
            agent_strains.add_strain(uids, 'pan')
    """

    PREFIX = 'carries_'

    def __init__(self, catalog):
        self.catalog = catalog
        self.names = [f'{self.PREFIX}{uid}' for uid in catalog.uids]
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
                    f'AgentStrains state {name!r} not found on TB; '
                    f'did you forget to include agent_strains.state_defs() in define_states()?'
                )
        return

    # ---- access helpers ----

    def _arr(self, strain):
        """Return the ``ss.BoolArr`` for *strain* (by uid or idx)."""
        if self._tb is None:
            raise RuntimeError('AgentStrains is not attached to a TB module.')
        return getattr(self._tb, self.names[self._to_idx(strain)])

    def _to_idx(self, strain):
        if isinstance(strain, str):
            return self.catalog.index(strain)
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

    def carries_any(self, uids):
        """Return boolean mask: True where agents carry at least one strain."""
        return self.n_strains_per_agent(uids) > 0

    def n_strains_per_agent(self, uids=None):
        """Return per-agent count of strains currently carried.

        When ``uids is None`` the result is positional over active agents;
        when ``uids`` is supplied the result is in the same order as ``uids``.
        """
        if self._tb is None:
            raise RuntimeError('AgentStrains is not attached to a TB module.')
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
            raise RuntimeError('AgentStrains is not attached to a TB module.')
        if len(uids) == 0:
            return np.zeros(0, dtype=float)
        fit = self.catalog.fitness
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
            raise RuntimeError('AgentStrains is not attached to a TB module.')
        n = len(source_uids)
        if n == 0:
            return np.zeros(0, dtype=int)

        ns = self.catalog.n
        weights = np.zeros((n, ns), dtype=float)
        fit = self.catalog.fitness
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
