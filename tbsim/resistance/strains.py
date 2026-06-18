"""Strain specifications and registry for the TB resistance overlay."""

import numpy as np

__all__ = ['StrainSpec', 'StrainRegistry']


class StrainSpec:
    """
    Declarative specification for a single TB strain.

    A StrainSpec is pure configuration: it does not own any simulation state.
    Per-agent state is stored in :class:`StrainProfile`. Per-step behavior is
    applied by :class:`ResistanceConnector` and (later) by progression,
    clearance, and acquisition resolvers.

    Args:
        uid          (str):   Unique identifier for the strain (e.g. 'pan_sus', 'rif_r').
        resistance   (dict):  Mapping of drug/class name to {0, 1}; 1 means resistant.
        fitness      (float): Multiplicative reduction on relative transmissibility (0-1).
            Default 1.0 (no fitness cost).
        init_prev    (float): Initial population prevalence of this strain at sim start.
            Default 0.0. Used only when StrainProfile seeds initial infections.
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


class StrainRegistry:
    """
    Catalog of strains used by a simulation.

    The registry expands a list of :class:`StrainSpec` into ordered numpy
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

            reg = StrainRegistry([
                StrainSpec('pan',   {'INH': 0, 'RIF': 0, 'BDQ': 0}),
                StrainSpec('inh_r', {'INH': 1, 'RIF': 0, 'BDQ': 0}, fitness=0.95),
            ])
            reg.index('inh_r')   # -> 1
    """

    def __init__(self, strains, drugs=None):
        if not strains:
            raise ValueError('StrainRegistry requires at least one StrainSpec.')

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
            raise ValueError(f'Duplicate strain uids in registry: {uids}')

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
                f'StrainRegistry init_prev sums to {init_prev.sum():g} > 1; '
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
        return (f"StrainRegistry(n={self.n}, drugs={self.drugs}, "
                f"uids={self.uids})")
