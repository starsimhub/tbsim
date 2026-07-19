"""
Strain registry for multi-strain (drug-resistance) TB.

A *strain* is an ``n``-bit resistance profile over a fixed, ordered set of drugs
(bit ``i`` = resistant to drug ``i``). For ``n`` drugs there are ``m = 2**n``
possible strains, enumerated as integer ids ``0..m-1`` (id 0 = pan-susceptible).

An *agent* carries a subset of the ``m`` strains, stored elsewhere as a single
integer ``strain_mask`` (bit ``j`` set = carries strain ``j``); see
``tbsim.resistance.TBResistant``. This registry holds the strain-level lookups
(fitness, resistance profiles, labels) and the bit helpers used to reason about masks.

The two-strain reference (``model-tests.md``, ``ode.r``) is the ``n=1`` special
case: ``drugs=['TX']`` gives strain 0 = A (susceptible) and strain 1 = B (resistant).
"""

import numpy as np

__all__ = ['Strains']


class Strains:
    """
    Registry of the possible strains for a given drug set.

    Args:
        drugs (list): ordered drug/class names; index = resistance bit position,
            e.g. ``['RIF', 'BDQ']`` (or ``['TX']`` for the two-strain reference).
        rel_fitness (dict): per-drug multiplicative transmission fitness cost
            ``r_i`` in ``[0, 1]``, e.g. ``{'RIF': 0.5}``. Drugs absent from the
            dict have no cost (``r_i = 1``). A strain's fitness is the product of
            the costs of the drugs it is resistant to (pan-susceptible = 1.0).

    Attributes:
        n (int): number of drugs.
        m (int): number of possible strains (``2**n``).
        profile (np.ndarray): ``(m, n)`` bool; ``profile[j, i]`` = strain ``j`` resistant to drug ``i``.
        fitness (np.ndarray): length-``m`` float; per-strain transmission fitness.
        labels (list): length-``m`` human-readable strain labels (e.g. ``'pan'``, ``'RIF+FQ'``).
    """

    def __init__(self, drugs, rel_fitness=None):
        self.drugs = list(drugs)
        self.n = len(self.drugs)
        if self.n == 0:
            raise ValueError('Strains requires at least one drug.')
        if len(set(self.drugs)) != self.n:
            raise ValueError(f'Duplicate drug names in {self.drugs}.')
        self.m = 2 ** self.n
        self.drug_idx = {d: i for i, d in enumerate(self.drugs)}

        rel_fitness = rel_fitness or {}
        for d, f in rel_fitness.items():
            if d not in self.drug_idx:
                raise ValueError(f'rel_fitness drug {d!r} not in drugs {self.drugs}.')
            if not 0.0 <= float(f) <= 1.0:
                raise ValueError(f'rel_fitness[{d!r}]={f!r} must be in [0, 1].')

        # profile[j, i] = bit i of strain id j
        ids = np.arange(self.m)
        self.profile = ((ids[:, None] >> np.arange(self.n)) & 1).astype(bool)  # (m, n)

        # Per-drug cost r_i, then per-strain fitness = product over resistant drugs.
        self.cost = np.array([rel_fitness.get(d, 1.0) for d in self.drugs], dtype=float)  # (n,)
        self.fitness = np.array([self.cost[self.profile[j]].prod() for j in range(self.m)])  # (m,)

        # Readable labels (id 0 -> 'pan'; otherwise '+'-joined resisted drugs, e.g. 'RIF+FQ').
        self.labels = [self._label(j) for j in range(self.m)]
        return

    def _label(self, j):
        """Human-readable label for strain id ``j``."""
        resisted = [self.drugs[i] for i in range(self.n) if self.profile[j, i]]
        return '+'.join(resisted) if resisted else 'pan'

    def validate_drugs(self, names, where=''):
        """Raise ``ValueError`` if any name in ``names`` (iterable of drug names, e.g. a list or the keys
        of a per-drug dict) is not a known drug — fail-fast on typos that otherwise resolve silently via
        ``.get()`` (L8). ``where`` labels the call site in the message.

        Example::

            strains.validate_drugs(['RIF', 'BDQ'], where='TxR.regimen_drugs')
        """
        unknown = [d for d in names if d not in self.drug_idx]
        if unknown:
            loc = f' in {where}' if where else ''
            raise ValueError(f'Unknown drug(s){loc}: {unknown}. Known drugs: {self.drugs}.')
        return

    def drug_bit(self, drug):
        """Resistance-profile bit (within a strain id) for a drug name."""
        return 1 << self.drug_idx[drug]

    def add_resistance(self, strain_ids, drug):
        """Return the strain id(s) obtained by adding resistance to ``drug``."""
        return strain_ids | self.drug_bit(drug)

    def carried(self, mask):
        """
        Decode agent strain masks into a boolean membership matrix.

        Args:
            mask (array): per-agent ``strain_mask`` integers.

        Returns:
            ``(len(mask), m)`` bool; ``[k, j]`` = agent ``k`` carries strain ``j``.
        """
        mask = np.asarray(mask)
        return ((mask[:, None] >> np.arange(self.m)) & 1).astype(bool)

    def max_fitness(self, mask):
        """Per-agent maximum strain fitness over carried strains (0 if none carried)."""
        return (self.carried(mask) * self.fitness).max(axis=1)

    def transmit_probs(self, mask, counts=None):
        """
        Per-agent probability of transmitting each strain, ``\\propto count × fitness`` over
        carried strains (the spec's ``r_j / \\sum r``, generalized so an agent carrying multiple
        copies of a strain is proportionally more likely to pass it). Rows for uninfected agents
        (mask 0) are left all-zero. The *overall* per-contact transmission probability
        (``rel_trans = max fitness``) does not depend on ``counts`` — only the which-strain split does.

        Args:
            mask (array): per-agent ``strain_mask`` integers.
            counts (array): optional ``(len(mask), m)`` per-strain multiplicity. If ``None``, every
                carried strain is weighted as a single copy (fitness-only, the pre-counter behavior).

        Returns:
            ``(len(mask), m)`` float, each infected row summing to 1.
        """
        w = self.carried(mask) * self.fitness  # (k, m)
        if counts is not None:
            w = w * np.asarray(counts)
        tot = w.sum(axis=1, keepdims=True)
        return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)

    def phenotype_any(self, mask):
        """
        Aggregate observed resistance phenotype per agent: OR of the resistance
        profiles over all carried strains.

        Returns:
            ``(len(mask), n)`` bool; ``[k, i]`` = agent ``k`` carries some strain resistant to drug ``i``.
        """
        carried = self.carried(mask)  # (k, m)
        return carried @ self.profile > 0  # (k, n)

    def resistant_frac(self, mask, drug):
        """Fraction of the given (infected) agents whose aggregate phenotype is resistant to ``drug``."""
        if len(mask) == 0:
            return 0.0
        return float(self.phenotype_any(mask)[:, self.drug_idx[drug]].mean())

    def mutate_one_susceptible(self, masks, drug, dist, uids, weighted=False):
        """Acquisition strain selection (L4): for each agent pick **one** carried strain susceptible to
        ``drug`` and replace it with its resistant (``| drug_bit``) counterpart (replacement).

        The strain is chosen uniformly at random among the agent's carried drug-susceptible strains, or
        ``\\propto`` fitness if ``weighted``, via the CRN ``dist`` (a ``choice2d``). Agents carrying no
        such strain are returned unchanged. Preserves "one mutation per hit" while removing the old
        lowest-id bias (which could never land on a strain already carrying other resistances).

        Args:
            masks (array): per-agent ``strain_mask`` integers for the hit agents.
            drug (str): the regimen drug whose resistance is acquired.
            dist (choice2d): a per-agent CRN choice distribution owned by the caller.
            uids (ss.uids): the hit agents' UIDs, aligned with ``masks`` rows (for CRN).
            weighted (bool): weight the selection by strain fitness instead of uniform.

        Returns:
            The mutated ``masks`` array (a copy).
        """
        masks = np.asarray(masks).copy()
        if len(masks) == 0:
            return masks
        dcol = self.drug_idx[drug]
        bit = self.drug_bit(drug)
        # carried AND susceptible to this drug (bit dcol of the strain id is 0)
        w = (self.carried(masks) & ~self.profile[:, dcol]).astype(float)  # (k, m)
        if weighted:
            w = w * self.fitness
        tot = w.sum(1, keepdims=True)
        has_target = tot[:, 0] > 0
        if not has_target.any():
            return masks
        probs = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
        probs[~has_target, 0] = 1.0  # dummy valid row for no-target agents (their draw is discarded)
        dist.set(a=np.arange(self.m), p=probs)
        chosen = np.asarray(dist.rvs(uids)).astype(int)
        sel = np.nonzero(has_target)[0]
        j = chosen[sel]
        sub = masks[sel]
        masks[sel] = (sub & ~(1 << j)) | (1 << (j | bit))
        return masks
