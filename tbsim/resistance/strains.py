"""
Strain registry for multi-strain (drug-resistance) TB.

A *strain* is an ``n``-bit resistance profile over a fixed, ordered set of drugs
(bit ``i`` = resistant to drug ``i``). For ``n`` drugs there are ``m = 2**n``
possible strains, enumerated as integer ids ``0..m-1`` (id 0 = pan-susceptible).

An *agent* carries a subset of the ``m`` strains, stored elsewhere as a single
integer ``strain_mask`` (bit ``j`` set = carries strain ``j``); see
``tbsim.resistance.TBResistant``. This registry holds the strain-level lookups
(fitness, resistance profiles) and the bit helpers used to reason about masks.

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
    """

    def __init__(self, drugs, rel_fitness=None):
        self.drugs = list(drugs)
        self.n = len(self.drugs)
        self.m = 2 ** self.n
        self.drug_idx = {d: i for i, d in enumerate(self.drugs)}
        rel_fitness = rel_fitness or {}

        # profile[j, i] = bit i of strain id j
        ids = np.arange(self.m)
        self.profile = ((ids[:, None] >> np.arange(self.n)) & 1).astype(bool)  # (m, n)

        # Per-drug cost r_i, then per-strain fitness = product over resistant drugs.
        self.cost = np.array([rel_fitness.get(d, 1.0) for d in self.drugs], dtype=float)  # (n,)
        self.fitness = np.array([self.cost[self.profile[j]].prod() for j in range(self.m)])  # (m,)
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

    def transmit_probs(self, mask):
        """
        Per-agent probability of transmitting each strain, ``\\propto`` fitness over
        carried strains (the spec's ``r_j / \\sum r``). Rows for uninfected agents
        (mask 0) are left all-zero.

        Returns:
            ``(len(mask), m)`` float, each infected row summing to 1.
        """
        w = self.carried(mask) * self.fitness  # (k, m)
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
