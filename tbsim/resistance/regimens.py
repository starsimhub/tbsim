"""Regimen specifications mapping drug classes to per-strain efficacy."""

import numpy as np

__all__ = ['Regimen']


class Regimen:
    """
    A regimen is a set of drug/class names plus per-strain efficacy.

    Per-strain cure probability is computed as a *minimum-effective-drug* model:
    among the drugs in the regimen, a strain is exposed to those drugs it is
    *not* resistant to. The per-strain cure probability is then::

        p_cure(strain) = base_efficacy * f(susceptible_drugs)

    where ``f`` defaults to ``max`` over per-drug efficacies (best drug in the
    regimen drives the cure). Strains that are resistant to every regimen drug
    have ``p_cure = 0``.

    Args:
        name (str): Regimen label, e.g. ``'first_line'``.
        drugs (list[str]): Drug/class names in this regimen.
        per_drug_efficacy (dict): Per-drug cure probability (0-1) for a
            strain susceptible to that drug. Default: 1.0 for each drug.
        base_efficacy (float): Multiplicative top-level efficacy applied to all
            strains. Default 1.0.
        combine (str): How to combine per-drug efficacies for strains
            susceptible to more than one regimen drug. Either ``'max'``
            (default; best drug drives cure) or ``'parallel'``
            (independent: ``1 - prod(1 - p_i)``).

    Example:
        ::

            first_line = Regimen('first_line', drugs=['INH', 'RIF'],
                                  per_drug_efficacy={'INH': 0.95, 'RIF': 0.95},
                                  base_efficacy=1.0)
            # Against a pan-susceptible strain  → p_cure = 0.95 (max)
            # Against an INH-resistant strain   → p_cure = 0.95 (RIF still works)
            # Against an MDR strain (INH+RIF)   → p_cure = 0.0
    """

    COMBINE_MODES = ('max', 'parallel')

    def __init__(self, name, drugs, per_drug_efficacy=None, base_efficacy=1.0,
                 combine='max'):
        if not isinstance(name, str) or not name:
            raise ValueError(f'Regimen name must be a non-empty string; got {name!r}')
        drugs = list(drugs)
        if not drugs:
            raise ValueError('Regimen must include at least one drug.')
        if combine not in self.COMBINE_MODES:
            raise ValueError(
                f'Regimen combine must be one of {self.COMBINE_MODES}; got {combine!r}'
            )
        if not (0.0 <= float(base_efficacy) <= 1.0):
            raise ValueError(
                f'Regimen base_efficacy must be in [0, 1]; got {base_efficacy!r}'
            )

        if per_drug_efficacy is None:
            per_drug_efficacy = {d: 1.0 for d in drugs}
        else:
            for d in drugs:
                if d not in per_drug_efficacy:
                    per_drug_efficacy[d] = 1.0
            for d, p in per_drug_efficacy.items():
                if not (0.0 <= float(p) <= 1.0):
                    raise ValueError(
                        f'per_drug_efficacy[{d!r}] must be in [0, 1]; got {p!r}'
                    )

        self.name = name
        self.drugs = drugs
        self.per_drug_efficacy = dict(per_drug_efficacy)
        self.base_efficacy = float(base_efficacy)
        self.combine = combine
        return

    def strain_cure_probs(self, registry):
        """
        Compute per-strain cure probability for this regimen against the registry.

        Args:
            registry (StrainRegistry): Registry of all strains.

        Returns:
            np.ndarray, shape (n_strains,): cure probability per strain.
        """
        probs = np.zeros(registry.n, dtype=float)
        # Map regimen drugs to registry-drug indices once
        try:
            drug_idx = [registry.drugs.index(d) for d in self.drugs]
        except ValueError as exc:
            unknown = [d for d in self.drugs if d not in registry.drugs]
            raise ValueError(
                f'Regimen {self.name!r} references drug(s) {unknown} '
                f'not in registry drugs {registry.drugs}'
            ) from exc

        per_drug = np.array([self.per_drug_efficacy[d] for d in self.drugs], dtype=float)

        for s_idx in range(registry.n):
            phenotype = registry.resistance[s_idx, drug_idx]  # 1 = resistant
            susceptible = phenotype == 0
            if not susceptible.any():
                probs[s_idx] = 0.0
                continue
            eff = per_drug[susceptible]
            if self.combine == 'max':
                p = float(eff.max())
            else:  # parallel
                p = float(1.0 - np.prod(1.0 - eff))
            probs[s_idx] = self.base_efficacy * p
        return probs

    def __repr__(self):
        return (f'Regimen(name={self.name!r}, drugs={self.drugs}, '
                f'base_efficacy={self.base_efficacy:g}, combine={self.combine!r})')
