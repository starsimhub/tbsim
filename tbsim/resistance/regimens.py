"""Regimen specifications mapping drug classes to per-strain efficacy."""

import numpy as np

__all__ = ['Regimen']


class Regimen:
    """
    A regimen is a set of drug/class names plus per-strain efficacy.

    Per-strain cure probability is computed as a drug-level efficacy model:
    among the drugs in the regimen, each drug contributes its configured
    efficacy, optionally reduced by a resistance penalty if the strain is
    resistant to that drug. The per-strain cure probability is then::

        p_cure(strain) = base_efficacy * f(effective_drug_efficacies)

    where ``f`` defaults to ``max`` over effective per-drug efficacies (best
    drug in the regimen drives the cure). By default resistant drugs contribute
    zero efficacy, preserving the historical "fully resistant means uncovered"
    behavior. Set ``resistance_penalty`` to allow reduced-but-nonzero efficacy
    against resistant strains, as in the two-strain ODE reference operator.

    Args:
        name (str): Regimen label, e.g. ``'first_line'``.
        drugs (list[str]): Drug/class names in this regimen.
        per_drug_efficacy (dict): Per-drug cure probability (0-1) for a
            strain susceptible to that drug. Default: 1.0 for each drug.
        resistance_penalty (dict): Optional per-drug multiplier applied to the
            drug's efficacy when a strain is resistant to that drug. Default:
            0.0 for each drug (resistant drug contributes no efficacy).
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

    def __init__(self, name, drugs, per_drug_efficacy=None,
                 resistance_penalty=None, base_efficacy=1.0, combine='max'):
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
        if resistance_penalty is None:
            resistance_penalty = {}
        for d, p in resistance_penalty.items():
            if d not in drugs:
                raise ValueError(
                    f'resistance_penalty[{d!r}] references a drug not in regimen {drugs}'
                )
            if not (0.0 <= float(p) <= 1.0):
                raise ValueError(
                    f'resistance_penalty[{d!r}] must be in [0, 1]; got {p!r}'
                )

        self.name = name
        self.drugs = drugs
        self.per_drug_efficacy = dict(per_drug_efficacy)
        self.resistance_penalty = {d: float(resistance_penalty.get(d, 0.0)) for d in drugs}
        self.base_efficacy = float(base_efficacy)
        self.combine = combine
        return

    def strain_cure_probs(self, catalog):
        """
        Compute per-strain cure probability for this regimen against the catalog.

        Args:
            catalog (StrainCatalog): Catalog of all strains.

        Returns:
            np.ndarray, shape (n_strains,): cure probability per strain.
        """
        probs = np.zeros(catalog.n, dtype=float)
        # Map regimen drugs to catalog-drug indices once
        try:
            drug_idx = [catalog.drugs.index(d) for d in self.drugs]
        except ValueError as exc:
            unknown = [d for d in self.drugs if d not in catalog.drugs]
            raise ValueError(
                f'Regimen {self.name!r} references drug(s) {unknown} '
                f'not in catalog drugs {catalog.drugs}'
            ) from exc

        per_drug = np.array([self.per_drug_efficacy[d] for d in self.drugs], dtype=float)
        penalties = np.array([self.resistance_penalty[d] for d in self.drugs], dtype=float)

        for s_idx in range(catalog.n):
            phenotype = catalog.resistance[s_idx, drug_idx]  # 1 = resistant
            eff = per_drug * np.where(phenotype == 1, penalties, 1.0)
            if self.combine == 'max':
                p = float(eff.max())
            else:  # parallel
                p = float(1.0 - np.prod(1.0 - eff))
            probs[s_idx] = self.base_efficacy * p
        return probs

    def __repr__(self):
        return (f'Regimen(name={self.name!r}, drugs={self.drugs}, '
                f'base_efficacy={self.base_efficacy:g}, combine={self.combine!r})')
