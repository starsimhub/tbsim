"""Diagnostic products for TB testing."""

import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['Dx', 'xpert', 'oral_swab', 'fujilam', 'cad_cxr']

# Optional filter columns recognized by Dx
_FILTER_COLS = ['age_min', 'age_max', 'hiv']


class Dx(ss.Product):
    """
    TB diagnostic product defined by a DataFrame of state-to-result probabilities.

    Args:
        df: DataFrame with required columns (state, result, probability) and
            optional filter columns (age_min, age_max, hiv). Only include
            filter columns relevant to the specific diagnostic.
        hierarchy: List of result strings in priority order, e.g. ['positive', 'negative'].
            Agents in states not listed in df get the last entry (default negative).
    """

    def __init__(self, df, hierarchy=None, **kwargs):
        super().__init__(**kwargs)
        self.df = df.copy()
        self.hierarchy = hierarchy if hierarchy is not None else list(df.result.unique())
        self._filter_cols = [c for c in _FILTER_COLS if c in df.columns]
        self._validate()

    def _validate(self):
        """Check that probabilities sum to 1.0 per group."""
        required = {'state', 'result', 'probability'}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        group_cols = ['state'] + self._filter_cols
        for name, group in self.df.groupby(group_cols):
            total = group.probability.sum()
            if not np.isclose(total, 1.0):
                raise ValueError(
                    f"Probabilities for {dict(zip(group_cols, name if isinstance(name, tuple) else [name]))} "
                    f"sum to {total}, not 1.0"
                )

    @property
    def default_value(self):
        return len(self.hierarchy) - 1

    def administer(self, sim, uids):
        """
        Administer diagnostic test to agents.

        Returns:
            dict keyed by hierarchy strings, e.g. {'positive': uids, 'negative': uids}
        """
        tb = tbsim.get_tb(sim)
        ppl = sim.people

        # Pre-fill with default (last hierarchy entry = most benign result)
        results = np.full(len(uids), self.default_value, dtype=int)

        # Get agent attributes needed for filtering
        tb_states = np.asarray(tb.state[uids])
        ages = np.asarray(ppl.age[uids]) if ('age_min' in self._filter_cols or 'age_max' in self._filter_cols) else None

        hiv_states = None
        if 'hiv' in self._filter_cols and hasattr(sim.diseases, 'hiv'):
            hiv_states = np.asarray(sim.diseases.hiv.infected[uids])

        # Group df rows by (state + filter columns)
        group_cols = ['state'] + self._filter_cols
        for group_key, group_df in self.df.groupby(group_cols):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            vals = dict(zip(group_cols, group_key))

            # Build mask for agents matching this group
            mask = tb_states == vals['state']
            if 'age_min' in vals:
                mask = mask & (ages >= vals['age_min'])
            if 'age_max' in vals:
                mask = mask & (ages < vals['age_max'])
            if 'hiv' in vals and hiv_states is not None:
                mask = mask & (hiv_states == vals['hiv'])

            matched_idx = np.where(mask)[0]
            if len(matched_idx) == 0:
                continue

            # Get probabilities in hierarchy order
            probs = []
            for r in self.hierarchy:
                row = group_df[group_df.result == r]
                probs.append(row.probability.values[0] if len(row) > 0 else 0.0)
            probs = np.array(probs)

            # Draw stochastic results using cumulative probabilities
            cumprobs = np.cumsum(probs)
            rand_vals = np.random.random(len(matched_idx))
            draws = np.searchsorted(cumprobs, rand_vals)
            draws = np.clip(draws, 0, len(self.hierarchy) - 1)

            # Take minimum (best/first result wins if agent matches multiple groups)
            results[matched_idx] = np.minimum(draws, results[matched_idx])

        # Convert to dict of UIDs per result
        output = {}
        for i, label in enumerate(self.hierarchy):
            output[label] = ss.uids(uids[results == i])
        return output
