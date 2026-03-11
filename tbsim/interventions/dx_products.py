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


def xpert():
    """Xpert MTB/RIF -- age-stratified (child/adult), TB-state-stratified."""
    df = pd.DataFrame([
        # Adults (age >= 15)
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.909, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.091, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.775, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.225, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.775, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.225, age_min=15, age_max=np.inf),
        # Children (age < 15)
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.27, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.27, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.73, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.27, age_min=0, age_max=15),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def oral_swab():
    """Oral swab -- age-stratified, TB-state-stratified."""
    df = pd.DataFrame([
        # Adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.80, age_min=15, age_max=np.inf),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.20, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.30, age_min=15, age_max=np.inf),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.70, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.30, age_min=15, age_max=np.inf),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.70, age_min=15, age_max=np.inf),
        # Children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.75, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.75, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.25, age_min=0, age_max=15),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.75, age_min=0, age_max=15),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def fujilam():
    """FujiLAM -- HIV-stratified, age-stratified."""
    df = pd.DataFrame([
        # HIV+, adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.75,  age_min=15, age_max=np.inf, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.25,  age_min=15, age_max=np.inf, hiv=True),
        # HIV-, adults
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.58,  age_min=15, age_max=np.inf, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.42,  age_min=15, age_max=np.inf, hiv=False),
        # HIV+, children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.579, age_min=0, age_max=15, hiv=True),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.421, age_min=0, age_max=15, hiv=True),
        # HIV-, children
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.51,  age_min=0, age_max=15, hiv=False),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.49,  age_min=0, age_max=15, hiv=False),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])


def cad_cxr():
    """CAD chest X-ray -- simple, no age/HIV stratification."""
    df = pd.DataFrame([
        dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.66),
        dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.34),
        dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.66),
        dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.34),
        dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.66),
        dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.34),
    ])
    return Dx(df=df, hierarchy=['positive', 'negative'])
