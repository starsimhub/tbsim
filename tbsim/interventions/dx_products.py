"""Diagnostic products for TB testing."""

import os
import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['Dx', 'Xpert', 'OralSwab', 'FujiLAM', 'CAD']

# Optional filter columns recognized by Dx
_FILTER_COLS = ['age_min', 'age_max', 'hiv']

# Path to bundled data files
_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


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
        super().__init__()
        self.df = df.copy()
        self.hierarchy = hierarchy if hierarchy is not None else list(df.result.unique())
        self._filter_cols = [c for c in _FILTER_COLS if c in df.columns]
        self._validate()
        self.rand_dist = ss.random()
        self.update_pars(**kwargs)
        return

    def _validate(self):
        """Check that probabilities sum to 1.0 per group."""
        required = {'state', 'result', 'probability'}
        missing = required - set(self.df.columns)
        if missing:
            errormsg = f"DataFrame missing required columns: {missing}"
            raise ValueError(errormsg)

        group_cols = ['state'] + self._filter_cols
        for name, group in self.df.groupby(group_cols):
            total = group.probability.sum()
            if not np.isclose(total, 1.0):
                errormsg = (
                    f"Probabilities for {dict(zip(group_cols, name if isinstance(name, tuple) else [name]))} "
                    f"sum to {total}, not 1.0"
                )
                raise ValueError(errormsg)

    @property
    def default_value(self):
        """Index of the default (most benign) result, i.e. the last entry in the hierarchy."""
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
        tb_states = tb.state[uids]
        ages = ppl.age[uids] if ('age_min' in self._filter_cols or 'age_max' in self._filter_cols) else None

        hiv_states = None
        if 'hiv' in self._filter_cols and hasattr(sim.diseases, 'hiv'):
            hiv_states = sim.diseases.hiv.infected[uids]

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

            matched_idx = np.where(np.asarray(mask))[0]
            if len(matched_idx) == 0:
                continue

            # Get the actual UIDs for matched agents (needed for CRN-safe rvs)
            matched_agent_uids = uids[matched_idx]

            # Get probabilities in hierarchy order
            probs = []
            for r in self.hierarchy:
                row = group_df[group_df.result == r]
                probs.append(row.probability.values[0] if len(row) > 0 else 0.0)
            probs = np.array(probs)

            # Draw stochastic results using cumulative probabilities
            cumprobs = np.cumsum(probs)
            rand_vals = self.rand_dist.rvs(matched_agent_uids)
            draws = np.searchsorted(cumprobs, rand_vals)
            draws = np.clip(draws, 0, len(self.hierarchy) - 1)

            # Take minimum (best/first result wins if agent matches multiple groups)
            results[matched_idx] = np.minimum(draws, results[matched_idx])

        # Convert to dict of UIDs per result
        output = {}
        for i, label in enumerate(self.hierarchy):
            output[label] = uids[results == i]
        return output


class Xpert(Dx):
    """Xpert MTB/RIF -- age-stratified (child/adult), TB-state-stratified."""

    def __init__(self, **kwargs):
        df = pd.read_csv(os.path.join(_DATA_DIR, 'xpert.csv'))
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class OralSwab(Dx):
    """Oral swab -- age-stratified, TB-state-stratified."""

    def __init__(self, **kwargs):
        df = pd.read_csv(os.path.join(_DATA_DIR, 'oral_swab.csv'))
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class FujiLAM(Dx):
    """FujiLAM -- HIV-stratified, age-stratified."""

    def __init__(self, **kwargs):
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
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class CAD(Dx):
    """CAD chest X-ray -- simple, no age/HIV stratification."""

    def __init__(self, **kwargs):
        df = pd.DataFrame([
            dict(state=TBSL.SYMPTOMATIC,    result='positive', probability=0.66),
            dict(state=TBSL.SYMPTOMATIC,    result='negative', probability=0.34),
            dict(state=TBSL.ASYMPTOMATIC,   result='positive', probability=0.66),
            dict(state=TBSL.ASYMPTOMATIC,   result='negative', probability=0.34),
            dict(state=TBSL.NON_INFECTIOUS, result='positive', probability=0.66),
            dict(state=TBSL.NON_INFECTIOUS, result='negative', probability=0.34),
        ])
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)
