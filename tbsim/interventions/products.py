"""Base product classes for DataFrame-driven multi-outcome products."""

import numpy as np
import sciris as sc
import starsim as ss
import tbsim
from tbsim import TBS

__all__ = ['ProductMulti']

# Optional filter columns recognized by ProductMulti
_FILTER_COLS = ['age_min', 'age_max', 'hiv']


class ProductMulti(ss.Product):
    """
    Base product with DataFrame-driven, multi-outcome probability logic.

    Supports per-agent outcome probabilities based on TB state and optional
    filter columns (age_min, age_max, hiv). Used as the base class for
    diagnostics (Dx) and multi-outcome treatments (TxMulti).

    Args:
        df: DataFrame with required columns (state, result, probability) and
            optional filter columns (age_min, age_max, hiv). Only include
            filter columns relevant to the specific product.
        hierarchy: List of result strings in priority order, e.g. ['positive', 'negative'].
            Agents in states not listed in df get the last entry (default).
    """

    def __init__(self, df, hierarchy=None, **kwargs):
        super().__init__()
        self.df = df.copy()
        self.hierarchy = hierarchy if hierarchy is not None else list(df.result.unique())
        self._filter_cols = [c for c in _FILTER_COLS if c in df.columns]
        self._validate()
        self.update_pars(**kwargs)
        self.outcome_dist = tbsim.choice2d(p=np.array([[1]])) # Probabilities set in administer()
        return

    @staticmethod
    def _load_data(df, datafile, columns, default_data):
        """
        Load product data from one of three sources.

        Args:
            df        (DataFrame) : use directly if provided
            datafile  (str)       : path to CSV file
            columns   (list)      : column names for default data
            default_data (list)   : list of lists with default values
        """
        if df is not None:
            return df
        if datafile is not None:
            return sc.dataframe.read_csv(datafile)
        return sc.dataframe(default_data, columns=columns)

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
        Administer product to agents.

        Returns:
            dict keyed by hierarchy strings, e.g. {'positive': uids, 'negative': uids}
        """
        # Create output, and return immediately if nothing to do
        output = {}
        if not len(uids):
            return output

        tb = tbsim.get_tb(sim)
        ppl = sim.people
        n_choices = len(self.hierarchy)

        # Build per-agent probability matrix; default: all probability on last hierarchy entry
        probs = np.zeros((len(uids), n_choices))
        probs[:, self.default_value] = 1.0

        # Get agent attributes needed for filtering
        tb_states = tb.state[uids]
        ages = ppl.age[uids] if ('age_min' in self._filter_cols or 'age_max' in self._filter_cols) else None

        hiv_states = None
        if 'hiv' in self._filter_cols and 'hiv' in sim.diseases:
            hiv_states = sim.diseases.hiv.infected[uids]

        # Group df rows by (state + filter columns) and fill per-agent probabilities
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

            # Get probabilities in hierarchy order
            row_probs = []
            for r in self.hierarchy:
                row = group_df[group_df.result == r]
                row_probs.append(row.probability.values[0] if len(row) > 0 else 0.0)
            probs[matched_idx] = row_probs

        # Draw results using choice2d
        self.outcome_dist.set(a=np.arange(probs.shape[1]), p=probs)
        temp_uids = ss.uids(np.arange(len(uids))) # TODO: WARNING: not CRN safe!! See https://github.com/starsimhub/starsim/issues/1254
        results = self.outcome_dist.rvs(temp_uids)

        # Convert to dict of UIDs per result
        for i, label in enumerate(self.hierarchy):
            output[label] = uids[results == i]
        return output
