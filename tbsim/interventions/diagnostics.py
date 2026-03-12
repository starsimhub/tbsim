"""Diagnostic products and delivery for TB testing."""

import numpy as np
import pandas as pd
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['Dx', 'Xpert', 'OralSwab', 'FujiLAM', 'CAD', 'DxDelivery']

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
        super().__init__()
        self.df = df.copy()
        self.hierarchy = hierarchy if hierarchy is not None else list(df.result.unique())
        self._filter_cols = [c for c in _FILTER_COLS if c in df.columns]
        self._validate()
        self.rand_dist = ss.random()
        self.update_pars(**kwargs)
        return

    @staticmethod
    def _load_data(df, datafile, columns, default_data):
        """
        Load diagnostic data from one of three sources.

        Args:
            df        (DataFrame) : use directly if provided
            datafile  (str)       : path to CSV file
            columns   (list)      : column names for default data
            default_data (list)   : list of lists with default values
        """
        if df is not None:
            return df
        if datafile is not None:
            return pd.read_csv(datafile)
        return pd.DataFrame(default_data, columns=columns)

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

    columns = 'state,result,probability,age_min,age_max'.split(',')
    default_data = [
        # Adults (15+)
        #state              result     prob  amin  amax
        [TBSL.SYMPTOMATIC,    'positive', 0.909, 15, np.inf],
        [TBSL.SYMPTOMATIC,    'negative', 0.091, 15, np.inf],
        [TBSL.ASYMPTOMATIC,   'positive', 0.775, 15, np.inf],
        [TBSL.ASYMPTOMATIC,   'negative', 0.225, 15, np.inf],
        [TBSL.NON_INFECTIOUS, 'positive', 0.775, 15, np.inf],
        [TBSL.NON_INFECTIOUS, 'negative', 0.225, 15, np.inf],
        # Children (<15)
        [TBSL.SYMPTOMATIC,    'positive', 0.73,  0,  15],
        [TBSL.SYMPTOMATIC,    'negative', 0.27,  0,  15],
        [TBSL.ASYMPTOMATIC,   'positive', 0.73,  0,  15],
        [TBSL.ASYMPTOMATIC,   'negative', 0.27,  0,  15],
        [TBSL.NON_INFECTIOUS, 'positive', 0.73,  0,  15],
        [TBSL.NON_INFECTIOUS, 'negative', 0.27,  0,  15],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class OralSwab(Dx):
    """Oral swab -- age-stratified, TB-state-stratified."""

    columns = 'state,result,probability,age_min,age_max'.split(',')
    default_data = [
        # Adults (15+)
        #state              result     prob  amin  amax
        [TBSL.SYMPTOMATIC,    'positive', 0.80, 15, np.inf],
        [TBSL.SYMPTOMATIC,    'negative', 0.20, 15, np.inf],
        [TBSL.ASYMPTOMATIC,   'positive', 0.30, 15, np.inf],
        [TBSL.ASYMPTOMATIC,   'negative', 0.70, 15, np.inf],
        [TBSL.NON_INFECTIOUS, 'positive', 0.30, 15, np.inf],
        [TBSL.NON_INFECTIOUS, 'negative', 0.70, 15, np.inf],
        # Children (<15)
        [TBSL.SYMPTOMATIC,    'positive', 0.25, 0,  15],
        [TBSL.SYMPTOMATIC,    'negative', 0.75, 0,  15],
        [TBSL.ASYMPTOMATIC,   'positive', 0.25, 0,  15],
        [TBSL.ASYMPTOMATIC,   'negative', 0.75, 0,  15],
        [TBSL.NON_INFECTIOUS, 'positive', 0.25, 0,  15],
        [TBSL.NON_INFECTIOUS, 'negative', 0.75, 0,  15],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class FujiLAM(Dx):
    """FujiLAM -- HIV-stratified, age-stratified."""

    columns = 'state,result,probability,age_min,age_max,hiv'.split(',')
    default_data = [
        # HIV+, adults
        #state              result     prob   amin  amax    hiv
        [TBSL.SYMPTOMATIC,    'positive', 0.75,  15, np.inf, True],
        [TBSL.SYMPTOMATIC,    'negative', 0.25,  15, np.inf, True],
        [TBSL.ASYMPTOMATIC,   'positive', 0.75,  15, np.inf, True],
        [TBSL.ASYMPTOMATIC,   'negative', 0.25,  15, np.inf, True],
        [TBSL.NON_INFECTIOUS, 'positive', 0.75,  15, np.inf, True],
        [TBSL.NON_INFECTIOUS, 'negative', 0.25,  15, np.inf, True],
        # HIV-, adults
        [TBSL.SYMPTOMATIC,    'positive', 0.58,  15, np.inf, False],
        [TBSL.SYMPTOMATIC,    'negative', 0.42,  15, np.inf, False],
        [TBSL.ASYMPTOMATIC,   'positive', 0.58,  15, np.inf, False],
        [TBSL.ASYMPTOMATIC,   'negative', 0.42,  15, np.inf, False],
        [TBSL.NON_INFECTIOUS, 'positive', 0.58,  15, np.inf, False],
        [TBSL.NON_INFECTIOUS, 'negative', 0.42,  15, np.inf, False],
        # HIV+, children
        [TBSL.SYMPTOMATIC,    'positive', 0.579, 0,  15,     True],
        [TBSL.SYMPTOMATIC,    'negative', 0.421, 0,  15,     True],
        [TBSL.ASYMPTOMATIC,   'positive', 0.579, 0,  15,     True],
        [TBSL.ASYMPTOMATIC,   'negative', 0.421, 0,  15,     True],
        [TBSL.NON_INFECTIOUS, 'positive', 0.579, 0,  15,     True],
        [TBSL.NON_INFECTIOUS, 'negative', 0.421, 0,  15,     True],
        # HIV-, children
        [TBSL.SYMPTOMATIC,    'positive', 0.51,  0,  15,     False],
        [TBSL.SYMPTOMATIC,    'negative', 0.49,  0,  15,     False],
        [TBSL.ASYMPTOMATIC,   'positive', 0.51,  0,  15,     False],
        [TBSL.ASYMPTOMATIC,   'negative', 0.49,  0,  15,     False],
        [TBSL.NON_INFECTIOUS, 'positive', 0.51,  0,  15,     False],
        [TBSL.NON_INFECTIOUS, 'negative', 0.49,  0,  15,     False],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class CAD(Dx):
    """CAD chest X-ray -- simple, no age/HIV stratification."""

    columns = 'state,result,probability'.split(',')
    default_data = [
        #state              result     prob
        [TBSL.SYMPTOMATIC,    'positive', 0.66],
        [TBSL.SYMPTOMATIC,    'negative', 0.34],
        [TBSL.ASYMPTOMATIC,   'positive', 0.66],
        [TBSL.ASYMPTOMATIC,   'negative', 0.34],
        [TBSL.NON_INFECTIOUS, 'positive', 0.66],
        [TBSL.NON_INFECTIOUS, 'negative', 0.34],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)


class DxDelivery(ss.Intervention):
    """
    Delivers a diagnostic product to eligible agents.

    Handles eligibility (default: sought care, not yet diagnosed, alive),
    coverage filtering, result-state setting, and false-negative retry logic.

    Args:
        product: A Dx product instance.
        coverage: Fraction of eligible agents tested (float or ss.Dist). Default 1.0.
        eligibility: Callable (sim) -> uids. Default: sought_care & ~diagnosed & alive.
        result_state: People-state name set True on positive result. Default 'diagnosed'.
        care_seeking_multiplier: Multiplier on care-seeking rate for false negatives. Default 1.0.
    """

    def __init__(self, product, coverage=1.0, eligibility=None, result_state='diagnosed',
                 care_seeking_multiplier=1.0, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self.result_state = result_state
        self.care_seeking_multiplier_value = care_seeking_multiplier

        self.define_pars(
            p_coverage = ss.bernoulli(p=coverage)
        )

        # Person-level states
        self.define_states(
            ss.BoolState('sought_care', default=False),
            ss.BoolState('diagnosed', default=False),
            ss.BoolArr('tested', default=False),
            ss.IntArr('n_times_tested', default=0),
            ss.BoolState('test_result', default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied', default=False),
        )
        self.update_pars(**kwargs)
        product.name = f'{self.name}_product'
        return

    def init_post(self):
        super().init_post()
        ppl = self.sim.people
        # Expose key states directly on People so that HealthSeekingBehavior
        # (and other interventions) can find them via 'sought_care' in ppl.states.
        if 'sought_care' not in ppl.states: # TODO: use People method (not currently implemented)
            ppl.states['sought_care'] = self.sought_care
            setattr(ppl, 'sought_care', self.sought_care)
        if 'diagnosed' not in ppl.states:
            ppl.states['diagnosed'] = self.diagnosed
            setattr(ppl, 'diagnosed', self.diagnosed)
        # For custom result_state, create and register the state on People
        if self.result_state not in ('diagnosed', 'sought_care'):
            if self.result_state not in ppl.states:
                state = ss.BoolState(self.result_state, default=False)
                state.link_people(ppl)
                state.init_vals()
                ppl.states[self.result_state] = state # TODO: use People method (not currently implemented)
                setattr(ppl, self.result_state, state)
        return

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self.eligibility is not None:
            return ss.uids(self.eligibility(sim))
        return (self.sought_care & (~self.diagnosed) & sim.people.alive).uids

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_tested', dtype=int),
            ss.Result('n_positive', dtype=int),
            ss.Result('n_negative', dtype=int),
            ss.Result('cum_positive', dtype=int),
            ss.Result('cum_negative', dtype=int),
        )
        return

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        eligible = self._get_eligible(sim)
        if len(eligible) == 0:
            return

        # Coverage filter
        selected = self.pars.p_coverage.filter(eligible)
        if len(selected) == 0:
            return

        # Administer diagnostic product
        results = self.product.administer(sim, selected)
        pos_uids = results.get('positive', ss.uids())
        neg_uids = results.get('negative', ss.uids())

        # Set result state
        result_arr = ppl[self.result_state]
        result_arr[pos_uids] = True

        # Update tracking states
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = np.isin(selected, pos_uids)

        # Handle false negatives: TB-positive agents who tested negative
        tb_states = tb.state[neg_uids]
        has_tb = np.isin(tb_states, TBSL.active_tb_states())
        false_neg_uids = neg_uids[has_tb]

        if len(false_neg_uids) > 0 and self.care_seeking_multiplier_value != 1.0:
            unboosted = false_neg_uids[~self.multiplier_applied[false_neg_uids]]
            if len(unboosted) > 0:
                self.care_seeking_multiplier[unboosted] *= self.care_seeking_multiplier_value
                self.multiplier_applied[unboosted] = True

        # Reset flags for false negatives to allow retry
        if len(false_neg_uids) > 0:
            self.sought_care[false_neg_uids] = False
            self.tested[false_neg_uids] = False

        # Store for results
        self.results.n_tested[self.ti] = len(selected)
        self.results.n_positive[self.ti] = len(pos_uids)
        self.results.n_negative[self.ti] = len(neg_uids)
        return

    def update_results(self):
        pass

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_positive[:] = np.cumsum(self.results.n_positive)
        self.results.cum_negative[:] = np.cumsum(self.results.n_negative)
        return
