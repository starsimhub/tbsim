"""Diagnostic products and delivery for TB testing."""

import numpy as np
import sciris as sc
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
        self.update_pars(**kwargs)
        self.outcome_dist = tbsim.choice2d(p=np.array([[1]])) # Probabilities set in administer()
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
        Administer diagnostic test to agents.

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


class Xpert(Dx):
    """Xpert MTB/RIF -- age-stratified (child/adult), TB-state-stratified."""

    columns = 'state,result,probability,age_min,age_max'.split(',')
    default_data = [
        #state                result      prob   min max
        # Adults (15+)
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
        #state                result      prob  min max
        # Adults (15+)
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
        #state                result      prob   min max      hiv
        # HIV+, adults
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
        #state                result      prob
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
            ss.BoolState(result_state, default=False),
            ss.BoolArr('tested', default=False),
            ss.IntArr('n_times_tested', default=0),
            ss.BoolState('test_result', default=False),
            ss.FloatArr('care_seeking_multiplier', default=1.0),
            ss.BoolState('multiplier_applied', default=False),
        )
        self.update_pars(**kwargs)
        product.name = f'{self.name}_product'
        return

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self.eligibility is not None:
            return ss.uids(self.eligibility(sim))
        # Default: use HSB sought_care if available, otherwise all alive
        hsb = sim.get_hsb()
        result_arr = self[self.result_state]
        if hsb is not None:
            return (hsb.sought_care & (~result_arr) & sim.people.alive).uids
        return (~result_arr & sim.people.alive).uids

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
        """ Main step method: see submethods for details. """
        self.step_select_eligible() # Coverage filter
        self.step_administer() # Administer diagnostic product
        self.step_update_states() # Update states based on results
        self.step_false_negatives() # Handle false negatives
        return
    
    def step_select_eligible(self):
        """Return eligible UIDs for this step."""
        sim = self.sim
        eligible = self._get_eligible(sim)
        if len(eligible):
            self._selected = self.pars.p_coverage.filter(eligible)
        else:
            self._selected = ss.uids()
        return self._selected
        
    def step_administer(self):
        """Administer diagnostic product to selected UIDs and return results."""
        results = self.product.administer(self.sim, self._selected)
        self._results = results
        self._pos = results.get('positive', ss.uids())
        self._neg = results.get('negative', ss.uids())
        return results
    
    def step_update_states(self):
        """ Update states associated with """
        selected = self._selected
        pos_uids = self._pos

        # Set result state
        result_arr = self[self.result_state]
        result_arr[pos_uids] = True

        # Update tracking states
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = np.isin(selected, pos_uids)

        return
    
    def step_false_negatives(self):
        # Handle false negatives: TB-positive agents who tested negative
        neg_uids = self._neg

        tb = tbsim.get_tb(self.sim)
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
            if hsb := self.sim.get_hsb():
                hsb.sought_care[false_neg_uids] = False
            self.tested[false_neg_uids] = False
        return
    
    def update_results(self):
        """ Store results """
        ti = self.ti
        self.results.n_tested[ti] = len(self._selected)
        self.results.n_positive[ti] = len(self._pos)
        self.results.n_negative[ti] = len(self._neg)
        return

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_positive[:] = np.cumsum(self.results.n_positive)
        self.results.cum_negative[:] = np.cumsum(self.results.n_negative)
        return
    
    def shrink(self):
        """ Delete temporary results """
        self._selected = None
        self._results = None
        self._pos = None
        self._neg = None
        return
