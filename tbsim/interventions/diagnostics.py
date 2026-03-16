"""Diagnostic products and delivery for TB testing."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL
from .products import ProductMulti # Not yet available for import in TBsim

__all__ = ['Dx', 'Xpert', 'OralSwab', 'FujiLAM', 'CAD', 'DxDelivery']


class Dx(ProductMulti):
    """
    TB diagnostic product defined by a DataFrame of state-to-result probabilities.

    Args:
        df: DataFrame with required columns (state, result, probability) and
            optional filter columns (age_min, age_max, hiv). Only include
            filter columns relevant to the specific diagnostic.
        hierarchy: List of result strings in priority order, e.g. ['positive', 'negative'].
            Agents in states not listed in df get the last entry (default negative).
    """
    pass


class Xpert(Dx):
    """Xpert MTB/RIF -- age-stratified (child/adult), TB-state-stratified."""

    columns = 'state,result,probability,age_min,age_max'.split(',')
    default_data = [
        #state                result      prob   min  max
        # Adults (15+)
        [TBSL.SYMPTOMATIC,    'positive', 0.909, 15,  99],
        [TBSL.SYMPTOMATIC,    'negative', 0.091, 15,  99],
        [TBSL.ASYMPTOMATIC,   'positive', 0.775, 15,  99],
        [TBSL.ASYMPTOMATIC,   'negative', 0.225, 15,  99],
        [TBSL.NON_INFECTIOUS, 'positive', 0.775, 15,  99],
        [TBSL.NON_INFECTIOUS, 'negative', 0.225, 15,  99],
        # Children (<15)
        [TBSL.SYMPTOMATIC,    'positive', 0.73,   0,  15],
        [TBSL.SYMPTOMATIC,    'negative', 0.27,   0,  15],
        [TBSL.ASYMPTOMATIC,   'positive', 0.73,   0,  15],
        [TBSL.ASYMPTOMATIC,   'negative', 0.27,   0,  15],
        [TBSL.NON_INFECTIOUS, 'positive', 0.73,   0,  15],
        [TBSL.NON_INFECTIOUS, 'negative', 0.27,   0,  15],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)
        return


class OralSwab(Dx):
    """Oral swab -- age-stratified, TB-state-stratified."""

    columns = 'state,result,probability,age_min,age_max'.split(',')
    default_data = [
        #state                result      prob  min  max
        # Adults (15+)
        [TBSL.SYMPTOMATIC,    'positive', 0.80, 15,  99],
        [TBSL.SYMPTOMATIC,    'negative', 0.20, 15,  99],
        [TBSL.ASYMPTOMATIC,   'positive', 0.30, 15,  99],
        [TBSL.ASYMPTOMATIC,   'negative', 0.70, 15,  99],
        [TBSL.NON_INFECTIOUS, 'positive', 0.30, 15,  99],
        [TBSL.NON_INFECTIOUS, 'negative', 0.70, 15,  99],
        # Children (<15)
        [TBSL.SYMPTOMATIC,    'positive', 0.25,  0,  15],
        [TBSL.SYMPTOMATIC,    'negative', 0.75,  0,  15],
        [TBSL.ASYMPTOMATIC,   'positive', 0.25,  0,  15],
        [TBSL.ASYMPTOMATIC,   'negative', 0.75,  0,  15],
        [TBSL.NON_INFECTIOUS, 'positive', 0.25,  0,  15],
        [TBSL.NON_INFECTIOUS, 'negative', 0.75,  0,  15],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)
        return


class FujiLAM(Dx):
    """FujiLAM -- HIV-stratified, age-stratified."""

    columns = 'state,result,probability,age_min,age_max,hiv'.split(',')
    default_data = [
        #state                result      prob   min  max  hiv
        # HIV+, adults
        [TBSL.SYMPTOMATIC,    'positive', 0.75,  15,  99,  True],
        [TBSL.SYMPTOMATIC,    'negative', 0.25,  15,  99,  True],
        [TBSL.ASYMPTOMATIC,   'positive', 0.75,  15,  99,  True],
        [TBSL.ASYMPTOMATIC,   'negative', 0.25,  15,  99,  True],
        [TBSL.NON_INFECTIOUS, 'positive', 0.75,  15,  99,  True],
        [TBSL.NON_INFECTIOUS, 'negative', 0.25,  15,  99,  True],
        # HIV-, adults
        [TBSL.SYMPTOMATIC,    'positive', 0.58,  15,  99,  False],
        [TBSL.SYMPTOMATIC,    'negative', 0.42,  15,  99,  False],
        [TBSL.ASYMPTOMATIC,   'positive', 0.58,  15,  99,  False],
        [TBSL.ASYMPTOMATIC,   'negative', 0.42,  15,  99,  False],
        [TBSL.NON_INFECTIOUS, 'positive', 0.58,  15,  99,  False],
        [TBSL.NON_INFECTIOUS, 'negative', 0.42,  15,  99,  False],
        # HIV+, children
        [TBSL.SYMPTOMATIC,    'positive', 0.579,  0,  15,  True],
        [TBSL.SYMPTOMATIC,    'negative', 0.421,  0,  15,  True],
        [TBSL.ASYMPTOMATIC,   'positive', 0.579,  0,  15,  True],
        [TBSL.ASYMPTOMATIC,   'negative', 0.421,  0,  15,  True],
        [TBSL.NON_INFECTIOUS, 'positive', 0.579,  0,  15,  True],
        [TBSL.NON_INFECTIOUS, 'negative', 0.421,  0,  15,  True],
        # HIV-, children
        [TBSL.SYMPTOMATIC,    'positive', 0.51,   0,  15,  False],
        [TBSL.SYMPTOMATIC,    'negative', 0.49,   0,  15,  False],
        [TBSL.ASYMPTOMATIC,   'positive', 0.51,   0,  15,  False],
        [TBSL.ASYMPTOMATIC,   'negative', 0.49,   0,  15,  False],
        [TBSL.NON_INFECTIOUS, 'positive', 0.51,   0,  15,  False],
        [TBSL.NON_INFECTIOUS, 'negative', 0.49,   0,  15,  False],
    ]

    def __init__(self, df=None, datafile=None, **kwargs):
        df = self._load_data(df, datafile, self.columns, self.default_data)
        super().__init__(df=df, hierarchy=['positive', 'negative'], **kwargs)
        return


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
        return


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
