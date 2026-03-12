"""Diagnostic delivery intervention for TB testing."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['DxDelivery']

_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


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
        # Give product a unique name based on this intervention's name to avoid
        # collisions when multiple DxDelivery instances use the same product class
        product.name = f'{self.name}_product'
        self.product = product
        self._coverage_val = coverage
        self._eligibility_fn = eligibility
        self.result_state = result_state
        self._csm_value = care_seeking_multiplier  # Store parameter value with private name

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

        # Tracking for results
        self._n_tested = 0
        self._n_positive = 0
        self._n_negative = 0
        self.update_pars(**kwargs)
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
        if self._eligibility_fn is not None:
            return ss.uids(self._eligibility_fn(sim))
        return (self.sought_care & (~self.diagnosed) & sim.people.alive).uids

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        eligible = self._get_eligible(sim)
        if len(eligible) == 0:
            self._n_tested = self._n_positive = self._n_negative = 0
            return

        # Coverage filter
        selected = self.pars.p_coverage.filter(eligible)
        if len(selected) == 0:
            self._n_tested = self._n_positive = self._n_negative = 0
            return

        # Administer diagnostic product
        results = self.product.administer(sim, selected)
        pos_uids = results.get('positive', ss.uids())
        neg_uids = results.get('negative', ss.uids())

        # Set result state
        result_arr = getattr(ppl, self.result_state)
        result_arr[pos_uids] = True

        # Update tracking states
        self.tested[selected] = True
        self.n_times_tested[selected] += 1
        self.test_result[selected] = np.isin(selected, pos_uids)

        # Handle false negatives: TB-positive agents who tested negative
        tb_states = np.asarray(tb.state[neg_uids])
        has_tb = np.isin(tb_states, _ACTIVE_TB_STATES)
        false_neg_uids = neg_uids[has_tb]

        if len(false_neg_uids) > 0 and self._csm_value != 1.0:
            unboosted = false_neg_uids[~self.multiplier_applied[false_neg_uids]]
            if len(unboosted) > 0:
                self.care_seeking_multiplier[unboosted] *= self._csm_value
                self.multiplier_applied[unboosted] = True

        # Reset flags for false negatives to allow retry
        if len(false_neg_uids) > 0:
            self.sought_care[false_neg_uids] = False
            self.tested[false_neg_uids] = False

        # Store for results
        self._n_tested = len(selected)
        self._n_positive = len(pos_uids)
        self._n_negative = len(neg_uids)
        return

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

    def update_results(self):
        self.results.n_tested[self.ti] = self._n_tested
        self.results.n_positive[self.ti] = self._n_positive
        self.results.n_negative[self.ti] = self._n_negative

        if self.ti > 0:
            self.results['cum_positive'][self.ti] = self.results['cum_positive'][self.ti - 1] + self._n_positive
            self.results['cum_negative'][self.ti] = self.results['cum_negative'][self.ti - 1] + self._n_negative
        else:
            self.results['cum_positive'][self.ti] = self._n_positive
            self.results['cum_negative'][self.ti] = self._n_negative

        self._n_tested = self._n_positive = self._n_negative = 0
        return
    
    def finalize_results(self):
        self.results.cum_positive[:] = self.results.n_positive.cumsum()
        self.results.cum_negative[:] = self.results.n_negative.cumsum()
        return
