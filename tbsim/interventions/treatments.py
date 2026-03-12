"""Treatment delivery intervention for TB."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL

__all__ = ['TxDelivery']

_ACTIVE_TB_STATES = [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC]


class TxDelivery(ss.Intervention):
    """
    Delivers a treatment product to eligible (diagnosed, active-TB) agents.

    Handles eligibility, treatment initiation via the TB module, success/failure
    outcomes from the product, and failure retry logic.

    Args:
        product: A Tx product instance.
        eligibility: Callable (sim) -> uids. Default: diagnosed & active TB & alive.
        reseek_multiplier: Care-seeking multiplier applied after treatment failure. Default 2.0.
        reset_flags: Whether to reset diagnosed/tested flags after failure. Default True.
    """

    def __init__(self, product, eligibility=None, reseek_multiplier=2.0,
                 reset_flags=True, **kwargs):
        super().__init__(**kwargs)
        product.name = f'{self.name}_product'
        self.product = product
        self._eligibility_fn = eligibility
        self._reseek_multiplier = reseek_multiplier
        self._reset_flags = reset_flags

        # Person-level states (only states owned by TxDelivery; shared states
        # like diagnosed/sought_care/tested are read from ppl, set by DxDelivery)
        self.define_states(
            ss.IntArr('n_times_treated', default=0),
            ss.BoolState('tb_treatment_success', default=False),
            ss.BoolArr('treatment_failure', default=False),
        )

        # Tracking for results
        self._n_treated = 0
        self._n_success = 0
        self._n_failure = 0

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self._eligibility_fn is not None:
            return ss.uids(self._eligibility_fn(sim))
        # Default: diagnosed, active TB, alive (read diagnosed from ppl, set by DxDelivery)
        ppl = sim.people
        diagnosed_uids = (ppl.diagnosed & ppl.alive).uids
        tb = tbsim.get_tb(sim)
        active_tb_uids = np.where(np.isin(np.asarray(tb.state), _ACTIVE_TB_STATES))[0]
        return ss.uids(np.intersect1d(diagnosed_uids, active_tb_uids))

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        uids = self._get_eligible(sim)
        if len(uids) == 0:
            self._n_treated = self._n_success = self._n_failure = 0
            return

        # Start treatment (moves active -> TREATMENT state)
        tb.start_treatment(uids)

        # Only proceed with agents actually on treatment
        tx_uids = uids[tb.on_treatment[uids]]
        if len(tx_uids) == 0:
            self._n_treated = self._n_success = self._n_failure = 0
            return

        self.n_times_treated[tx_uids] += 1

        # Administer product to get success/failure
        outcomes = self.product.administer(sim, tx_uids)
        success_uids = outcomes.get('success', ss.uids())
        failure_uids = outcomes.get('failure', ss.uids())

        # Successful treatment clears infection
        if len(success_uids) > 0:
            tb.state[success_uids] = TBSL.CLEARED
            tb.on_treatment[success_uids] = False
            tb.susceptible[success_uids] = True
            tb.infected[success_uids] = False
            ppl.diagnosed[success_uids] = False
            self.tb_treatment_success[success_uids] = True

        # Handle failures
        if len(failure_uids) > 0:
            self.treatment_failure[failure_uids] = True

            if self._reset_flags:
                ppl.diagnosed[failure_uids] = False
                if 'tested' in ppl.states:
                    ppl.tested[failure_uids] = False

            # Trigger renewed care-seeking
            if 'sought_care' in ppl.states:
                ppl.sought_care[failure_uids] = False
            if 'care_seeking_multiplier' in ppl.states:
                ppl.care_seeking_multiplier[failure_uids] *= self._reseek_multiplier

        # Store for results
        self._n_treated = len(tx_uids)
        self._n_success = len(success_uids)
        self._n_failure = len(failure_uids)

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_success', dtype=int),
            ss.Result('n_failure', dtype=int),
            ss.Result('cum_success', dtype=int),
            ss.Result('cum_failure', dtype=int),
        )

    def update_results(self):
        self.results['n_treated'][self.ti] = self._n_treated
        self.results['n_success'][self.ti] = self._n_success
        self.results['n_failure'][self.ti] = self._n_failure

        if self.ti > 0:
            self.results['cum_success'][self.ti] = self.results['cum_success'][self.ti - 1] + self._n_success
            self.results['cum_failure'][self.ti] = self.results['cum_failure'][self.ti - 1] + self._n_failure
        else:
            self.results['cum_success'][self.ti] = self._n_success
            self.results['cum_failure'][self.ti] = self._n_failure

        self._n_treated = self._n_success = self._n_failure = 0
