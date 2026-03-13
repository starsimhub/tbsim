"""Treatment products and delivery for TB."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL
from .drug_types import drug_params

__all__ = ['Tx', 'DOTS', 'DOTSImproved', 'FirstLine', 'SecondLine', 'TxDelivery']


class Tx(ss.Product):
    """
    TB treatment product that encapsulates drug efficacy.

    Args:
        efficacy: Probability of treatment success (0-1). Default 0.85.
        drug_type: If provided (e.g. 'dots', 'first_line_combo'), overrides efficacy with drug-specific cure probability.
    """

    def __init__(self, efficacy=0.85, drug_type=None, **kwargs):
        super().__init__()
        if drug_type is not None:
            self.efficacy = drug_params[drug_type]['cure_prob']
        else:
            self.efficacy = efficacy

        self.define_pars(
            p_success = ss.bernoulli(self.efficacy)
        )
        self.update_pars(**kwargs)
        return

    def administer(self, sim, uids):
        """
        Administer treatment to agents.

        Returns:
            dict with 'success' and 'failure' UIDs.
        """
        success_uids, failure_uids = self.pars.p_success.filter(uids, both=True)
        return {'success': success_uids, 'failure': failure_uids}


class DOTS(Tx):
    """Standard DOTS (85% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='dots', **kwargs)


class DOTSImproved(Tx):
    """Enhanced DOTS (90% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='dots_improved', **kwargs)


class FirstLine(Tx):
    """First-line combination therapy (95% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='first_line_combo', **kwargs)


class SecondLine(Tx):
    """Second-line therapy for MDR-TB (75% cure)."""

    def __init__(self, **kwargs):
        super().__init__(drug_type='second_line_combo', **kwargs)


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
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self.reseek_multiplier = reseek_multiplier
        self.reset_flags = reset_flags

        # Person-level states (only states owned by TxDelivery; shared states
        # like diagnosed/sought_care/tested are read from ppl, set by DxDelivery)
        self.define_states(
            ss.IntArr('n_times_treated', default=0),
            ss.BoolState('tb_treatment_success', default=False),
            ss.BoolArr('treatment_failure', default=False),
        )
        self.update_pars(**kwargs)
        product.name = f'{self.name}_product'
        return

    def init_post(self):
        super().init_post()
        self._dx = self.sim.get_dx(result_state='diagnosed')

    def _get_eligible(self, sim):
        """Get eligible UIDs using custom or default eligibility."""
        if self.eligibility is not None:
            return ss.uids(self.eligibility(sim))
        # Default: diagnosed, active TB, alive (read diagnosed from DxDelivery)
        if self._dx is None:
            return ss.uids()
        diagnosed_uids = (self._dx.diagnosed & sim.people.alive).uids
        tb = tbsim.get_tb(sim)
        active_tb_mask = np.isin(tb.state, TBSL.active_tb_states())
        active_tb_uids = ss.uids(np.where(active_tb_mask)[0])
        return diagnosed_uids & active_tb_uids

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_success', dtype=int),
            ss.Result('n_failure', dtype=int),
            ss.Result('cum_success', dtype=int),
            ss.Result('cum_failure', dtype=int),
        )
        return

    def step(self):
        sim = self.sim
        ppl = sim.people
        tb = tbsim.get_tb(sim)

        uids = self._get_eligible(sim)
        if len(uids) == 0:
            return

        # Start treatment (moves active -> TREATMENT state)
        tb.start_treatment(uids)

        # Only proceed with agents actually on treatment
        tx_uids = uids[tb.on_treatment[uids]]
        if len(tx_uids) == 0:
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
            if self._dx is not None:
                self._dx.diagnosed[success_uids] = False
            self.tb_treatment_success[success_uids] = True

        # Handle failures
        if len(failure_uids) > 0:
            self.treatment_failure[failure_uids] = True

            if self.reset_flags and self._dx is not None:
                self._dx.diagnosed[failure_uids] = False
                self._dx.tested[failure_uids] = False

            # Trigger renewed care-seeking
            if hsb := sim.get_hsb():
                hsb.sought_care[failure_uids] = False
            if self._dx is not None:
                self._dx.care_seeking_multiplier[failure_uids] *= self.reseek_multiplier

        # Store for results
        self.results.n_treated[self.ti] = len(tx_uids)
        self.results.n_success[self.ti] = len(success_uids)
        self.results.n_failure[self.ti] = len(failure_uids)
        return

    def update_results(self):
        pass

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_success[:] = np.cumsum(self.results.n_success)
        self.results.cum_failure[:] = np.cumsum(self.results.n_failure)
        return
