"""Treatment products and delivery for TB."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBSL
__all__ = ['drug_params', 'Tx', 'DOTS', 'DOTSImproved', 'FirstLine', 'SecondLine', 'TxDelivery']

# Drug parameters keyed by drug type name.
# Each entry maps to a dict of clinical parameters.
drug_params = dict(
    dots                = dict(cure_prob=0.85, inactivation_rate=0.10, resistance_rate=0.02, relapse_rate=0.05, mortality_rate=0.80, duration=180, adherence_rate=0.85, cost_per_course=100),
    dots_improved       = dict(cure_prob=0.90, inactivation_rate=0.08, resistance_rate=0.015, relapse_rate=0.03, mortality_rate=0.85, duration=180, adherence_rate=0.90, cost_per_course=150),
    empiric_treatment   = dict(cure_prob=0.70, inactivation_rate=0.15, resistance_rate=0.05, relapse_rate=0.10, mortality_rate=0.60, duration=90,  adherence_rate=0.75, cost_per_course=80),
    first_line_combo    = dict(cure_prob=0.95, inactivation_rate=0.05, resistance_rate=0.01, relapse_rate=0.02, mortality_rate=0.90, duration=120, adherence_rate=0.88, cost_per_course=200),
    second_line_combo   = dict(cure_prob=0.75, inactivation_rate=0.12, resistance_rate=0.03, relapse_rate=0.08, mortality_rate=0.70, duration=240, adherence_rate=0.80, cost_per_course=500),
    third_line_combo    = dict(cure_prob=0.60, inactivation_rate=0.20, resistance_rate=0.08, relapse_rate=0.15, mortality_rate=0.50, duration=360, adherence_rate=0.70, cost_per_course=1000),
    latent_treatment    = dict(cure_prob=0.90, inactivation_rate=0.02, resistance_rate=0.005, relapse_rate=0.01, mortality_rate=0.95, duration=90,  adherence_rate=0.85, cost_per_course=50),
)


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
        """ Coordinate everything that happens on the step; details are in methods below """
        self.step_eligibility() # Check eligibility
        self.step_start_treatment() # Start treatment for eligible agents
        self.step_administer() # Administer product and handle outcomes
        self.step_success() # Handle success
        self.step_failures() # Handle failures
        return
    
    def step_eligibility(self):
        """ Check agents for eligibility, and start treatment for those eligible """
        self._elig_uids = self._get_eligible(self.sim)
        return self._elig_uids

    def step_start_treatment(self):
        """ Start treatment for eligible agents """
        tb = self.sim.get_tb()
        uids = self._elig_uids

        # ACUTE or INFECTION: clear (NB, acute may not be present in all models, but fine to check)
        latent = uids[np.isin(tb.state[uids], [TBSL.ACUTE, TBSL.INFECTION])]
        tb.state[latent] = TBSL.CLEARED
        tb.rr_reinfection[latent] = tb.pars.rr_reinfection_cleared
        if tb.pars.dur_reinfection_protection is not None and len(latent):
            self.ti_rr_reinfection_wane[latent] = self.ti + tb.pars.dur_reinfection_protection.rvs(latent)
        tb.infected[latent] = False
        tb.susceptible[latent] = True

        # Active TB: put on treatment
        active = uids[np.isin(tb.state[uids], [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        tb.state[active] = TBSL.TREATMENT
        tb.on_treatment[active] = True
        tb.results['new_notifications_15+'][tb.ti] += np.count_nonzero(self.sim.people.age[active] >= 15) # TODO: should this result be in the intervention instead?

        # Only proceed with agents actually on treatment
        tx_uids = uids[tb.on_treatment[uids]]
        self.n_times_treated[tx_uids] += 1
        self._tx_uids = tx_uids
        return
    
    def step_administer(self):
        """ Administer product to get success/failure """
        outcomes = self.product.administer(self.sim, self._tx_uids)
        self._outcomes = outcomes
        self._success = outcomes.get('success', ss.uids())
        self._fail = outcomes.get('failure', ss.uids())
        return
    
    def step_success(self):
        """ Successful treatment clears infection """
        tb = self.sim.get_tb()
        success_uids = self._success
        if len(success_uids) > 0:
            tb.state[success_uids] = TBSL.CLEARED
            tb.on_treatment[success_uids] = False
            tb.susceptible[success_uids] = True
            tb.infected[success_uids] = False
            if self._dx is not None:
                self._dx.diagnosed[success_uids] = False
            self.tb_treatment_success[success_uids] = True
        return
    
    def step_failures(self):
        """ Handle failures """
        failure_uids = self._fail
        if len(failure_uids) > 0:
            self.treatment_failure[failure_uids] = True

            if self.reset_flags and self._dx is not None:
                self._dx.diagnosed[failure_uids] = False
                self._dx.tested[failure_uids] = False

            # Trigger renewed care-seeking
            if hsb := self.sim.get_hsb():
                hsb.sought_care[failure_uids] = False
            if self._dx is not None:
                self._dx.care_seeking_multiplier[failure_uids] *= self.reseek_multiplier
        return

    def update_results(self):
        self.results.n_treated[self.ti] = len(self._tx_uids)
        self.results.n_success[self.ti] = len(self._success)
        self.results.n_failure[self.ti] = len(self._fail)
        return

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_success[:] = np.cumsum(self.results.n_success)
        self.results.cum_failure[:] = np.cumsum(self.results.n_failure)
        return
    
    def shrink(self):
        """ Remove temporary results """
        self._outcomes = None
        self._elig_uids = None
        self._tx_uids = None
        self._success = None
        self._fail = None
        self._dx = None # Link to treatment module
        return
