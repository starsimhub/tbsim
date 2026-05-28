"""Treatment products and delivery for TB."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
from .products import ProductMulti # Not yet available for import in TBsim

__all__ = ['drug_params', 'Tx', 'TxMulti', 'DOTS', 'DOTSImproved', 'FirstLine', 'SecondLine', 'TxDelivery']

# Drug parameters keyed by drug type name.
# Each entry maps to a dict of clinical parameters.
drug_params = dict(
    dots                = dict(cure_prob=0.85, inactivation_rate=0.10, resistance_rate=0.02, relapse_rate=0.05, mortality_rate=0.80, duration=ss.days(180), adherence_rate=0.85, cost_per_course=100),
    dots_improved       = dict(cure_prob=0.90, inactivation_rate=0.08, resistance_rate=0.015, relapse_rate=0.03, mortality_rate=0.85, duration=ss.days(180), adherence_rate=0.90, cost_per_course=150),
    empiric_treatment   = dict(cure_prob=0.70, inactivation_rate=0.15, resistance_rate=0.05, relapse_rate=0.10, mortality_rate=0.60, duration=ss.days(90),  adherence_rate=0.75, cost_per_course=80),
    first_line_combo    = dict(cure_prob=0.95, inactivation_rate=0.05, resistance_rate=0.01, relapse_rate=0.02, mortality_rate=0.90, duration=ss.days(120), adherence_rate=0.88, cost_per_course=200),
    second_line_combo   = dict(cure_prob=0.75, inactivation_rate=0.12, resistance_rate=0.03, relapse_rate=0.08, mortality_rate=0.70, duration=ss.days(240), adherence_rate=0.80, cost_per_course=500),
    third_line_combo    = dict(cure_prob=0.60, inactivation_rate=0.20, resistance_rate=0.08, relapse_rate=0.15, mortality_rate=0.50, duration=ss.days(360), adherence_rate=0.70, cost_per_course=1000),
    latent_treatment    = dict(cure_prob=0.90, inactivation_rate=0.02, resistance_rate=0.005, relapse_rate=0.01, mortality_rate=0.95, duration=ss.days(90),  adherence_rate=0.85, cost_per_course=50),
)


class Tx(ss.Product):
    """
    TB treatment product that encapsulates drug efficacy, duration, adherence, and relapse.

    Args:
        efficacy: Probability of cure given adherence (0-1). Default 0.85.
        dur_treatment: Treatment duration distribution. Default 180 days.
        adherence: Probability of completing the full course (0-1). Default 0.85.
        p_relapse: Probability that a successfully cured agent relapses to active TB.
            Default 0.05 (matches drug_params['dots']['relapse_rate']).
        dur_relapse: Distribution for time-to-relapse (days), measured from the
            end of successful treatment. Default constant 1.5 years.
        drug_type: If provided (e.g. 'dots', 'first_line_combo'), overrides
            efficacy, dur_treatment, adherence, and p_relapse with drug-specific values.
    """

    def __init__(self, efficacy=0.85, dur_treatment=None, adherence=0.85,
                 p_relapse=0.05, dur_relapse=None, drug_type=None, **kwargs):
        super().__init__()
        if drug_type is not None:
            dp = drug_params[drug_type]
            efficacy = dp['cure_prob']
            if dur_treatment is None:
                dur_treatment = ss.constant(v=dp['duration'].days)
            adherence = dp['adherence_rate']
            p_relapse = dp['relapse_rate']
        else:
            if dur_treatment is None:
                dur_treatment = ss.constant(v=ss.days(180).days)

        if dur_relapse is None:
            dur_relapse = ss.constant(v=ss.years(1.5).days)

        self.define_pars(
            p_adherence = ss.bernoulli(adherence),
            p_success = ss.bernoulli(efficacy),
            p_relapse = ss.bernoulli(p_relapse),
            dur_treatment = dur_treatment,
            dur_relapse = dur_relapse,
        )
        self.update_pars(**kwargs)
        return

    def administer(self, sim, uids):
        """
        Roll treatment outcomes for agents starting treatment.

        First rolls adherence (will they complete the course?), then rolls
        cure probability for adherent agents. Non-adherent agents are failures.
        Among successful agents, a subset is pre-rolled for future relapse.

        Returns:
            dict with 'success', 'failure', and 'relapse' UIDs.
            'relapse' is a subset of 'success'.
        """
        # TODO: non-adherent agents currently get zero efficacy; consider partial
        # efficacy for incomplete courses (e.g. scaled by fraction of course completed)
        adherent, non_adherent = self.pars.p_adherence.filter(uids, both=True)
        success_uids, adherent_fail = self.pars.p_success.filter(adherent, both=True)
        failure_uids = ss.uids.cat([non_adherent, adherent_fail])
        relapse_uids = self.pars.p_relapse.filter(success_uids)
        return {'success': success_uids, 'failure': failure_uids, 'relapse': relapse_uids}


class TxMulti(ProductMulti):
    """
    TB treatment product that supports more than one outcome (e.g. success, failure, partial).

    Uses a DataFrame of state-to-outcome probabilities, just like Dx.
    See ProductMulti for full details on the df format.

    Args:
        df: DataFrame with required columns (state, result, probability) and
            optional filter columns (age_min, age_max, hiv).
        hierarchy: List of result strings in priority order, e.g. ['success', 'failure'].
    """
    pass


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
            ss.FloatArr('ti_treatment_start'),
            ss.FloatArr('ti_treatment_end'),
            ss.BoolArr('pending_success', default=False),
            ss.BoolArr('pending_failure', default=False),
            ss.BoolArr('pending_relapse', default=False),
            ss.FloatArr('ti_relapse'),
            ss.IntArr('prior_state', default=int(TBS.SUSCEPTIBLE)),
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
        active_tb_mask = np.isin(tb.state, TBS.active_tb_states())
        active_tb_uids = ss.uids(np.where(active_tb_mask)[0])
        return diagnosed_uids & active_tb_uids

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_success', dtype=int),
            ss.Result('n_failure', dtype=int),
            ss.Result('n_relapsed', dtype=int),
            ss.Result('cum_success', dtype=int),
            ss.Result('cum_failure', dtype=int),
            ss.Result('cum_relapsed', dtype=int),
        )
        return

    def step(self):
        """ Coordinate everything that happens on the step; details are in methods below """
        self.step_eligibility()        # Check eligibility
        self.step_start_treatment()    # Start treatment for newly eligible agents
        self.step_check_completion()   # Check if any on-treatment agents have completed
        self.step_success()            # Handle completed successes
        self.step_failures()           # Handle completed failures
        self.step_relapses()           # Trigger relapses scheduled at earlier steps
        return

    def step_eligibility(self):
        """ Check agents for eligibility, and start treatment for those eligible """
        self._elig_uids = self._get_eligible(self.sim)
        return self._elig_uids

    def step_start_treatment(self):
        """
        Start treatment for eligible agents.

        Latent/acute agents are cleared immediately. Active TB agents are put
        into TREATMENT state and their outcome (success/failure) is pre-rolled
        via the product, but not resolved until dur_treatment has elapsed.
        """
        tb = self.sim.get_tb()
        uids = self._elig_uids

        # ACUTE or INFECTION: clear immediately (no treatment course needed)
        latent = uids[np.isin(tb.state[uids], [TBS.ACUTE, TBS.INFECTION])]
        tb.state[latent] = TBS.CLEARED
        tb.rr_reinfection[latent] = tb.pars.rr_reinfection_cleared
        if tb.pars.dur_reinfection_protection is not None and len(latent):
            tb.ti_rr_reinfection_wane[latent] = self.ti + tb.pars.dur_reinfection_protection.rvs(latent)
        tb.infected[latent] = False
        tb.susceptible[latent] = True

        # Active TB: put on treatment
        active = uids[np.isin(tb.state[uids], [TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC])]
        if len(active) == 0:
            self._newly_treated = ss.uids()
            return

        self.prior_state[active] = tb.state[active]
        tb.state[active] = TBS.TREATMENT
        tb.on_treatment[active] = True
        # Clear any stale relapse schedule from a prior treatment cycle.
        # The new cycle will roll a fresh schedule below if the agent is selected.
        self.pending_relapse[active] = False
        tb.results['new_notifications_15+'][tb.ti] += np.count_nonzero(self.sim.people.age[active] >= 15)

        # Record treatment timing (convert dur from days to timesteps)
        dur_days = self.product.pars.dur_treatment.rvs(active)
        dur_steps = dur_days / self.dt.days
        self.ti_treatment_start[active] = self.ti
        self.ti_treatment_end[active] = self.ti + dur_steps
        self.n_times_treated[active] += 1

        # Pre-roll outcomes (stored as pending, resolved when treatment completes)
        outcomes = self.product.administer(self.sim, active)
        self.pending_success[outcomes.get('success', ss.uids())] = True
        self.pending_failure[outcomes.get('failure', ss.uids())] = True

        # Pre-roll relapses (subset of successes). Schedule the relapse time now;
        # the actual transition back to SYMPTOMATIC is resolved in step_relapses().
        relapse_uids = outcomes.get('relapse', ss.uids())
        if len(relapse_uids):
            self.pending_relapse[relapse_uids] = True
            relapse_days = self.product.pars.dur_relapse.rvs(relapse_uids)
            relapse_steps = relapse_days / self.dt.days
            # Relapse clock starts when treatment ends (not when it began)
            self.ti_relapse[relapse_uids] = self.ti_treatment_end[relapse_uids] + relapse_steps

        self._newly_treated = active
        return

    def step_check_completion(self):
        """ Find agents whose treatment course has completed this step """
        tb = self.sim.get_tb()
        on_tx = tb.on_treatment.uids
        if len(on_tx) == 0:
            self._success = ss.uids()
            self._fail = ss.uids()
            return

        completed = on_tx[self.ti >= self.ti_treatment_end[on_tx]]
        self._success = completed[self.pending_success[completed]]
        self._fail = completed[self.pending_failure[completed]]
        return

    def step_success(self):
        """ Successful treatment clears infection and sets reinfection protection """
        tb = self.sim.get_tb()
        success_uids = self._success
        if len(success_uids) == 0:
            return

        tb.state[success_uids] = TBS.CLEARED
        tb.on_treatment[success_uids] = False
        tb.susceptible[success_uids] = True
        tb.infected[success_uids] = False
        if self._dx is not None:
            self._dx.diagnosed[success_uids] = False
        self.tb_treatment_success[success_uids] = True
        self.pending_success[success_uids] = False

        # Reinfection protection
        tb.rr_reinfection[success_uids] = tb.pars.rr_reinfection_treat
        if tb.pars.dur_reinfection_protection is not None and len(success_uids):
            tb.ti_rr_reinfection_wane[success_uids] = tb.ti + tb.pars.dur_reinfection_protection.rvs(success_uids)
        return

    def step_failures(self):
        """ Handle failures: return to prior TB state and trigger re-care-seeking """
        tb = self.sim.get_tb()
        failure_uids = self._fail
        if len(failure_uids) == 0:
            return

        tb.state[failure_uids] = self.prior_state[failure_uids]
        tb.on_treatment[failure_uids] = False
        self.treatment_failure[failure_uids] = True
        self.pending_failure[failure_uids] = False

        if self.reset_flags and self._dx is not None:
            self._dx.diagnosed[failure_uids] = False
            self._dx.tested[failure_uids] = False

        # Trigger renewed care-seeking
        if hsb := self.sim.get_hsb():
            hsb.sought_care[failure_uids] = False
        if self._dx is not None:
            self._dx.care_seeking_multiplier[failure_uids] *= self.reseek_multiplier
        return

    def step_relapses(self):
        """
        Trigger relapses for agents whose scheduled relapse time has arrived.

        Eligible pre-states: CLEARED (no active disease), INFECTION (latently
        reinfected, no active disease yet), NON_INFECTIOUS (early active TB).
        Agents in ASYMPTOMATIC, SYMPTOMATIC, or TREATMENT are silently dropped
        - they're already on an active disease trajectory (or being treated),
        and forcing a state change would corrupt counters or pull them out of
        an active treatment course.

        Relapsing agents return to SYMPTOMATIC active TB and have their diagnosed
        flag reset so they can be re-detected and re-treated.
        """
        tb = self.sim.get_tb()
        pending = self.pending_relapse.uids
        if len(pending) == 0:
            self._relapsed = ss.uids()
            return

        due = pending[(self.ti >= self.ti_relapse[pending]) & self.sim.people.alive[pending]]
        if len(due) == 0:
            self._relapsed = ss.uids()
            return

        # Always clear the schedule for due agents, whether or not they actually relapse.
        # Otherwise stale schedules would re-fire on subsequent steps.
        eligible_states = [TBS.CLEARED, TBS.INFECTION, TBS.NON_INFECTIOUS]
        relapsing = due[np.isin(tb.state[due], eligible_states)]
        self.pending_relapse[due] = False

        if len(relapsing) == 0:
            self._relapsed = ss.uids()
            return

        tb.state[relapsing] = TBS.SYMPTOMATIC
        tb.infected[relapsing] = True
        tb.susceptible[relapsing] = False

        # Reset diagnosed flag so they can be re-detected by DxDelivery and re-treated
        if self._dx is not None:
            self._dx.diagnosed[relapsing] = False
            self._dx.tested[relapsing] = False

        self._relapsed = relapsing
        return

    def update_results(self):
        self.results.n_treated[self.ti] = len(self._newly_treated)
        self.results.n_success[self.ti] = len(self._success)
        self.results.n_failure[self.ti] = len(self._fail)
        self.results.n_relapsed[self.ti] = len(getattr(self, '_relapsed', ss.uids()))
        return

    def finalize_results(self):
        super().finalize_results()
        self.results.cum_success[:] = np.cumsum(self.results.n_success)
        self.results.cum_failure[:] = np.cumsum(self.results.n_failure)
        self.results.cum_relapsed[:] = np.cumsum(self.results.n_relapsed)
        return

    def shrink(self):
        """ Remove temporary results """
        self._elig_uids = None
        self._newly_treated = None
        self._success = None
        self._fail = None
        self._relapsed = None
        self._dx = None
        return
