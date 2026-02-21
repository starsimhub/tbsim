"""
BCG vaccination: Product + Intervention following the Starsim Vx pattern.

``BCGVx``      — vaccine product  (biological effect, per-agent state)
``BCGRoutine`` — delivery intervention (eligibility, coverage, scheduling)
"""

import numpy as np
import starsim as ss

__all__ = ['BCGVx', 'BCGRoutine']


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class BCGVx(ss.Vx):
    """
    BCG vaccine product.

    Handles immunological take, per-agent modifier sampling, protection
    duration tracking, and per-step modifier application to disease
    ``rr_activation``, ``rr_clearance``, and ``rr_death`` arrays.

    Parameters
    ----------
    disease : str
        Key of the disease module to target (default ``'tb'``).
    p_take : ss.bernoulli
        Probability of immunological response post-vaccination.
    dur_immune : ss.Dist
        Duration of protection (sampled per individual).
    activation_modifier / clearance_modifier / death_modifier : ss.Dist
        Per-individual risk-modifier distributions.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            disease='tb',
            p_take=ss.bernoulli(p=0.8),
            dur_immune=ss.constant(v=ss.years(10)),
            activation_modifier=ss.uniform(0.5, 0.65),
            clearance_modifier=ss.uniform(1.3, 1.5),
            death_modifier=ss.uniform(0.05, 0.15),
        )
        self.update_pars(pars)

        self.define_states(
            ss.BoolArr('bcg_protected', default=False),
            ss.FloatArr('ti_bcg_protection_expires'),
            ss.FloatArr('bcg_activation_modifier_applied'),
            ss.FloatArr('bcg_clearance_modifier_applied'),
            ss.FloatArr('bcg_death_modifier_applied'),
        )

    def administer(self, people, uids):
        """
        Administer BCG to *uids* (already accepted by the delivery intervention).

        Applies ``p_take`` filter, samples per-agent modifiers, and sets
        protection expiry.  Does **not** touch ``rr_*`` arrays — that happens
        in :meth:`apply_protection` each step.

        Returns
        -------
        ss.uids
            UIDs of agents who responded immunologically.
        """
        if len(uids) == 0:
            return ss.uids()

        responders = self.pars.p_take.filter(uids)
        if len(responders) > 0:
            dur = self.pars.dur_immune.rvs(responders)
            self.ti_bcg_protection_expires[responders] = self.ti + dur
            self.bcg_protected[responders] = True
            self.bcg_activation_modifier_applied[responders] = self.pars.activation_modifier.rvs(responders)
            self.bcg_clearance_modifier_applied[responders] = self.pars.clearance_modifier.rvs(responders)
            self.bcg_death_modifier_applied[responders] = self.pars.death_modifier.rvs(responders)

        return responders

    def expire_protection(self):
        """Remove protection for agents past their expiry time."""
        protected_uids = self.bcg_protected.uids
        if len(protected_uids) > 0:
            expired = protected_uids[self.ti > self.ti_bcg_protection_expires[protected_uids]]
            self.bcg_protected[expired] = False

    def apply_protection(self):
        """Multiply ``rr_*`` arrays for all currently protected agents."""
        protected = self.bcg_protected.uids
        if len(protected) > 0:
            tb = self.sim.diseases[self.pars.disease]
            tb.rr_activation[protected] *= self.bcg_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.bcg_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.bcg_death_modifier_applied[protected]


# ---------------------------------------------------------------------------
# Delivery intervention
# ---------------------------------------------------------------------------

class BCGRoutine(ss.Intervention):
    """
    Routine BCG vaccination delivery.

    Handles age-based eligibility, one-time-offer semantics, coverage
    filtering, and orchestrates per-step product operations (expiry,
    new vaccination, modifier re-application).

    Parameters
    ----------
    product : BCGVx
        The BCG vaccine product (created automatically if not provided).
    coverage : ss.bernoulli
        Fraction of eligible individuals vaccinated (applied once per person).
    start / stop : ss.date
        Campaign window.
    age_range : list
        ``[min_age, max_age]`` for eligibility.
    """

    def __init__(self, product=None, pars=None, **kwargs):
        super().__init__(**kwargs)

        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
            coverage=ss.bernoulli(p=0.5),
            age_range=[0, 5],
        )
        self.update_pars(pars)
        self.min_age = self.pars.age_range[0]
        self.max_age = self.pars.age_range[1]

        self.product = product if product is not None else BCGVx()

        self.define_states(
            ss.BoolArr('bcg_offered', default=False),
            ss.BoolArr('bcg_vaccinated', default=False),
            ss.FloatArr('ti_bcg_vaccinated'),
        )

    def check_eligibility(self):
        """Select eligible individuals (coverage applied once per person)."""
        newly_eligible = (
            (self.sim.people.age >= self.min_age) &
            (self.sim.people.age <= self.max_age) &
            ~self.bcg_vaccinated &
            ~self.bcg_offered
        ).uids
        self.bcg_offered[newly_eligible] = True
        selected = self.pars.coverage.filter(newly_eligible)
        return selected

    def step(self):
        """
        Two-phase BCG step:

        Phase A — Update vaccination roster (expire old, add new).
        Phase B — Apply stored rr modifiers for all ``bcg_protected``.
        """
        now = self.sim.now
        now_date = now.date() if hasattr(now, 'date') else now
        if now_date < self.pars.start.date() or now_date > self.pars.stop.date():
            return

        # --- Phase A: Update vaccination roster ---

        # 1. Expire protection
        self.product.expire_protection()

        # 2. New vaccinations
        eligible = self.check_eligibility()
        if len(eligible) > 0:
            self.bcg_vaccinated[eligible] = True
            self.ti_bcg_vaccinated[eligible] = self.ti
            self.product.administer(self.sim.people, eligible)

        # --- Phase B: Apply rr modifiers for all currently protected ---
        self.product.apply_protection()

    def init_results(self):
        super().init_results()
        if hasattr(self, 'results') and 'n_newly_vaccinated' in self.results:
            return
        self.define_results(
            ss.Result('n_newly_vaccinated', dtype=int),
            ss.Result('n_protected', dtype=int),
        )

    def update_results(self):
        newly_vaccinated = np.sum((self.ti_bcg_vaccinated == self.ti) & self.bcg_vaccinated)
        self.results['n_newly_vaccinated'][self.ti] = newly_vaccinated
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.bcg_protected)
