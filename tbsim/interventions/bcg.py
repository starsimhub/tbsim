import numpy as np
import starsim as ss
import logging

__all__ = ['BCGProtection']
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class BCGProtection(ss.Intervention):
    """
    BCG vaccination intervention for tuberculosis prevention.

    Identifies individuals within a configurable age range, vaccinates a fraction
    based on ``coverage``, and applies individual-level risk modifiers to TB
    activation, clearance, and death rates for the duration of ``dur_immune``.

    Parameters
    ----------
    coverage : ss.bernoulli
        Fraction of eligible individuals vaccinated (applied once per person).
    start / stop : ss.date
        Campaign window.
    p_take : ss.bernoulli
        Probability of immunological response post-vaccination.
    dur_immune : ss.Dist
        Duration of protection (default: constant 10 years). Sampled per-individual.
    age_range : list
        [min_age, max_age] for eligibility.
    activation_modifier / clearance_modifier / death_modifier : ss.Dist
        Per-individual risk modifier distributions.
    """

    def __init__(self, pars={}, **kwargs):
        super().__init__(**kwargs)

        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
            coverage=ss.bernoulli(p=pars.get('coverage', 0.5)),
            p_take=ss.bernoulli(p=0.8),
            dur_immune=ss.constant(v=ss.years(10)),
            age_range=[0, 5],
            activation_modifier=ss.uniform(0.5, 0.65),
            clearance_modifier=ss.uniform(1.3, 1.5),
            death_modifier=ss.uniform(0.05, 0.15),
        )
        self.update_pars(pars)
        self.min_age = self.pars.age_range[0]
        self.max_age = self.pars.age_range[1]

        self.define_states(
            ss.BoolArr('bcg_offered', default=False),
            ss.BoolArr('bcg_vaccinated', default=False),
            ss.BoolArr('bcg_protected', default=False),
            ss.FloatArr('ti_bcg_vaccinated'),
            ss.FloatArr('ti_bcg_protection_expires'),
            ss.FloatArr('bcg_activation_modifier_applied'),
            ss.FloatArr('bcg_clearance_modifier_applied'),
            ss.FloatArr('bcg_death_modifier_applied'),
        )

    def check_eligibility(self):
        """Select eligible individuals for vaccination (coverage applied once per person)."""
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
          Phase A — Update vaccination roster (expire old, add new)
          Phase B — Apply stored rr modifiers for all bcg_protected
        """
        # Temporal eligibility
        now = self.sim.now
        now_date = now.date() if hasattr(now, 'date') else now
        if now_date < self.pars.start.date() or now_date > self.pars.stop.date():
            return

        current_time = self.ti

        # --- Phase A: Update vaccination roster ---

        # 1. Expire protection
        protected_uids = self.bcg_protected.uids
        if len(protected_uids) > 0:
            expired = protected_uids[current_time > self.ti_bcg_protection_expires[protected_uids]]
            self.bcg_protected[expired] = False

        # 2. New vaccinations
        eligible = self.check_eligibility()
        if len(eligible) > 0:
            self.bcg_vaccinated[eligible] = True
            self.ti_bcg_vaccinated[eligible] = current_time

            # Only responders get protection
            responders = self.pars.p_take.filter(eligible)
            if len(responders) > 0:
                dur = self.pars.dur_immune.rvs(responders)
                self.ti_bcg_protection_expires[responders] = current_time + dur
                self.bcg_protected[responders] = True
                self.bcg_activation_modifier_applied[responders] = self.pars.activation_modifier.rvs(responders)
                self.bcg_clearance_modifier_applied[responders] = self.pars.clearance_modifier.rvs(responders)
                self.bcg_death_modifier_applied[responders] = self.pars.death_modifier.rvs(responders)

        # --- Phase B: Apply rr modifiers for all currently protected ---

        protected = self.bcg_protected.uids
        if len(protected) > 0:
            tb = self.sim.diseases.tb
            tb.rr_activation[protected] *= self.bcg_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.bcg_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.bcg_death_modifier_applied[protected]

    def init_results(self):
        super().init_results()
        if hasattr(self, 'results') and 'n_newly_vaccinated' in self.results:
            return
        self.define_results(
            ss.Result('n_newly_vaccinated', dtype=int),
            ss.Result('n_protected', dtype=int),
        )

    def update_results(self):
        current_time = self.ti
        newly_vaccinated = np.sum((self.ti_bcg_vaccinated == current_time) & self.bcg_vaccinated)
        self.results['n_newly_vaccinated'][self.ti] = newly_vaccinated
        self.results['n_protected'][self.ti] = np.count_nonzero(self.bcg_protected)
