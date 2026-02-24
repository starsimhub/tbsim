"""
Shared intervention infrastructure for TBsim.

``TBProductRoutine`` — generic routine delivery for TB vaccine/treatment products.
"""

import numpy as np
import starsim as ss

__all__ = ['TBProductRoutine']


class TBProductRoutine(ss.Intervention):
    """
    Generic routine delivery for TB vaccine/treatment products.

    Handles date-windowed delivery with composable eligibility filters
    (age range, disease state, user-provided callable) and orchestrates
    per-step product operations.

    Any product that exposes ``update_roster()``, ``administer(people, uids)``,
    and ``apply_protection()`` can be plugged into this delivery.

    Parameters
    ----------
    product : ss.Vx
        The vaccine/treatment product.
    coverage : ss.bernoulli
        Fraction of eligible individuals who accept (applied once per person).
    start / stop : ss.date
        Campaign window.
    age_range : list or None
        ``[min_age, max_age]`` filter, or ``None`` to skip.
    eligible_states : list or None
        List of disease-state values (e.g. ``[TBSL.INFECTION]``) to filter on,
        or ``None`` to skip.
    eligibility : callable or None
        User-provided ``fn(sim) → BoolArr | uids`` for custom filtering
        (passed through to ``ss.Intervention``).
    """

    def __init__(self, product=None, pars=None, **kwargs):
        """Initialize with a product and delivery parameters (coverage, age range, date window)."""
        super().__init__(**kwargs)

        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
            coverage=ss.bernoulli(p=0.5),
            age_range=None,
            eligible_states=None,
        )
        self.update_pars(pars)

        self.product = product

        self.define_states(
            ss.BoolArr('offered', default=False),
            ss.BoolArr('initiated', default=False),
            ss.FloatArr('ti_initiated'),
        )

    def check_eligibility(self):
        """
        Combine built-in filters (age, disease state) with optional
        user-provided eligibility callable. Coverage applied once per person.
        """
        # Start with Starsim's eligibility callable (or all agents)
        eligible = super().check_eligibility()

        # Age filter
        if self.pars.age_range is not None:
            min_age, max_age = self.pars.age_range[0], self.pars.age_range[1]
            age_ok = (self.sim.people.age >= min_age) & (self.sim.people.age <= max_age)
            eligible = eligible.intersect(age_ok.uids)

        # Disease state filter
        if self.pars.eligible_states is not None:
            tb = self.sim.diseases[self.product.pars.disease]
            state_ok = np.isin(np.asarray(tb.state[eligible]), self.pars.eligible_states)
            eligible = eligible[state_ok]

        # Not yet offered (one-time offer)
        eligible = eligible.intersect((~self.offered).uids)
        self.offered[eligible] = True

        # Coverage filter
        return self.pars.coverage.filter(eligible)

    def step(self):
        """
        Routine delivery step:

        1. Check date window.
        2. Product handles internal phase transitions (``update_roster``).
        3. Find and deliver to newly eligible agents.
        4. Product applies ``rr_*`` modifiers for all currently protected.
        """
        now = self.sim.now
        now_date = now.date() if hasattr(now, 'date') else now
        if now_date < self.pars.start.date() or now_date > self.pars.stop.date():
            return

        # Product-internal transitions (expire protection, complete treatment, etc.)
        self.product.update_roster()

        # Deliver to newly eligible
        eligible = self.check_eligibility()
        if len(eligible) > 0:
            self.initiated[eligible] = True
            self.ti_initiated[eligible] = self.ti
            self.product.administer(self.sim.people, eligible)

        # Apply modifiers for all currently protected
        self.product.apply_protection()

    def init_results(self):
        """Define result channels for newly initiated and currently protected counts."""
        super().init_results()
        if hasattr(self, 'results') and 'n_newly_initiated' in self.results:
            return
        self.define_results(
            ss.Result('n_newly_initiated', dtype=int),
            ss.Result('n_protected', dtype=int),
        )

    def update_results(self):
        """Record number of newly initiated individuals this timestep."""
        newly_initiated = np.sum((self.ti_initiated == self.ti) & self.initiated)
        self.results['n_newly_initiated'][self.ti] = newly_initiated
