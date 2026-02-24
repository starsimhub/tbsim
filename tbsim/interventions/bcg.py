"""
BCG vaccination: Product + Delivery following the Starsim Vx pattern.

``BCGVx``      — vaccine product  (biological effect, per-agent state)
``BCGRoutine`` — delivery intervention (thin wrapper over TBProductRoutine)
"""

import numpy as np
import starsim as ss
from .interventions import TBProductRoutine

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

    Args:
        disease (str): Key of the disease module to target (default ``'tb'``).
        p_take (ss.bernoulli): Probability of immunological response post-vaccination.
        dur_immune (ss.Dist): Duration of protection (sampled per individual).
        activation_modifier / clearance_modifier / death_modifier (ss.Dist): Per-individual risk-modifier distributions.
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize BCG product with default efficacy and duration parameters."""
        super().__init__(**kwargs)
        self.define_pars(
            disease=0, # Assume the first disease is TB
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
        return

    def administer(self, people, uids):
        """
        Administer BCG to *uids* (already accepted by the delivery intervention).

        Applies ``p_take`` filter, samples per-agent modifiers, and sets
        protection expiry.  Does **not** touch ``rr_*`` arrays — that happens
        in :meth:`apply_protection` each step.

        Returns:
            ss.uids: UIDs of agents who responded immunologically.
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

    def update_roster(self):
        """Expire protection for agents past their expiry time."""
        protected_uids = self.bcg_protected.uids
        if len(protected_uids) > 0:
            expired = protected_uids[self.ti > self.ti_bcg_protection_expires[protected_uids]]
            self.bcg_protected[expired] = False
        return

    def apply_protection(self):
        """Multiply ``rr_*`` arrays for all currently protected agents."""
        protected = self.bcg_protected.uids
        if len(protected) > 0:
            tb = self.sim.diseases[self.pars.disease]
            tb.rr_activation[protected] *= self.bcg_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.bcg_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.bcg_death_modifier_applied[protected]
        return


# ---------------------------------------------------------------------------
# Delivery intervention
# ---------------------------------------------------------------------------

class BCGRoutine(TBProductRoutine):
    """
    Routine BCG vaccination delivery.

    Thin wrapper over :class:`TBProductRoutine` with pediatric defaults
    (age 0–5, 50 % coverage).  Creates a :class:`BCGVx` product
    automatically if none is provided.

    Args:
        product (BCGVx): The BCG vaccine product (created automatically if not provided).
        pars (dict): Overrides for delivery parameters (``coverage``, ``age_range``,
            ``start``, ``stop``).

    Example::

        import starsim as ss
        import tbsim
        from tbsim.interventions.bcg import BCGRoutine

        tb  = tbsim.TB_LSHTM(name='tb')
        bcg = BCGRoutine()
        sim = ss.Sim(diseases=tb, interventions=bcg, pars=dict(start='2000', stop='2020'))
        sim.run()
    """

    def __init__(self, product=None, pars=None, **kwargs):
        """Initialize BCG routine delivery; defaults to age 0-5 with a standard BCGVx product."""
        super().__init__(
            product=product if product is not None else BCGVx(),
            pars=pars,
            **kwargs,
        )
        # BCG-specific defaults (age 0-5)
        if pars is None or 'age_range' not in (pars or {}):
            self.pars.age_range = [0, 5]
        return

    def update_results(self):
        """Record number of currently protected individuals."""
        super().update_results()
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.bcg_protected)
        return
