"""
Tuberculosis Preventive Therapy (TPT): Product + Delivery.

``TPTTx``      — treatment product  (biological effect, per-agent state)
``TPTSimple``  — simple delivery    (targets latently infected by default)
``TPTRegimes`` — CDC 2024 regimen duration constants
"""

import numpy as np
import starsim as ss

from .interventions import TBProductRoutine
from ..tb_lshtm import TBSL

__all__ = ['TPTTx', 'TPTSimple']


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------

class TPTTx(ss.Product):
    """
    TPT treatment product.

    Models a two-phase intervention: a **treatment phase** (antibiotic course)
    followed by a **protection phase** (residual risk reduction via ``rr_*``
    modifiers).  During treatment, no protection is applied; after treatment
    completes, ``rr_activation``, ``rr_clearance``, and ``rr_death`` are
    modified each step until protection expires.

    Parameters
    ----------
    disease : str
        Key of the disease module to target (default ``'tb'``).
    dur_treatment : ss.Dist
        Duration of the antibiotic course (default 3 months).
    dur_protection : ss.Dist
        Duration of post-treatment protection (default 2 years).
    activation_modifier / clearance_modifier / death_modifier : ss.Dist
        Per-individual risk-modifier distributions.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            disease='tb',
            dur_treatment=ss.constant(v=ss.months(3)),
            dur_protection=ss.constant(v=ss.years(2)),
            activation_modifier=ss.uniform(0.3, 0.5),
            clearance_modifier=ss.uniform(1.2, 1.4),
            death_modifier=ss.uniform(0.1, 0.3),
        )
        self.update_pars(pars)

        self.define_states(
            ss.BoolArr('tpt_protected', default=False),
            ss.FloatArr('ti_protection_starts'),
            ss.FloatArr('ti_protection_expires'),
            ss.FloatArr('tpt_activation_modifier_applied'),
            ss.FloatArr('tpt_clearance_modifier_applied'),
            ss.FloatArr('tpt_death_modifier_applied'),
        )

    def administer(self, people, uids):
        """
        Start TPT for *uids* (already accepted by the delivery intervention).

        Samples per-agent modifiers and schedules the protection window
        (starts after treatment completes, expires after ``dur_protection``).

        Returns
        -------
        ss.uids
            UIDs of agents who started treatment.
        """
        if len(uids) == 0:
            return ss.uids()

        dur_tx = self.pars.dur_treatment.rvs(uids)
        dur_prot = self.pars.dur_protection.rvs(uids)

        self.ti_protection_starts[uids] = self.ti + dur_tx
        self.ti_protection_expires[uids] = self.ti + dur_tx + dur_prot

        self.tpt_activation_modifier_applied[uids] = self.pars.activation_modifier.rvs(uids)
        self.tpt_clearance_modifier_applied[uids] = self.pars.clearance_modifier.rvs(uids)
        self.tpt_death_modifier_applied[uids] = self.pars.death_modifier.rvs(uids)

        return uids

    def update_roster(self):
        """
        Start protection for agents whose treatment completed;
        expire protection for agents past their expiry time.
        """
        # Start protection for agents whose treatment just completed
        not_yet_protected = (~self.tpt_protected).uids
        if len(not_yet_protected) > 0:
            starts = self.ti_protection_starts[not_yet_protected]
            ready = not_yet_protected[
                ~np.isnan(starts) & (self.ti >= starts)
            ]
            self.tpt_protected[ready] = True

        # Expire protection
        protected_uids = self.tpt_protected.uids
        if len(protected_uids) > 0:
            expired = protected_uids[
                self.ti > self.ti_protection_expires[protected_uids]
            ]
            self.tpt_protected[expired] = False

    def apply_protection(self):
        """Multiply ``rr_*`` arrays for all currently protected agents."""
        protected = self.tpt_protected.uids
        if len(protected) > 0:
            tb = self.sim.diseases[self.pars.disease]
            tb.rr_activation[protected] *= self.tpt_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.tpt_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.tpt_death_modifier_applied[protected]


# ---------------------------------------------------------------------------
# Delivery intervention
# ---------------------------------------------------------------------------

class TPTSimple(TBProductRoutine):
    """
    Simple TPT delivery targeting latently infected agents.

    Thin wrapper over :class:`TBProductRoutine` that defaults to targeting
    agents in ``TBSL.INFECTION`` state (latent TB).  Creates a
    :class:`TPTTx` product automatically if none is provided.

    Parameters
    ----------
    product : TPTTx
        The TPT treatment product (created automatically if not provided).
    pars : dict
        Overrides for delivery parameters (``coverage``, ``age_range``,
        ``eligible_states``, ``start``, ``stop``).
    """

    def __init__(self, product=None, pars=None, **kwargs):
        super().__init__(
            product=product if product is not None else TPTTx(),
            pars=pars,
            **kwargs,
        )
        # Default: target latently infected agents
        if pars is None or 'eligible_states' not in (pars or {}):
            self.pars.eligible_states = [TBSL.INFECTION]

    def update_results(self):
        super().update_results()
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.tpt_protected)


