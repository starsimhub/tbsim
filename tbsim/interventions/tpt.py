"""
Tuberculosis Preventive Therapy (TPT): Product + Delivery.

``TPTTx``         — treatment product  (biological effect, per-agent state)
``TPTSimple``     — simple delivery    (targets latently infected by default)
``TPTHousehold``  — household contact tracing delivery
"""

import numpy as np
import starsim as ss
from .interventions import TBProductRoutine
from ..tb_lshtm import TBSL

__all__ = ['TPTTx', 'TPTSimple', 'TPTHousehold']


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

    Args:
        disease (str): Key of the disease module to target (default ``'tb'``).
        dur_treatment (ss.Dist): Duration of the antibiotic course (default 3 months).
        dur_protection (ss.Dist): Duration of post-treatment protection (default 2 years).
        activation_modifier / clearance_modifier / death_modifier (ss.Dist): Per-individual risk-modifier distributions.
        exclude_on_treatment (bool): If True, skip agents currently on TB treatment (default True).
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize TPT product with default treatment duration and efficacy parameters."""
        super().__init__(**kwargs)
        self.define_pars(
            disease='tb',
            dur_treatment=ss.constant(v=ss.months(3)),
            dur_protection=ss.constant(v=ss.years(2)),
            activation_modifier=ss.uniform(0.3, 0.5),
            clearance_modifier=ss.uniform(1.2, 1.4),
            death_modifier=ss.uniform(0.1, 0.3),
            exclude_on_treatment=True,
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
        return

    def administer(self, people, uids):
        """
        Start TPT for *uids*.

        Filters out ineligible agents (already on TB treatment, already
        protected), then samples per-agent modifiers and schedules the
        protection window.

        Returns:
            ss.uids: UIDs of agents who actually started TPT.
        """
        if len(uids) == 0:
            return ss.uids()

        # Product-side eligibility: skip agents on TB treatment
        if self.pars.exclude_on_treatment:
            tb = self.sim.diseases[self.pars.disease]
            uids = uids[~tb.on_treatment[uids]]

        # Skip agents already protected (don't restart their clock)
        uids = uids[~self.tpt_protected[uids]]

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
        return

    def apply_protection(self):
        """Multiply ``rr_*`` arrays for all currently protected agents."""
        protected = self.tpt_protected.uids
        if len(protected) > 0:
            tb = self.sim.diseases[self.pars.disease]
            tb.rr_activation[protected] *= self.tpt_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.tpt_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.tpt_death_modifier_applied[protected]
        return


# ---------------------------------------------------------------------------
# Simple delivery
# ---------------------------------------------------------------------------

class TPTSimple(TBProductRoutine):
    """
    Simple TPT delivery targeting latently infected agents.

    Thin wrapper over :class:`TBProductRoutine` that defaults to targeting
    agents in ``TBSL.INFECTION`` state (latent TB).  Creates a
    :class:`TPTTx` product automatically if none is provided.

    Args:
        product (TPTTx): The TPT treatment product (created automatically if not provided).
        pars (dict): Overrides for delivery parameters (``coverage``, ``age_range``,
            ``eligible_states``, ``start``, ``stop``).

    Example::

        import starsim as ss
        import tbsim
        from tbsim.interventions.tpt import TPTSimple

        tb  = tbsim.TB_LSHTM(name='tb')
        tpt = TPTSimple()
        sim = ss.Sim(diseases=tb, interventions=tpt, pars=dict(start='2000', stop='2020'))
        sim.run()
    """

    def __init__(self, product=None, pars=None, **kwargs):
        """Initialize simple TPT delivery; defaults to targeting latently infected agents."""
        super().__init__(
            product=product if product is not None else TPTTx(),
            pars=pars,
            **kwargs,
        )
        # Default: target latently infected agents
        if pars is None or 'eligible_states' not in (pars or {}):
            self.pars.eligible_states = [TBSL.INFECTION]
        return

    def update_results(self):
        """Record number of currently protected individuals."""
        super().update_results()
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.tpt_protected)
        return


# ---------------------------------------------------------------------------
# Household contact tracing delivery
# ---------------------------------------------------------------------------

class TPTHousehold(TBProductRoutine):
    """
    Household contact tracing TPT delivery.

    Detects new TB treatment starts each step, traces household contacts
    of followed-up index cases, and offers TPT to all contacts (product
    handles eligibility filtering).

    Requires a household network with ``household_ids`` (e.g.
    ``HouseholdDHSNet`` or ``EvolvingHouseholdDHSNet`` from
    ``starsim_examples``).

    Args:
        product (TPTTx): The TPT treatment product (created automatically if not provided).
        pars (dict): Overrides. ``coverage`` controls the probability that a given index
            case's household is followed up (per-index-case Bernoulli).
    """

    def __init__(self, product=None, pars=None, **kwargs):
        """Initialize household contact-tracing TPT delivery."""
        super().__init__(
            product=product if product is not None else TPTTx(),
            pars=pars,
            **kwargs,
        )
        self.define_states(
            ss.BoolArr('prev_on_treatment', default=False),
        )
        self.hh_net = None
        return

    def find_household_net(self):
        """Find the network that has ``household_ids``."""
        for net in self.sim.networks.values():
            if hasattr(net, 'household_ids'):
                return net
        raise ValueError("No household network with household_ids found in sim")

    def check_eligibility(self):
        """
        Detect new treatment starts → follow up index cases → trace
        household contacts.
        """
        tb = self.sim.diseases[self.product.pars.disease]
        if self.hh_net is None:
            self.hh_net = self.find_household_net()
        hh_net = self.hh_net

        # 1. Detect NEW treatment starts (on_treatment: False → True)
        newly_on_treatment = (tb.on_treatment & ~self.prev_on_treatment).uids
        self.prev_on_treatment[:] = tb.on_treatment

        if len(newly_on_treatment) == 0:
            return ss.uids()

        # 2. Coverage: per index case (does health system follow up?)
        followed_up = self.pars.coverage.filter(newly_on_treatment)
        if len(followed_up) == 0:
            return ss.uids()

        # 3. Find their household IDs
        hhids = np.asarray(hh_net.household_ids[followed_up])
        target_hhids = np.unique(hhids[~np.isnan(hhids)])

        if len(target_hhids) == 0:
            return ss.uids()

        # 4. Find household contacts (excluding index cases)
        all_hh = np.isin(np.asarray(hh_net.household_ids), target_hhids)
        contacts = ss.uids(all_hh)
        contacts = contacts.remove(followed_up)

        # 5. Age filter (if set)
        if self.pars.age_range is not None:
            min_age, max_age = self.pars.age_range[0], self.pars.age_range[1]
            ppl = self.sim.people
            age_ok = (ppl.age[contacts] >= min_age) & (ppl.age[contacts] <= max_age)
            contacts = contacts[age_ok]

        return contacts

    def update_results(self):
        """Record number of currently protected individuals."""
        super().update_results()
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.tpt_protected)
        return

    def shrink(self):
        """ Delete link to household net """
        self.hh_net = None
        return
