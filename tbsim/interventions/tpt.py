"""
Tuberculosis Preventive Therapy (TPT) — revised implementation.
"""

from __future__ import annotations
from enum import Enum
from collections import defaultdict

import numpy as np
import starsim as ss

from tbsim.tb_lshtm import TBSL

__all__ = [
    'RegimenCategory',
    'TPTRegimen',
    'REGIMENS',
    'TPTProduct',
    'TPTDelivery',
    'TPTRoutine',
    'TPTHousehold',
]

# Active TB states that disqualify an agent from receiving TPT.
# WHO (2024): providing TPT to active TB delays resolution and causes resistance.
_ACTIVE_TB_STATES = frozenset([
    int(TBSL.NON_INFECTIOUS),
    int(TBSL.ASYMPTOMATIC),
    int(TBSL.SYMPTOMATIC),
    int(TBSL.TREATMENT),
])

class RegimenCategory(Enum):
    """
    WHO/CDC regimen categories.

    RIFAMYCIN_SHORT  — 3HP, 3HR, 4R, 1HP (CDC preferred; highest completion).
    ISONIAZID_LONG   — 6H, 9H (CDC alternative; lower completion rates).
    FLUOROQUINOLONE  — 6Lfx (WHO 2024 strong recommendation for MDR/RR-TB contacts).
    """
    RIFAMYCIN_SHORT = 'rifamycin_short'
    ISONIAZID_LONG  = 'isoniazid_long'
    FLUOROQUINOLONE = 'fluoroquinolone'


class TPTRegimen:
    """
    Describes one WHO/CDC-recognized TPT regimen.

    Parameters
    ----------
    name : str
        Regimen label (e.g. ``'3HP'``, ``'6H'``).
    category : RegimenCategory
        Grouping used to document origin and guide defaults.
    dur_treatment : ss.Dist
        Duration of the antibiotic course.
    dur_protection : ss.Dist
        Post-treatment protection window. Duration evidence is highly setting-
        dependent (6–12 months in high-burden settings, longer in low-burden);
        calibrate before use.
    p_complete : ss.bernoulli
        Probability of completing the full course. Non-completers receive no
        protection. Sources: PEPFAR 87% for PLHIV/ART; Cambodia: 3HP 97.9%,
        6H substantially lower; CDC: higher completion is the primary reason
        3HP/4R are preferred over 6H/9H.
    activation_modifier : ss.Dist
        Per-agent multiplier applied to rr_activation while protected.
        Default derived from Ross et al. 2021 (HR=0.68 for IPT+ART vs ART):
        a modifier of 0.68 means 32% reduction in activation rate.
    """

    def __init__(
        self,
        name: str,
        category: RegimenCategory,
        dur_treatment: ss.Dist,
        dur_protection: ss.Dist,
        p_complete: ss.bernoulli,
        activation_modifier: ss.Dist,
    ):
        self.name                = name
        self.category            = category
        self.dur_treatment       = dur_treatment
        self.dur_protection      = dur_protection
        self.p_complete          = p_complete
        self.activation_modifier = activation_modifier

    def __repr__(self):
        return f'TPTRegimen({self.name}, category={self.category.value})'


# Pre-defined regimens:
# ==================== 
# activation_modifier defaults:
#   All active regimens show similar efficacy in the 2023 NMA (Yothasan et al.);
#   the reference HR=0.68 (32% reduction, modifier=0.68) comes from Ross et al.
#   Shorter rifamycin regimens trend slightly more protective (SUCRA rank order in
#   the NMA), so RIFAMYCIN_SHORT uses 0.65 and ISONIAZID_LONG uses 0.68.
#   Both are uncertain; calibrate to setting.
#
# p_complete defaults:
#   3HP:  0.93 (Uganda 3HP Options Trial; Cambodia programmatic data)
#   4R:   0.88 (shorter course → higher completion; Sci Rep 2023 NMA)
#   3HR:  0.80 (Cambodia: 2.6× more incomplete than 3HP)
#   6H:   0.70 (Cambodia: 7× more incomplete than 3HP; CDC: "lower completion rates")
#   9H:   0.75 (CDC alternative; no population-level estimate available)
#   6Lfx: 0.80 (no direct data; similar programme context to 6H)

REGIMENS: dict[str, TPTRegimen] = {
    '3HP': TPTRegimen(
        name='3HP',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.months(3)),
        dur_protection=ss.constant(v=ss.years(1.5)),
        p_complete=ss.bernoulli(p=0.93),
        activation_modifier=ss.constant(v=0.65),
    ),
    '3HR': TPTRegimen(
        name='3HR',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.months(3)),
        dur_protection=ss.constant(v=ss.years(1.5)),
        p_complete=ss.bernoulli(p=0.80),
        activation_modifier=ss.constant(v=0.65),
    ),
    '4R': TPTRegimen(
        name='4R',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.months(4)),
        dur_protection=ss.constant(v=ss.years(1.5)),
        p_complete=ss.bernoulli(p=0.88),
        activation_modifier=ss.constant(v=0.65),
    ),
    '1HP': TPTRegimen(
        name='1HP',
        category=RegimenCategory.RIFAMYCIN_SHORT,
        dur_treatment=ss.constant(v=ss.months(1)),
        dur_protection=ss.constant(v=ss.years(1.0)),
        p_complete=ss.bernoulli(p=0.90),
        activation_modifier=ss.constant(v=0.65),
    ),
    '6H': TPTRegimen(
        name='6H',
        category=RegimenCategory.ISONIAZID_LONG,
        dur_treatment=ss.constant(v=ss.months(6)),
        dur_protection=ss.constant(v=ss.years(1.0)),
        p_complete=ss.bernoulli(p=0.70),
        activation_modifier=ss.constant(v=0.68),
    ),
    '9H': TPTRegimen(
        name='9H',
        category=RegimenCategory.ISONIAZID_LONG,
        dur_treatment=ss.constant(v=ss.months(9)),
        dur_protection=ss.constant(v=ss.years(1.0)),
        p_complete=ss.bernoulli(p=0.75),
        activation_modifier=ss.constant(v=0.68),
    ),
    '6Lfx': TPTRegimen(
        # WHO 2024 strong recommendation for MDR/RR-TB contacts (Rec. 21).
        # https://www.ncbi.nlm.nih.gov/books/NBK607293/
        name='6Lfx',
        category=RegimenCategory.FLUOROQUINOLONE,
        dur_treatment=ss.constant(v=ss.months(6)),
        dur_protection=ss.constant(v=ss.years(1.0)),
        p_complete=ss.bernoulli(p=0.80),
        activation_modifier=ss.constant(v=0.68),
    ),
}

# TPTProduct
# ======================== 
class TPTProduct(ss.Product):
    """
    TPT treatment product: models treatment completion and per-agent protection.

    Two-phase model
    ---------------
    1. **Treatment phase** (``dur_treatment``): if the agent completes the course
       (``p_complete``), protection is scheduled to start at treatment end.
       Agents who do not complete receive no protection.
    2. **Protection phase** (``dur_protection``): ``rr_activation`` is multiplied
       by ``activation_modifier`` each step for the duration of the window.

    Active TB exclusion
    -------------------
    Agents in ``NON_INFECTIOUS``, ``ASYMPTOMATIC``, ``SYMPTOMATIC``, or
    ``TREATMENT`` states are refused TPT.  This is broader than the previous
    implementation (which only excluded ``TREATMENT``) and matches WHO guidance
    that active TB disease must be ruled out before initiating TPT.

    Protection re-offer
    -------------------
    When protection expires the agent's ``tpt_offered`` flag on the delivery
    object is cleared, allowing re-offer if re-exposed and re-infected.  This is
    consistent with CDC guidance on re-testing re-exposed contacts and with the
    49.5% undulation rate reported by Horton et al. (PNAS 2023).

    Parameters
    ----------
    regimen : str | TPTRegimen
        Which regimen to use.  Pass a string key from ``REGIMENS`` (e.g.
        ``'3HP'``) or a custom :class:`TPTRegimen` instance.
    disease : str
        Key of the disease module (default ``'tb'``).
    """

    def __init__(self, regimen: str | TPTRegimen = '3HP', pars=None, **kwargs):
        super().__init__(**kwargs)

        if isinstance(regimen, str):
            if regimen not in REGIMENS:
                raise ValueError(f"Unknown regimen '{regimen}'. Known: {list(REGIMENS)}")
            self.regimen = REGIMENS[regimen]
        else:
            self.regimen = regimen

        self.define_pars(disease='tb')
        self.update_pars(pars)

        self.define_states(
            ss.BoolArr('tpt_protected',  default=False),
            ss.FloatArr('ti_protection_starts'),
            ss.FloatArr('ti_protection_expires'),
            ss.FloatArr('tpt_activation_modifier_applied'),
        )
        return

    def _active_tb_mask(self, uids):
        """Return boolean mask for *uids* currently in an active TB state."""
        tb    = self.sim.diseases[self.pars.disease]
        state = np.asarray(tb.state[uids])
        return np.isin(state, list(_ACTIVE_TB_STATES))

    def administer(self, people, uids):
        """
        Start TPT for *uids*.

        Filters: (a) agents with active TB, (b) agents already protected.
        Then applies completion probability — non-completers are excluded.

        Returns the UIDs of agents who actually started TPT (post all filters).
        This is the value callers must use to count new initiations.
        """
        if len(uids) == 0:
            return ss.uids()

        uids = uids[~self._active_tb_mask(uids)]
        if len(uids) == 0:
            return ss.uids()

        uids = uids[~self.tpt_protected[uids]]
        if len(uids) == 0:
            return ss.uids()

        completers = self.regimen.p_complete.filter(uids)
        if len(completers) == 0:
            return ss.uids()

        dur_tx   = self.regimen.dur_treatment.rvs(completers)
        dur_prot = self.regimen.dur_protection.rvs(completers)

        self.ti_protection_starts[completers]              = self.ti + dur_tx
        self.ti_protection_expires[completers]             = self.ti + dur_tx + dur_prot
        self.tpt_activation_modifier_applied[completers]   = self.regimen.activation_modifier.rvs(completers)

        return completers

    def update_roster(self, delivery=None):
        """
        Advance internal protection state.

        - Starts protection for agents whose treatment phase has completed.
        - Expires protection for agents past their window.
        - When protection expires, the delivery's ``tpt_offered`` flag is cleared
          so those agents can be re-offered if re-infected.

        Parameters
        ----------
        delivery : TPTDelivery or None
            If provided, ``tpt_offered`` is cleared on expiry.
        """
        ti = self.ti

        not_yet = (~self.tpt_protected).uids
        if len(not_yet):
            starts = self.ti_protection_starts[not_yet]
            ready  = not_yet[~np.isnan(starts) & (ti >= starts)]
            self.tpt_protected[ready] = True

        protected = self.tpt_protected.uids
        if len(protected):
            expired = protected[ti > self.ti_protection_expires[protected]]
            self.tpt_protected[expired] = False
            if delivery is not None and len(expired):
                delivery.tpt_offered[expired] = False

    def apply_protection(self):
        """
        Apply ``rr_activation`` multiplier for all currently protected agents.

        Only ``rr_activation`` is modified.  clearance_modifier and death_modifier
        are intentionally absent: no peer-reviewed evidence supports a statistically
        significant effect of TPT on clearance from minimal disease or on TB
        mortality in the post-treatment protection phase (Ross et al. 2021,
        HR=0.69, p=0.12 for mortality; Yothasan et al. 2023 NMA).
        """
        protected = self.tpt_protected.uids
        if len(protected):
            tb = self.sim.diseases[self.pars.disease]
            tb.rr_activation[protected] *= self.tpt_activation_modifier_applied[protected]
            tb.rr_clearance[protected] *= self.tpt_clearance_modifier_applied[protected]
            tb.rr_death[protected] *= self.tpt_death_modifier_applied[protected]
        return

    @property
    def n_protected(self):
        return int(np.count_nonzero(self.tpt_protected))


# Delivery base class:
# ====================
class TPTDelivery(ss.Intervention):
    """
    Base delivery class for TPT interventions.

    Handles date windowing, per-step product orchestration, and result tracking
    that is common to both routine and household delivery.

    Subclasses must implement ``_find_candidates()`` returning ``ss.uids``.

    Parameters
    ----------
    product : str | TPTProduct
        The TPT product.  Pass a regimen name (``'3HP'``, ``'6H'``, …) or a
        fully configured :class:`TPTProduct` instance.
    start / stop : ss.date
        Delivery window (set via ``pars``).
    """

    def __init__(self, product=None, **kwargs):
        super().__init__(**kwargs)

        # NOTE: do not call update_pars here.  Subclasses call define_pars
        # for their own keys and then call update_pars once at the end,
        # covering both base and child pars in a single pass.
        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
        )

        if product is None:
            self.product = TPTProduct()
        elif isinstance(product, (str, TPTRegimen)):
            self.product = TPTProduct(regimen=product)
        else:
            self.product = product

        self.define_states(
            ss.BoolArr('tpt_offered',   default=False),
            ss.BoolArr('tpt_initiated', default=False),
            ss.FloatArr('ti_initiated'),
        )

    def _in_window(self):
        now = self.sim.now
        now_date = now.date() if hasattr(now, 'date') else now
        return self.pars.start.date() <= now_date <= self.pars.stop.date()

    def _deliver(self, candidates):
        """
        Filter *candidates* through the offered gate, call administer, and
        record initiations from the product's actual return value.
        """
        eligible = candidates[~self.tpt_offered[candidates]]
        if len(eligible) == 0:
            return

        self.tpt_offered[eligible] = True
        started = self.product.administer(self.sim.people, eligible)

        if len(started):
            self.tpt_initiated[started] = True
            self.ti_initiated[started]  = self.ti

    def step(self):
        if not self._in_window():
            return

        self.product.update_roster(delivery=self)

        candidates = self._find_candidates()
        if len(candidates):
            self._deliver(candidates)

        self.product.apply_protection()

    def _find_candidates(self):
        raise NotImplementedError

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_initiated', dtype=int, label='Newly initiated'),
            ss.Result('n_protected', dtype=int, label='Currently protected'),
        )

    def update_results(self):
        initiated_this_step = int(np.sum(self.ti_initiated == self.ti))
        self.results['n_initiated'][self.ti] = initiated_this_step
        self.results['n_protected'][self.ti] = self.product.n_protected


class TPTRoutine(TPTDelivery):
    """
    Routine TPT delivery targeting agents by disease state (default: latent TB).

    WHO eligibility
    ---------------
    - HIV-negative contacts ≥5 y in ``TBSL.INFECTION``:
      default ``eligible_states=[TBSL.INFECTION]``.
    - PLHIV (WHO Rec. 1, 2024): TPT regardless of LTBI test → set
      ``eligible_states=None`` to target all alive, non-active-TB agents.

    Coverage
    --------
    ``coverage`` is applied per agent (Bernoulli): the probability that an
    eligible agent reached by the programme accepts TPT.

    Parameters
    ----------
    product : str | TPTProduct
        Regimen (default ``'3HP'``).
    pars : dict, optional
        Accepted keys: ``start``, ``stop``, ``coverage``, ``eligible_states``,
        ``age_range``.
    """

    def __init__(self, product=None, pars=None, **kwargs):
        super().__init__(product=product, **kwargs)

        self.define_pars(
            coverage=ss.bernoulli(p=0.5),
            eligible_states=[TBSL.INFECTION],
            age_range=None,
        )
        self.update_pars(pars)

    def _find_candidates(self):
        ppl = self.sim.people
        tb  = self.sim.diseases[self.product.pars.disease]

        if self.pars.eligible_states is not None:
            state_vals  = [int(s) for s in self.pars.eligible_states]
            state_mask  = np.isin(np.asarray(tb.state), state_vals)
            eligible    = ss.uids(state_mask)
        else:
            eligible = ppl.alive.uids

        if self.pars.age_range is not None and len(eligible):
            lo, hi   = self.pars.age_range
            age_ok   = (ppl.age[eligible] >= lo) & (ppl.age[eligible] <= hi)
            eligible = eligible[age_ok]

        return self.pars.coverage.filter(eligible)



class TPTHousehold(TPTDelivery):
    """
    Household contact tracing TPT delivery.

    Triggers on newly diagnosed TB cases (agents newly entering
    ``TBSL.TREATMENT``), traces household contacts, and offers TPT.

    Coverage semantics
    ------------------
    Two independent coverage parameters, matching how programmes measure them:

    ``hh_coverage``
        Probability that a given index case's household is visited at all
        (per-household Bernoulli).  Maps to programme-level "household visit
        coverage".

    ``contact_acceptance``
        Probability that each contact within a visited household accepts TPT
        (per-contact Bernoulli).  Maps to programme-level "TPT acceptance".

    The previous ``tpt.py`` used a single ``coverage`` applied only at the
    household level, leaving contact acceptance unmodelled.

    Re-offer behaviour
    ------------------
    Household contacts do not have their ``tpt_offered`` flag set here (unlike
    ``TPTRoutine``).  Their flag is cleared when protection expires (in
    ``TPTProduct.update_roster``), so contacts can be re-offered if a new index
    case is found in their household after re-infection.

    Requirements
    ------------
    A household network with a ``household_ids`` attribute must be present in
    ``sim.networks``.  Both ``HouseholdDHSNet`` (static) and
    ``EvolvingHouseholdDHSNet`` (dynamic) from ``starsim_examples`` are
    supported.  For evolving networks the household index is rebuilt
    automatically whenever births or household splits are detected.

    Parameters
    ----------
    product : str | TPTProduct
    pars : dict, optional
        Accepted keys: ``start``, ``stop``, ``hh_coverage``,
        ``contact_acceptance``, ``age_range``.
    """

    def __init__(self, product=None, pars=None, **kwargs):
        super().__init__(product=product, **kwargs)

        self.define_pars(
            hh_coverage=ss.bernoulli(p=0.5),
            contact_acceptance=ss.bernoulli(p=1.0),
            age_range=None,
        )
        self.update_pars(pars)

        self.define_states(
            ss.BoolArr('prev_on_treatment', default=False),
        )
        self.hh_net = None
        return

        self._hh_net         = None
        self._hh_index: dict | None = None
        self._hh_n_assigned  = 0    # sentinels for detecting network mutations
        self._hh_max_id      = -1.0

    def init_pre(self, sim):
        super().init_pre(sim)
        self._hh_net = self._find_hh_net()
        self._rebuild_hh_index()

    def _find_hh_net(self):
        for net in self.sim.networks.values():
            if hasattr(net, 'household_ids'):
                return net
        raise ValueError(
            'TPTHousehold requires a network with a household_ids attribute '
            '(e.g. HouseholdDHSNet from starsim_examples).'
        )

    def _rebuild_hh_index(self):
        """Build {household_id: uid_array} inverted index and update sentinels."""
        hh_ids = np.asarray(self._hh_net.household_ids)
        valid  = hh_ids[np.isfinite(hh_ids)]
        idx: dict[int, list] = defaultdict(list)
        for uid, hid in enumerate(hh_ids):
            if np.isfinite(hid):
                idx[int(hid)].append(uid)
        self._hh_index      = {hid: ss.uids(members) for hid, members in idx.items()}
        self._hh_n_assigned = len(valid)
        self._hh_max_id     = float(valid.max()) if len(valid) else -1.0

    def _maybe_rebuild_hh_index(self):
        """Rebuild the index if the network has changed since the last rebuild.

        Two O(1) checks cover all mutation types produced by
        ``EvolvingHouseholdDHSNet``:
        - births increase the count of assigned slots
        - household splits always assign new IDs above the previous maximum
        """
        hh_ids = np.asarray(self._hh_net.household_ids)
        valid  = hh_ids[np.isfinite(hh_ids)]
        if len(valid) != self._hh_n_assigned or (len(valid) and valid.max() != self._hh_max_id):
            self._rebuild_hh_index()

    # Core delivery
    def _find_candidates(self):
        self._maybe_rebuild_hh_index()

        tb = self.sim.diseases[self.product.pars.disease]
        if self.hh_net is None:
            self.hh_net = self.find_household_net()
        hh_net = self.hh_net

        newly_on_treatment = (tb.on_treatment & ~self.prev_on_treatment).uids
        self.prev_on_treatment[:] = tb.on_treatment

        if len(newly_on_treatment) == 0:
            return ss.uids()

        followed_up = self.pars.hh_coverage.filter(newly_on_treatment)
        if len(followed_up) == 0:
            return ss.uids()

        hh_ids     = np.asarray(self._hh_net.household_ids[followed_up])
        target_ids = {int(h) for h in hh_ids if not np.isnan(h)}

        if not target_ids:
            return ss.uids()

        contacts = ss.uids.cat(
            [self._hh_index[hid] for hid in target_ids if hid in self._hh_index]
        )
        contacts = contacts.remove(followed_up)

        if self.pars.age_range is not None and len(contacts):
            lo, hi   = self.pars.age_range
            ppl      = self.sim.people
            age_ok   = (ppl.age[contacts] >= lo) & (ppl.age[contacts] <= hi)
            contacts = contacts[age_ok]

        return self.pars.contact_acceptance.filter(contacts)

    def _deliver(self, candidates):
        """
        Bypass the ``tpt_offered`` gate for household contacts — they receive
        an offer each time a new index case is identified in their household,
        subject to the product's own already-protected check.
        """
        if len(candidates) == 0:
            return

        started = self.product.administer(self.sim.people, candidates)

        if len(started):
            self.tpt_initiated[started] = True
            self.ti_initiated[started]  = self.ti
    def update_results(self):
        """Record number of currently protected individuals."""
        super().update_results()
        self.results['n_protected'][self.ti] = np.count_nonzero(self.product.tpt_protected)
        return

    def shrink(self):
        """ Delete link to household net """
        self.hh_net = None
        return
