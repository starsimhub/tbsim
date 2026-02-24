"""Tuberculosis Preventive Therapy (TPT)."""
from collections import defaultdict
import numpy as np
import starsim as ss

from tbsim.tb_lshtm import TBSL

__all__ = ['REGIMENS', 'TPTProduct', 'TPTDelivery', 'TPTRoutine', 'TPTHousehold']


class TPTProduct(ss.Product):
    """
    Handles both oral and injectable TPT — same class, different pars.

    For oral regimens the gap between efficacy (what the drug does) and
    effectiveness (what actually happens in the field) mostly comes down to
    p_complete: people don't finish their pills.  For LAI, that role shifts to
    p_attend for follow-up injections — the first one always goes in.

    Active TB is checked before anything.  You don't want to give preventive
    therapy to someone already sick — it's just basic clinical sense and 
    there are recomendations from instsitutions like WHO and CDC.

    When protection expires, tpt_offered resets so contacts can be re-offered
    if they get re-infected later.  This matters more than it might seem 
    on reinfection dynamics in high-burden settings. (— see
    Horton et al. (PNAS 2023) for more details.)

    effects is the mechanism: a dict of {disease attribute: distribution}.
    Default is a 32% reduction in activation rate (Ross et al. 2021, HR=0.68).
    If your regimen also affects clearance or death, just add those keys.
    Anything not in effects is intentionally left alone.
    """

    def __init__(self, effects=None, pars=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            disease='tb',   # get_tb() new function ?
            dur_treatment=ss.constant(v=ss.months(3)),
            dur_protection=ss.constant(v=ss.years(1.5)),
            p_complete=ss.bernoulli(p=0.93),
            n_doses=1,                                    # >1 activates LAI scheduling
            dosing_interval=ss.constant(v=ss.months(1)),
            p_attend=ss.bernoulli(p=1.0),
        )
        self.update_pars(pars)
        self.effects = effects if effects is not None else {'rr_activation': ss.constant(v=0.65)}
        self.define_states(
            ss.BoolArr('tpt_protected',  default=False),
            ss.FloatArr('ti_protection_starts'),
            ss.FloatArr('ti_protection_expires'),
            ss.FloatArr('ti_next_dose'),
            ss.IntArr('doses_received',  default=0),
            *[ss.FloatArr(f'_eff_{k}') for k in self.effects],
        )

    def administer(self, people, uids):
        if not len(uids):
            return ss.uids()

        tb    = self.sim.diseases[self.pars['disease']]
        state = np.asarray(tb.state[uids])
        uids  = uids[~np.isin(state, list(TBSL.active_tb_states()))]
        if not len(uids):
            return ss.uids()

        uids = uids[~self.tpt_protected[uids]]
        if not len(uids):
            return ss.uids()

        started = self.pars['p_complete'].filter(uids)
        if not len(started):
            return ss.uids()

        dur_tx = self.pars['dur_treatment'].rvs(started)
        self.doses_received[started]       = 1
        self.ti_protection_starts[started] = self.ti + dur_tx

        if self.pars['n_doses'] == 1:
            self.ti_protection_expires[started] = self.ti + dur_tx + self.pars['dur_protection'].rvs(started)
        else:
            interval = self.pars['dosing_interval'].rvs(started)
            self.ti_protection_expires[started] = self.ti + dur_tx + interval + self.pars['dur_protection'].rvs(started)
            self.ti_next_dose[started]          = self.ti + dur_tx + interval

        for key, dist in self.effects.items():
            getattr(self, f'_eff_{key}')[started] = dist.rvs(started)

        return started

    def update_roster(self, delivery=None):
        ti = self.ti

        # Follow-up injections (LAI multi-dose only)
        if self.pars['n_doses'] > 1:
            due = ss.uids(ti >= self.ti_next_dose)
            if len(due):
                attending = self.pars['p_attend'].filter(due)
                missed    = due.remove(attending)

                if len(attending):
                    self.doses_received[attending] = np.asarray(self.doses_received[attending]) + 1
                    done    = attending[np.asarray(self.doses_received[attending]) >= self.pars['n_doses']]
                    ongoing = attending[np.asarray(self.doses_received[attending]) <  self.pars['n_doses']]

                    if len(done):
                        self.ti_protection_expires[done] = ti + self.pars['dur_protection'].rvs(done)
                        self.ti_next_dose[done]          = np.nan

                    if len(ongoing):
                        interval = self.pars['dosing_interval'].rvs(ongoing)
                        self.ti_protection_expires[ongoing] = ti + interval + self.pars['dur_protection'].rvs(ongoing)
                        self.ti_next_dose[ongoing]          = ti + interval

                if len(missed):
                    self.ti_next_dose[missed] = np.nan

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
        protected = self.tpt_protected.uids
        if not len(protected):
            return
        tb = self.sim.diseases[self.pars['disease']]
        for key in self.effects:
            getattr(tb, key)[protected] *= getattr(self, f'_eff_{key}')[protected]

    @property
    def n_protected(self):
        return int(np.count_nonzero(self.tpt_protected))


class TPTDelivery(ss.Intervention):
    """
    The plumbing that gets the product to people.

    Handles the date window, the tpt_offered gate, and the per-step product
    orchestration.  Subclasses decide who's eligible (_find_candidates).

    product accepts a string ('3HP', 'LAI-RPT', ...), a dict of pars,
    or any ss.Product directly.
    """

    def __init__(self, product=None, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            start=ss.date('1900-01-01'),
            stop=ss.date('2100-12-31'),
        )

        if product is None:
            self.product = TPTProduct()
        elif isinstance(product, str):
            if product not in REGIMENS:
                raise KeyError(f"Unknown regimen '{product}'. Known: {list(REGIMENS)}")
            self.product = TPTProduct(**REGIMENS[product])
        elif isinstance(product, dict):
            self.product = TPTProduct(**product)
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
        return self.pars['start'].date() <= now_date <= self.pars['stop'].date()

    def _deliver(self, candidates):
        eligible = candidates[~self.tpt_offered[candidates]]
        if not len(eligible):
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
        self.results['n_initiated'][self.ti] = int(np.sum(self.ti_initiated == self.ti))
        self.results['n_protected'][self.ti] = getattr(self.product, 'n_protected', 0)


class TPTRoutine(TPTDelivery):
    """
    The workhorse: find everyone in eligible_states and offer them TPT.

    Default targets latent TB (TBSL.INFECTION) — standard for HIV-negative
    contacts.  For PLHIV (WHO Rec. 1, 2024) set eligible_states=None; they
    get TPT regardless of LTBI test result.

    coverage is acceptance probability per eligible agent.
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
        tb  = self.sim.diseases[self.product.pars['disease']]

        if self.pars['eligible_states'] is not None:
            state_vals = [int(s) for s in self.pars['eligible_states']]
            eligible   = ss.uids(np.isin(np.asarray(tb.state), state_vals))
        else:
            eligible = ppl.alive.uids

        if self.pars['age_range'] is not None and len(eligible):
            lo, hi   = self.pars['age_range']
            eligible = eligible[(ppl.age[eligible] >= lo) & (ppl.age[eligible] <= hi)]

        return self.pars['coverage'].filter(eligible)


class TPTHousehold(TPTDelivery):
    """
    Contact tracing delivery: someone starts TB treatment → find their
    household → offer TPT to whoever is there.

    Two coverage parameters because that's how programmes actually measure
    this: did you even visit the household (hh_coverage), and if you did,
    did contacts accept (contact_acceptance).

    Contacts don't get the tpt_offered gate set, so they can receive a
    fresh offer each time a new index case turns up in their household —
    which matters in high-transmission settings where reinfection is common.

    Needs a network with a household_ids attribute (HouseholdDHSNet or
    EvolvingHouseholdDHSNet from starsim_examples).  The household index
    rebuilds itself when households change.
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
        self._hh_net        = None
        self._hh_index      = None
        self._hh_n_assigned = 0
        self._hh_max_id     = -1.0

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
        hh_ids = np.asarray(self._hh_net.household_ids)
        valid  = hh_ids[np.isfinite(hh_ids)]
        idx = defaultdict(list)
        for uid, hid in enumerate(hh_ids):
            if np.isfinite(hid):
                idx[int(hid)].append(uid)
        self._hh_index      = {hid: ss.uids(members) for hid, members in idx.items()}
        self._hh_n_assigned = len(valid)
        self._hh_max_id     = float(valid.max()) if len(valid) else -1.0

    def _maybe_rebuild_hh_index(self):
        # Two O(1) sentinels cover all mutations from EvolvingHouseholdDHSNet:
        # births raise the assigned count; splits push the max id upward.
        hh_ids = np.asarray(self._hh_net.household_ids)
        valid  = hh_ids[np.isfinite(hh_ids)]
        if len(valid) != self._hh_n_assigned or (len(valid) and valid.max() != self._hh_max_id):
            self._rebuild_hh_index()

    def _find_candidates(self):
        self._maybe_rebuild_hh_index()

        tb = self.sim.diseases[self.product.pars['disease']]
        newly_on_treatment = (tb.on_treatment & ~self.prev_on_treatment).uids
        self.prev_on_treatment[:] = tb.on_treatment

        if not len(newly_on_treatment):
            return ss.uids()

        followed_up = self.pars['hh_coverage'].filter(newly_on_treatment)
        if not len(followed_up):
            return ss.uids()

        hh_ids     = np.asarray(self._hh_net.household_ids[followed_up])
        target_ids = {int(h) for h in hh_ids if not np.isnan(h)}
        if not target_ids:
            return ss.uids()

        contacts = ss.uids.cat([self._hh_index[hid] for hid in target_ids if hid in self._hh_index])
        contacts = contacts.remove(followed_up)

        if self.pars['age_range'] is not None and len(contacts):
            lo, hi   = self.pars['age_range']
            ppl      = self.sim.people
            contacts = contacts[(ppl.age[contacts] >= lo) & (ppl.age[contacts] <= hi)]

        return self.pars['contact_acceptance'].filter(contacts)

    def _deliver(self, candidates):
        # Household contacts bypass the tpt_offered gate — they may receive
        # a new offer for each new index case found in their household.
        if not len(candidates):
            return
        started = self.product.administer(self.sim.people, candidates)
        if len(started):
            self.tpt_initiated[started] = True
            self.ti_initiated[started]  = self.ti

    def shrink(self):
        self._hh_net = None


REGIMENS = {
    '3HP': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(3)),
            dur_protection=ss.constant(v=ss.years(1.5)),
            p_complete=ss.bernoulli(p=0.93),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    '3HR': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(3)),
            dur_protection=ss.constant(v=ss.years(1.5)),
            p_complete=ss.bernoulli(p=0.80),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    '4R': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(4)),
            dur_protection=ss.constant(v=ss.years(1.5)),
            p_complete=ss.bernoulli(p=0.88),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    '1HP': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(1)),
            dur_protection=ss.constant(v=ss.years(1.0)),
            p_complete=ss.bernoulli(p=0.90),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    '6H': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(6)),
            dur_protection=ss.constant(v=ss.years(1.0)),
            p_complete=ss.bernoulli(p=0.70),
        ),
        effects={'rr_activation': ss.constant(v=0.68)},
    ),
    '9H': dict(
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(9)),
            dur_protection=ss.constant(v=ss.years(1.0)),
            p_complete=ss.bernoulli(p=0.75),
        ),
        effects={'rr_activation': ss.constant(v=0.68)},
    ),
    '6Lfx': dict(  # WHO 2024 strong recommendation for MDR/RR-TB contacts (Rec. 21)
        pars=dict(
            dur_treatment=ss.constant(v=ss.months(6)),
            dur_protection=ss.constant(v=ss.years(1.0)),
            p_complete=ss.bernoulli(p=0.80),
        ),
        effects={'rr_activation': ss.constant(v=0.68)},
    ),

    'LAI-RFB': dict(
        pars=dict(
            dur_treatment=ss.constant(v=0),
            dur_protection=ss.constant(v=ss.months(4)),
            p_complete=ss.bernoulli(p=1.0),
            n_doses=1,
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
    'LAI-RPT': dict( # LA-rifapentine (LONGEVITY project): 2 monthly IM injections with
        pars=dict(
            dur_treatment=ss.constant(v=0),
            dur_protection=ss.constant(v=ss.months(2)),
            p_complete=ss.bernoulli(p=1.0),
            n_doses=2,
            dosing_interval=ss.constant(v=ss.months(1)),
            p_attend=ss.bernoulli(p=0.85),
        ),
        effects={'rr_activation': ss.constant(v=0.65)},
    ),
}
