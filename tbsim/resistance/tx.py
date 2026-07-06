"""Strain-aware TB treatment product and delivery."""

import numpy as np
import starsim as ss
import tbsim
from tbsim import TBS
from tbsim.interventions.treatments import Tx, TxDelivery
from .regimens import Regimen

__all__ = ['StrainAwareTx', 'StrainAwareTxDelivery']


class StrainAwareTx(Tx):
    """
    Treatment product that resolves cure independently per strain.

    Extends :class:`tbsim.Tx` with a :class:`Regimen` describing which drugs
    are administered and per-drug efficacy. Per-strain cure probability comes
    from :meth:`Regimen.strain_cure_probs`. Selective acquisition on failed
    treatment is delegated to an :class:`AcquisitionResolver`.

    Adherence is drawn once per agent per episode and applied across all
    strains, preserving agent-level correlation (Decision 9 in the findings).
    Non-adherent agents are treated as if every strain failed.

    Args:
        regimen (Regimen): The regimen definition.
        catalog (StrainCatalog): Strain catalog.
        p_selective_acquisition (dict): Optional per-drug selective acquisition
            probability on failure (forwarded to AcquisitionResolver).
        adherence (float): Per-agent course-completion probability.
        dur_treatment (ss.Dist): Course duration distribution.
        p_relapse (float): Per-agent post-cure relapse probability (independent
            of strain identity, as in the base ``Tx``).
        dur_relapse (ss.Dist): Time-to-relapse distribution.
    """

    def __init__(self, regimen, catalog, p_selective_acquisition=None,
                 acq_state_modifiers=None,
                 adherence=0.85, dur_treatment=None, p_relapse=0.05,
                 dur_relapse=None, **kwargs):
        if not isinstance(regimen, Regimen):
            raise TypeError(
                f'regimen must be a Regimen; got {type(regimen).__name__}'
            )
        # First, build the standard Tx pars (p_adherence, p_success,
        # p_relapse, dur_*). Do NOT pass user kwargs yet — we'll add
        # per-strain pars and then run update_pars(**kwargs) once at the end
        # so user overrides hit a fully-populated pars dict.
        super().__init__(
            efficacy=regimen.base_efficacy,
            dur_treatment=dur_treatment,
            adherence=adherence,
            p_relapse=p_relapse,
            dur_relapse=dur_relapse,
        )
        self.regimen = regimen
        self._catalog = catalog
        self._strain_cure_p = regimen.strain_cure_probs(catalog)

        # Per-strain Bernoullis are added as proper Starsim pars so they
        # participate in the standard RNG plumbing. Naming convention:
        # ``p_cure_strain_<i>`` for the i-th strain in the catalog.
        per_strain = {
            f'p_cure_strain_{i}': ss.bernoulli(p=float(self._strain_cure_p[i]))
            for i in range(catalog.n)
        }
        self.define_pars(**per_strain)

        # Acquisition resolver — state-dependent ω_R,d (Phase 4).
        from .resolvers import AcquisitionResolver
        self._acq_resolver = AcquisitionResolver(
            p_selective=p_selective_acquisition,
            state_modifiers=acq_state_modifiers,
        )

        self.update_pars(**kwargs)
        return

    def administer(self, sim, uids):
        """Roll per-strain cure outcomes; adherence is correlated within an
        agent.

        Returns the same shape dict as :meth:`Tx.administer` plus a
        ``per_strain`` key giving per-strain cure masks.
        """
        n = len(uids)
        # Agent-level adherence draw (correlated across strains).
        adherent_mask = np.asarray(self.pars.p_adherence.rvs(uids), dtype=bool)

        tb = tbsim.get_tb(sim)
        profile = tb.agent_strains

        # Per-strain cure rolls, gated by adherence and carrier status.
        cure_masks = {}
        for s_idx in range(self._catalog.n):
            dist = self.pars[f'p_cure_strain_{s_idx}']
            roll = np.asarray(dist.rvs(uids), dtype=bool)
            strain_arr = getattr(profile._tb, profile.names[s_idx])
            # Idiomatic Starsim: BoolArr.uids ∩ uids → carrier mask in `uids` order.
            carrier_uids = strain_arr.uids.intersect(uids)
            carriers = np.isin(uids, carrier_uids) if len(carrier_uids) else np.zeros(n, dtype=bool)
            cure_masks[s_idx] = adherent_mask & roll & carriers

        # An agent is a "success" if every carried strain is cured.
        carried_total = profile.n_strains_per_agent(uids)
        cured_per_agent = np.zeros(n, dtype=np.int32)
        for mask in cure_masks.values():
            cured_per_agent += mask.astype(np.int32)
        all_cured = (cured_per_agent == carried_total) & (carried_total > 0)
        any_failure = ~all_cured

        success_uids = uids[all_cured]
        failure_uids = uids[any_failure]
        relapse_uids = self.pars.p_relapse.filter(success_uids)
        return {
            'success': success_uids,
            'failure': failure_uids,
            'relapse': relapse_uids,
            'per_strain': cure_masks,
        }


class StrainAwareTxDelivery(TxDelivery):
    """
    TxDelivery that updates strain profiles per the spec.

    On treatment start, per-strain cure outcomes are pre-rolled by the product.
    On treatment completion, cured strains are removed from the recipient.
    On treatment failure (any strain remains), the agent reverts to their
    prior TB state, the resolver applies selective acquisition for the regimen
    drugs, and re-care-seeking is triggered (as in the base ``TxDelivery``).

    This class subclasses :class:`tbsim.TxDelivery` and only overrides the
    pieces that need strain logic, so existing eligibility and HSB integration
    continue to work unchanged.
    """

    def __init__(self, product, **kwargs):
        if not isinstance(product, StrainAwareTx):
            raise TypeError(
                f'StrainAwareTxDelivery requires a StrainAwareTx product; '
                f'got {type(product).__name__}'
            )
        super().__init__(product=product, **kwargs)
        self._pending_per_strain = None  # set on each start
        # Snapshot of strain identities for scheduled relapses.
        # Key: uid (int) -> tuple[strain_idx, ...]
        self._relapse_strains_by_uid = {}
        return

    def step_start_treatment(self):
        tb = self.sim.get_tb()
        uids = self._elig_uids

        if tb.agent_strains is None:
            # Fall through to the base behavior; should be configured though.
            return super().step_start_treatment()

        # INFECTION: per-strain clearance (susceptible strains drop; resistant remain latent)
        latent = uids[np.isin(tb.state[uids], [TBS.INFECTION])]
        if len(latent):
            self._clear_susceptible_strains(latent)
            cleared = latent[tb.agent_strains.n_strains_per_agent(latent) == 0]
            still = latent[tb.agent_strains.n_strains_per_agent(latent) > 0]
            if len(cleared):
                tb.state[cleared] = TBS.CLEARED
                tb.rr_reinfection[cleared] = tb.pars.rr_reinfection_cleared
                if tb.pars.dur_reinfection_protection is not None:
                    tb.ti_rr_reinfection_wane[cleared] = self.ti + tb.pars.dur_reinfection_protection.rvs(cleared)
                tb.infected[cleared] = False
                tb.susceptible[cleared] = True
            # `still` agents remain in INFECTION carrying resistant strain(s); selective
            # acquisition for latent treatment is a future extension.

        # Active TB: same flow as base, but using the per-strain administer.
        active = uids[np.isin(tb.state[uids], [TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC])]
        if len(active) == 0:
            self._newly_treated = ss.uids()
            return

        self.prior_state[active] = tb.state[active]
        tb.state[active] = TBS.TREATMENT
        tb.on_treatment[active] = True
        self.pending_relapse[active] = False
        tb.results['new_notifications_15+'][tb.ti] += np.count_nonzero(self.sim.people.age[active] >= 15)

        dur_days = self.product.pars.dur_treatment.rvs(active)
        dur_steps = dur_days / self.dt.days
        self.ti_treatment_start[active] = self.ti
        self.ti_treatment_end[active] = self.ti + dur_steps
        self.n_times_treated[active] += 1

        outcomes = self.product.administer(self.sim, active)
        self.pending_success[outcomes.get('success', ss.uids())] = True
        self.pending_failure[outcomes.get('failure', ss.uids())] = True

        # Store per-strain cure outcomes against the *active* uids for use at
        # resolution. We translate to a sparse dict keyed by strain index.
        per_strain = outcomes.get('per_strain', {})
        self._pending_per_strain = {
            s_idx: active[mask] for s_idx, mask in per_strain.items() if mask.any()
        }

        relapse_uids = outcomes.get('relapse', ss.uids())
        if len(relapse_uids):
            self._capture_relapse_strains(relapse_uids)
            self.pending_relapse[relapse_uids] = True
            relapse_days = self.product.pars.dur_relapse.rvs(relapse_uids)
            relapse_steps = relapse_days / self.dt.days
            self.ti_relapse[relapse_uids] = self.ti_treatment_end[relapse_uids] + relapse_steps

        self._newly_treated = active
        return

    def _clear_susceptible_strains(self, uids):
        """Remove from carriers any strain susceptible to *all* regimen drugs.

        For latent (INFECTION) treatment we apply a simple model: any carried
        strain whose catalog phenotype is susceptible to every drug in the
        regimen is cleared. Resistant strains persist.
        """
        tb = tbsim.get_tb(self.sim)
        profile = tb.agent_strains
        catalog = profile.catalog
        regimen = self.product.regimen
        drug_cols = [catalog.drugs.index(d) for d in regimen.drugs if d in catalog.drugs]
        if not drug_cols:
            return
        # A strain is "covered" if it is susceptible to every regimen drug.
        covered = np.all(catalog.resistance[:, drug_cols] == 0, axis=1)
        for s_idx in np.where(covered)[0]:
            profile.remove_strain(uids, int(s_idx))
        return

    def step_success(self):
        """Clear all remaining strains for successful agents (full cure)."""
        super().step_success()
        tb = self.sim.get_tb()
        if tb.agent_strains is not None and len(self._success):
            tb.agent_strains.clear_all(self._success)
        return

    def step_failures(self):
        """Run selective acquisition on failure, then revert state via base class."""
        tb = self.sim.get_tb()
        failure_uids = self._fail
        if tb.agent_strains is not None and len(failure_uids):
            # Apply per-strain cure outcomes from the original pre-roll
            if self._pending_per_strain:
                for s_idx, cured_uids in self._pending_per_strain.items():
                    # UIDs that *would* have been cured per pre-roll but the
                    # agent ultimately failed — clear them from the profile.
                    sub = cured_uids.intersect(failure_uids)
                    if len(sub):
                        tb.agent_strains.remove_strain(sub, int(s_idx))
            # Selective acquisition on the regimen drugs — state-dependent ω
            self.product._acq_resolver.selective_acquisition(
                tb.agent_strains, failure_uids, self.product.regimen.drugs,
                tb=tb,
            )
        super().step_failures()
        return

    def step_relapses(self):
        """Restore pre-cure strain identity for agents who relapse."""
        super().step_relapses()
        tb = self.sim.get_tb()
        relapsed = getattr(self, '_relapsed', ss.uids())
        if tb.agent_strains is not None and len(relapsed):
            # Re-assign the strain(s) carried at treatment start to prevent
            # symptomatic relapse without a strain identity.
            by_strain = {}
            for uid in relapsed:
                for s_idx in self._relapse_strains_by_uid.get(int(uid), ()): 
                    by_strain.setdefault(int(s_idx), []).append(int(uid))
            for s_idx, raw_uids in by_strain.items():
                tb.agent_strains.add_strain(ss.uids(raw_uids), int(s_idx))

            # Updated spec: relapse is an unsuccessful treatment outcome that
            # can drive selective acquisition under regimen pressure.
            self.product._acq_resolver.selective_acquisition(
                tb.agent_strains, relapsed, self.product.regimen.drugs,
                tb=tb,
            )

        # Drop snapshots for agents whose relapse episode is no longer pending
        # (due and resolved, ineligible, dead, etc.).
        ended = [uid for uid in self._relapse_strains_by_uid
                 if not bool(self.pending_relapse[ss.uids([uid])][0])]
        for uid in ended:
            self._relapse_strains_by_uid.pop(uid, None)
        return

    def _capture_relapse_strains(self, relapse_uids):
        """Store per-agent strain identities to restore if/when relapse occurs."""
        tb = self.sim.get_tb()
        profile = tb.agent_strains
        if profile is None or len(relapse_uids) == 0:
            return
        carriers = {
            int(s_idx): set(getattr(profile._tb, profile.names[s_idx]).uids.intersect(relapse_uids).tolist())
            for s_idx in range(profile.catalog.n)
        }
        for uid in relapse_uids:
            uid_i = int(uid)
            self._relapse_strains_by_uid[uid_i] = tuple(
                s_idx for s_idx, uidset in carriers.items() if uid_i in uidset
            )
        return

    def shrink(self):
        super().shrink()
        self._pending_per_strain = None
        self._relapse_strains_by_uid = None
        return
