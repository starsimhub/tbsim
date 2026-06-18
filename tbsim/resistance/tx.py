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
        registry (StrainRegistry): Strain registry.
        p_selective_acquisition (dict): Optional per-drug selective acquisition
            probability on failure (forwarded to AcquisitionResolver).
        adherence (float): Per-agent course-completion probability.
        dur_treatment (ss.Dist): Course duration distribution.
        p_relapse (float): Per-agent post-cure relapse probability (independent
            of strain identity, as in the base ``Tx``).
        dur_relapse (ss.Dist): Time-to-relapse distribution.
    """

    def __init__(self, regimen, registry, p_selective_acquisition=None,
                 adherence=0.85, dur_treatment=None, p_relapse=0.05,
                 dur_relapse=None, **kwargs):
        if not isinstance(regimen, Regimen):
            raise TypeError(f'regimen must be a Regimen; got {type(regimen).__name__}')
        super().__init__(
            efficacy=regimen.base_efficacy,
            dur_treatment=dur_treatment,
            adherence=adherence,
            p_relapse=p_relapse,
            dur_relapse=dur_relapse,
            **kwargs,
        )
        self.regimen = regimen
        self._registry = registry
        self._strain_cure_p = regimen.strain_cure_probs(registry)

        # Per-strain RNGs for independent cure rolls
        self._rng_per_strain = [
            ss.bernoulli(p=float(self._strain_cure_p[i]),
                         name=f'tx_cure_{regimen.name}_{registry.uids[i]}')
            for i in range(registry.n)
        ]

        # Acquisition resolver lazily — instantiated only if needed
        from .resolvers import AcquisitionResolver
        self._acq_resolver = AcquisitionResolver(p_selective=p_selective_acquisition)
        return

    def administer(self, sim, uids):
        """Roll per-strain cure outcomes; adherence is correlated within an agent.

        Returns the same shape dict as :meth:`Tx.administer` plus a
        ``per_strain`` key giving per-strain cure masks.
        """
        n = len(uids)
        # Agent-level adherence draw (correlated across strains).
        adherent_mask = np.asarray(self.pars.p_adherence.rvs(uids), dtype=bool)
        # Per-strain cure rolls, gated by adherence.
        cure_masks = {}
        tb = sim.diseases['tb']
        profile = tb.strain_profile
        for s_idx, dist in enumerate(self._rng_per_strain):
            roll = np.asarray(dist.rvs(uids), dtype=bool)
            carriers = np.asarray(
                getattr(profile._tb, profile.names[s_idx])[uids], dtype=bool
            )
            cure_masks[s_idx] = adherent_mask & roll & carriers

        # An agent is a "success" if every carried strain is cured.
        carried_total = profile.n_strains_per_agent(uids)
        cured_per_agent = np.zeros(n, dtype=np.int32)
        for s_idx, mask in cure_masks.items():
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
        return

    def step_start_treatment(self):
        tb = self.sim.get_tb()
        uids = self._elig_uids

        if tb.strain_profile is None:
            # Fall through to the base behavior; should be configured though.
            return super().step_start_treatment()

        # INFECTION: per-strain clearance (susceptible strains drop; resistant remain latent)
        latent = uids[np.isin(tb.state[uids], [TBS.INFECTION])]
        if len(latent):
            self._clear_susceptible_strains(latent)
            cleared = latent[tb.strain_profile.n_strains_per_agent(latent) == 0]
            still = latent[tb.strain_profile.n_strains_per_agent(latent) > 0]
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
            self.pending_relapse[relapse_uids] = True
            relapse_days = self.product.pars.dur_relapse.rvs(relapse_uids)
            relapse_steps = relapse_days / self.dt.days
            self.ti_relapse[relapse_uids] = self.ti_treatment_end[relapse_uids] + relapse_steps

        self._newly_treated = active
        return

    def _clear_susceptible_strains(self, uids):
        """Remove from carriers any strain susceptible to *all* regimen drugs.

        For latent (INFECTION) treatment we apply a simple model: any carried
        strain whose registry phenotype is susceptible to every drug in the
        regimen is cleared. Resistant strains persist.
        """
        tb = tbsim.get_tb(self.sim)
        profile = tb.strain_profile
        registry = profile.registry
        regimen = self.product.regimen
        drug_cols = [registry.drugs.index(d) for d in regimen.drugs if d in registry.drugs]
        if not drug_cols:
            return
        # A strain is "covered" if it is susceptible to every regimen drug.
        covered = np.all(registry.resistance[:, drug_cols] == 0, axis=1)
        for s_idx in np.where(covered)[0]:
            profile.remove_strain(uids, int(s_idx))
        return

    def step_success(self):
        """Clear all remaining strains for successful agents (full cure)."""
        super().step_success()
        tb = self.sim.get_tb()
        if tb.strain_profile is not None and len(self._success):
            tb.strain_profile.clear_all(self._success)
        return

    def step_failures(self):
        """Run selective acquisition on failure, then revert state via base class."""
        tb = self.sim.get_tb()
        failure_uids = self._fail
        if tb.strain_profile is not None and len(failure_uids):
            # Apply per-strain cure outcomes from the original pre-roll
            if self._pending_per_strain:
                for s_idx, cured_uids in self._pending_per_strain.items():
                    # UIDs that *would* have been cured per pre-roll but the
                    # agent ultimately failed — clear them from the profile.
                    sub = cured_uids.intersect(failure_uids)
                    if len(sub):
                        tb.strain_profile.remove_strain(sub, int(s_idx))
            # Selective acquisition on the regimen drugs
            self.product._acq_resolver.selective_acquisition(
                tb.strain_profile, failure_uids, self.product.regimen.drugs,
            )
        super().step_failures()
        return

    def shrink(self):
        super().shrink()
        self._pending_per_strain = None
        return
