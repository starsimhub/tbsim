"""Strain-aware TPT treatment product."""

import numpy as np
import starsim as ss
from tbsim.interventions.tpt import TPTTx
from tbsim import TBS
from .regimens import Regimen

__all__ = ['StrainAwareTPTTx']


class StrainAwareTPTTx(TPTTx):
    """
    TPT product that applies per-strain sterilization based on a regimen.

    Extends :class:`tbsim.TPTTx` so that sterilization clears only susceptible
    strains (those the regimen covers). Resistant strains remain. The agent
    moves to ``CLEARED`` only when *all* carried strains have been removed
    (Decision 10 in the findings).

    Suppression (rr_*) modifiers continue to apply at the agent level; the
    spec's per-strain suppression option is not yet implemented and would
    require strain-level rr_* arrays.

    TPT-driven acquisition (a configurable per-drug acquisition probability
    on TPT failure) is delegated to an :class:`AcquisitionResolver`.

    Args:
        regimen (Regimen): TPT regimen (drugs + per-drug efficacy).
        registry (StrainRegistry): Strain registry.
        p_tpt_acquisition (dict): Optional per-drug acquisition probability
            applied to surviving strains in agents whose TPT is ineffective.
        acq_state_modifiers (dict): Optional per-state modifier on the
            acquisition probability. Defaults follow the spec:
            very low for INFECTION (0.05), medium for NON_INFECTIOUS (0.5),
            high for ASYMPTOMATIC and SYMPTOMATIC (1.0).
    """

    # Spec §"TPT": TPT-driven acquisition risk varies by TB state.
    DEFAULT_TPT_STATE_MODIFIERS = {
        'infection':      0.05,
        'non_infectious': 0.5,
        'asymptomatic':   1.0,
        'symptomatic':    1.0,
        'treatment':      0.0,
        'cleared':        0.0,
    }

    def __init__(self, regimen, registry, p_tpt_acquisition=None,
                 acq_state_modifiers=None, **kwargs):
        if not isinstance(regimen, Regimen):
            raise TypeError(f'regimen must be a Regimen; got {type(regimen).__name__}')
        super().__init__(**kwargs)
        self.regimen = regimen
        self._registry = registry
        self._cover_mask = self._compute_cover_mask(registry, regimen)
        from .resolvers import AcquisitionResolver
        if acq_state_modifiers is None:
            acq_state_modifiers = dict(self.DEFAULT_TPT_STATE_MODIFIERS)
        self._acq_resolver = AcquisitionResolver(
            p_selective=p_tpt_acquisition,
            state_modifiers=acq_state_modifiers,
        )
        return

    @staticmethod
    def _compute_cover_mask(registry, regimen):
        """Return a boolean array of length n_strains: True if regimen covers strain."""
        cols = [registry.drugs.index(d) for d in regimen.drugs if d in registry.drugs]
        if not cols:
            return np.zeros(registry.n, dtype=bool)
        return np.all(registry.resistance[:, cols] == 0, axis=1)

    def _apply_sterilization(self, uids):
        """Per-strain sterilization: clear susceptible strains; resistant strains remain.

        TPT-driven acquisition runs *before* sterilization: a susceptible
        carried strain may mutate to its resistant counterpart with
        probability ``p_tpt_acquisition[drug]`` (state-modified). Mutated
        strains are by construction resistant to the regimen and therefore
        survive the subsequent sterilization step. This models the biology
        of suboptimal drug pressure selecting resistance rather than
        clearance.

        Acquisition trials run on **all** sterilize-branch agents regardless
        of TB state — the spec's per-state modifier (``acq_state_modifiers``)
        is what governs whether NI/ASY/SYM agents actually acquire. The
        clearance step (state → CLEARED) is restricted to agents still in
        ``INFECTION``, matching base TPT semantics.
        """
        tb = self.sim.diseases[self.pars.disease]
        profile = getattr(tb, 'strain_profile', None)
        if profile is None:
            # Fall back to the base agent-level sterilization if no overlay.
            return super()._apply_sterilization(uids)

        # TPT-driven acquisition: mutate a fraction of susceptible carried
        # strains to their resistant counterpart *before* sterilization. Must
        # run on the pre-sterilization profile, otherwise the only strains
        # still present would be regimen-resistant and have nothing to mutate.
        # Runs on the full uids cohort so per-state ω modifiers actually
        # apply to non-INFECTION agents (NI/ASY/SYM) per spec.
        self._acq_resolver.selective_acquisition(
            profile, uids, self.regimen.drugs, tb=tb,
        )

        # Sterilization (strain removal + state → CLEARED) only applies to
        # agents still in INFECTION — sterilize→CLEARED is a latent-only
        # transition per base TPT semantics.
        still_infected = uids[tb.state[uids] == TBS.INFECTION]
        if len(still_infected) > 0:
            for s_idx in np.where(self._cover_mask)[0]:
                profile.remove_strain(still_infected, int(s_idx))

            # Agents who now carry zero strains have been fully cleared.
            counts = profile.n_strains_per_agent(still_infected)
            fully_cleared = still_infected[counts == 0]

            if len(fully_cleared):
                tb.state[fully_cleared] = TBS.CLEARED
                tb.rr_reinfection[fully_cleared] = tb.pars.rr_reinfection_cleared
                if tb.pars.dur_reinfection_protection is not None:
                    tb.ti_rr_reinfection_wane[fully_cleared] = self.ti + tb.pars.dur_reinfection_protection.rvs(fully_cleared)
                tb.infected[fully_cleared] = False
                tb.susceptible[fully_cleared] = True

        # Partial-clearance agents (still carrying resistant or mutated strains),
        # and any non-INFECTION agents in this branch, remain in their prior
        # state and may transmit later.
        self.tpt_resolved[uids] = True
        return

    def _apply_neither_branch(self, uids):
        """Per-spec: TPT was completely ineffective — apply acquisition.

        Spec §"TPT": "some percentage of agents for whom TPT was not
        effective (provided neither clearance nor longer-term protection
        from progression) have a probability of acquiring resistance to
        the drugs/classes included in the TPT regimen."

        The "neither" branch is exactly this cohort, so we run a selective
        acquisition trial against the regimen's drugs. Per-state modifiers
        (``acq_state_modifiers``) make this a no-op when configured to 0
        for a given state.
        """
        tb = self.sim.diseases[self.pars.disease]
        profile = getattr(tb, 'strain_profile', None)
        if profile is not None and len(uids) > 0:
            self._acq_resolver.selective_acquisition(
                profile, uids, self.regimen.drugs, tb=tb,
            )
        return super()._apply_neither_branch(uids)
