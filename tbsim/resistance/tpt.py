"""
Strain-aware TB preventive therapy (TPT).

``TPTRx`` extends the base ``tbsim.TPTTx`` product so that sterilization is applied
*per strain*: only strains susceptible to every drug in the TPT regimen are cleared;
resistant strains remain and may then progress to disease and transmit (the spec's
"TPT clears a susceptible strain → resistant strain goes on to progress/transmit"
dynamic). An agent moves to ``CLEARED`` only once all carried strains are removed.

Suppression (the base product's ``rr_*`` progression-protection branch) is likewise applied
*per strain*: the protection an agent receives is scaled by the fraction of its carried strains
the regimen actually covers, so a surviving resistant strain is not shielded from progressing
(spec §TPT: efficacy "varies by regimen and resistance/strain" — the same resistance-unmasking
dynamic, via the progression channel rather than clearance).

TPT drug pressure can also *select* resistance among agents for whom TPT was ineffective
(neither cleared nor protected): a susceptible carried strain mutates to its resistant
counterpart with probability ``p_tpt_acq[drug]`` × a per-TB-state RR (spec: highest for
ASYMPTOMATIC/SYMPTOMATIC, low-medium for NON_INFECTIOUS, very low for INFECTION).

Wrap ``TPTRx`` in any TPT delivery (e.g. ``tbsim.TPTSimple(product=TPTRx(...))``); set the
product's ``p_sterilize`` > 0 so the strain-aware sterilization path is exercised.
"""

import numpy as np
import starsim as ss

from ..interventions.tpt import TPTTx
from ..tb import TBS
from .tb_resistant import TBResistant

__all__ = ['TPTRx']


class TPTRx(TPTTx):
    """
    Strain-aware TPT product. See module docstring for the mechanism.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        regimen_drugs (list): drugs in the TPT regimen (default: all drugs in ``strains``).
            A strain is cleared by sterilization only if susceptible to *every* regimen drug.
        p_tpt_acq (dict): per-drug ``{drug: prob}`` of a susceptible strain acquiring resistance
            under TPT drug pressure (replacement). Default none (off). Each carried drug-susceptible
            strain rolls independently, so multiple strains may acquire resistance under one course.
        acq_state_rr (dict): per-TB-state multiplier on ``p_tpt_acq``. Default follows the spec:
            INFECTION 0.05, NON_INFECTIOUS 0.5, ASYMPTOMATIC 1.0, SYMPTOMATIC 1.0.
        pars, **kwargs: forwarded to :class:`tbsim.TPTTx` (``efficacy``, ``p_sterilize``, durations, …).
    """

    # Spec §"TPT": acquisition risk varies by TB state at time of TPT failure.
    DEFAULT_ACQ_STATE_RR = {
        int(TBS.INFECTION):      0.05,
        int(TBS.NON_INFECTIOUS): 0.5,
        int(TBS.ASYMPTOMATIC):   1.0,
        int(TBS.SYMPTOMATIC):    1.0,
    }

    def __init__(self, strains, regimen_drugs=None, p_tpt_acq=None, acq_state_rr=None, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.strains = strains
        self.regimen_drugs = list(regimen_drugs) if regimen_drugs is not None else list(strains.drugs)
        self.p_tpt_acq = dict(p_tpt_acq) if p_tpt_acq else {}
        # Fail-fast on mistyped drug names (L8).
        strains.validate_drugs(self.regimen_drugs, where='TPTRx.regimen_drugs')
        strains.validate_drugs(self.p_tpt_acq, where='TPTRx.p_tpt_acq')
        self.acq_state_rr = dict(self.DEFAULT_ACQ_STATE_RR)
        if acq_state_rr:
            self.acq_state_rr.update({int(k): float(v) for k, v in acq_state_rr.items()})

        # Strains covered by the regimen (susceptible to every regimen drug); used to weight per-strain
        # protection. The clearance itself is delegated to TBResistant.sterilize_covered.
        self._covered_row = strains.covered(self.regimen_drugs)  # (m,) True = covered by regimen

        # One independent uniform stream per regimen drug for TPT-driven acquisition (state-scaled at draw).
        self._acq_rngs = [ss.random(name=f'tpt_acq_{d}') for d in self.regimen_drugs]

        # Origin-flux accounting (L2): a monotonic count of TPT-acquired resistance events. The
        # per-step delta is written to the ``n_acquired`` result and read by ``ResistanceStats``.
        self._cum_tpt_acquired = 0
        self._prev_cum = 0
        return

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result('n_acquired', dtype=int, label='Resistance acquired under TPT pressure'))
        return

    def update_results(self):
        super().update_results()
        # Delta since the last update_results = this step's acquisitions (order-independent w.r.t. the
        # delivery step, which is what actually calls _acquire).
        self.results.n_acquired[self.ti] = self._cum_tpt_acquired - self._prev_cum
        self._prev_cum = self._cum_tpt_acquired
        return

    def _tb(self):
        return self.sim.diseases[self.pars.disease]

    def _acquire(self, uids):
        """TPT drug pressure selects resistance: each carried drug-susceptible strain independently
        mutates to its resistant counterpart (replacement) with prob ``p_tpt_acq[drug]`` × per-state RR,
        transferring multiplicity source→target (see :meth:`TBResistant._apply_acquisition`)."""
        if len(uids) == 0 or not self.p_tpt_acq:
            return
        tb = self._tb()
        rr = np.array([self.acq_state_rr.get(int(s), 0.0) for s in tb.state[uids]], dtype=float)
        counts0 = tb._counts(uids)
        before_mask = np.asarray(tb.strain_mask[uids])
        drug_idxs, q_by_di, rng_by_di = [], {}, {}
        for pos, drug in enumerate(self.regimen_drugs):
            p = self.p_tpt_acq.get(drug, 0.0)
            if p > 0:
                di = self.strains.drug_idx[drug]
                drug_idxs.append(di)
                q_by_di[di] = p
                rng_by_di[di] = self._acq_rngs[pos]
        def hit_fn(di, cu, rows):
            u = np.asarray(rng_by_di[di].rvs(cu), dtype=float)
            return u < (q_by_di[di] * rr[rows])
        counts, mask, _ = tb._apply_acquisition(uids, counts0, drug_idxs, hit_fn, mixed=False)
        tb._write_counts(uids, counts)
        tb.strain_mask[uids] = mask
        self._cum_tpt_acquired += int(np.count_nonzero(mask != before_mask))  # agents whose profile changed
        return

    def _apply_sterilization(self, uids):
        """Per-strain sterilization: regimen-susceptible strains are cleared for still-latent agents;
        an agent left carrying no strain moves to CLEARED, while resistant strains persist and may
        later progress/transmit (the spec's TPT resistance-unmasking dynamic)."""
        tb = self._tb()
        if not isinstance(tb, TBResistant):
            return super()._apply_sterilization(uids)  # non-strain TB: fall back to whole-agent clearance

        # Regimen-susceptible strains are cleared only for agents still latent (base TPT semantics).
        still = uids[tb.state[uids] == TBS.INFECTION]
        tb.sterilize_covered(still, self.regimen_drugs)
        self.tpt_resolved[uids] = True
        return

    def _apply_neither_branch(self, uids):
        """TPT completely ineffective → run an acquisition trial against the regimen drugs, then resolve."""
        tb = self._tb()
        if isinstance(tb, TBResistant):
            self._acquire(uids)
        self.tpt_resolved[uids] = True
        return

    def apply_protection(self):
        """Scale TPT progression-protection by the fraction of an agent's carried strains the regimen covers.

        Base :class:`tbsim.TPTTx` protects a suppressed agent as a whole, multiplying ``rr_activation`` /
        ``rr_clearance`` / ``rr_death`` by the sampled modifiers. For a mixed infection that over-protects
        any surviving *resistant* strain, muting the spec's resistance-unmasking dynamic
        (§TPT: efficacy varies by regimen and resistance/strain). Here each modifier is blended toward 1
        (no effect) by the uncovered fraction: with coverage weight ``w`` an agent carrying only
        regimen-resistant strains (``w = 0``) gets no protection, an all-susceptible agent (``w = 1``) gets
        full protection, and mixed infections get partial protection. ``w`` is recomputed each step from the
        agent's current strains, so it tracks strain changes (bottleneck, de-novo acquisition)."""
        tb = self._tb()
        if not isinstance(tb, TBResistant):
            return super().apply_protection()  # non-strain TB: whole-agent protection

        protected = self.tpt_protected.uids
        if len(protected) == 0:
            return

        # Coverage weight w = (carried strains susceptible to the whole regimen) / (carried strains).
        carried = self.strains.carried(tb.strain_mask[protected])  # (k, m) bool
        n_carried = carried.sum(axis=1)
        n_covered = (carried & self._covered_row).sum(axis=1)
        w = np.divide(n_covered, n_carried, out=np.zeros(len(protected)), where=n_carried > 0)

        tb.rr_activation[protected] *= 1.0 - w * (1.0 - self.tpt_activation_modifier_applied[protected])
        tb.rr_clearance[protected]  *= 1.0 - w * (1.0 - self.tpt_clearance_modifier_applied[protected])
        tb.rr_death[protected]      *= 1.0 - w * (1.0 - self.tpt_death_modifier_applied[protected])
        return
