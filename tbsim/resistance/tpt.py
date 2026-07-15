"""
Strain-aware TB preventive therapy (TPT).

``TPTRx`` extends the base ``tbsim.TPTTx`` product so that sterilization is applied
*per strain*: only strains susceptible to every drug in the TPT regimen are cleared by
default; resistant strains remain and may then progress to disease and transmit (the
spec's "TPT clears a susceptible strain → resistant strain goes on to progress/transmit"
dynamic). An agent moves to ``CLEARED`` only once all carried strains are removed.

Optional ``resist_penalty`` enables graded clearance of partially resistant strains
(default ``{}`` preserves binary unmasking: unspecified regimen drugs contribute factor 0).

TPT drug pressure can also *select* resistance among agents for whom TPT was ineffective
(neither cleared nor protected): a susceptible carried strain mutates to its resistant
counterpart with probability ``p_tpt_acq[drug]`` × a per-TB-state RR (spec: highest for
ASYMPTOMATIC/SYMPTOMATIC, low-medium for NON_INFECTIOUS, very low for INFECTION).
Acquisition replaces one carried susceptible strain per drug per episode (same bottleneck
as ``TxR.acquire``; differs from multi-strain de-novo).

Wrap ``TPTRx`` in any TPT delivery (e.g. ``tbsim.TPTSimple(product=TPTRx(...))``); set the
product's ``p_sterilize`` > 0 so the strain-aware sterilization path is exercised.
"""

import numpy as np
import starsim as ss

from ..interventions.tpt import TPTTx
from ..tb import TBS
from .tb_resistant import TBResistant
from .dst import reset_dst_on_cure

__all__ = ['TPTRx']


class TPTRx(TPTTx):
    """
    Strain-aware TPT product. See module docstring for the mechanism.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        regimen_drugs (list): drugs in the TPT regimen (default: all drugs in ``strains``).
        resist_penalty (dict): optional per-drug clearance factors for strains resistant to
            regimen drugs. Default ``{}`` → binary unmasking (unspecified drugs contribute
            factor ``0``, so any regimen resistance blocks clearance). Unlike ``TxR`` (default
            factor 1 = no penalty), TPT defaults unknown drugs to full penalty to preserve the
            Mills–Cohen unmasking default.
        p_tpt_acq (dict): per-drug ``{drug: prob}`` of a susceptible strain acquiring resistance
            under TPT drug pressure (replacement). Default none (off).
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

    def __init__(self, strains, regimen_drugs=None, resist_penalty=None, p_tpt_acq=None,
                 acq_state_rr=None, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)
        self.strains = strains
        self.regimen_drugs = list(regimen_drugs) if regimen_drugs is not None else list(strains.drugs)
        self.resist_penalty = dict(resist_penalty) if resist_penalty else {}
        self.p_tpt_acq = dict(p_tpt_acq) if p_tpt_acq else {}
        self.acq_state_rr = dict(self.DEFAULT_ACQ_STATE_RR)
        if acq_state_rr:
            self.acq_state_rr.update({int(k): float(v) for k, v in acq_state_rr.items()})

        # Clearance factor per strain: ∏ resist_penalty[d] over regimen drugs the strain resists.
        # Unspecified regimen drugs default to 0 (binary unmasking when resist_penalty={}).
        penalty = np.zeros(strains.n)  # default 0 for unspecified drugs
        for d, f in self.resist_penalty.items():
            penalty[strains.drug_idx[d]] = f
        in_regimen = np.array([d in self.regimen_drugs for d in strains.drugs])
        self.clear_factor_by_id = np.zeros(strains.m)
        for j in range(strains.m):
            resists = strains.profile[j] & in_regimen
            if not resists.any():
                self.clear_factor_by_id[j] = 1.0  # fully susceptible to regimen
            else:
                self.clear_factor_by_id[j] = float(penalty[resists].prod())

        # Legacy bitmask: strains with factor==1 (fully covered) — used when all factors in {0,1}
        self._covered_mask = int(sum(1 << j for j in range(strains.m) if self.clear_factor_by_id[j] >= 1.0))

        # One independent uniform stream per regimen drug for TPT-driven acquisition.
        self._acq_rngs = [ss.random(name=f'tpt_acq_{d}') for d in self.regimen_drugs]
        self._clear_rngs = [ss.bernoulli(name=f'tpt_clear_{j}', p=float(max(self.clear_factor_by_id[j], 0.0)))
                            for j in range(strains.m)]
        self._n_acquired = 0
        return

    def update_roster(self):
        """Reset per-step acquisition counter, then run base mechanism completion."""
        self._n_acquired = 0
        super().update_roster()
        return

    def _tb(self):
        return self.sim.diseases[self.pars.disease]

    def _acquire(self, uids):
        """TPT drug pressure selects resistance: a susceptible carried strain mutates to its
        resistant counterpart (replacement) with prob ``p_tpt_acq[drug]`` × per-state RR."""
        if len(uids) == 0 or not self.p_tpt_acq:
            return
        tb = self._tb()
        m = self.strains
        rr = np.array([self.acq_state_rr.get(int(s), 0.0) for s in tb.state[uids]], dtype=float)
        before = np.asarray(tb.strain_mask[uids]).copy()
        surv = before.copy()
        for di, drug in enumerate(self.regimen_drugs):
            p = self.p_tpt_acq.get(drug, 0.0)
            if p <= 0:
                continue
            dcol = m.drug_idx[drug]
            bit = m.drug_bit(drug)
            sus_ids = [j for j in range(m.m) if not m.profile[j, dcol]]  # strains susceptible to this drug
            u = np.asarray(self._acq_rngs[di].rvs(uids), dtype=float)
            hit = u < (p * rr)
            if not hit.any():
                continue
            sub = surv[hit].copy()
            done = np.zeros(len(sub), dtype=bool)  # replace only the first susceptible carried strain
            for j in sus_ids:
                has_j = (((sub >> j) & 1).astype(bool)) & ~done
                if not has_j.any():
                    continue
                sub[has_j] = (sub[has_j] & ~(1 << j)) | (1 << (j | bit))
                done |= has_j
            surv[hit] = sub
        tb.strain_mask[uids] = surv
        self._n_acquired += int(np.count_nonzero(surv != before))
        return

    def _apply_sterilization(self, uids):
        """Per-strain sterilization: each carried strain clears with probability ``clear_factor_by_id``.

        Factor 1 (fully susceptible) / 0 (resistant, default) reproduce binary clearance; graded
        ``resist_penalty`` yields intermediate factors. Agents left with no strains go to CLEARED.
        """
        tb = self._tb()
        if not isinstance(tb, TBResistant):
            return super()._apply_sterilization(uids)  # non-strain TB: fall back to whole-agent clearance

        still = uids[tb.state[uids] == TBS.INFECTION]
        if len(still):
            masks = np.asarray(tb.strain_mask[still]).copy()
            surv = masks.copy()
            binary = np.all((self.clear_factor_by_id == 0) | (self.clear_factor_by_id >= 1.0))
            if binary:
                surv &= ~self._covered_mask
            else:
                for j in range(self.strains.m):
                    p = self.clear_factor_by_id[j]
                    if p <= 0:
                        continue
                    carrier = ((masks >> j) & 1).astype(bool)
                    if not carrier.any():
                        continue
                    if p >= 1.0:
                        surv[carrier] &= ~(1 << j)
                        continue
                    cleared = self._clear_rngs[j].rvs(still[carrier])
                    sub = surv[carrier]
                    sub[cleared] &= ~(1 << j)
                    surv[carrier] = sub
            tb.strain_mask[still] = surv
            cleared = still[surv == 0]
            if len(cleared):
                tb.state[cleared] = TBS.CLEARED
                tb.rr_reinfection[cleared] = tb.pars.rr_reinfection_cleared
                if tb.pars.dur_reinfection_protection is not None:
                    tb.ti_rr_reinfection_wane[cleared] = self.ti + tb.pars.dur_reinfection_protection.rvs(cleared)
                tb.infected[cleared] = False
                tb.susceptible[cleared] = True
                reset_dst_on_cure(self.sim, cleared)
        self.tpt_resolved[uids] = True
        return

    def _apply_neither_branch(self, uids):
        """TPT completely ineffective → run an acquisition trial against the regimen drugs, then resolve."""
        tb = self._tb()
        if isinstance(tb, TBResistant):
            self._acquire(uids)
        self.tpt_resolved[uids] = True
        return
