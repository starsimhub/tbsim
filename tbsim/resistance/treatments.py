"""
Strain-aware TB treatment.

``TxR`` (product) holds the regimen's per-strain efficacy, agent-level adherence,
and acquisition-on-failure rules. ``TxDeliveryR`` (delivery) initiates treatment
from the active-disease states at state-specific rates, freezes the outcome at
initiation, and resolves it after a fixed course.

Per the reference treatment operator (``model-tests.md`` §6): each carried strain
is cured independently (efficacy reduced for resistant strains), the whole course's
outcomes are correlated through a single agent-level adherence draw, failures return
to the state treatment was initiated from with the surviving strains, and among
surviving treatment-susceptible strains resistance is acquired with probability
``q_acq`` as **replacement** (the surviving A strain becomes B). This reproduces the
ODE's ``π(m → s)`` outcome table exactly for the two-strain case.
"""

import numpy as np
import starsim as ss

from ..tb import TBS, get_tb
from .tb_resistant import TBResistant

__all__ = ['TxR', 'TxDeliveryR']


class TxR(ss.Product):
    """
    Strain-aware treatment product.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        base_efficacy (float): per-course cure probability for a treatment-susceptible strain.
        resist_penalty (dict): per-drug multiplicative efficacy penalty applied for each drug a
            strain is resistant to, e.g. ``{'TX': 0.333}`` gives a resistant strain 1/3 the cure
            probability. Drugs absent from the dict have no penalty (penalty 1).
        adherence (float): probability an agent completes the course; non-adherent agents clear no
            strains this course (the mechanism correlating outcomes across an agent's strains).
        q_acq (float): probability a surviving treatment-susceptible strain acquires resistance to
            each regimen drug on a failed course (always replacement).
        regimen_drugs (list): drugs the regimen acts on; determines which strains can be cured/acquire
            resistance. Default: all drugs in ``strains``.
    """

    def __init__(self, strains, base_efficacy=0.85, resist_penalty=None, adherence=1.0,
                 q_acq=0.0, regimen_drugs=None, **kwargs):
        super().__init__(**kwargs)
        self.strains = strains
        self.base_efficacy = base_efficacy
        self.resist_penalty = resist_penalty or {}
        self.regimen_drugs = list(regimen_drugs) if regimen_drugs is not None else list(strains.drugs)

        # Per-strain cure probability = base × ∏ penalty over the strain's resistant drugs.
        penalty = np.ones(strains.n)
        for d, f in self.resist_penalty.items():
            penalty[strains.drug_idx[d]] = f
        self.eff_by_id = np.array([base_efficacy * penalty[strains.profile[j]].prod() for j in range(strains.m)])

        # CRN distributions (a list of Dists is discovered by sc.search like any attribute).
        self._adh_rng = ss.bernoulli(name='txr_adherence', p=adherence)
        self._cure_rngs = [ss.bernoulli(name=f'txr_cure_{j}', p=float(self.eff_by_id[j])) for j in range(strains.m)]
        self._acq_rng = ss.bernoulli(name='txr_acquire', p=q_acq)
        return

    def roll_survivors(self, tb, uids):
        """Pre-roll the surviving strain mask for each treated agent (adherence, then per-strain cure)."""
        m = self.strains
        masks = tb.strain_mask[uids]
        surv = masks.copy()
        adherent = self._adh_rng.rvs(uids)  # per-agent, position-aligned with uids
        for j in range(m.m):
            if self.eff_by_id[j] <= 0:
                continue
            carrier = (((masks >> j) & 1).astype(bool)) & adherent
            if not carrier.any():
                continue
            cured = self._cure_rngs[j].rvs(uids[carrier])  # per-carrier cure draw
            sub = surv[carrier]
            sub[cured] &= ~(1 << j)
            surv[carrier] = sub
        return surv

    def acquire(self, uids, surv):
        """Apply acquisition-on-failure (replacement) to the surviving masks of failed courses."""
        if len(uids) == 0 or self._acq_rng.pars.p == 0:
            return surv
        m = self.strains
        for drug in self.regimen_drugs:
            di = m.drug_idx[drug]
            bit = m.drug_bit(drug)
            for j in range(m.m):
                if m.profile[j, di]:
                    continue  # strain j already resistant to this drug
                carrier = ((surv >> j) & 1).astype(bool)
                if not carrier.any():
                    continue
                acq = self._acq_rng.rvs(uids[carrier])
                sub = surv[carrier]
                # Replacement: strain j becomes j|bit (drop the old strain, add the resistant one).
                sub[acq] = (sub[acq] & ~(1 << j)) | (1 << (j | bit))
                surv[carrier] = sub
        return surv


class TxDeliveryR(ss.Intervention):
    """
    Rate-based strain-aware treatment delivery.

    Each step, active-TB agents not already on treatment start a course at state-specific
    rates (``rate_asym`` from ASYMPTOMATIC, ``rate_sym`` from SYMPTOMATIC). The per-strain
    outcome is frozen at initiation and resolved after ``dur_treatment``: fully cleared
    courses go to CLEARED (with post-treatment reinfection protection); otherwise the agent
    returns to the state it was treated from carrying the surviving (and possibly newly
    resistant) strains.

    Args:
        product (TxR): the strain-aware treatment product.
        rate_asym (ss.rate): treatment initiation rate from ASYMPTOMATIC.
        rate_sym (ss.rate): treatment initiation rate from SYMPTOMATIC.
        dur_treatment (ss.dur): course duration (fixed).
        eligibility (callable): optional ``sim -> uids`` override; if given, those agents start
            treatment (subject to not already being on treatment) instead of the rate-based rule.
    """

    def __init__(self, product, rate_asym=ss.peryear(0.0), rate_sym=ss.peryear(2.0),
                 dur_treatment=ss.months(6), eligibility=None, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self.define_pars(
            rate_asym=rate_asym,
            rate_sym=rate_sym,
            dur_treatment=dur_treatment,
        )
        self.update_pars(**kwargs)
        self.define_states(
            ss.FloatArr('ti_treatment_end'),
            ss.IntArr('pending_surv', default=0),
            ss.IntArr('prior_state', default=int(TBS.SUSCEPTIBLE)),
        )
        self._init_rng = ss.bernoulli(name='txdr_init', p=0.5)
        product.name = f'{self.name}_product'
        return

    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('n_treated', dtype=int),
            ss.Result('n_success', dtype=int),
            ss.Result('n_failure', dtype=int),
            ss.Result('n_acquired', dtype=int, label='Resistance acquired on treatment failure'),
        )
        return

    def step(self):
        """Resolve completed courses, then initiate new ones."""
        self._resolve()
        self._initiate()
        return

    def _initiate(self):
        tb = get_tb(self.sim, which=TBResistant)
        if self.eligibility is not None:
            start = ss.uids(self.eligibility(self.sim))
            start = start[tb.state[start] != TBS.TREATMENT]
        else:
            asy = tb.asymptomatic.uids
            sym = tb.symptomatic.uids
            self._init_rng.set(p=self.pars.rate_asym.to_prob(self.t.dt))
            start_a = self._init_rng.filter(asy)
            self._init_rng.set(p=self.pars.rate_sym.to_prob(self.t.dt))
            start_y = self._init_rng.filter(sym)
            start = start_a | start_y

        self._n_treated = len(start)
        if len(start) == 0:
            return

        self.prior_state[start] = tb.state[start]
        self.pending_surv[start] = self.product.roll_survivors(tb, start)
        tb.state[start] = TBS.TREATMENT
        dur_steps = self.pars.dur_treatment / self.t.dt
        self.ti_treatment_end[start] = self.ti + dur_steps
        tb.results['new_notifications_15+'][tb.ti] += np.count_nonzero(self.sim.people.age[start] >= 15)
        return

    def _resolve(self):
        tb = get_tb(self.sim, which=TBResistant)
        self._n_success = self._n_failure = self._n_acquired = 0
        on_tx = (tb.state == TBS.TREATMENT).uids
        done = on_tx[self.ti >= self.ti_treatment_end[on_tx]]
        if len(done) == 0:
            return

        surv = self.pending_surv[done]
        cured = done[surv == 0]
        failed = done[surv != 0]

        # Cured: clear all strains, go to CLEARED with post-treatment reinfection protection.
        if len(cured):
            tb.state[cured] = TBS.CLEARED
            tb.strain_mask[cured] = 0
            tb.rr_reinfection[cured] = tb.pars.rr_reinfection_treat
            tb._set_reinfection_wane(cured)
        self._n_success = len(cured)

        # Failed: acquisition (replacement), then return to the state treatment was initiated from.
        if len(failed):
            surv_f = self.pending_surv[failed].copy()
            before = surv_f.copy()
            surv_f = self.product.acquire(failed, surv_f)
            self._n_acquired = int(np.count_nonzero(surv_f != before))
            tb.strain_mask[failed] = surv_f
            tb.state[failed] = self.prior_state[failed]
        self._n_failure = len(failed)
        return

    def update_results(self):
        ti = self.ti
        self.results.n_treated[ti] = self._n_treated
        self.results.n_success[ti] = self._n_success
        self.results.n_failure[ti] = self._n_failure
        self.results.n_acquired[ti] = self._n_acquired
        return
