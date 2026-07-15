"""
Strain-aware TB treatment.

``TxR`` (product) holds the regimen's per-strain efficacy, agent-level adherence,
and acquisition-on-failure rules. ``TxDeliveryR`` (delivery) initiates treatment
from the active-disease states at state-specific rates (or from a custom eligibility
callable), freezes the outcome at initiation, and resolves it after a fixed course.
It also supports treatment monitoring / regimen switching via ``interrupt`` +
``supersedes`` (see ``treatment_monitoring_eligibility``).

Per the reference treatment operator (``model-tests.md`` §6): each carried strain is
cured independently (efficacy reduced for strains resistant to *regimen* drugs), the
whole course's outcomes are correlated through a single agent-level adherence draw,
failures return to the state treatment was initiated from with the surviving strains,
and among surviving treatment-susceptible strains resistance is acquired with probability
``q_acq[drug]`` (× a per-state RR) as **replacement** (the surviving A strain becomes B).
This reproduces the ODE's ``π(m → s)`` outcome table exactly for the two-strain case.

Acquisition-on-failure replaces one carried susceptible strain per drug per episode
(lowest-id candidate); this intentional bottleneck differs from de-novo acquisition,
which mutates every carried strain independently.
"""

import numpy as np
import starsim as ss

from ..tb import TBS, get_tb
from .tb_resistant import TBResistant
from .dst import reset_dst_on_cure

__all__ = ['TxR', 'TxDeliveryR', 'treatment_monitoring_eligibility']


class TxR(ss.Product):
    """
    Strain-aware treatment product.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        base_efficacy (float): per-course cure probability for a treatment-susceptible strain.
        resist_penalty (dict): per-drug multiplicative efficacy penalty applied for each *regimen*
            drug a strain is resistant to, e.g. ``{'TX': 0.333}`` gives a resistant strain 1/3 the
            cure probability. Resistance to a drug outside the regimen does not reduce efficacy.
            Unspecified regimen drugs contribute factor 1 (no penalty) for active courses.
        adherence (float): probability an agent completes the course; non-adherent agents clear no
            strains this course (the mechanism correlating outcomes across an agent's strains).
        q_acq (dict): per-drug probability ``{drug: prob}`` that a surviving treatment-susceptible
            strain acquires resistance to that regimen drug on a failed course (always replacement).
            Default none (off).
        acq_state_rr (dict): per-TB-state multiplier on ``q_acq`` at time of failure (spec's RR on q).
            Default 1 for ASYMPTOMATIC/SYMPTOMATIC, 0 for all other states.
        regimen_drugs (list): drugs the regimen acts on; determines which strains can be cured/acquire
            resistance. Default: all drugs in ``strains``.
    """

    def __init__(self, strains, base_efficacy=0.85, resist_penalty=None, adherence=1.0,
                 q_acq=None, acq_state_rr=None, regimen_drugs=None, **kwargs):
        super().__init__(**kwargs)
        self.strains = strains
        self.base_efficacy = base_efficacy
        self.resist_penalty = resist_penalty or {}
        self.regimen_drugs = list(regimen_drugs) if regimen_drugs is not None else list(strains.drugs)
        self.q_acq = dict(q_acq) if q_acq else {}

        # Acquisition RR by TB state at treatment failure (spec: 1 for ASY/SYM, 0 otherwise).
        self.acq_state_rr = {int(TBS.ASYMPTOMATIC): 1.0, int(TBS.SYMPTOMATIC): 1.0}
        if acq_state_rr:
            self.acq_state_rr.update({int(k): float(v) for k, v in acq_state_rr.items()})

        # Per-strain cure probability = base × ∏ penalty over the *regimen* drugs the strain resists
        # (resistance to a non-regimen drug leaves this regimen's efficacy unchanged).
        penalty = np.ones(strains.n)
        for d, f in self.resist_penalty.items():
            penalty[strains.drug_idx[d]] = f
        in_regimen = np.array([d in self.regimen_drugs for d in strains.drugs])
        self.eff_by_id = np.array([base_efficacy * penalty[strains.profile[j] & in_regimen].prod() for j in range(strains.m)])
        # Latent clear factor = ∏ penalty only (no base_efficacy): susceptible → 1; resistant → reduced.
        self.latent_clear_by_id = np.array([float(penalty[strains.profile[j] & in_regimen].prod()) for j in range(strains.m)])

        # CRN distributions (a list of Dists is discovered by sc.search like any attribute).
        self._adh_rng = ss.bernoulli(name='txr_adherence', p=adherence)
        self._cure_rngs = [ss.bernoulli(name=f'txr_cure_{j}', p=float(self.eff_by_id[j])) for j in range(strains.m)]
        self._latent_rngs = [ss.bernoulli(name=f'txr_latent_{j}', p=float(self.latent_clear_by_id[j])) for j in range(strains.m)]
        # One independent uniform stream per regimen drug for acquisition-on-failure (state-scaled at draw).
        self._acq_rngs = [ss.random(name=f'txr_acq_{d}') for d in self.regimen_drugs]
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

    def clear_latent(self, tb, uids):
        """Clear strains in latent (INFECTION) agents without a timed course.

        Regimen-susceptible strains clear with probability 1; strains resistant to regimen drugs
        clear at ``∏ resist_penalty`` (no ``base_efficacy`` multiplier). Returns the surviving
        strain masks (caller may run ``acquire`` and route agents to CLEARED / residual state).
        """
        m = self.strains
        masks = np.asarray(tb.strain_mask[uids]).copy()
        surv = masks.copy()
        for j in range(m.m):
            p = self.latent_clear_by_id[j]
            if p <= 0:
                continue
            carrier = ((masks >> j) & 1).astype(bool)
            if not carrier.any():
                continue
            if p >= 1.0:
                surv[carrier] &= ~(1 << j)
                continue
            cured = self._latent_rngs[j].rvs(uids[carrier])
            sub = surv[carrier]
            sub[cured] &= ~(1 << j)
            surv[carrier] = sub
        return surv

    def acquire(self, uids, surv, states=None):
        """Apply acquisition-on-failure (replacement) to the surviving masks of failed courses.

        One trial per agent per regimen drug (spec: once per treatment episode), scaled by the
        per-agent state RR (``acq_state_rr``; default 0 outside ASYMPTOMATIC/SYMPTOMATIC). A hit
        replaces one carried strain susceptible to that drug with its resistant counterpart
        (first/lowest-id candidate — intentional bottleneck vs multi-strain de-novo).
        """
        if len(uids) == 0 or not self.q_acq:
            return surv
        m = self.strains
        rr = np.ones(len(uids))
        if states is not None:
            rr = np.array([self.acq_state_rr.get(int(s), 0.0) for s in states], dtype=float)
        for di, drug in enumerate(self.regimen_drugs):
            p = self.q_acq.get(drug, 0.0)
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
            done = np.zeros(len(sub), dtype=bool)  # replace only the first susceptible carried strain per agent
            for j in sus_ids:
                has_j = (((sub >> j) & 1).astype(bool)) & ~done
                if not has_j.any():
                    continue
                sub[has_j] = (sub[has_j] & ~(1 << j)) | (1 << (j | bit))
                done |= has_j
            surv[hit] = sub
        return surv


class TxDeliveryR(ss.Intervention):
    """
    Rate-based strain-aware treatment delivery.

    Each step, active-TB agents not already on treatment start a course at state-specific
    rates (``rate_asym`` from ASYMPTOMATIC, ``rate_sym`` from SYMPTOMATIC), or from a custom
    ``eligibility`` callable (e.g. a DST-routed regimen). The per-strain outcome is frozen at
    initiation and resolved after ``dur_treatment``: fully cleared courses go to CLEARED (with
    post-treatment reinfection protection); otherwise the agent returns to the state it was
    treated from carrying the surviving (and possibly newly resistant) strains.

    Custom eligibility is gated to current active TB or latent INFECTION (not CLEARED), so
    sticky DST profiles cannot re-treat cured agents. Latent agents take an immediate
    resistance-aware clear path (see ``TxR.clear_latent``) rather than a timed course.

    Treatment monitoring / regimen switching: a delivery given ``supersedes=[name, ...]`` will
    ``interrupt`` any ongoing course on those deliveries for its eligible agents before starting
    them, so a second-line regimen can take over an in-progress first-line course.

    Args:
        product (TxR): the strain-aware treatment product.
        rate_asym (ss.rate): treatment initiation rate from ASYMPTOMATIC.
        rate_sym (ss.rate): treatment initiation rate from SYMPTOMATIC.
        dur_treatment (ss.dur): course duration (fixed).
        eligibility (callable): optional ``sim -> uids`` override; if given, those agents start
            treatment instead of the rate-based rule (still gated to treatable states).
        supersedes (str/list): name(s) of other ``TxDeliveryR`` whose ongoing course is interrupted
            for eligible agents before this delivery starts them (regimen switching).
    """

    def __init__(self, product, rate_asym=ss.peryear(0.0), rate_sym=ss.peryear(2.0),
                 dur_treatment=ss.months(6), eligibility=None, supersedes=None, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self.supersedes = [supersedes] if isinstance(supersedes, str) else list(supersedes or [])
        self.define_pars(
            rate_asym=rate_asym,
            rate_sym=rate_sym,
            dur_treatment=dur_treatment,
        )
        self.update_pars(**kwargs)
        self.define_states(
            ss.FloatArr('ti_treatment_start'),
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
        self._n_treated = self._n_success = self._n_failure = self._n_acquired = 0
        self._resolve()
        self._initiate()
        return

    def interrupt(self, uids):
        """Prematurely stop this delivery's ongoing courses for *uids* (treatment monitoring / switch).

        Reverts interrupted agents to the state treatment was initiated from, keeping their current
        strains, so another delivery can re-treat them. Only agents actually on *this* delivery's
        course (state TREATMENT with a finite ``ti_treatment_end``) are affected.
        """
        uids = ss.uids(uids)
        if len(uids) == 0:
            return ss.uids()
        tb = get_tb(self.sim, which=TBResistant)
        mine = uids[(tb.state[uids] == TBS.TREATMENT) & np.isfinite(self.ti_treatment_end[uids])]
        if len(mine) == 0:
            return mine
        tb.state[mine] = self.prior_state[mine]
        self.ti_treatment_start[mine] = np.nan
        self.ti_treatment_end[mine] = np.nan
        self.pending_surv[mine] = 0
        return mine

    def _initiate(self):
        tb = get_tb(self.sim, which=TBResistant)
        if self.eligibility is not None:
            start = ss.uids(self.eligibility(self.sim))
            # Regimen switch: interrupt superseded deliveries' ongoing courses so these agents
            # leave TREATMENT and can be re-started here.
            if len(start) and self.supersedes:
                switching = start[tb.state[start] == TBS.TREATMENT]
                for name in self.supersedes:
                    other = self.sim.interventions.get(name)
                    if other is not None and len(switching):
                        other.interrupt(switching)
            start = start[tb.state[start] != TBS.TREATMENT]
            # Gate custom eligibility to treatable disease (blocks CLEARED phantom re-treatment).
            if len(start):
                start = start[tb.active_tb[start] | tb.latent[start]]
        else:
            asy = tb.asymptomatic.uids
            sym = tb.symptomatic.uids
            self._init_rng.set(p=self.pars.rate_asym.to_prob(self.t.dt))
            start_a = self._init_rng.filter(asy)
            self._init_rng.set(p=self.pars.rate_sym.to_prob(self.t.dt))
            start_y = self._init_rng.filter(sym)
            start = start_a | start_y

        if len(start) == 0:
            return

        # Latent (INFECTION): immediate resistance-aware clear — no timed TREATMENT course.
        latent = start[tb.latent[start]]
        active = start[tb.active_tb[start]]
        if len(latent):
            self._initiate_latent(tb, latent)
        if len(active):
            self._initiate_active(tb, active)
        return

    def _initiate_latent(self, tb, uids):
        """Clear latent infection immediately; apply acquisition on surviving resistant strains."""
        self._n_treated += len(uids)
        surv = self.product.clear_latent(tb, uids)
        before = surv.copy()
        surv = self.product.acquire(uids, surv, states=np.full(len(uids), int(TBS.INFECTION)))
        self._n_acquired += int(np.count_nonzero(surv != before))
        cured = uids[surv == 0]
        failed = uids[surv != 0]
        if len(cured):
            tb.state[cured] = TBS.CLEARED
            tb.strain_mask[cured] = 0
            tb.rr_reinfection[cured] = tb.pars.rr_reinfection_cleared
            tb._set_reinfection_wane(cured)
            tb.infected[cured] = False
            tb.susceptible[cured] = True
            reset_dst_on_cure(self.sim, cured)
        if len(failed):
            tb.strain_mask[failed] = surv[surv != 0]
            tb.state[failed] = TBS.INFECTION
            tb.infected[failed] = True
            tb.susceptible[failed] = False
        self._n_success += len(cured)
        self._n_failure += len(failed)
        return

    def _initiate_active(self, tb, uids):
        """Start a timed TREATMENT course for active-TB agents."""
        self._n_treated += len(uids)
        self.prior_state[uids] = tb.state[uids]
        self.pending_surv[uids] = self.product.roll_survivors(tb, uids)
        tb.state[uids] = TBS.TREATMENT
        dur_steps = self.pars.dur_treatment / self.t.dt
        self.ti_treatment_start[uids] = self.ti
        self.ti_treatment_end[uids] = self.ti + dur_steps
        tb.results['new_notifications_15+'][tb.ti] += np.count_nonzero(self.sim.people.age[uids] >= 15)
        return

    def _resolve(self):
        tb = get_tb(self.sim, which=TBResistant)
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
            reset_dst_on_cure(self.sim, cured)
        self._n_success += len(cured)

        # Failed: acquisition (replacement, state-scaled), then return to the state treated from.
        if len(failed):
            surv_f = self.pending_surv[failed].copy()
            before = surv_f.copy()
            surv_f = self.product.acquire(failed, surv_f, states=self.prior_state[failed])
            self._n_acquired += int(np.count_nonzero(surv_f != before))
            tb.strain_mask[failed] = surv_f
            tb.state[failed] = self.prior_state[failed]
        self._n_failure += len(failed)
        return

    def update_results(self):
        ti = self.ti
        self.results.n_treated[ti] = self._n_treated
        self.results.n_success[ti] = self._n_success
        self.results.n_failure[ti] = self._n_failure
        self.results.n_acquired[ti] = self._n_acquired
        return


def treatment_monitoring_eligibility(tx_name, after_steps=4, every_steps=None, extra=None):
    """
    Eligibility callable selecting agents on ``tx_name``'s course for at least ``after_steps`` steps.

    Feed as ``eligibility=`` to a monitoring ``DxDelivery`` (to flag still-bacteriologically-positive
    agents by TB state), or — combined with ``supersedes=[tx_name]`` on a second-line ``TxDeliveryR`` —
    to switch regimens mid-course (spec §"Treatment monitoring").

    Compose with DST or other filters via ``extra`` (a ``sim → uids`` callable intersected after the
    time-on-treatment gate), e.g.::

        treatment_monitoring_eligibility('first', after_steps=2, extra=dst.matches(RIF=True))

    Args:
        tx_name (str): the ``name`` of the ``TxDeliveryR`` to monitor.
        after_steps (int): minimum sim steps since ``ti_treatment_start`` before eligibility. Default 4.
        every_steps (int/None): if given, re-test every N steps after the first; else a single test at ``after_steps``.
        extra (callable/None): optional ``sim → uids`` filter intersected with the time-on-treatment set.
    """
    def _elig(sim):
        tx = sim.interventions.get(tx_name)
        if tx is None:
            return ss.uids()
        tb = get_tb(sim, which=TBResistant)
        on_tx = (tb.state == TBS.TREATMENT).uids
        if len(on_tx) == 0:
            return on_tx
        start = np.asarray(tx.ti_treatment_start[on_tx], dtype=float)  # finite only for tx's own patients
        elapsed = sim.ti - start
        ready = np.isfinite(start) & (elapsed >= after_steps)
        if every_steps:
            ready &= ((elapsed - after_steps) % every_steps == 0)
        out = on_tx[ready]
        if extra is not None and len(out):
            out = out.intersect(ss.uids(extra(sim)))
        return out
    _elig.__name__ = f'monitoring_after_{after_steps}_steps'
    return _elig
