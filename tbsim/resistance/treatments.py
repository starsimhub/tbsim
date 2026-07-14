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
"""

import numpy as np
import starsim as ss

from ..tb import TBS, get_tb, choice2d
from .tb_resistant import TBResistant

__all__ = ['TxR', 'TxDeliveryR', 'treatment_monitoring_eligibility', 'eligibility_all', 'eligibility_any', 'will_fail']


class TxR(ss.Product):
    """
    Strain-aware treatment product.

    Args:
        strains (Strains): the strain registry (e.g. ``tb.strains``).
        base_efficacy (float): per-course cure probability for a treatment-susceptible strain.
        resist_penalty (dict): per-drug multiplicative efficacy penalty applied for each *regimen*
            drug a strain is resistant to, e.g. ``{'TX': 0.333}`` gives a resistant strain 1/3 the
            cure probability. Resistance to a drug outside the regimen does not reduce efficacy.
        efficacy_by_strain (array): optional explicit per-strain cure-probability vector
            ``T_l = {t_1,l, ..., t_m,l}`` (length ``strains.m``, spec §"Treatment efficacy"). When given
            it *is* the per-strain efficacy and ``base_efficacy``/``resist_penalty`` are ignored; use it
            when the constrained ``base × ∏penalty`` parameterization cannot express the desired vector.
        adherence (float or callable): per-course completion probability correlating outcomes across an
            agent's strains (non-completers clear no strains this course). A float applies one regimen-level
            probability to every agent; a callable ``uids -> per-agent probability`` makes adherence a
            *regimen-level distribution that varies by agent* and is applied across all that agent's strains
            (spec §"Treatment efficacy"), e.g. ``adherence=lambda uids: my_dist.rvs(uids)``.
        q_acq (dict): per-drug probability ``{drug: prob}`` that a surviving treatment-susceptible
            strain acquires resistance to that regimen drug on a failed course (always replacement).
            Default none (off).
        acq_state_rr (dict): per-TB-state multiplier on ``q_acq`` at time of failure (spec's RR on q).
            Default 1 for ASYMPTOMATIC/SYMPTOMATIC, 0 for all other states.
        regimen_drugs (list): drugs the regimen acts on; determines which strains can be cured/acquire
            resistance. Default: all drugs in ``strains``.
        acq_select (str): how the carried strain that acquires resistance on failure is chosen among
            the drug-susceptible carried strains — ``'random'`` (uniform, default) or ``'fitness'``
            (∝ fitness). See L4 / implementation-decisions.md D-L4.
    """

    def __init__(self, strains, base_efficacy=0.85, resist_penalty=None, efficacy_by_strain=None,
                 adherence=1.0, q_acq=None, acq_state_rr=None, regimen_drugs=None, acq_select='random', **kwargs):
        super().__init__(**kwargs)
        self.strains = strains
        self.base_efficacy = base_efficacy
        self.resist_penalty = resist_penalty or {}
        self.regimen_drugs = list(regimen_drugs) if regimen_drugs is not None else list(strains.drugs)
        self.q_acq = dict(q_acq) if q_acq else {}
        self.acq_select = acq_select
        # Fail-fast on mistyped drug names (L8) before they silently resolve via .get().
        strains.validate_drugs(self.regimen_drugs, where='TxR.regimen_drugs')
        strains.validate_drugs(self.q_acq, where='TxR.q_acq')
        strains.validate_drugs(self.resist_penalty, where='TxR.resist_penalty')

        # Acquisition RR by TB state at treatment failure (spec: 1 for ASY/SYM, 0 otherwise).
        self.acq_state_rr = {int(TBS.ASYMPTOMATIC): 1.0, int(TBS.SYMPTOMATIC): 1.0}
        if acq_state_rr:
            self.acq_state_rr.update({int(k): float(v) for k, v in acq_state_rr.items()})

        # Per-strain cure-probability vector T_l = {t_1,l, ..., t_m,l}. Either taken verbatim from an
        # explicit `efficacy_by_strain` (spec's arbitrary vector), or derived as base × ∏ penalty over
        # the *regimen* drugs the strain resists (resistance to a non-regimen drug leaves efficacy
        # unchanged) — the constrained parameterization that covers the common case.
        if efficacy_by_strain is not None:
            eff = np.asarray(efficacy_by_strain, dtype=float)
            if eff.shape != (strains.m,):
                raise ValueError(f'efficacy_by_strain must have length strains.m={strains.m}, got {eff.shape}.')
            if np.any((eff < 0) | (eff > 1)):
                raise ValueError(f'efficacy_by_strain values must be in [0, 1], got {eff}.')
            self.eff_by_id = eff
        else:
            penalty = np.ones(strains.n)
            for d, f in self.resist_penalty.items():
                penalty[strains.drug_idx[d]] = f
            in_regimen = np.array([d in self.regimen_drugs for d in strains.drugs])
            self.eff_by_id = np.array([base_efficacy * penalty[strains.profile[j] & in_regimen].prod() for j in range(strains.m)])

        # Adherence: a regimen-level probability (float) shared by all agents, or a regimen-level
        # *distribution* (callable uids -> per-agent probability) that varies by agent. Either way a
        # single per-agent completion draw correlates outcomes across all that agent's strains.
        self.adherence_distribution = adherence if callable(adherence) else None
        # CRN distributions (a list of Dists is discovered by sc.search like any attribute).
        self._adh_rng = ss.bernoulli(name='txr_adherence', p=(0.5 if self.adherence_distribution else adherence))
        self._cure_rngs = [ss.bernoulli(name=f'txr_cure_{j}', p=float(self.eff_by_id[j])) for j in range(strains.m)]
        # One independent uniform stream per regimen drug for acquisition-on-failure (state-scaled at draw),
        # plus one choice stream per drug to pick which carried susceptible strain mutates (L4).
        self._acq_rngs = [ss.random(name=f'txr_acq_{d}') for d in self.regimen_drugs]
        self._acq_select = [choice2d(p=np.ones((1, strains.m)) / strains.m, name=f'txr_acqsel_{d}') for d in self.regimen_drugs]
        return

    def roll_survivors(self, tb, uids):
        """Pre-roll the surviving strain mask for each treated agent (adherence, then per-strain cure)."""
        m = self.strains
        masks = tb.strain_mask[uids]
        surv = masks.copy()
        if self.adherence_distribution is not None:  # per-agent completion probability from the distribution
            self._adh_rng.set(p=np.asarray(self.adherence_distribution(uids), dtype=float))
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

    def acquire(self, uids, surv, states=None):
        """Apply acquisition-on-failure (replacement) to the surviving masks of failed courses.

        One trial per agent per regimen drug (spec: once per treatment episode), scaled by the
        per-agent state RR (``acq_state_rr``; default 0 outside ASYMPTOMATIC/SYMPTOMATIC). A hit
        replaces one carried strain susceptible to that drug — chosen per ``acq_select`` (L4) — with
        its resistant counterpart.
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
            u = np.asarray(self._acq_rngs[di].rvs(uids), dtype=float)
            hit = u < (p * rr)
            if not hit.any():
                continue
            surv[hit] = m.mutate_one_susceptible(surv[hit], drug, self._acq_select[di], uids[hit],
                                                 weighted=(self.acq_select == 'fitness'))
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

    Treatment monitoring / regimen switching: a delivery given ``supersedes=[name, ...]`` will
    ``interrupt`` any ongoing course on those deliveries for its eligible agents before starting
    them, so a second-line regimen can take over an in-progress first-line course.

    Args:
        product (TxR): the strain-aware treatment product.
        rate_asym (ss.rate): treatment initiation rate from ASYMPTOMATIC.
        rate_sym (ss.rate): treatment initiation rate from SYMPTOMATIC.
        dur_treatment (ss.dur): course duration (fixed).
        eligibility (callable): optional ``sim -> uids`` override; if given, those agents start
            treatment instead of the rate-based rule.
        supersedes (str/list): name(s) of other ``TxDeliveryR`` whose ongoing course is interrupted
            for eligible agents before this delivery starts them (regimen switching).
        retreat_after (ss.dur): refractory period after a course ends before the *same* agent may be
            re-treated by this delivery. Prevents a failing agent from being re-treated on every
            resolution off a single stale DST result (L1). Default ``None`` = no guard.
        treat_latent (bool): how latent (``INFECTION``) agents selected for treatment are handled.
            ``False`` (default): cleared immediately to ``CLEARED`` with no course (matching base
            ``tbsim.TxDelivery``); they never enter ``TREATMENT`` and cannot acquire resistance.
            ``True``: run through a full course that can fail / select for resistance. See L3 /
            implementation-decisions.md D-L3. (The default rate-based eligibility never selects latent
            agents, so this only affects custom / DST-routed eligibilities.)
    """

    def __init__(self, product, rate_asym=ss.peryear(0.0), rate_sym=ss.peryear(2.0),
                 dur_treatment=ss.months(6), eligibility=None, supersedes=None, retreat_after=None,
                 treat_latent=False, **kwargs):
        super().__init__()
        self.product = product
        self.eligibility = eligibility
        self.supersedes = [supersedes] if isinstance(supersedes, str) else list(supersedes or [])
        self.retreat_after = retreat_after
        self.treat_latent = treat_latent
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

    @staticmethod
    def failure_case_eligibility(within, base=None, new_case=False):
        """Classify a later treatment episode as *treatment failure/retreatment* vs a *new case*.

        Implements the spec's requirement (§Diagnostics) to "track time since last treatment initiation
        to inform whether later treatment is managed as treatment failure, with need for DST/second-line
        treatments, or as a new case." Reads the durable, cross-regimen ``tb.ti_last_treatment`` written by
        every :class:`TxDeliveryR` at initiation.

        Returns a ``sim -> uids`` eligibility callable selecting agents whose most recent treatment
        initiation was within ``within`` (an ``ss.dur``) of the current step — i.e. to be managed as a
        treatment failure (route to DST / second-line). Pass ``new_case=True`` for the complement (agents
        with no treatment within the window → managed as a new case). ``base`` optionally restricts the
        candidate pool (default: current active TB), e.g. ``base=dst.matches(RIF=True)``.

        Feed as ``eligibility=`` to a DST or ``TxDeliveryR`` (optionally via :func:`eligibility_all`)::

            failed = TxDeliveryR.failure_case_eligibility(within=ss.years(2))
            second_line = tbsim.TxDeliveryR(eligibility=failed, supersedes=['first'], product=...)
        """
        def _elig(sim):
            tb = get_tb(sim, which=TBResistant)
            cand = ss.uids(base(sim)) if base is not None else tb.active_tb.uids
            if len(cand) == 0:
                return cand
            last = np.asarray(tb.ti_last_treatment[cand], dtype=float)
            window = within / sim.t.dt
            recent = np.isfinite(last) & ((sim.ti - last) <= window)
            return cand[~recent] if new_case else cand[recent]
        _elig.__name__ = 'new_case_eligibility' if new_case else 'failure_case_eligibility'
        return _elig

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
        else:
            asy = tb.asymptomatic.uids
            sym = tb.symptomatic.uids
            self._init_rng.set(p=self.pars.rate_asym.to_prob(self.t.dt))
            start_a = self._init_rng.filter(asy)
            self._init_rng.set(p=self.pars.rate_sym.to_prob(self.t.dt))
            start_y = self._init_rng.filter(sym)
            start = start_a | start_y

        # Retreatment refractory guard (L1): skip agents whose most recent course on THIS delivery
        # ended fewer than `retreat_after` steps ago, so a failing agent isn't re-treated immediately.
        # Switching agents (superseded) are on another delivery, so their end time here is nan → kept.
        if self.retreat_after is not None and len(start):
            end = self.ti_treatment_end[start]
            window = self.retreat_after / self.t.dt
            start = start[~(np.isfinite(end) & ((self.ti - end) < window))]

        # Latent-treatment divergence (L3): by default, latent agents selected for treatment are
        # cleared immediately (base-tbsim behavior) instead of running a course that could fail /
        # acquire resistance, and are excluded from the course-based `start` (not counted as treated).
        if not self.treat_latent and len(start):
            is_latent = tb.latent[start]
            latent = start[is_latent]
            if len(latent):
                tb.state[latent] = TBS.CLEARED
                tb.strain_mask[latent] = 0
                tb._reset_counts(latent)
                tb.rr_reinfection[latent] = tb.pars.rr_reinfection_cleared
                tb._set_reinfection_wane(latent)
                tb.infected[latent] = False
                tb.susceptible[latent] = True
                start = start[~is_latent]

        self._n_treated = len(start)
        if len(start) == 0:
            return

        self.prior_state[start] = tb.state[start]
        self.pending_surv[start] = self.product.roll_survivors(tb, start)
        tb.ti_last_treatment[start] = self.ti  # durable cross-regimen history (failure-vs-new-case; spec §Diagnostics)
        tb.state[start] = TBS.TREATMENT
        dur_steps = self.pars.dur_treatment / self.t.dt
        self.ti_treatment_start[start] = self.ti
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

        # Cured: clear all strains (and their counts, spec §5), go to CLEARED with post-treatment protection.
        if len(cured):
            tb.state[cured] = TBS.CLEARED
            tb.strain_mask[cured] = 0
            tb._reset_counts(cured)
            tb.rr_reinfection[cured] = tb.pars.rr_reinfection_treat
            tb._set_reinfection_wane(cured)
        self._n_success = len(cured)

        # Failed: acquisition (replacement, state-scaled), then return to the state treated from. The
        # per-strain counter is re-synced from the pre-course mask: cured strains → count 0, an acquired
        # (emergent) strain → count 1, surviving strains keep their count (spec §5, D-COUNTER).
        if len(failed):
            before = np.asarray(tb.strain_mask[failed]).copy()
            surv_f = self.pending_surv[failed].copy()
            before_acq = surv_f.copy()
            surv_f = self.product.acquire(failed, surv_f, states=self.prior_state[failed])
            self._n_acquired = int(np.count_nonzero(surv_f != before_acq))
            tb.strain_mask[failed] = surv_f
            tb._sync_counts_to_mask(failed, before)
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


def eligibility_all(*callables):
    """Combinator (L5): return a ``sim -> uids`` selecting agents returned by **all** of ``callables``
    (set intersection). E.g. ``eligibility_all(treatment_monitoring_eligibility('first'), dst.matches(RIF=True))``
    switches only agents that are both far enough into first-line and observed RIF-resistant."""
    def _elig(sim):
        if not callables:
            return ss.uids()
        out = ss.uids(callables[0](sim))
        for c in callables[1:]:
            if len(out) == 0:
                break
            out = out & ss.uids(c(sim))
        return out
    _elig.__name__ = 'eligibility_all'
    return _elig


def eligibility_any(*callables):
    """Combinator (L5): return a ``sim -> uids`` selecting agents returned by **any** of ``callables``
    (set union)."""
    def _elig(sim):
        out = ss.uids()
        for c in callables:
            out = out | ss.uids(c(sim))
        return out
    _elig.__name__ = 'eligibility_any'
    return _elig


def will_fail(tx_name):
    """Eligibility factory (L5): on-treatment agents on ``tx_name`` whose pre-rolled course outcome is a
    failure (``pending_surv != 0``). Because the outcome is frozen at initiation, this is an *oracle* —
    it selects agents whose course will fail before it completes — useful for constructing
    failure-contingent regimen-switch scenarios."""
    def _elig(sim):
        tx = sim.interventions.get(tx_name)
        if tx is None:
            return ss.uids()
        tb = get_tb(sim, which=TBResistant)
        on_tx = (tb.state == TBS.TREATMENT).uids
        if len(on_tx) == 0:
            return on_tx
        return on_tx[np.asarray(tx.pending_surv[on_tx]) != 0]
    _elig.__name__ = f'will_fail_{tx_name}'
    return _elig


def treatment_monitoring_eligibility(tx_name, after_steps=4, every_steps=None, require=None):
    """
    Eligibility callable selecting agents on ``tx_name``'s course for at least ``after_steps`` steps.

    Feed as ``eligibility=`` to a monitoring ``DxDelivery`` (to flag still-bacteriologically-positive
    agents by TB state), or — combined with ``supersedes=[tx_name]`` on a second-line ``TxDeliveryR`` —
    to switch regimens mid-course (spec §"Treatment monitoring").

    Args:
        tx_name (str): the ``name`` of the ``TxDeliveryR`` to monitor.
        after_steps (int): minimum sim steps since ``ti_treatment_start`` before eligibility. Default 4.
        every_steps (int/None): if given, re-test every N steps after the first; else a single test at ``after_steps``.
        require (callable): optional additional ``sim -> uids`` AND-ed in (sugar over
            :func:`eligibility_all`), e.g. ``require=dst.matches(RIF=True, exclude_on_treatment=False)``
            to make monitoring contingent on an observed DST profile (L5). Note monitored agents are on
            treatment, so pass ``exclude_on_treatment=False`` to a ``matches`` used here.
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
        if require is not None and len(out):
            out = out & ss.uids(require(sim))
        return out
    _elig.__name__ = f'monitoring_after_{after_steps}_steps'
    return _elig
