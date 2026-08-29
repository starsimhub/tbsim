"""
Multi-strain (drug-resistance) TB natural history.

``TBResistant`` extends the single-strain ``tbsim.TB`` state machine with a
strain-membership overlay: each agent carries a set of strains encoded as a
single integer ``strain_mask`` (bit ``j`` = carries strain ``j``; see
``tbsim.resistance.Strains``). The agent-level ``TB.state`` machine is unchanged
— resistance is an overlay on top of it.

Transmission reuses Starsim's common-random-number force-of-infection engine
(``ss.Infection.infect``): ``step_bookkeeping`` sets state-dependent
``susceptible`` / ``rel_sus`` / ``rel_trans`` so the engine does the transmission
arithmetic, and ``set_prognoses`` picks *which* strain is passed (∝ fitness),
enforces identical-strain blocking, and applies superinfection. ``step_transitions``
overlays the strain-aware natural history: de-novo resistance acquisition (``p_rand``,
per drug, per carried strain), the progression bottleneck (``p_multi``), clear-all-strains
on natural clearance, and the superinfection rate modifiers (``rr_prog_super`` ψ,
``rr_clear_super`` ω).

This reproduces the two-strain reference ODE (``tbsim.compartmental.TwoStrainODE``) in the
``n=1`` case; see ``tests/test_resistance.py``.
"""

import numpy as np
import starsim as ss

from ..tb import TB, TBS, choice2d
from .strains import Strains

__all__ = ['TBResistant']


class TBResistant(TB):
    """
    Strain-aware TB. See module docstring for the architecture.

    Args:
        pars (dict): TB and resistance parameter overrides.
        drugs (list): ordered drug/class names defining the strain space
            (default ``['TX']`` = the two-strain A/B reference).
        rel_fitness (dict): per-drug transmission fitness cost ``r_i`` (default none).

    Resistance parameters (``pars``):
        - ``rr_reinfection_inf`` (σ_L): susceptibility of an ``INFECTION`` agent to a 2nd strain.
          Default ``None`` → ``rr_reinfection_rec`` (the spec's coupling; set to 1.0 for the ODE null).
        - ``rr_reinfection_non`` (σ_N): of a ``NON_INFECTIOUS`` agent. Default ``None`` → ``rr_reinfection_inf``.
        - ``rr_reinfection_asy`` (σ_A): of a mono ``ASYMPTOMATIC`` agent. Default 0.
        - ``rr_reinfection_sym`` (σ_Y): of a mono ``SYMPTOMATIC`` agent. Default 0.
        - ``p_multi``: prob. both strains co-progress at ``→ASYMPTOMATIC`` (1 = no bottleneck). Default 1.
        - ``rr_prog_super`` (ψ): progression-rate multiplier for multi-strain agents. Default 1.
        - ``rr_clear_super`` (ω): natural-clearance multiplier for multi-strain agents. Default 1.
        - ``p_rand``: de-novo resistance ``{drug: prob}`` per not-yet-resistant drug at each
          ``INFECTION→NON_INFECTIOUS`` and ``INFECTION→ASYMPTOMATIC`` progression; each carried
          strain mutates independently. Default none (off).
        - ``prog_resist_mode``: de-novo mechanism, ``'mixed'`` (→ superinfection) or ``'replacement'``. Default ``'mixed'``.
        - ``prog_select``: strain selected when one progresses under the bottleneck, ``'random'`` or ``'fitness'``. Default ``'random'``.
        - ``init_strains``: probability vector over strain ids for seeded infections (default all on id 0, pan-susceptible).
    """

    def __init__(self, pars=None, drugs=None, rel_fitness=None, name=None, label=None, **kwargs):
        # Let TB define its own pars/states/RNGs with defaults first. Default the module name to
        # 'tb' so the standard tbsim interventions (TPT, HSB, Dx) that key on disease 'tb' just work.
        super().__init__(name=name if name is not None else 'tb', label=label)

        self.strains = Strains(drugs if drugs is not None else ['TX'], rel_fitness)

        self.define_pars(
            # σ_L (rr_reinfection_inf) and σ_N (rr_reinfection_non) are defined on base TB, which also
            # resolves their default coupling in init_pre. TBResistant adds only the active-state factors.
            rr_reinfection_asy = 0.0,    # σ_A
            rr_reinfection_sym = 0.0,    # σ_Y
            p_multi            = 1.0,
            rr_prog_super      = 1.0,    # ψ
            rr_clear_super     = 1.0,    # ω
            p_rand             = None,   # de-novo resistance {drug: prob} per not-yet-resistant drug
            prog_resist_mode   = 'mixed',       # 'mixed' | 'replacement'
            prog_select        = 'random',      # 'random' | 'fitness'
            init_strains       = None,          # prob over strain ids for seeds
        )
        self.update_pars(pars, **kwargs)
        # TB validates shape pars in its init, but TBResistant applies pars after super().__init__,
        # so re-run the shared validation here (one rule, one place; see TB._validate_pars). The σ_L/σ_N
        # reinfection default coupling is resolved in TB.init_pre once all pars are final.
        self._validate_pars()

        # Per-agent strain membership (bit j = carries strain j; 0 = uninfected) and the per-strain
        # multiplicity counter (one IntArr per strain id; count>0 ⟺ bit set — see the D-COUNTER note
        # in implementation-decisions.md). Multiplicity >1 arises only from repeated transmission/seeding
        # of an already-carried strain; it feeds only the transmission multinomial and the progression
        # bottleneck, never DST / treatment / acquisition.
        self.strain_counts = [ss.IntArr(f'strain_count_{j}', default=0) for j in range(self.strains.m)]
        # Durable, cross-regimen time of the agent's most recent treatment initiation (nan = never treated),
        # written by every TxDeliveryR at initiation. Powers failure-vs-new-case classification for later
        # DST / second-line routing (spec §"Diagnostics"); see TxDeliveryR.failure_case_eligibility.
        self.define_states(ss.IntArr('strain_mask', default=0),
                           ss.FloatArr('ti_last_treatment', default=np.nan),
                           *self.strain_counts)

        # CRN-safe distributions. choice2d holds per-agent probabilities set each use.
        m = self.strains.m
        self._strain_dist = choice2d(p=np.ones((1, m)) / m)   # which strain is transmitted / seeded
        self._prog_dist   = choice2d(p=np.ones((1, m)) / m)   # which strain progresses under the bottleneck
        self._rng_pmulti  = ss.bernoulli(name='tb_rng_pmulti', p=float(self.pars.p_multi))
        # One independent CRN stream per drug for de-novo acquisition (keeps per-drug draws uncorrelated).
        p_rand = self.pars.p_rand or {}
        self._denovo_rngs = [ss.bernoulli(name=f'tb_denovo_{d}', p=float(p_rand.get(d, 0.0))) for d in self.strains.drugs]

        # Per-step resistance-origin counters (written to results in update_results).
        self._n_identical_superinf = 0
        self._n_denovo = 0
        self._n_trans_resist = 0
        return

    @classmethod
    def agnostic(cls, pars=None, **kwargs):
        """Construct a ``TBResistant`` configured to behave like single-strain ``tbsim.TB`` (L7).

        Sets the "effectively single-strain" defaults — one drug, all seed infections pan-susceptible
        (``init_strains=[1, 0]``) and no de-novo resistance (``p_rand=None``) — so no resistant strain
        ever arises (and the counter never activates). Reinfection is left at base ``tbsim.TB``'s
        defaults (``rr_reinfection_inf``/``_non`` coupling), so a latent/non-infectious agent can be
        re-exposed exactly as in single-strain TB; because the only circulating strain is pan-susceptible,
        such re-exposure just resets the infection clock and never creates resistance. This gives a
        one-liner for strain-aware-vs-agnostic comparison runs without hand-tuning. It does **not** create
        a true ``m=1`` strain space (the bitmask needs ``m = 2**n``); it is the documented convenience
        recipe, not a separate mode.

        Args:
            pars (dict): extra parameter overrides merged over the agnostic defaults.
            **kwargs: forwarded to ``TBResistant`` (e.g. ``name``).

        Example::

            tb = tbsim.TBResistant.agnostic(pars=dict(beta=ss.permonth(0.2), init_prev=ss.bernoulli(0.05)))
        """
        agn = dict(init_strains=[1.0, 0.0], p_rand=None)
        agn.update(pars or {})
        return cls(drugs=['TX'], rel_fitness=None, pars=agn, **kwargs)

    # ------------------------------------------------------------------ per-strain counter helpers
    def _counts(self, uids):
        """Return the ``(len(uids), m)`` per-strain multiplicity matrix for ``uids``."""
        return np.stack([c[uids] for c in self.strain_counts], axis=1)

    def _write_counts(self, uids, mat):
        """Write an ``(len(uids), m)`` multiplicity matrix back to the per-strain count arrays."""
        for j, c in enumerate(self.strain_counts):
            c[uids] = mat[:, j]
        return

    def _reset_counts(self, uids):
        """Zero every per-strain count for ``uids`` (natural clearance / successful cure / death)."""
        if len(uids) == 0:
            return
        for c in self.strain_counts:
            c[uids] = 0
        return

    def _sync_counts_to_mask(self, uids, before_mask):
        """Restore the invariant count>0 ⟺ bit-set after a ``strain_mask`` edit that does *not* transfer
        multiplicity (the progression bottleneck, TPT/latent sterilization).

        A strain whose bit is newly set gets count 1; a strain whose bit was cleared gets count 0;
        unchanged strains keep their count. Resistance *acquisition* (de-novo / treatment / TPT) instead
        goes through :meth:`_apply_acquisition`, which transfers the source strain's count to the target
        (see D-COUNTER)."""
        if len(uids) == 0:
            return
        js = np.arange(self.strains.m)
        after = np.asarray(self.strain_mask[uids])[:, None]
        before = np.asarray(before_mask)[:, None]
        bits_after = ((after >> js) & 1).astype(bool)
        bits_before = ((before >> js) & 1).astype(bool)
        counts = self._counts(uids)
        counts[bits_after & ~bits_before] = 1
        counts[bits_before & ~bits_after] = 0
        self._write_counts(uids, counts)
        return

    def _apply_acquisition(self, uids, counts0, drug_idxs, hit_fn, mixed):
        """Shared per-strain resistance-acquisition operator for de-novo, treatment, and TPT.

        Each carried *source* strain ``j`` independently rolls, for every drug index in ``drug_idxs`` it
        is still susceptible to, whether it acquires that resistance: ``hit_fn(di, cu, rows)`` returns a
        bool array over ``cu`` (the uids carrying ``j``; ``rows`` are their positions in ``uids``). The
        drug bits a source strain acquires this event combine into one resistant *target* ``j | bits``.

        Multiplicity is **transferred** from source to target (spec §2/§3): the target's count is
        incremented by the source strain's *pre-event* count (accumulating across multiple sources and any
        pre-existing target), and — in **replacement** mode (``mixed=False``) — the source's own copies
        are removed. In **mixed** mode (de-novo superinfection) the source keeps its count and the target
        additionally receives a copy of it, so total per-agent load rises when a new lineage emerges.

        Args:
            uids (ss.uids): agents to operate on (rows of ``counts0``).
            counts0 (np.ndarray): ``(len(uids), m)`` per-strain count snapshot to transform.
            drug_idxs (list[int]): strain-profile drug indices eligible for acquisition.
            hit_fn (callable): ``(di, cu, rows) -> bool array`` over ``cu``.
            mixed (bool): keep the source strain (superinfection) vs replace it.

        Returns:
            ``(counts, mask, n_events)`` — the new count matrix, the strain mask rederived from it
            (count>0 ⟺ bit set), and the number of (agent, source-strain) mutation events.
        """
        m = self.strains
        counts = counts0.copy()
        n_events = 0
        for j in range(m.m):
            carrier = counts0[:, j] > 0
            if not carrier.any():
                continue
            rows = np.nonzero(carrier)[0]
            cu = uids[rows]
            acquired = np.zeros(len(cu), dtype=np.int64)  # OR of drug bits acquired for source strain j
            for di in drug_idxs:
                if m.profile[j, di]:
                    continue  # strain j is already resistant to this drug
                hit = hit_fn(di, cu, rows)  # bool array over cu
                acquired[hit] |= (1 << di)
            got = acquired > 0
            if not got.any():
                continue
            g_rows = rows[got]
            targets = (j | acquired[got]).astype(np.int64)  # resistant target strain id per agent
            amt = counts0[g_rows, j]
            np.add.at(counts, (g_rows, targets), amt)  # target gains the source's pre-event count
            if not mixed:
                counts[g_rows, j] -= amt                # replacement removes the source's own copies
            n_events += int(got.sum())
        js = np.arange(m.m)
        mask = ((counts > 0).astype(np.int64) * (1 << js)).sum(axis=1)  # rederive: count>0 ⟺ bit set
        return counts, mask, n_events

    # ------------------------------------------------------------------ helpers
    @property
    def _init_strain_probs(self):
        """Probability vector over strain ids used to assign strains to seed infections."""
        p = self.pars.init_strains
        if p is None:
            p = np.zeros(self.strains.m)
            p[0] = 1.0
        else:
            p = np.array(p, dtype=float)
        return p / p.sum()

    # ------------------------------------------------------------------ step
    def step(self):
        """Reset per-step counters, then run the TB step (transmission → transitions → bookkeeping)."""
        self._n_identical_superinf = 0
        self._n_denovo = 0
        self._n_trans_resist = 0
        super().step()
        return

    def set_prognoses(self, uids, sources=None):
        """
        Assign strains to newly infected / superinfected agents.

        Seeds (``sources`` is ``None`` or a scalar) draw a single strain from ``init_strains`` and
        start it at count 1. Transmission events (``sources`` is a UID array, one per target) draw the
        transmitted strain from the source's carried strains ∝ ``count × fitness``. A target already
        carrying the drawn strain is **superinfected with an identical strain** — its count for that
        strain is incremented (spec §1; previously this was blocked). Otherwise the strain is added
        (entering ``INFECTION`` from a susceptible state, or keeping the current state for a
        superinfection) at count 1, regardless of how many copies the source carried.
        """
        super().set_prognoses(uids, sources)  # Record the new infections; also clears `susceptible`
        if len(uids) == 0:
            return
        ti = self.ti
        m = self.strains

        # --- Seeding: single strain per seed, no source ---
        if sources is None or np.isscalar(sources):
            probs = np.tile(self._init_strain_probs, (len(uids), 1))
            self._strain_dist.set(a=np.arange(m.m), p=probs)
            ids = self._strain_dist.rvs(uids).astype(int)
            self.strain_mask[uids] = (1 << ids)
            onehot = np.zeros((len(uids), m.m), dtype=int)
            onehot[np.arange(len(uids)), ids] = 1  # founding strain starts at count 1, all others 0
            self._write_counts(uids, onehot)
            self.state[uids] = TBS.INFECTION
            self.infected[uids] = True
            self.ever_infected[uids] = True
            self.ti_infected[uids] = ti
            self.susceptible[uids] = False
            return

        # --- Transmission: pick which strain each source passes (∝ count × fitness) ---
        sources = ss.uids(sources)
        probs = m.transmit_probs(self.strain_mask[sources], counts=self._counts(sources))
        self._strain_dist.set(a=np.arange(m.m), p=probs)
        drawn = self._strain_dist.rvs(uids).astype(int)

        tgt_masks = self.strain_mask[uids]
        already = ((tgt_masks >> drawn) & 1).astype(bool)  # target already carries the drawn strain

        # Identical-strain superinfection: mask unchanged, increment that strain's count, and reset the
        # infection clock (spec: every successful exposure resets time-since-infection).
        ident = uids[already]
        self._n_identical_superinf += len(ident)
        self.ti_infected[ident] = ti
        if len(ident):
            counts = self._counts(ident)
            counts[np.arange(len(ident)), drawn[already]] += 1
            self._write_counts(ident, counts)

        # New strain acquired (a founding infection, or superinfection with a not-yet-carried strain).
        acq = ~already
        acq_uids = uids[acq]
        if len(acq_uids):
            new_ids = drawn[acq]
            was_uninfected = tgt_masks[acq] == 0
            self.strain_mask[acq_uids] = tgt_masks[acq] | (1 << new_ids)
            self.infected[acq_uids] = True
            self.ever_infected[acq_uids] = True
            self.ti_infected[acq_uids] = ti
            self.susceptible[acq_uids] = False
            # Primary infection (from a susceptible/cleared state) enters latent;
            # superinfection of an already-infected agent keeps the current state.
            self.state[acq_uids[was_uninfected]] = TBS.INFECTION
            # Founding strain starts at count 1 regardless of the source's multiplicity (spec §1).
            counts = self._counts(acq_uids)
            counts[np.arange(len(acq_uids)), new_ids] = 1
            self._write_counts(acq_uids, counts)
            # Track transmitted resistance (a resistant strain, id != 0, was acquired).
            self._n_trans_resist += int(np.count_nonzero(new_ids != 0))
        return

    # ------------------------------------------------------------------ natural history
    def step_transitions(self):
        """Strain-aware natural-history transitions (mirrors the reference ODE)."""
        ti = self.ti
        m = self.strains
        p = self.pars

        # --- INFECTION (latent) ---
        u = self.latent.uids
        if len(u):
            multi = m.carried(self.strain_mask[u]).sum(1) >= 2
            psi = np.where(multi, p.rr_prog_super, 1.0)
            omega = np.where(multi, p.rr_clear_super, 1.0)
            inf_non, inf_asy = self.progression_rates(u)
            self.transition(u, to={
                TBS.CLEARED:        p.inf_cle * omega,
                TBS.NON_INFECTIOUS: inf_non,
                TBS.ASYMPTOMATIC:   inf_asy * psi,
            }, rng=self._rng_inf)
            dest = self.state[u]
            cleared = u[dest == TBS.CLEARED]
            self._enter_cleared(cleared, p.rr_reinfection_cleared)  # natural clearance drops all strains + counts

            # De-novo acquisition (at →NON_INFECTIOUS and →ASYMPTOMATIC) and the progression
            # bottleneck (only at →ASYMPTOMATIC). Subsets are computed from the pre-edit mask so
            # the two operators don't interfere.
            self._progress(u[dest == TBS.ASYMPTOMATIC], bottleneck=True, denovo=True)
            self._progress(u[dest == TBS.NON_INFECTIOUS], bottleneck=False, denovo=True)

        # --- NON_INFECTIOUS ---
        u = self.non_infectious.uids
        if len(u):
            multi = m.carried(self.strain_mask[u]).sum(1) >= 2
            psi = np.where(multi, p.rr_prog_super, 1.0)
            omega = np.where(multi, p.rr_clear_super, 1.0)
            self.transition(u, to={
                TBS.CLEARED:      p.non_rec * self.rr_clearance[u] * omega,
                TBS.ASYMPTOMATIC: p.non_asy * psi,
            }, rng=self._rng_non)
            dest = self.state[u]
            cleared = u[dest == TBS.CLEARED]
            self._enter_cleared(cleared, p.rr_reinfection_rec)  # drops all strains + counts (spec §3)
            self._progress(u[dest == TBS.ASYMPTOMATIC], bottleneck=True, denovo=False)  # no de-novo out of NON_INFECTIOUS

        # --- ASYMPTOMATIC (A→Y keeps both strains; ψ scales the A→Y rate for multi) ---
        u = self.asymptomatic.uids
        if len(u):
            psi = np.where(m.carried(self.strain_mask[u]).sum(1) >= 2, p.rr_prog_super, 1.0)
            self.transition(u, to={
                TBS.NON_INFECTIOUS: p.asy_non,
                TBS.SYMPTOMATIC:    p.asy_sym * psi,
            }, rng=self._rng_asy)

        # --- SYMPTOMATIC ---
        u = self.symptomatic.uids
        if len(u):
            self.transition(u, to={
                TBS.ASYMPTOMATIC: p.sym_asy,
                TBS.DEAD:         p.sym_dead * self.rr_death[u],
            }, rng=self._rng_sym)
        return

    def _progress(self, prog_uids, bottleneck, denovo):
        """Apply de-novo acquisition and/or the progression bottleneck to progressing agents."""
        if len(prog_uids) == 0:
            return
        if denovo:
            self._denovo(prog_uids)
        if bottleneck:
            # Recompute counts after de-novo (mixed de-novo may have created a second strain).
            nstr = self.strains.carried(self.strain_mask[prog_uids]).sum(1)
            self._bottleneck(prog_uids[nstr >= 2])
        return

    def _bottleneck(self, uids):
        """Multi-strain agents entering ASYMPTOMATIC keep all strains w.p. ``p_multi``, else one strain progresses."""
        if len(uids) == 0 or self.pars.p_multi >= 1:
            return
        _, reduce = self._rng_pmulti.filter(uids, both=True)
        if len(reduce) == 0:
            return
        before = np.asarray(self.strain_mask[reduce]).copy()
        probs = self._select_probs(self.strain_mask[reduce], counts=self._counts(reduce))
        self._prog_dist.set(a=np.arange(self.strains.m), p=probs)
        chosen = self._prog_dist.rvs(reduce).astype(int)
        self.strain_mask[reduce] = (1 << chosen)
        self._sync_counts_to_mask(reduce, before)  # dropped strains → count 0; survivor keeps its count
        return

    def _select_probs(self, masks, counts=None):
        """Per-agent probability of each carried strain being the one that progresses.

        Weighted by ``count`` (a multi-copy strain is proportionally more likely to be the survivor;
        spec §2) and, if ``prog_select='fitness'``, additionally by fitness. ``counts=None`` falls back
        to one-copy-per-carried-strain weighting."""
        w = self.strains.carried(masks).astype(float)
        if counts is not None:
            w = w * np.asarray(counts)
        if self.pars.prog_select == 'fitness':
            w = w * self.strains.fitness
        return w / w.sum(1, keepdims=True)

    def _denovo(self, uids):
        """
        De-novo resistance at progression out of INFECTION: each carried strain independently
        acquires resistance to each not-yet-resistant drug with probability ``p_rand[drug]``. The
        drug bits a source strain acquires this step combine into one resistant target strain,
        added (``prog_resist_mode='mixed'`` → superinfection) or swapped in (``'replacement'``).
        Multiplicity transfers from the source strain to the resistant target (see
        :meth:`_apply_acquisition`); the bitmask guarantees the resistant target always exists.
        """
        p_rand = self.pars.p_rand or {}
        if len(uids) == 0 or not p_rand:
            return
        m = self.strains
        drug_idxs = [di for di, drug in enumerate(m.drugs) if p_rand.get(drug, 0.0) > 0]
        if not drug_idxs:
            return
        counts0 = self._counts(uids)  # snapshot: carriers/source amounts read from this
        def hit_fn(di, cu, rows):
            return self._denovo_rngs[di].rvs(cu)  # bernoulli → bool array
        mixed = self.pars.prog_resist_mode == 'mixed'
        counts, mask, n_events = self._apply_acquisition(uids, counts0, drug_idxs, hit_fn, mixed)
        self._write_counts(uids, counts)
        self.strain_mask[uids] = mask
        self._n_denovo += n_events
        return

    def _enter_cleared(self, uids, rr):
        """Extend base clearance bookkeeping: also drop all carried strains and their counts (spec §3)."""
        if len(uids) == 0:
            return
        super()._enter_cleared(uids, rr)
        self.strain_mask[uids] = 0
        self._reset_counts(uids)
        return

    def sterilize_covered(self, uids, regimen_drugs):
        """Clear (with certainty) every carried strain susceptible to *all* of ``regimen_drugs``; any
        regimen-resistant strain persists. An agent left carrying no strain moves to ``CLEARED`` (with
        post-clearance reinfection protection). Shared by strain-aware TPT sterilization and latent
        treatment (TR-5). Returns the uids that were fully cleared.
        """
        uids = ss.uids(uids)
        if len(uids) == 0:
            return ss.uids()
        covered_mask = self.strains.covered_mask(regimen_drugs)
        before = self.strain_mask[uids].copy()
        self.strain_mask[uids] &= ~covered_mask
        self._sync_counts_to_mask(uids, before)  # sterilized strains → count 0; survivors keep their count
        cleared = uids[self.strain_mask[uids] == 0]
        if len(cleared):
            self.state[cleared] = TBS.CLEARED
            self._enter_cleared(cleared, self.pars.rr_reinfection_cleared)
            self.infected[cleared] = False
            self.susceptible[cleared] = True
        return cleared

    # ------------------------------------------------------------------ transmission set-up
    def step_bookkeeping(self):
        """Set flags, request TB deaths, reset modifiers, and set strain-aware ``susceptible`` / ``rel_sus`` / ``rel_trans``."""
        st = self.state
        p = self.pars

        # infected ⇔ carries at least one strain
        self.infected[:] = self.strain_mask != 0
        self.on_treatment[:] = (st == TBS.TREATMENT)

        # Susceptible = primary-infection- or superinfection-eligible states. Whether a *new*
        # strain is actually acquired (vs identical-strain blocked) is resolved in set_prognoses.
        sus = st.isin((TBS.SUSCEPTIBLE, TBS.CLEARED, TBS.INFECTION, TBS.NON_INFECTIOUS))
        if p.rr_reinfection_asy > 0:
            sus = sus | (st == TBS.ASYMPTOMATIC)
        if p.rr_reinfection_sym > 0:
            sus = sus | (st == TBS.SYMPTOMATIC)
        self.susceptible[:] = sus

        # TB deaths
        dead = ss.uids((st == TBS.DEAD) & self.sim.people.alive)
        self.sim.people.request_death(dead)
        self.results['new_deaths'][self.ti] = len(dead)
        self.results['new_deaths_15+'][self.ti] = np.count_nonzero(self.sim.people.age[dead] >= 15)

        # Reset per-agent risk modifiers (interventions set fresh values next step)
        self.rr_activation[:] = 1
        self.rr_clearance[:] = 1
        self.rr_death[:] = 1

        # Relative susceptibility by state (σ superinfection factors; ρ reinfection for CLEARED)
        self.rel_sus[:] = 0.0
        self.rel_sus[st == TBS.SUSCEPTIBLE] = 1.0
        cleared = ss.uids(st == TBS.CLEARED)
        if p.dur_reinfection_protection is not None and len(cleared):
            waned = cleared[self.ti >= self.ti_rr_reinfection_wane[cleared]]
            self.rr_reinfection[waned] = 1.0
            self.ti_rr_reinfection_wane[waned] = np.inf
        self.rel_sus[cleared] = self.rr_reinfection[cleared]
        self.rel_sus[st == TBS.INFECTION] = p.rr_reinfection_inf
        self.rel_sus[st == TBS.NON_INFECTIOUS] = p.rr_reinfection_non
        self.rel_sus[st == TBS.ASYMPTOMATIC] = p.rr_reinfection_asy
        self.rel_sus[st == TBS.SYMPTOMATIC] = p.rr_reinfection_sym

        # Relative transmissibility = max fitness over carried strains (× κ for asymptomatic).
        # Non-infectious agents are zeroed by the `infectious` mask in Infection.infect().
        self.rel_trans[:] = self.strains.max_fitness(self.strain_mask.values)
        self.rel_trans[st == TBS.ASYMPTOMATIC] *= p.trans_asymp
        return

    def step_die(self, uids):
        """Clear strains (and their counts) on death, then apply the base TB death handling."""
        if len(uids):
            self.strain_mask[uids] = 0
            self._reset_counts(uids)
        super().step_die(uids)
        return

    # ------------------------------------------------------------------ results
    def init_results(self):
        super().init_results()
        results = [
            ss.Result('new_identical_superinf', dtype=int, label='Identical-strain superinfections (count increments)'),
            ss.Result('new_denovo_resistance', dtype=int, label='De-novo resistance acquisitions'),
            ss.Result('new_transmitted_resistance', dtype=int, label='Transmitted resistant infections'),
            ss.Result('frac_resist', dtype=float, scale=False, label='Resistant fraction of active TB'),
            ss.Result('frac_super', dtype=float, scale=False, label='Superinfected fraction of active TB'),
        ]
        for drug in self.strains.drugs:
            results.append(ss.Result(f'frac_resist_{drug}', dtype=float, scale=False, label=f'Active TB resistant to {drug}'))
        self.define_results(*results)
        return

    def update_results(self):
        super().update_results()
        res = self.results
        ti = self.ti
        res['new_identical_superinf'][ti] = self._n_identical_superinf
        res['new_denovo_resistance'][ti] = self._n_denovo
        res['new_transmitted_resistance'][ti] = self._n_trans_resist

        active = self.active_tb.uids
        if len(active):
            masks = self.strain_mask[active]
            pheno = self.strains.phenotype_any(masks)  # (n_active, n_drug)
            res['frac_resist'][ti] = float(pheno.any(1).mean())
            res['frac_super'][ti] = float((self.strains.carried(masks).sum(1) >= 2).mean())
            for i, drug in enumerate(self.strains.drugs):
                res[f'frac_resist_{drug}'][ti] = float(pheno[:, i].mean())
        return
