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
overlays the strain-aware natural history: the progression bottleneck (``p_multi``),
de-novo resistance acquisition (``q_prog``), clear-all-strains on natural clearance,
and the superinfection rate modifiers (``rr_prog_super`` ψ, ``rr_clear_super`` ω).

This reproduces the two-strain reference ODE (``ode.r`` / ``model-tests.md``) in the
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
        - ``rr_reinfection_inf`` (σ_L): susceptibility of a mono ``INFECTION`` agent to a 2nd strain. Default 1.
        - ``rr_reinfection_non`` (σ_N): of a mono ``NON_INFECTIOUS`` agent. Default 1.
        - ``rr_reinfection_asy`` (σ_A): of a mono ``ASYMPTOMATIC`` agent. Default 0.
        - ``rr_reinfection_sym`` (σ_Y): of a mono ``SYMPTOMATIC`` agent. Default 0.
        - ``p_multi``: prob. both strains co-progress at ``→ASYMPTOMATIC`` (1 = no bottleneck). Default 1.
        - ``rr_prog_super`` (ψ): progression-rate multiplier for multi-strain agents. Default 1.
        - ``rr_clear_super`` (ω): natural-clearance multiplier for multi-strain agents. Default 1.
        - ``q_prog``: de-novo resistance prob. per not-yet-resistant drug at each ``INFECTION→NON_INFECTIOUS``
          and ``INFECTION→ASYMPTOMATIC`` progression (mono-strain agents only). Default 0.
        - ``prog_resist_mode``: de-novo mechanism, ``'mixed'`` (→ superinfection) or ``'replacement'``. Default ``'mixed'``.
        - ``prog_select``: strain selected when one progresses under the bottleneck, ``'random'`` or ``'fitness'``. Default ``'random'``.
        - ``init_strains``: probability vector over strain ids for seeded infections (default all on id 0, pan-susceptible).
    """

    def __init__(self, pars=None, drugs=None, rel_fitness=None, name=None, label=None, **kwargs):
        # Let TB define its own pars/states/RNGs with defaults first.
        super().__init__(name=name, label=label)

        self.strains = Strains(drugs if drugs is not None else ['TX'], rel_fitness)

        self.define_pars(
            rr_reinfection_inf = 1.0,   # σ_L
            rr_reinfection_non = 1.0,   # σ_N
            rr_reinfection_asy = 0.0,   # σ_A
            rr_reinfection_sym = 0.0,   # σ_Y
            p_multi            = 1.0,
            rr_prog_super      = 1.0,   # ψ
            rr_clear_super     = 1.0,   # ω
            q_prog             = 0.0,   # de-novo acquisition prob per drug
            prog_resist_mode   = 'mixed',       # 'mixed' | 'replacement'
            prog_select        = 'random',      # 'random' | 'fitness'
            init_strains       = None,          # prob over strain ids for seeds
        )
        self.update_pars(pars, **kwargs)

        # Per-agent strain membership (bit j = carries strain j; 0 = uninfected).
        self.define_states(ss.IntArr('strain_mask', default=0))

        # CRN-safe distributions. choice2d holds per-agent probabilities set each use.
        m = self.strains.m
        self._strain_dist = choice2d(p=np.ones((1, m)) / m)   # which strain is transmitted / seeded
        self._prog_dist   = choice2d(p=np.ones((1, m)) / m)   # which strain progresses under the bottleneck
        self._rng_pmulti  = ss.bernoulli(name='tb_rng_pmulti', p=float(self.pars.p_multi))
        self._rng_denovo  = ss.bernoulli(name='tb_rng_denovo', p=float(self.pars.q_prog))

        # Per-step resistance-origin counters (written to results in update_results).
        self._n_blocked = 0
        self._n_denovo = 0
        self._n_trans_resist = 0
        return

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
        self._n_blocked = 0
        self._n_denovo = 0
        self._n_trans_resist = 0
        super().step()
        return

    def set_prognoses(self, uids, sources=None):
        """
        Assign strains to newly infected / superinfected agents.

        Seeds (``sources`` is ``None`` or a scalar) draw a single strain from
        ``init_strains``. Transmission events (``sources`` is a UID array, one per
        target) draw the transmitted strain from the source's carried strains ∝
        fitness, block identical-strain re-exposure, and otherwise add the strain
        (entering ``INFECTION`` from a susceptible state, or keeping the current
        state for a superinfection).
        """
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
            self.state[uids] = TBS.INFECTION
            self.infected[uids] = True
            self.ever_infected[uids] = True
            self.ti_infected[uids] = ti
            self.susceptible[uids] = False
            return

        # --- Transmission: pick which strain each source passes (∝ fitness) ---
        sources = ss.uids(sources)
        probs = m.transmit_probs(self.strain_mask[sources])
        self._strain_dist.set(a=np.arange(m.m), p=probs)
        drawn = self._strain_dist.rvs(uids).astype(int)

        tgt_masks = self.strain_mask[uids]
        already = ((tgt_masks >> drawn) & 1).astype(bool)  # target already carries drawn strain

        # Blocked identical-strain exposures: no change, but reset the infection clock
        # (spec: every successful exposure resets time-since-infection; a no-op until
        # time-varying progression exists).
        blocked = uids[already]
        self._n_blocked += len(blocked)
        self.ti_infected[blocked] = ti

        # Acquired strains
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
            self.transition(u, to={
                TBS.CLEARED:        p.inf_cle * omega,
                TBS.NON_INFECTIOUS: p.inf_non * self.rr_activation[u],
                TBS.ASYMPTOMATIC:   p.inf_asy * self.rr_activation[u] * psi,
            }, rng=self._rng_inf)
            dest = self.state[u]
            cleared = u[dest == TBS.CLEARED]
            self.rr_reinfection[cleared] = p.rr_reinfection_cleared
            self._set_reinfection_wane(cleared)
            self.strain_mask[cleared] = 0  # natural clearance removes all strains

            # Progression bottleneck (only at →ASYMPTOMATIC) and de-novo acquisition
            # (at →NON_INFECTIOUS and →ASYMPTOMATIC, mono-strain agents only). Subsets
            # are computed from the pre-edit mask so the two operators don't interfere.
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
            self.rr_reinfection[cleared] = p.rr_reinfection_rec
            self._set_reinfection_wane(cleared)
            self.strain_mask[cleared] = 0
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
        """Apply the progression bottleneck and/or de-novo acquisition to progressing agents."""
        if len(prog_uids) == 0:
            return
        nstr = self.strains.carried(self.strain_mask[prog_uids]).sum(1)
        if bottleneck:
            self._bottleneck(prog_uids[nstr >= 2])
        if denovo:
            self._denovo(prog_uids[nstr == 1])
        return

    def _bottleneck(self, uids):
        """Multi-strain agents entering ASYMPTOMATIC keep all strains w.p. ``p_multi``, else one strain progresses."""
        if len(uids) == 0 or self.pars.p_multi >= 1:
            return
        _, reduce = self._rng_pmulti.filter(uids, both=True)
        if len(reduce) == 0:
            return
        probs = self._select_probs(self.strain_mask[reduce])
        self._prog_dist.set(a=np.arange(self.strains.m), p=probs)
        chosen = self._prog_dist.rvs(reduce).astype(int)
        self.strain_mask[reduce] = (1 << chosen)
        return

    def _select_probs(self, masks):
        """Per-agent probability of each carried strain being the one that progresses (random or fitness-weighted)."""
        w = self.strains.carried(masks).astype(float)
        if self.pars.prog_select == 'fitness':
            w = w * self.strains.fitness
        return w / w.sum(1, keepdims=True)

    def _denovo(self, uids):
        """
        Mono-strain agents progressing out of INFECTION acquire de-novo resistance to each
        not-yet-resistant drug with probability ``q_prog`` (mixed → superinfection, replacement → switch).
        """
        if len(uids) == 0 or self.pars.q_prog <= 0:
            return
        m = self.strains
        strain_id = m.carried(self.strain_mask[uids]).argmax(1)  # exactly one carried strain
        mixed = self.pars.prog_resist_mode == 'mixed'
        for di, drug in enumerate(m.drugs):
            # NB: per-drug draws reuse one CRN stream, so for n>1 the drug acquisitions are
            # correlated within an agent; negligible since q_prog is tiny and n=1 is the reference.
            elig = uids[~m.profile[strain_id, di]]
            if len(elig) == 0:
                continue
            acq = self._rng_denovo.filter(elig)
            if len(acq) == 0:
                continue
            cur = self.strain_mask[acq]
            new_strain = m.carried(cur).argmax(1) | m.drug_bit(drug)
            self.strain_mask[acq] = (cur | (1 << new_strain)) if mixed else (1 << new_strain)
            self._n_denovo += len(acq)
        return

    def _set_reinfection_wane(self, uids):
        """Schedule reinfection-protection waning for newly cleared agents, if enabled."""
        if self.pars.dur_reinfection_protection is not None and len(uids):
            self.ti_rr_reinfection_wane[uids] = self.ti + self.pars.dur_reinfection_protection.rvs(uids)
        return

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
        """Clear strains on death, then apply the base TB death handling."""
        if len(uids):
            self.strain_mask[uids] = 0
        super().step_die(uids)
        return

    # ------------------------------------------------------------------ results
    def init_results(self):
        super().init_results()
        results = [
            ss.Result('new_blocked_superinf', dtype=int, label='Blocked identical-strain superinfections'),
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
        res['new_blocked_superinf'][ti] = self._n_blocked
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
