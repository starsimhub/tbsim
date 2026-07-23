"""TB natural history model. State definitions and transition diagram are in the API docs (tbsim.tb)."""

from enum import IntEnum

import numpy as np
import starsim as ss
from .plots import plot as _tbsim_plot


__all__ = ['TB', 'TBS', 'get_tb', 'choice2d']


class TBS(IntEnum):
    """
    TB state labels.

    - Each agent is in exactly one of these states.
    - Transitions are driven by exponential rates in `TB`.
    """
    SUSCEPTIBLE     = 0     # Never infected (agents who clear/recover/treat remain in their last state, not here)
    INFECTION       = 1     # Latent infection (not yet active TB)
    CLEARED         = 2     # Post-infection: cleared latent, recovered from non-infectious, or completed treatment
    NON_INFECTIOUS  = 3     # Non-infectious TB (early/smear-negative)
    ASYMPTOMATIC    = 4     # Active TB, asymptomatic (infectious)
    SYMPTOMATIC     = 5     # Active TB, symptomatic (infectious)
    TREATMENT       = 6     # On TB treatment
    DEAD            = 7     # Dead (TB-caused via sym_dead; general mortality via step_die also sets this)
    REMOVED         = 8     # Removed from the active population (e.g. emigration)


# State groups for fast membership tests, e.g. ``tb.state.isin(TBS.ACTIVE)``.
TBS.ACTIVE       = (TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC)  # active TB disease
TBS.TERMINAL     = (TBS.DEAD, TBS.REMOVED)                                  # no longer in the active population
TBS.CARE_SEEKING = (TBS.SYMPTOMATIC,)                                       # eligible to seek care (clinical symptoms only)

# Number of bins for one-pass state counting via np.bincount (TBS codes are contiguous from 0).
_TBS_NBINS = int(max(TBS)) + 1


class BaseTB(ss.Infection):
    """Base class for TB natural history models."""
    pass


class TB(BaseTB):
    """
    Agent-based TB natural history adapting the LSHTM compartmental structure [1] (Schwalb et al. 2025).
    States in `TBS` span the spectrum from susceptibility to active disease and treatment.
    Infectious states are `TBS.ASYMPTOMATIC` and `TBS.SYMPTOMATIC`; the force
    of infection depends on `pars.beta` and the prevalence of those states, with
    `pars.trans_asymp` (kappa) giving the relative infectiousness of asymptomatic vs symptomatic TB.
    Progression out of latent `INFECTION` uses constant per-year hazards by default, but can optionally
    be *front-loaded* via `k_asy` (and `k_non`): the `INFECTION -> ASYMPTOMATIC` (and
    `INFECTION -> NON_INFECTIOUS`) hazard then declines exponentially with time since infection
    (see `progression_rates`).
    Reinfectable state (`TBS.CLEARED`) uses per-agent `rr_reinfection`, set on entry from each
    pathway (`rr_reinfection_cleared`, `rr_reinfection_rec`, `rr_reinfection_treat`). Per-agent modifiers
    ``rr_activation``, ``rr_clearance``, ``rr_death`` scale selected rates.

    Args (pars):
        *Transmission and reinfection*

        - ``init_prev``:   Initial seed infections into latent ``INFECTION`` (prevalence).
        - ``init_prev_active``: Initial seed infections into active ``SYMPTOMATIC`` TB (prevalence). Default 0. Mirrors the ODE model's symptomatic seeding, so that both models can start from identical active-TB prevalence.
        - ``beta``:        Transmission rate per year.
        - ``trans_asymp``: Relative transmissibility, asymptomatic vs symptomatic. (kappa)
        - ``rr_reinfection_rec``:     Relative risk of reinfection after recovering from NON_INFECTIOUS. (pi)
        - ``rr_reinfection_treat``:   Relative risk of reinfection after completing treatment. (rho)
        - ``rr_reinfection_cleared``: Relative risk of reinfection after clearing latent infection. Default: 1.0
        - ``dur_reinfection_protection``: Distribution of duration of reinfection protection. If None, never wanes.

        *From INFECTION (latent)*

        - ``inf_cle``:     Infection -> Cleared (no active TB).
        - ``inf_non``:     Infection -> Non-infectious TB.
        - ``inf_asy``:     Infection -> Asymptomatic TB.
        - ``k_asy``:       Optional exponential decline shape parameter of ``inf_asy`` with time since infection tau: ``inf_asy(tau) = inf_asy * exp(-k_asy * tau)``. Default 0 (constant hazard). Set > 0 to front-load progression to active TB (the recommended one-parameter form).
        - ``k_non``:       Optional exponential decline shape parameter of ``inf_non`` with time since infection, analogous to ``k_asy``. Default 0 (constant hazard).

        *From NON_INFECTIOUS*

        - ``non_rec``:    Non-infectious -> Recovered.
        - ``non_asy``:    Non-infectious -> Asymptomatic.

        *From ASYMPTOMATIC*

        - ``asy_non``:    Asymptomatic -> Non-infectious.
        - ``asy_sym``:    Asymptomatic -> Symptomatic.

        *From SYMPTOMATIC*

        - ``sym_asy``:     Symptomatic -> Asymptomatic.
        - ``sym_dead``:    Symptomatic -> Dead (TB-specific mortality, mu_TB).

        *Background (general mortality is handled by* ``ss.Deaths`` *demographics, not this module)*

        - ``cxr_asymp_sens``:   CXR sensitivity for screening asymptomatic (0-1).

    Attributes:
        *Infection flags*

        - ``susceptible``    (BoolState, default=True):   Whether the agent is susceptible to TB.
        - ``infected``       (BoolState, default=False):  Whether the agent is currently infected.
        - ``ever_infected``  (BoolState, default=False):  Whether the agent has ever been infected.
        - ``on_treatment``   (BoolState, default=False):  Whether the agent is on TB treatment.

        *TB state machine*

        - ``state``          (FloatArr, default=TBS.SUSCEPTIBLE):  Current TB state (`TBS` value).
        - ``ti_infected``    (FloatArr, default=-inf):              Time of infection (never infected = -inf).

        *Transmission modifiers*

        - ``rel_sus``        (FloatArr, default=1.0):  Relative susceptibility to TB.
        - ``rel_trans``      (FloatArr, default=1.0):  Relative transmissibility of TB.

        *Per-agent risk modifiers*

        - ``rr_activation``         (FloatArr, default=1.0):   Multiplier on INFECTION -> NON_INFECTIOUS / ASYMPTOMATIC.
        - ``rr_clearance``          (FloatArr, default=1.0):   Multiplier on NON_INFECTIOUS -> CLEARED.
        - ``rr_death``              (FloatArr, default=1.0):   Multiplier on SYMPTOMATIC -> DEAD.
        - ``rr_reinfection``        (FloatArr, default=1.0):   Per-agent relative reinfection risk; set on entry to CLEARED.
        - ``ti_rr_reinfection_wane`` (FloatArr, default=np.inf): Sim time at which ``rr_reinfection`` resets to 1.0.

    Example:
        ::

            import starsim as ss
            import tbsim

            sim = ss.Sim(diseases=tbsim.TB(), pars=dict(start='2000', stop='2020'))
            sim.run()
            sim.plot()

    References:
        [1] Schwalb et al. (2025) Potential impact, costs, and benefits of population-wide
        screening interventions for tuberculosis in Viet Nam. PLOS Glob Public Health.
        https://doi.org/10.1371/journal.pgph.0005050

    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with default natural history parameters; override via ``pars``."""
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        # --- Transmission and reinfection ---
        self.define_pars(
            init_prev=ss.bernoulli(0.05),       # Initial seed infections into latent INFECTION (prevalence)
            init_prev_active=ss.bernoulli(0.0), # Initial seed infections into active SYMPTOMATIC TB (prevalence); mirrors the ODE's symptomatic seeding
            beta=ss.permonth(0.2),              # Transmission rate per month
            trans_asymp=0.82,                   # κ kappa: rel. transmissibility asymptomatic vs symptomatic
            rr_reinfection_rec=0.21,            # π pi: RR reinfection after NON_INFECTIOUS → CLEARED
            rr_reinfection_treat=3.15,          # ρ rho: RR reinfection after TREATMENT → CLEARED
            rr_reinfection_cleared=1.0,         # RR reinfection after INFECTION → CLEARED (latent cleared); also applies to agents cleared via TPT sterilization
            rr_reinfection_inf=None,            # σ_L: relative susceptibility of a latent INFECTION agent to reinfection (clock reset). None → rr_reinfection_rec
            rr_reinfection_non=None,            # σ_N: relative susceptibility of a NON_INFECTIOUS agent to reinfection (clock reset). None → rr_reinfection_inf
            dur_reinfection_protection=None,    # Distribution of protection duration; None = never wanes
            # --- From INFECTION (latent) ---
            inf_cle=ss.peryear(1.90),            # Clear infection (no active TB)
            inf_non=ss.peryear(0.16),            # Progress to non-infectious TB
            inf_asy=ss.peryear(0.06),            # Progress to asymptomatic active TB
            # Optional front-loading of progression: the INFECTION-exit hazards may decline with
            # time since infection tau (years), rate(tau) = rate * exp(-k * tau). Both default to 0
            # (constant hazard = unchanged behaviour); set k_asy > 0 for the recommended 1-parameter
            # front-loaded INFECTION -> ASYMPTOMATIC (see progression_rates).
            k_asy=0.0,                           # Exponential decay shape parameter for inf_asy; 0 = constant
            k_non=0.0,                           # Exponential decay shape parameter for inf_non; 0 = constant
            # --- From NON_INFECTIOUS ---
            non_rec=ss.peryear(0.18),            # Non-infectious → CLEARED
            non_asy=ss.peryear(0.25),            # Progress to asymptomatic
            # --- From ASYMPTOMATIC ---
            asy_non=ss.peryear(1.66),            # Revert to non-infectious
            asy_sym=ss.peryear(0.88),            # Progress to symptomatic
            # --- From SYMPTOMATIC ---
            sym_asy=ss.peryear(0.54),            # Regress to asymptomatic (still active TB; does not enter CLEARED)
            sym_dead=ss.peryear(0.34),           # μ_TB: symptomatic → dead (TB mortality)
            # --- Background ---
            cxr_asymp_sens=1.0,                 # CXR sensitivity for screening asymptomatic (0–1)
        )
        self.update_pars(pars, **kwargs)
        self._validate_pars()

        # CRN-safe RNG distributions for per-step transition draws (one per source state)
        self._rng_inf = ss.random(name='tb_rng_inf')   # INFECTION exits
        self._rng_non = ss.random(name='tb_rng_non')   # NON_INFECTIOUS exits
        self._rng_asy = ss.random(name='tb_rng_asy')   # ASYMPTOMATIC exits
        self._rng_sym = ss.random(name='tb_rng_sym')   # SYMPTOMATIC exits

        # Per-agent state: redefine base Infection states and add TB-specific ones
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('infected'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_trans', default=1.0),
            ss.FloatArr('ti_infected', default=-np.inf),
            ss.FloatArr('state', default=TBS.SUSCEPTIBLE),
            ss.FloatArr('ti_asymp', default=np.nan),                # Time of last entry to ASYMPTOMATIC (for new_active tracking)
            ss.BoolState('on_treatment', default=False),
            ss.BoolState('ever_infected', default=False),
            # Risk modifiers
            ss.FloatArr('rr_activation', default=1.0),
            ss.FloatArr('rr_clearance', default=1.0),
            ss.FloatArr('rr_death', default=1.0),
            # Reinfection protection
            ss.FloatArr('rr_reinfection', default=1.0),
            ss.FloatArr('ti_rr_reinfection_wane', default=np.inf),
            reset=True,
        )

        return

    def _validate_pars(self):
        """Validate parameter values that share the same rule across ``TB`` and ``TBResistant``.

        Called after ``update_pars`` in each subclass' ``__init__`` so the checks live in one place
        (``TBResistant`` applies its own pars after ``super().__init__``, hence the shared re-use).
        """
        # Decline parameters are shape terms for exp(-k*tau); require k >= 0.
        if self.pars.k_asy < 0:
            raise ValueError(f'k_asy must be >= 0, got {self.pars.k_asy}')
        if self.pars.k_non < 0:
            raise ValueError(f'k_non must be >= 0, got {self.pars.k_non}')
        return

    def _set_reinfection_wane(self, uids):
        """Schedule reinfection-protection waning for newly cleared ``uids``, if waning is enabled."""
        if self.pars.dur_reinfection_protection is not None and len(uids):
            self.ti_rr_reinfection_wane[uids] = self.ti + self.pars.dur_reinfection_protection.rvs(uids)
        return

    def _enter_cleared(self, uids, rr):
        """Bookkeeping shared by every pathway that moves agents into ``CLEARED``.

        Sets the pathway-specific per-agent reinfection RR and schedules protection waning. Callers set
        ``state`` (and any infected/susceptible flags) themselves. ``TBResistant`` extends this to also
        drop all carried strains and their counts.
        """
        if len(uids) == 0:
            return
        self.rr_reinfection[uids] = rr
        self._set_reinfection_wane(uids)
        return

    @property
    def infectious(self):
        """
        Boolean array: True for agents who can transmit TB.

        In this model only ASYMPTOMATIC and SYMPTOMATIC states are infectious.
        Used by the base `starsim.Infection` for transmission.
        """
        return (self.state == TBS.ASYMPTOMATIC) | (self.state == TBS.SYMPTOMATIC)

    # Read-only boolean views of the categorical `state` (the single source of truth).
    # These give the boolean idiom at call sites (e.g. ``tb.active_tb.uids``,
    # ``tb.latent[uids]``) with no extra storage and no risk of desync; to *write*
    # state, assign ``self.state[uids] = TBS.X`` as before.
    @property
    def latent(self):
        """Latent infection (`TBS.INFECTION`)."""
        return self.state == TBS.INFECTION

    @property
    def non_infectious(self):
        """Non-infectious TB (`TBS.NON_INFECTIOUS`)."""
        return self.state == TBS.NON_INFECTIOUS

    @property
    def asymptomatic(self):
        """Asymptomatic active TB (`TBS.ASYMPTOMATIC`)."""
        return self.state == TBS.ASYMPTOMATIC

    @property
    def symptomatic(self):
        """Symptomatic active TB (`TBS.SYMPTOMATIC`)."""
        return self.state == TBS.SYMPTOMATIC

    @property
    def active_tb(self):
        """Active TB disease (non-infectious, asymptomatic, or symptomatic)."""
        return self.state.isin(TBS.ACTIVE)

    @property
    def terminal(self):
        """No longer participating in transmission or care flows (dead or removed)."""
        return self.state.isin(TBS.TERMINAL)

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for newly infected agents (called when transmission occurs).

        The base `starsim.Infection` calls this when a susceptible agent acquires infection. A primary
        infection (from ``SUSCEPTIBLE`` / ``CLEARED``) enters latent ``INFECTION``; a *reinfection* of an
        already-infected latent (``INFECTION``) or non-infectious (``NON_INFECTIOUS``) agent only resets
        the ``ti_infected`` clock — its state is left unchanged (clock reset, no state change). Transitions
        are evaluated per dt each timestep in `step`.
        """
        super().set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        # Reinfection resets everyone's infection clock (raising the progression hazard under
        # front-loading); the state is only (re)set to INFECTION for agents entering from a non-infected
        # state, so a reinfected NON_INFECTIOUS agent is not wrongly knocked back to latent.
        was_uninfected = ~self.infected[uids]
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.state[uids[was_uninfected]] = TBS.INFECTION

        return

    def init_pre(self, sim):
        """Resolve the reinfection-RR default coupling once all pars are final (spec §reinfection).

        ``rr_reinfection_inf`` (σ_L) defaults to ``rr_reinfection_rec``; ``rr_reinfection_non`` (σ_N)
        defaults to ``rr_reinfection_inf``. Done here (not in ``__init__``) so it uses the fully-resolved
        pars regardless of subclass construction order (``TBResistant`` applies its pars after
        ``super().__init__``).
        """
        super().init_pre(sim)
        if self.pars.rr_reinfection_inf is None:
            self.pars.rr_reinfection_inf = float(self.pars.rr_reinfection_rec)
        if self.pars.rr_reinfection_non is None:
            self.pars.rr_reinfection_non = float(self.pars.rr_reinfection_inf)
        return

    def init_post(self):
        """
        Seed initial infections, then (optionally) promote a fraction to active TB.

        The base `starsim.Infection.init_post` seeds ``init_prev`` cases into latent
        ``INFECTION`` (via `set_prognoses`). We additionally seed ``init_prev_active``
        cases directly into ``SYMPTOMATIC`` active TB, matching the compartmental ODE,
        which seeds its initial cases as symptomatic rather than latent. Active seeding
        draws from agents still susceptible after latent seeding, so the two seeds do
        not overlap.
        """
        super().init_post()

        active_cases = self.pars.init_prev_active.filter(self.susceptible.uids)
        if len(active_cases):
            self.set_prognoses(active_cases)                # sets infection flags + ti_infected (state → INFECTION)
            self.state[active_cases] = TBS.SYMPTOMATIC      # override latent → symptomatic active TB
        return

    def transition(self, uids, to, rng):
        """
        Evaluate competing exponential transitions over one dt and apply immediately.

        For each agent in *uids*, computes the probability of transitioning
        to each destination in *to* during one timestep. Uses a single uniform
        draw per agent to decide (a) whether the agent transitions and
        (b) which destination it goes to. State is updated **immediately**.

        """
        if len(uids) == 0:
            return

        dt = self.sim.t.dt
        n = len(uids)
        states = list(to.keys())
        n_dest = len(states)

        # Convert each rate to dimensionless "rate per dt" using its own unit
        rates_per_dt = np.zeros((n_dest, n))
        for idx, rate_val in enumerate(to.values()):
            factor = dt / rate_val.unit
            v = rate_val.value
            rates_per_dt[idx] = np.broadcast_to(v, n) * factor

        # Total exit rate per agent (dimensionless, per dt)
        total_rate_dt = rates_per_dt.sum(axis=0)

        # Probability of any transition in one dt
        p_any = 1 - np.exp(-total_rate_dt)

        # Build CDF bins for destination selection
        with np.errstate(divide='ignore', invalid='ignore'):
            fractions = np.where(total_rate_dt > 0, rates_per_dt / total_rate_dt, 0)
        cum_p = np.cumsum(fractions, axis=0) * p_any

        # Single uniform draw per agent
        u = rng.rvs(uids)

        # Find agents that transition: u < p_any
        transitioning = u < p_any
        if not np.any(transitioning):
            return

        # For transitioning agents, determine destination from CDF
        t_idx = transitioning.nonzero()[0]
        t_uids = uids[t_idx]
        t_u = u[t_idx]
        t_cum_p = cum_p[:, t_idx]
        dest_idx = (t_u[None, :] < t_cum_p).argmax(axis=0)
        dest_states = np.array(states)[dest_idx]

        # Apply state change immediately
        self.state[t_uids] = dest_states

        # Record ti_asymp for new-active tracking
        newly_asymp = t_uids[dest_states == TBS.ASYMPTOMATIC]
        if len(newly_asymp):
            self.ti_asymp[newly_asymp] = self.ti

        return

    def step(self):
        """
        Advance TB state machine one timestep.

        1. **Transmission** (via ``super().step()``): handles force of infection.
        2. **Evaluate transitions** (``step_transitions``): competing-risk transitions,
           applied immediately; agents may cascade through multiple states in one step.
        3. **Bookkeeping** (``step_bookkeeping``): update flags, modifiers, deaths, and
           the transmission state (``susceptible``/``rel_sus``/``rel_trans``) for next step.

        The work is split into ``step_transitions`` and ``step_bookkeeping`` so that
        strain-aware subclasses (see ``tbsim.resistance``) can override the natural
        history and the transmission set-up independently.
        """
        super().step()
        self.step_transitions()
        self.step_bookkeeping()
        return

    def progression_rates(self, uids):
        """
        Per-agent INFECTION-exit rates to NON_INFECTIOUS and ASYMPTOMATIC.

        Returns ``(inf_non, inf_asy)`` as rate objects with per-agent values, applying
        ``rr_activation`` and the optional time-since-infection decline::

            inf_non(tau) = inf_non * rr_activation * exp(-k_non * tau)
            inf_asy(tau) = inf_asy * rr_activation * exp(-k_asy * tau)

        where ``tau`` is years since each agent entered INFECTION (from ``ti_infected``,
        so reinfection restarts the clock). With ``k_non = k_asy = 0`` (default) the decline
        factors are exactly 1 and the rates reduce to the constant ``inf_non``/``inf_asy``
        scaled only by ``rr_activation``.
        """
        rr = self.rr_activation[uids]
        inf_non = self.pars.inf_non * rr
        inf_asy = self.pars.inf_asy * rr
        k_non, k_asy = self.pars.k_non, self.pars.k_asy
        if k_non or k_asy:  # front-load progression by declining with time since infection
            tau = (self.ti - self.ti_infected[uids]) * self.t.dt_year
            if k_non:
                inf_non = inf_non * np.exp(-k_non * tau)
            if k_asy:
                inf_asy = inf_asy * np.exp(-k_asy * tau)
        return inf_non, inf_asy

    def step_transitions(self):
        """ Evaluate the natural-history state transitions (each mutates ``self.state`` in place). """
        # For transitions that lead to CLEARED, we snapshot the pre-transition CLEARED mask
        # and compare after to identify agents newly entering CLEARED from each source state,
        # so we can assign the correct pathway-specific rr_reinfection to each new entrant.

        u = self.latent.uids
        if len(u):
            inf_non, inf_asy = self.progression_rates(u)  # constant, or front-loaded via k_non/k_asy
            self.transition(u, to={
                TBS.CLEARED:        self.pars.inf_cle,
                TBS.NON_INFECTIOUS: inf_non,
                TBS.ASYMPTOMATIC:   inf_asy,
            }, rng=self._rng_inf)
            newly_cleared = u[self.state[u] == TBS.CLEARED]  # agents cleared from INFECTION this step
            self._enter_cleared(newly_cleared, self.pars.rr_reinfection_cleared)

        u = self.non_infectious.uids
        if len(u):
            self.transition(u, to={
                TBS.CLEARED:      self.pars.non_rec * self.rr_clearance[u],
                TBS.ASYMPTOMATIC: self.pars.non_asy,
            }, rng=self._rng_non)
            newly_cleared = u[self.state[u] == TBS.CLEARED]  # agents cleared from NON_INFECTIOUS this step
            self._enter_cleared(newly_cleared, self.pars.rr_reinfection_rec)

        u = self.asymptomatic.uids
        if len(u):
            self.transition(u, to={
                TBS.NON_INFECTIOUS: self.pars.asy_non,
                TBS.SYMPTOMATIC:    self.pars.asy_sym,
            }, rng=self._rng_asy)

        u = self.symptomatic.uids
        if len(u):
            self.transition(u, to={
                TBS.ASYMPTOMATIC: self.pars.sym_asy,
                TBS.DEAD:         self.pars.sym_dead * self.rr_death[u],
            }, rng=self._rng_sym)

        # NOTE: TREATMENT outcomes (success → CLEARED, failure → SYMPTOMATIC) are
        # handled by TxDelivery, not the natural history. Agents in TREATMENT state
        # without a TxDelivery intervention will remain in TREATMENT indefinitely.
        return

    def step_bookkeeping(self):
        """ Update infection flags, request TB deaths, reset risk modifiers, and set the
        transmission state (``susceptible``/``rel_sus``/``rel_trans``) used next step. """
        # --- Bookkeep from current state ---
        # Derive the transmission flags from the categorical state (identical values to,
        # but faster than, the previous np.isin re-derivation).
        st = self.state
        # Reinfection-eligible states: never-infected/cleared, plus latent and non-infectious agents
        # (a re-exposure resets their infection clock; see set_prognoses and rr_reinfection_inf/non).
        self.susceptible[:] = st.isin((TBS.SUSCEPTIBLE, TBS.CLEARED, TBS.INFECTION, TBS.NON_INFECTIOUS))
        self.infected[:] = st.isin((TBS.INFECTION, TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC, TBS.TREATMENT))
        self.on_treatment[:] = (st == TBS.TREATMENT)

        # TB deaths
        dead = ss.uids((st == TBS.DEAD) & self.sim.people.alive)
        self.sim.people.request_death(dead)
        self.results['new_deaths'][self.ti] = len(dead)
        self.results['new_deaths_15+'][self.ti] = np.count_nonzero(self.sim.people.age[dead] >= 15)

        # Reset rr_* (interventions set fresh values next step)
        self.rr_activation[:] = 1
        self.rr_clearance[:] = 1
        self.rr_death[:] = 1

        # rel_sus / rel_trans. rel_sus is per-state: 1 for fully-susceptible, the (waning) per-agent
        # rr_reinfection for CLEARED, and the σ_L / σ_N reinfection factors for latent / non-infectious.
        self.rel_sus[:] = 1
        cleared = ss.uids(self.state == TBS.CLEARED)
        if self.pars.dur_reinfection_protection is not None and len(cleared):
            # Waning: agents whose protection period has elapsed revert to full susceptibility
            waned = cleared[self.ti >= self.ti_rr_reinfection_wane[cleared]]
            self.rr_reinfection[waned] = 1.0
            self.ti_rr_reinfection_wane[waned] = np.inf
        self.rel_sus[cleared] *= self.rr_reinfection[cleared]
        self.rel_sus[st == TBS.INFECTION] = self.pars.rr_reinfection_inf
        self.rel_sus[st == TBS.NON_INFECTIOUS] = self.pars.rr_reinfection_non
        self.rel_trans[:] = 1
        self.rel_trans[self.asymptomatic] = self.pars.trans_asymp

        return

    def step_die(self, uids):
        """
        Apply death for the given agents and update TB state.

        Called by the framework when agents die (e.g. background mortality).
        Sets state to DEAD, clears flags, and stops transmission.
        """
        if len(uids) == 0:
            return

        removed = self.state[uids] == TBS.REMOVED
        super().step_die(uids)
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.on_treatment[uids] = False
        self.rel_sus[uids] = 0
        self.rel_trans[uids] = 0
        if np.any(~removed):
            self.state[uids[~removed]] = TBS.DEAD
        if np.any(removed):
            self.state[uids[removed]] = TBS.REMOVED
        return

    def init_results(self):
        """Define result time series."""
        super().init_results()

        results = []
        for state in TBS:
            results.append(ss.Result(f'n_{state.name}', dtype=int, label=state.name))
            results.append(ss.Result(f'n_{state.name}_15+', dtype=int, label=f'{state.name} (15+)'))

        self.define_results(*results)
        self.define_results(
            ss.Result('n_infectious',      dtype=int, label='Number Infectious'),
            ss.Result('n_infectious_15+',  dtype=int, label='Number Infectious, 15+'),
            ss.Result('new_active',        dtype=int, label='New Active'),
            ss.Result('new_active_15+',    dtype=int, label='New Active, 15+'),
            ss.Result('cum_active',        dtype=int, label='Cumulative Active'),
            ss.Result('cum_active_15+',    dtype=int, label='Cumulative Active, 15+'),
            ss.Result('new_deaths',        dtype=int, label='New Deaths'),
            ss.Result('new_deaths_15+',    dtype=int, label='New Deaths, 15+'),
            ss.Result('cum_deaths',        dtype=int, label='Cumulative Deaths'),
            ss.Result('cum_deaths_15+',    dtype=int, label='Cumulative Deaths, 15+'),
            ss.Result('prevalence_active', dtype=float, scale=False, label='Prevalence (Active)'),
            ss.Result('incidence_kpy',     dtype=float, scale=False, label='Incidence per 1,000 person-years'),
            ss.Result('deaths_ppy',        dtype=float, label='Death per person-year'),
            ss.Result('new_notifications_15+', dtype=int, label='New TB notifications, 15+'),
            ss.Result('n_detectable_15+', dtype=float, scale=False, label='Symptomatic plus cxr_asymp_sens * Asymptomatic (15+)'),
        )
        return

    def update_results(self):
        """Record current time-step values for all result series."""
        super().update_results()
        res = self.results
        ti = self.ti
        dty = self.sim.t.dt_year

        # Cache commonly reused arrays
        age15 = self.sim.people.age >= 15
        n_alive = self.sim.people.alive.count()
        new_asymp = self.ti_asymp == ti

        # Count every state in a single pass with np.bincount (TBS codes index bins directly),
        # once over all agents and once restricted to 15+. The per-state, infectious, and
        # detectable series are then plain lookups -- no per-state ==/& passes.
        codes = self.state.values.astype(np.intp)
        adult = age15.values
        n_all = np.bincount(codes, minlength=_TBS_NBINS)
        n_15  = np.bincount(codes[adult], minlength=_TBS_NBINS)
        for state in TBS:
            res[f'n_{state.name}'][ti] = n_all[state]
            res[f'n_{state.name}_15+'][ti] = n_15[state]

        asy = int(TBS.ASYMPTOMATIC)
        sym = int(TBS.SYMPTOMATIC)
        res.n_infectious[ti] = n_all[asy] + n_all[sym]                       # infectious == asymptomatic | symptomatic
        res['n_infectious_15+'][ti] = n_15[asy] + n_15[sym]
        res.prevalence_active[ti] = res.n_infectious[ti] / n_alive if n_alive else 0
        res.incidence_kpy[ti] = 1_000 * (self.ti_infected == ti).count() / (n_alive * dty) if n_alive else 0
        res.deaths_ppy[ti] = res.new_deaths[ti] / (n_alive * dty) if n_alive else 0

        # New active: agents whose ti_asymp == this step
        res['new_active'][ti] = new_asymp.count()
        res['new_active_15+'][ti] = (new_asymp & age15).count()
        res['n_detectable_15+'][ti] = n_15[sym] + self.pars.cxr_asymp_sens * n_15[asy]
        return

    def finalize_results(self):
        """Compute cumulative series from new-event series after the run."""
        super().finalize_results()
        res = self.results
        res['cum_deaths'] = np.cumsum(res['new_deaths'])
        res['cum_deaths_15+'] = np.cumsum(res['new_deaths_15+'])
        res['cum_active'] = np.cumsum(res['new_active'])
        res['cum_active_15+'] = np.cumsum(res['new_active_15+'])
        return

    def plot(self, **kwargs):
        """Plot TB result time series using tbsim.plot().

        Args:
            **kwargs: Forwarded to :func:`tbsim.plots.plot`. Common options
                include ``select``, ``title``, ``n_cols``, ``row_height``,
                ``style``, ``filename``, and ``show``.

        Returns:
            matplotlib.figure.Figure
        """
        return _tbsim_plot(self.sim, **kwargs)


def get_tb(sim, which=None):
    """ Helper to get the TB infection module from a sim

    Args:
        sim (Sim): the simulation to search for the TB module
        which (type, optional): the class of TB module to get (e.g. TB); if None, returns the first BaseTB subclass found
    """
    if which is None:
        which = BaseTB
    for disease in sim.diseases.values():
        if isinstance(disease, which):
            return disease
    raise ValueError("No TB module found in sim.diseases")

class choice2d(ss.choice):
    """ 
    Version of ss.choice() allowing different per-agent probabilities.

    Temporary location; to be ported to Starsim and potentially merged
    with ss.choice().

    Args:
        a (1D array): the values to choose from (default: np.arange(p.shape[1]))
        p (2D array): the probability of each choice for each agent

    **Example**:
        # Choose between specified options each with a specified probability (must sum to 1)
        p = np.array([ # Per-agent array of outcome probabilities across 3 options
            [0.1, 0.5, 0.4],
            [0.2, 0.3, 0.5],
            [0.3, 0.4, 0.3],
            [0.4, 0.2, 0.4],
            [0.5, 0.3, 0.2],
            [0.8, 0.1, 0.1],
        ])
        n = len(p)
        choices = tbsim.choice2d(p=p, strict=False)(n)
    """
    valid_pars = ['a', 'p', 'replace', 'dtype']
    scaling = ss.distributions.scale_types.false

    def __init__(self, a=None, p=None, replace=True, **kwargs):
        if p is None and a is not None:
            if np.ndim(a) == 2:
                p = a # Swap, to allow calling choice2d(p) directly
                a = np.arange(p.shape[1])
            else:
                errormsg = f'Must supply p as a 2D array of probabilities (n_agents x n_choices); got a={a} (shape {np.shape(a)})'
                raise ValueError(errormsg)
        if a is None:
            if p is None or np.ndim(p) != 2:
                errormsg = f'Must supply p as a 2D array of probabilities'
                raise ValueError(errormsg)
            a = np.arange(p.shape[1])
        ss.Dist.__init__(self, distname='choice2d', a=a, p=p, replace=replace, **kwargs)
        self._use_ppf = True # Always use array parameters
        return

    def ppf(self, rands):
        """ Always use ppf to allow per-agent probabilities """
        pars = self._pars
        pcum = np.cumsum(pars.p, axis=1) # Sum probabilities to equal one
        inds = (rands[:, np.newaxis] >= pcum).sum(axis=1) # 2D equivalent of np.searchsorted
        rvs = pars.a[inds] # Map to outcomes
        return rvs