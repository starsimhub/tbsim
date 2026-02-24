"""LSHTM TB natural history model. State definitions and transition diagram are in the API docs (tbsim.tb_lshtm)."""

from enum import IntEnum
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss

__all__ = ['TB_LSHTM', 'TB_LSHTM_Acute', 'TBSL', 'get_tb']


class TBSL(IntEnum):
    """
    TB state labels for the LSHTM model.

    - Each agent is in exactly one of these states.
    - Transitions are driven by exponential rates in `TB_LSHTM`
      (and `TB_LSHTM_Acute`).
    """
    SUSCEPTIBLE     = -1    # Never infected (agents who clear/recover/treat remain in their last state, not here)
    INFECTION       = 0     # Latent infection (not yet active TB)
    CLEARED         = 1     # Cleared infection without developing active TB
    NON_INFECTIOUS  = 2     # Non-infectious TB (early/smear-negative, corresponds to LSHTM diagram)
    RECOVERED       = 3     # Recovered from non-infectious TB (susceptible to reinfection)
    ASYMPTOMATIC    = 4     # Active TB, asymptomatic (infectious)
    SYMPTOMATIC     = 5     # Active TB, symptomatic (infectious)
    TREATMENT       = 6     # On TB treatment
    TREATED         = 7     # Completed treatment (susceptible to reinfection)
    DEAD            = 8     # Dead (TB-caused via sym_dead; general mortality via step_die also sets this)
    ACUTE           = 9     # Acute infection immediately after exposure (TB_LSHTM_Acute only)

    @staticmethod
    def care_seeking_eligible():
        """States eligible for care-seeking: only SYMPTOMATIC.
        Only individuals with clinical symptoms (cough, fever, night sweats, etc.)
        recognise their illness and seek healthcare."""
        return np.array([TBSL.SYMPTOMATIC])


class BaseTB(ss.Infection):
    """Base class for TB natural history models."""
    pass


class TB_LSHTM(BaseTB):
    """
    Agent-based TB natural history adapting the LSHTM compartmental structure [1] (Schwalb et al. 2025).
    States in `TBSL` span the spectrum from susceptibility to active disease and treatment.
    Infectious states are `TBSL.ASYMPTOMATIC` and `TBSL.SYMPTOMATIC`; the force
    of infection depends on `pars.beta` and the prevalence of those states, with
    `pars.trans_asymp` (kappa) giving the relative infectiousness of asymptomatic vs symptomatic TB.
    Reinfectable states (`TBSL.CLEARED`, `TBSL.RECOVERED`, `TBSL.TREATED`) use
    `pars.rr_rec` (pi) and `pars.rr_treat` (rho). Per-agent modifiers ``rr_activation``, ``rr_clearance``,
    ``rr_death`` scale selected rates. Interventions call `start_treatment`.

    Args (pars):
        *Transmission and reinfection*

        - ``init_prev``:   Initial seed infections (prevalence).
        - ``beta``:        Transmission rate per year.
        - ``trans_asymp``: Relative transmissibility, asymptomatic vs symptomatic. (kappa)
        - ``rr_rec``:      Relative risk of reinfection for RECOVERED.  (pi)
        - ``rr_treat``:    Relative risk of reinfection for TREATED. (rho)

        *From INFECTION (latent)*

        - ``inf_cle``:     Infection -> Cleared (no active TB).
        - ``inf_non``:     Infection -> Non-infectious TB.
        - ``inf_asy``:     Infection -> Asymptomatic TB.

        *From NON_INFECTIOUS*

        - ``non_rec``:    Non-infectious -> Recovered.
        - ``non_asy``:    Non-infectious -> Asymptomatic.

        *From ASYMPTOMATIC*

        - ``asy_non``:    Asymptomatic -> Non-infectious.
        - ``asy_sym``:    Asymptomatic -> Symptomatic.

        *From SYMPTOMATIC*

        - ``sym_asy``:     Symptomatic -> Asymptomatic.
        - ``sym_treat``:   Symptomatic -> Treatment. (theta)
        - ``sym_dead``:    Symptomatic -> Dead (TB-specific mortality, mu_TB).

        *From TREATMENT*

        - ``fail_rate``:     Treatment -> Symptomatic (failure). (phi)
        - ``complete_rate``: Treatment -> Treated (completion).  (delta)

        *Background (general mortality is handled by* ``ss.Deaths`` *demographics, not this module)*

        - ``cxr_asymp_sens``:   CXR sensitivity for screening asymptomatic (0-1).

    Attributes:
        *Infection flags*

        - ``susceptible``    (BoolState, default=True):   Whether the agent is susceptible to TB.
        - ``infected``       (BoolState, default=False):  Whether the agent is currently infected.
        - ``ever_infected``  (BoolState, default=False):  Whether the agent has ever been infected.
        - ``on_treatment``   (BoolState, default=False):  Whether the agent is on TB treatment.

        *TB state machine*

        - ``state``          (FloatArr, default=TBSL.SUSCEPTIBLE):  Current TB state (`TBSL` value).
        - ``ti_infected``    (FloatArr, default=-inf):              Time of infection (never infected = -inf).

        *Transmission modifiers*

        - ``rel_sus``        (FloatArr, default=1.0):  Relative susceptibility to TB.
        - ``rel_trans``      (FloatArr, default=1.0):  Relative transmissibility of TB.

        *Per-agent risk modifiers*

        - ``rr_activation``  (FloatArr, default=1.0):  Multiplier on INFECTION -> NON_INFECTIOUS / ASYMPTOMATIC.
        - ``rr_clearance``   (FloatArr, default=1.0):  Multiplier on NON_INFECTIOUS -> RECOVERED.
        - ``rr_death``       (FloatArr, default=1.0):  Multiplier on SYMPTOMATIC -> DEAD.

    Example:
        ::

            import starsim as ss
            import tbsim

            sim = ss.Sim(diseases=tbsim.TB_LSHTM(), pars=dict(start='2000', stop='2020'))
            sim.run()
            sim.plot()

    References:
        [1] Schwalb et al. (2025) Potential impact, costs, and benefits of population-wide
        screening interventions for tuberculosis in Viet Nam. PLOS Glob Public Health.
        https://doi.org/10.1371/journal.pgph.0005050

    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with default LSHTM natural history parameters; override via ``pars``."""
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        # --- Transmission and reinfection ---
        self.define_pars(
            init_prev=ss.bernoulli(0.05),       # Initial seed infections (prevalence)
            beta=ss.permonth(0.2),              # Transmission rate per year
            trans_asymp=0.82,                   # κ kappa: rel. transmissibility asymptomatic vs symptomatic
            rr_rec=0.21,                        # π pi: RR reinfection after recovery
            rr_treat=3.15,                      # ρ rho: RR reinfection after treatment
            # --- From INFECTION (latent) ---
            inf_cle=ss.peryear(1.90),            # Clear infection (no active TB)
            inf_non=ss.peryear(0.16),            # Progress to non-infectious TB
            inf_asy=ss.peryear(0.06),            # Progress to asymptomatic active TB
            # --- From NON_INFECTIOUS ---
            non_rec=ss.peryear(0.18),            # Recover (→ RECOVERED)
            non_asy=ss.peryear(0.25),            # Progress to asymptomatic
            # --- From ASYMPTOMATIC ---
            asy_non=ss.peryear(1.66),            # Revert to non-infectious
            asy_sym=ss.peryear(0.88),            # Progress to symptomatic
            # --- From SYMPTOMATIC ---
            sym_asy=ss.peryear(0.54),            # Regress to asymptomatic (still active, not recovered)
            sym_treat=ss.peryear(0.46),          # θ theta: symptomatic → treatment
            sym_dead=ss.peryear(0.34),           # μ_TB: symptomatic → dead (TB mortality)
            # --- From TREATMENT ---
            fail_rate=ss.peryear(0.63),          # φ phi: treatment failure (→ symptomatic)
            complete_rate=ss.peryear(2.00),      # δ delta: treatment completion (→ treated)
            # --- Background ---
            cxr_asymp_sens=1.0,                 # CXR sensitivity for screening asymptomatic (0–1)
            # --- For ACUTE ---
            rate_acute_latent=None,             # ACUTE → INFECTION
            trans_acute=None,                   # α alpha: rel. transmissibility acute vs symptomatic
        )
        self.update_pars(pars, **kwargs)

        # CRN-safe RNG distributions for per-step transition draws (one per source state)
        self._rng_inf = ss.random(name='tb_rng_inf')   # INFECTION exits
        self._rng_non = ss.random(name='tb_rng_non')   # NON_INFECTIOUS exits
        self._rng_asy = ss.random(name='tb_rng_asy')   # ASYMPTOMATIC exits
        self._rng_sym = ss.random(name='tb_rng_sym')   # SYMPTOMATIC exits
        self._rng_trt = ss.random(name='tb_rng_trt')   # TREATMENT exits

        # Per-agent state: redefine base Infection states and add TB-specific ones
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('infected'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_trans', default=1.0),
            ss.FloatArr('ti_infected', default=-np.inf),
            ss.FloatArr('state', default=TBSL.SUSCEPTIBLE),
            ss.FloatArr('ti_asymp', default=np.nan),                # Time of last entry to ASYMPTOMATIC (for new_active tracking)
            ss.BoolState('on_treatment', default=False),
            ss.BoolState('ever_infected', default=False),
            # Risk modifiers
            ss.FloatArr('rr_activation', default=1.0),
            ss.FloatArr('rr_clearance', default=1.0),
            ss.FloatArr('rr_death', default=1.0),
            reset=True,
        )

        return

    @property
    def infectious(self):
        """
        Boolean array: True for agents who can transmit TB.

        In this model only ASYMPTOMATIC and SYMPTOMATIC states are infectious.
        Used by the base `starsim.Infection` for transmission.
        """
        return (self.state == TBSL.ASYMPTOMATIC) | (self.state == TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for newly infected agents (called when transmission occurs).

        The base `starsim.Infection` calls this when a susceptible agent
        acquires infection. We mark agents infected and set state to INFECTION
        (latent). Transitions are evaluated per dt each timestep in `step`.
        """
        super().set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.state[uids] = TBSL.INFECTION

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
        newly_asymp = t_uids[dest_states == TBSL.ASYMPTOMATIC]
        if len(newly_asymp):
            self.ti_asymp[newly_asymp] = self.ti

        return

    def step(self):
        """
        Advance TB state machine one timestep.

        1. **Transmission** (via ``super().step()``): handles force of infection.
        2. **Reset RR multipliers** for all agents (interventions set fresh each step).
        3. **Evaluate transitions**: for each state group, evaluate competing-risk
           transitions and apply immediately. Agents may cascade through multiple
           states in one step.
        4. **Bookkeeping**: update flags, modifiers, deaths, results.
        """
        super().step()

        # --- Evaluate transitions (each mutates self.state in place) ---

        u = ss.uids(self.state == TBSL.INFECTION)
        if len(u):
            self.transition(u, to={
                TBSL.CLEARED:        self.pars.inf_cle,
                TBSL.NON_INFECTIOUS: self.pars.inf_non * self.rr_activation[u],
                TBSL.ASYMPTOMATIC:   self.pars.inf_asy * self.rr_activation[u],
            }, rng=self._rng_inf)

        u = ss.uids(self.state == TBSL.NON_INFECTIOUS)
        if len(u):
            self.transition(u, to={
                TBSL.RECOVERED:    self.pars.non_rec * self.rr_clearance[u],
                TBSL.ASYMPTOMATIC: self.pars.non_asy,
            }, rng=self._rng_non)

        u = ss.uids(self.state == TBSL.ASYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBSL.NON_INFECTIOUS: self.pars.asy_non,
                TBSL.SYMPTOMATIC:    self.pars.asy_sym,
            }, rng=self._rng_asy)

        u = ss.uids(self.state == TBSL.SYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBSL.ASYMPTOMATIC: self.pars.sym_asy,
                TBSL.TREATMENT:    self.pars.sym_treat,
                TBSL.DEAD:         self.pars.sym_dead * self.rr_death[u],
            }, rng=self._rng_sym)

        u = ss.uids(self.state == TBSL.TREATMENT)
        if len(u):
            self.transition(u, to={
                TBSL.SYMPTOMATIC: self.pars.fail_rate,
                TBSL.TREATED:     self.pars.complete_rate,
            }, rng=self._rng_trt)

        # --- Bookkeep from current state ---

        self.infected[:] = ~np.isin(self.state,
            [TBSL.SUSCEPTIBLE, TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED, TBSL.DEAD])
        self.susceptible[:] = np.isin(self.state,
            [TBSL.SUSCEPTIBLE, TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])
        self.on_treatment[:] = (self.state == TBSL.TREATMENT)

        # TB deaths
        dead = ss.uids((self.state == TBSL.DEAD) & self.sim.people.alive)
        self.sim.people.request_death(dead)
        self.results['new_deaths'][self.ti] = len(dead)
        self.results['new_deaths_15+'][self.ti] = np.count_nonzero(self.sim.people.age[dead] >= 15)

        # Reset rr_* (interventions set fresh values next step)
        self.rr_activation[:] = 1
        self.rr_clearance[:] = 1
        self.rr_death[:] = 1

        # rel_sus / rel_trans
        self.rel_sus[:] = 1
        self.rel_sus[self.state == TBSL.RECOVERED] = self.pars.rr_rec
        self.rel_sus[self.state == TBSL.TREATED] = self.pars.rr_treat
        self.rel_trans[:] = 1
        self.rel_trans[self.state == TBSL.ASYMPTOMATIC] = self.pars.trans_asymp

        return

    def start_treatment(self, uids):
        """
        Move specified agents onto TB treatment (or clear latent infection).

        Called by interventions (e.g. screening/case-finding) when an agent is
        identified for treatment. State is changed immediately.

        - Latent (INFECTION): cleared without treatment (→ CLEARED).
        - Active (NON_INFECTIOUS, ASYMPTOMATIC, SYMPTOMATIC): moved to TREATMENT.
        - Records new notifications (15+) for active cases starting treatment.
        """
        if len(uids) == 0:
            return

        # Latent infection: clear without active disease
        latent = uids[self.state[uids] == TBSL.INFECTION]
        self.state[latent] = TBSL.CLEARED
        self.infected[latent] = False
        self.susceptible[latent] = True

        # Active TB: put on treatment
        active = uids[np.isin(self.state[uids], [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state[active] = TBSL.TREATMENT
        self.on_treatment[active] = True

        self.results['new_notifications_15+'][self.ti] += np.count_nonzero(self.sim.people.age[active] >= 15)

        return

    def step_die(self, uids):
        """
        Apply death for the given agents and update TB state.

        Called by the framework when agents die (e.g. background mortality).
        Sets state to DEAD, clears flags, and stops transmission.
        """
        if len(uids) == 0:
            return

        super().step_die(uids)
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.state[uids] = TBSL.DEAD
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """Define result time series."""
        super().init_results()

        results = []
        for state in TBSL:
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
        infectious = self.infectious
        n_alive = self.sim.people.alive.count()
        new_asymp = self.ti_asymp == ti

        in_state = {}
        for state in TBSL:
            in_state[state] = self.state == state
            res[f'n_{state.name}'][ti] = in_state[state].count()
            res[f'n_{state.name}_15+'][ti] = (age15 & in_state[state]).count()

        res.n_infectious[ti] = infectious.count()
        res['n_infectious_15+'][ti] = (infectious & age15).count()
        res.prevalence_active[ti] = res.n_infectious[ti] / n_alive
        res.incidence_kpy[ti] = 1_000 * (self.ti_infected == ti).count() / (n_alive * dty)
        res.deaths_ppy[ti] = res.new_deaths[ti] / (n_alive * dty)

        # New active: agents whose ti_asymp == this step
        res['new_active'][ti] = new_asymp.count()
        res['new_active_15+'][ti] = (new_asymp & age15).count()
        res['n_detectable_15+'][ti] = (age15 * (in_state[TBSL.SYMPTOMATIC] + self.pars.cxr_asymp_sens*in_state[TBSL.ASYMPTOMATIC])).sum()
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

    def plot(self):
        """Plot all result time series on one figure."""
        fig = plt.figure()
        for rkey in self.results.keys():
            if rkey == 'timevec':
                continue
            plt.plot(self.results['timevec'], self.results[rkey], label=rkey.title())
        plt.legend()
        return fig


class TB_LSHTM_Acute(TB_LSHTM):
    """
    LSHTM TB model with an acute infection state immediately after exposure.

    Extends `TB_LSHTM` by inserting an ACUTE state between infection and
    the usual INFECTION (latent) state. New infections enter ACUTE first, then
    transition to INFECTION at rate `pars.rate_acute_latent`. Acute cases are
    infectious with relative transmissibility `pars.trans_acute` (alpha).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(pars=pars, **kwargs)

        self.define_pars(
            rate_acute_latent=ss.peryear(4.0),
            trans_acute=0.9,
        )
        self.update_pars(pars, **kwargs)

        self._rng_acu = ss.random(name='tb_rng_acu')
        return

    @property
    def infectious(self):
        """Includes ACUTE in addition to ASYMPTOMATIC and SYMPTOMATIC."""
        return (self.state == TBSL.ACUTE) | (self.state == TBSL.ASYMPTOMATIC) | (self.state == TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, sources=None):
        """New infections enter ACUTE (not INFECTION)."""
        super(TB_LSHTM, self).set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.state[uids] = TBSL.ACUTE

        return

    def step(self):
        """
        Advance TB state machine one timestep (acute variant).

        Same single-pass structure as `TB_LSHTM.step`, but adds
        ACUTE -> INFECTION transition and treats ACUTE as infectious.
        """
        super(TB_LSHTM, self).step()

        # --- Evaluate transitions ---

        u = ss.uids(self.state == TBSL.ACUTE)
        if len(u):
            self.transition(u, to={TBSL.INFECTION: self.pars.rate_acute_latent}, rng=self._rng_acu)

        u = ss.uids(self.state == TBSL.INFECTION)
        if len(u):
            self.transition(u, to={
                TBSL.CLEARED:        self.pars.inf_cle,
                TBSL.NON_INFECTIOUS: self.pars.inf_non * self.rr_activation[u],
                TBSL.ASYMPTOMATIC:   self.pars.inf_asy * self.rr_activation[u],
            }, rng=self._rng_inf)

        u = ss.uids(self.state == TBSL.NON_INFECTIOUS)
        if len(u):
            self.transition(u, to={
                TBSL.RECOVERED:    self.pars.non_rec * self.rr_clearance[u],
                TBSL.ASYMPTOMATIC: self.pars.non_asy,
            }, rng=self._rng_non)

        u = ss.uids(self.state == TBSL.ASYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBSL.NON_INFECTIOUS: self.pars.asy_non,
                TBSL.SYMPTOMATIC:    self.pars.asy_sym,
            }, rng=self._rng_asy)

        u = ss.uids(self.state == TBSL.SYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBSL.ASYMPTOMATIC: self.pars.sym_asy,
                TBSL.TREATMENT:    self.pars.sym_treat,
                TBSL.DEAD:         self.pars.sym_dead * self.rr_death[u],
            }, rng=self._rng_sym)

        u = ss.uids(self.state == TBSL.TREATMENT)
        if len(u):
            self.transition(u, to={
                TBSL.SYMPTOMATIC: self.pars.fail_rate,
                TBSL.TREATED:     self.pars.complete_rate,
            }, rng=self._rng_trt)

        # --- Bookkeep from current state ---

        self.infected[:] = ~np.isin(self.state,
            [TBSL.SUSCEPTIBLE, TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED, TBSL.DEAD])
        self.susceptible[:] = np.isin(self.state,
            [TBSL.SUSCEPTIBLE, TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])
        self.on_treatment[:] = (self.state == TBSL.TREATMENT)

        dead = ss.uids((self.state == TBSL.DEAD) & self.sim.people.alive)
        self.sim.people.request_death(dead)
        self.results['new_deaths'][self.ti] = len(dead)
        self.results['new_deaths_15+'][self.ti] = np.count_nonzero(self.sim.people.age[dead] >= 15)

        self.rr_activation[:] = 1
        self.rr_clearance[:] = 1
        self.rr_death[:] = 1

        self.rel_sus[:] = 1
        self.rel_sus[self.state == TBSL.RECOVERED] = self.pars.rr_rec
        self.rel_sus[self.state == TBSL.TREATED] = self.pars.rr_treat
        self.rel_trans[:] = 1
        self.rel_trans[self.state == TBSL.ACUTE] = self.pars.trans_acute
        self.rel_trans[self.state == TBSL.ASYMPTOMATIC] = self.pars.trans_asymp

        return

    def start_treatment(self, uids):
        """ACUTE or INFECTION -> CLEARED; active -> TREATMENT."""
        if len(uids) == 0:
            return

        # ACUTE or INFECTION: clear
        latent = uids[(self.state[uids] == TBSL.ACUTE) | (self.state[uids] == TBSL.INFECTION)]
        self.state[latent] = TBSL.CLEARED
        self.infected[latent] = False
        self.susceptible[latent] = True

        # Active TB: put on treatment
        active = uids[np.isin(self.state[uids], [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state[active] = TBSL.TREATMENT
        self.on_treatment[active] = True

        self.results['new_notifications_15+'][self.ti] += np.count_nonzero(self.sim.people.age[active] >= 15)

        return


def get_tb(sim, which=None): # TODO: Create tbsim.Sim and move this to sim.get_tb()
    """ Helper to get the TB_LSHTM infection module from a sim
    
    Args:
        sim (Sim): the simulation to search for the TB module
        which (type, optional): the class of TB module to get (e.g. TB_LSHTM; if None, returns the first BaseTB subclass found
    """
    if which is None:
        which = BaseTB
    for disease in sim.diseases:
        if isinstance(disease, which):
            return disease
    raise ValueError("No TB module found in sim.diseases")
