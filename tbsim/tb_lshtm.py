"""LSHTM TB natural history model. State definitions and transition diagram are in the API docs (tbsim.tb_lshtm)."""

import numpy as np
import starsim as ss
import matplotlib.pyplot as plt

from enum import IntEnum

__all__ = ['TB_LSHTM', 'TB_LSHTM_Acute', 'TBSL']


class TBSL(IntEnum):
    """
    TB state labels for the LSHTM model.

    - Each agent is in exactly one of these states.
    - Transitions are driven by exponential rates in :class:`TB_LSHTM`
      (and :class:`TB_LSHTM_Acute`).
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


def _get_rate_from_base(base_rate):
    # ss.expon stores scale=1/rate; try kwds first, fall back to positional args
    try:
        dist = base_rate.dist
        scale = dist.kwds.get('scale') or dist.args[0]
        return 1.0 / scale
    except Exception as e:
        raise ValueError(f'Cannot extract rate from base_rate: {base_rate!r}') from e


class ScaledRate:
    """A rate distribution scaled per-agent by a risk-ratio callable."""

    def __init__(self, base_rate, rr_callable):
        self._base_rate = base_rate
        self._rr_callable = rr_callable

    def sample_waiting_times(self, uids):
        # Per-agent rate ratio from callable (e.g. rr_activation[uids]); rr_i=0 => never transition
        rr = np.asarray(self._rr_callable(uids), dtype=float)
        # Base rate = 1/scale from expon; effective_rate_i = base * rr_i
        base_rate_val = _get_rate_from_base(self._base_rate)
        effective_rate = base_rate_val * rr
        # Inverse-transform: u ~ U(0,1) => T = -log(u)/λ ~ Exp(λ)
        u = self._base_rate.rand(len(uids))
        # rr=0 => div by 0; set t=inf (never transition), suppress warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(rr > 0, -np.log(u) / effective_rate, np.inf)
        return t


def make_scaled_rate(base_rate, rr_callable):
    return ScaledRate(base_rate, rr_callable)


class TB_LSHTM(ss.Infection):
    #region IDE Collapsable section for documentation
    """
    Agent-based TB natural history adapting the LSHTM compartmental structure [1] (Schwalb et al. 2025). 
    States in :class:`TBSL` span the spectrum from susceptibility to active disease and treatment.  
    Infectious states are :class:`TBSL.ASYMPTOMATIC` and :class:`TBSL.SYMPTOMATIC`; the force
    of infection depends on :attr:`pars.beta` and the prevalence of those states, with
    :attr:`pars.trans_asymp` (κ kappa) giving the relative infectiousness of asymptomatic vs symptomatic TB.
    Reinfectable states (:class:`TBSL.CLEARED`, :class:`TBSL.RECOVERED`, :class:`TBSL.TREATED`) use
    :attr:`pars.rr_rec` (π pi) and :attr:`pars.rr_treat` (ρ rho). Per-agent modifiers ``rr_activation``, ``rr_clearance``,
    ``rr_death`` scale selected rates. Interventions call :meth:`start_treatment`.

    Parameters (pars)
    -----------------
    *Transmission and reinfection*

    - ``init_prev``:   Initial seed infections (prevalence). 
    - ``beta``:        Transmission rate per year. 
    - ``trans_asymp``: Relative transmissibility, asymptomatic vs symptomatic. (κ kappa) 
    - ``rr_rec``:      Relative risk of reinfection for RECOVERED.  (π pi) 
    - ``rr_treat``:    Relative risk of reinfection for TREATED. (ρ rho) 

    *From INFECTION (latent)*

    - ``inf_cle``:     Infection → Cleared (no active TB). 
    - ``inf_non``:     Infection → Non-infectious TB. 
    - ``inf_asy``:     Infection → Asymptomatic TB. 

    *From NON_INFECTIOUS*

    - ``non_rec``:    Non-infectious → Recovered. 
    - ``non_asy``:    Non-infectious → Asymptomatic.

    *From ASYMPTOMATIC*

    - ``asy_non``:    Asymptomatic → Non-infectious.
    - ``asy_sym``:    Asymptomatic → Symptomatic.

    *From SYMPTOMATIC*

    - ``sym_asy``:     Symptomatic → Asymptomatic. 
    - ``sym_treat``:   Symptomatic → Treatment. (θ theta)
    - ``sym_dead``:    Symptomatic → Dead (TB-specific mortality, μ_TB).

    *From TREATMENT*

    - ``fail_rate``:     Treatment → Symptomatic (failure). (φ phi) 
    - ``complete_rate``: Treatment → Treated (completion).  (δ delta) 

    *Background (general mortality is handled by* ``ss.Deaths`` *demographics, not this module)*

    - ``cxr_asymp_sens``:   CXR sensitivity for screening asymptomatic (0–1). 

    Agent states (array-like, one per agent)
    ---------------------------------------

    *Infection flags*

    - ``susceptible``    (BoolState, default=True):   Whether the agent is susceptible to TB.
    - ``infected``       (BoolState, default=False):  Whether the agent is currently infected.
    - ``ever_infected``  (BoolState, default=False):  Whether the agent has ever been infected.
    - ``on_treatment``   (BoolState, default=False):  Whether the agent is on TB treatment.

    *TB state machine*

    - ``state``          (FloatArr, default=TBSL.SUSCEPTIBLE):  Current TB state (:class:`TBSL` value).
    - ``state_next``     (FloatArr, default=TBSL.INFECTION):    Scheduled next state.
    - ``ti_next``        (FloatArr, default=inf):               Time of next transition.
    - ``ti_infected``    (FloatArr, default=-inf):              Time of infection (never infected = -inf).

    *Transmission modifiers*

    - ``rel_sus``        (FloatArr, default=1.0):  Relative susceptibility to TB.
    - ``rel_trans``      (FloatArr, default=1.0):  Relative transmissibility of TB.

    *Per-agent risk modifiers*

    - ``rr_activation``  (FloatArr, default=1.0):  Multiplier on INFECTION → NON_INFECTIOUS / ASYMPTOMATIC.
    - ``rr_clearance``   (FloatArr, default=1.0):  Multiplier on NON_INFECTIOUS → RECOVERED.
    - ``rr_death``       (FloatArr, default=1.0):  Multiplier on SYMPTOMATIC → DEAD.

    References
    ----------
    .. [1] Schwalb et al. (2025) Potential impact, costs, and benefits of population-wide
       screening interventions for tuberculosis in Viet Nam. PLOS Glob Public Health.
       https://doi.org/10.1371/journal.pgph.0005050

    """
    #endregion Documentation

    def __init__(self, pars=None, **kwargs):
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        # --- Transmission and reinfection ---
        self.define_pars(
            init_prev=ss.bernoulli(0.01),       # Initial seed infections (prevalence)
            beta=ss.peryear(0.25),              # Transmission rate per year
            trans_asymp=0.82,                   # κ kappa: rel. transmissibility asymptomatic vs symptomatic
            rr_rec=0.21,                        # π pi: RR reinfection after recovery
            rr_treat=3.15,                      # ρ rho: RR reinfection after treatment
            # --- From INFECTION (latent) ---
            inf_cle=ss.years(ss.expon(1/1.90)),  # Clear infection (no active TB)
            inf_non=ss.years(ss.expon(1/0.16)),  # Progress to non-infectious TB
            inf_asy=ss.years(ss.expon(1/0.06)),  # Progress to asymptomatic active TB
            # --- From NON_INFECTIOUS ---
            non_rec=ss.years(ss.expon(1/0.18)),  # Recover (→ RECOVERED)
            non_asy=ss.years(ss.expon(1/0.25)),  # Progress to asymptomatic
            # --- From ASYMPTOMATIC ---
            asy_non=ss.years(ss.expon(1/1.66)),  # Revert to non-infectious
            asy_sym=ss.years(ss.expon(1/0.88)),  # Progress to symptomatic
            # --- From SYMPTOMATIC ---
            sym_asy=ss.years(ss.expon(1/0.54)),  # Regress to asymptomatic (still active, not recovered)
            sym_treat=ss.years(ss.expon(1/0.46)),   # θ theta: symptomatic → treatment
            sym_dead=ss.years(ss.expon(1/0.34)),    # μ_TB: symptomatic → dead (TB mortality)
            # --- From TREATMENT ---
            fail_rate=ss.years(ss.expon(1/0.63)),   # φ phi: treatment failure (→ symptomatic)
            complete_rate=ss.years(ss.expon(1/2.00)),  # δ delta: treatment completion (→ treated)
            # --- Background ---
            cxr_asymp_sens=1.0,                 # CXR sensitivity for screening asymptomatic (0–1)
            # --- For ACUTE ---
            rate_acute_latent=None,             # ACUTE → INFECTION
            trans_acute=None,                   # α alpha: rel. transmissibility acute vs symptomatic
        )
        self.update_pars(pars, **kwargs)

        # Per-agent state: redefine base Infection states and add TB-specific ones
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('infected'),
            ss.FloatArr('rel_sus', default=1.0),                 # Relative susceptibility to TB (default 1.0)
            ss.FloatArr('rel_trans', default=1.0),               # Relative transmissibility of TB (default 1.0)
            ss.FloatArr('ti_infected', default=-np.inf),
            ss.FloatArr('state', default=TBSL.SUSCEPTIBLE),       # Current TB state
            ss.FloatArr('state_next', default=TBSL.INFECTION),   # Scheduled next state
            ss.FloatArr('ti_next', default=np.inf),              # Time of next transition
            ss.BoolState('on_treatment', default=False),
            ss.BoolState('ever_infected', default=False),
            # Risk modifiers
            ss.FloatArr('rr_activation', default=1.0),           # Multiplier on infection-to-active rate (INFECTION → NON_INFECTIOUS, ASYMPTOMATIC)
            ss.FloatArr('rr_clearance', default=1.0),            # Multiplier on active-to-clearance rate (NON_INFECTIOUS → RECOVERED)
            ss.FloatArr('rr_death', default=1.0),                # Multiplier on active-to-death rate (SYMPTOMATIC → DEAD)
            reset=True,  # Replace base Infection states so we can set ti_infected default
        )

        return

    @property
    def infectious(self):
        """
        Boolean array: True for agents who can transmit TB.

        In this model only ASYMPTOMATIC and SYMPTOMATIC states are infectious.
        Used by the base :class:`starsim.Infection` for transmission.
        """
        return (self.state == TBSL.ASYMPTOMATIC) | (self.state == TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for newly infected agents (called when transmission occurs).

        The base :class:`starsim.Infection` calls this when a susceptible agent
        acquires infection. We:

        - Mark agents infected and set state to INFECTION (latent)
        - Schedule the first competing transition: CLEARED, NON_INFECTIOUS, or ASYMPTOMATIC
        - Leave the actual transition to :meth:`step` when ``ti >= ti_next``

        Parameters
        ----------
        uids : array-like
            Agent indices that have just been infected.
        sources : array-like or int, optional
            Source agents (base class uses for seeding; -1 for initial cases).
        """
        super().set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        # Mark as infected and record infection time
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti

        self.state[uids] = TBSL.INFECTION

        # Schedule first transition: latent → cleared, non-infectious, or asymptomatic
        # rr_activation multiplies the infection-to-active rates (inf_non, inf_asy).
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.CLEARED: self.pars.inf_cle,
            TBSL.NON_INFECTIOUS: make_scaled_rate(self.pars.inf_non, lambda uids: self.rr_activation[uids]),
            TBSL.ASYMPTOMATIC: make_scaled_rate(self.pars.inf_asy, lambda uids: self.rr_activation[uids]),
        })

        return

    def transition(self, uids, to):
        """
        Sample competing exponential transitions and return next state and time.

        For each agent in `uids`, draws one exponential waiting time per
        destination in `to`; the destination with the minimum time is chosen
        (competing risks). So each agent has exactly one next state and one
        transition time.

        Parameters
        ----------
        uids : array-like
            Agent indices to transition.
        to : dict
            Mapping from TBSL state -> rate-like. Keys are possible next states;
            values provide waiting times per agent via the object returned by :func:`make_scaled_rate` (``.sample_waiting_times(uids)``)
            or (Starsim dists) ``.rvs(uids)``.

        Returns
        -------
        state_next : np.ndarray
            Next state for each agent (one of the keys of `to`).
        ti_next : np.ndarray
            Simulated time of transition for each agent.

        Notes
        -----
        Competing risks: if an agent can go to A (rate λ_A) or B (rate λ_B):

        - Sample T_A ~ Exp(λ_A) and T_B ~ Exp(λ_B)
        - Agent goes to the state whose time is smallest
        - Equivalent to the usual competing-exponential formulation
        """
        if len(uids) == 0:
            return np.array([]), np.array([])

        ti = self.ti
        # One row per possible destination; column j = transition time for uids[j]
        ti_state = np.zeros((len(to), len(uids)))
        for idx, rate in enumerate(to.values()):
            draw = rate.sample_waiting_times(uids) if isinstance(rate, ScaledRate) else rate.rvs(uids)
            ti_state[idx, :] = ti + draw

        # Soonest transition wins (index into keys gives next state)
        state_next_idx = ti_state.argmin(axis=0)
        state_next = np.array(list(to.keys()))[state_next_idx]
        ti_next = ti_state.min(axis=0)

        return state_next, ti_next

    def step(self):
        """
        Advance TB state for all agents whose scheduled transition time has been reached.

        Order of operations (important for correctness):

        - Find agents with ti >= ti_next (due for a transition).
        - Record outcomes from *state_next* (new active, deaths) before overwriting state.
        - Apply the scheduled transition: state = state_next; set susceptible/infected
          and on_treatment from the *new* state.
        - Process TB deaths (request_death) so they leave the population.
        - Set rel_sus and rel_trans for the *new* state (used by base for transmission).
        - For each current state with spontaneous transitions, call transition() to
          schedule the next (state_next, ti_next). CLEARED/RECOVERED/TREATED have
          no scheduled transition until reinfected (transmission).

        Reinfection is not done here; the base class handles transmission and
        calls set_prognoses for newly infected agents.
        """
        super().step()
        ti = self.ti

        # Agents whose scheduled transition time has arrived (ti_next <= ti)
        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return

        # --- Record outcomes from state_next *before* we overwrite state ---
        # Only count first entry to ASYMPTOMATIC from non-active states; exclude SYMPTOMATIC → ASYMPTOMATIC (regression)
        new_asymp_uids = uids[
            (self.state_next[uids] == TBSL.ASYMPTOMATIC) &
            (self.state[uids] != TBSL.SYMPTOMATIC)
        ]
        self.results['new_active'][ti] = len(new_asymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_asymp_uids] >= 15)

        # Base class uses infected/susceptible for transmission; keep in sync with state
        new_inf_uids = uids[self.state_next[uids] == TBSL.INFECTION]
        self.infected[new_inf_uids] = True
        new_clr_uids = uids[np.isin(self.state_next[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])]
        self.infected[new_clr_uids] = False

        # --- Apply the scheduled transition ---
        self.state[uids] = self.state_next[uids]
        self.ti_next[uids] = np.inf  # Clear until we set next transition below (avoids double-firing)
        self.on_treatment[uids] = (self.state[uids] == TBSL.TREATMENT)

        # Only these states are susceptible to (re)infection; transmission handles the rest
        self.susceptible[uids] = np.isin(self.state[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])

        # --- TB deaths: request removal from population this step ---
        new_death_uids = uids[self.state_next[uids] == TBSL.DEAD]
        self.sim.people.request_death(new_death_uids)
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # rel_sus: multiplier for susceptibility; RECOVERED/TREATED at higher risk (π pi, ρ rho)
        self.rel_sus[uids] = 1
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.rr_rec
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rr_treat

        # rel_trans: multiplier for infectiousness; ASYMPTOMATIC < SYMPTOMATIC (κ kappa typically < 1)
        self.rel_trans[uids] = 1
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.trans_asymp

        # --- Schedule next transition: only states with spontaneous exits get a new (state_next, ti_next) ---
        # rr_activation: INFECTION → NON_INFECTIOUS or ASYMPTOMATIC (infection-to-active)
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.inf_cle,
            TBSL.NON_INFECTIOUS: make_scaled_rate(self.pars.inf_non, lambda uids: self.rr_activation[uids]),
            TBSL.ASYMPTOMATIC: make_scaled_rate(self.pars.inf_asy, lambda uids: self.rr_activation[uids]),
        })

        # rr_clearance: NON_INFECTIOUS → RECOVERED (non-infectious-to-recovered)
        u = uids[self.state[uids] == TBSL.NON_INFECTIOUS]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: make_scaled_rate(self.pars.non_rec, lambda uids: self.rr_clearance[uids]),
            TBSL.ASYMPTOMATIC: self.pars.non_asy,
        })

        # ASYMPTOMATIC → NON_INFECTIOUS or SYMPTOMATIC (no rr modifier)
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.NON_INFECTIOUS: self.pars.asy_non,
            TBSL.SYMPTOMATIC: self.pars.asy_sym
        })

        # rr_death: SYMPTOMATIC → DEAD (symptomatic-to-death)
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.sym_asy,
            TBSL.TREATMENT: self.pars.sym_treat,
            TBSL.DEAD: make_scaled_rate(self.pars.sym_dead, lambda uids: self.rr_death[uids]),
        })

        # TREATMENT → SYMPTOMATIC (failure) or TREATED (completion)
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.fail_rate,
            TBSL.TREATED: self.pars.complete_rate
        })

        # CLEARED, RECOVERED, TREATED: no scheduled transition; reinfection via transmission only

        # Reset rate multipliers states to 1
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        return

    def start_treatment(self, uids):
        """
        Move specified agents onto TB treatment (or clear latent infection).

        Called by interventions (e.g. screening/case-finding) when an agent is
        identified for treatment. We set state_next and ti_next to the *current*
        time so the change takes effect on the next :meth:`step` (no extra delay).

        - Latent (INFECTION): cleared without treatment (→ CLEARED).
        - Active (NON_INFECTIOUS, ASYMPTOMATIC, SYMPTOMATIC): moved to TREATMENT.
        - Records new notifications (15+) for active cases starting treatment.

        Parameters
        ----------
        uids : array-like
            Agent indices to start treatment (or clear if latent).
        """
        if len(uids) == 0:
            return 0

        # Latent infection: clear without active disease
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u] = TBSL.CLEARED
        self.ti_next[u] = self.ti

        # Active TB (non-infectious, asymptomatic, symptomatic): put on treatment
        u = uids[np.isin(self.state[uids], [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti] += np.count_nonzero(self.sim.people.age[u] >= 15)

        return

    def step_die(self, uids):
        """
        Apply death for the given agents and update TB state so they no longer participate.

        Called by the framework when agents die (e.g. background mortality or
        other modules). We:

        - Set state to DEAD; clear susceptible and infected flags
        - Set rel_trans=0 so they do not transmit
        - Set ti_next=inf so no further TB transitions are scheduled
        """
        if len(uids) == 0:
            return

        super().step_die(uids)
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.state[uids] = TBSL.DEAD
        self.ti_next[uids] = np.inf
        self.rel_trans[uids] = 0
        return

    def init_results(self):
        """
        Define result time series (counts by state, incidence, prevalence, etc.).

        Called once when the simulation is set up. Registers:

        - Per-state counts (n_* and n_*_15+) for each TBSL state
        - Infectious counts, new/cumulative active cases and deaths
        - Prevalence, incidence per 1000 person-years, deaths per person-year
        - Notifications and detectable (symptomatic + CXR sensitivity * asymptomatic) for 15+

        Actual values are filled in :meth:`update_results` each step and
        :meth:`finalize_results` after the run.
        """
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
        """
        Record current time-step values for all result series.

        Called each time step after :meth:`step`. Fills in:

        - State counts (all ages and 15+) for each TBSL state
        - Infectious counts, prevalence (active / alive)
        - Incidence per 1000 person-years (new infections this step / person-years at risk)
        - Deaths per person-year
        - Detectable 15+ (symptomatic + CXR sens * asymptomatic, for screening)
        """
        super().update_results()
        res = self.results
        ti = self.ti
        ti_infctd = self.ti_infected
        dty = self.sim.t.dt_year

        for state in TBSL:
            res[f'n_{state.name}'][ti] = np.count_nonzero(self.state == state)
            res[f'n_{state.name}_15+'][ti] = np.count_nonzero((self.sim.people.age >= 15) & (self.state == state))

        res.n_infectious[ti] = np.count_nonzero(self.infectious)
        res['n_infectious_15+'][ti] = np.count_nonzero(self.infectious & (self.sim.people.age >= 15))
        res.prevalence_active[ti] = res.n_infectious[ti] / np.count_nonzero(self.sim.people.alive)
        # Incidence: new infections at time ti, per 1000 person-years (denom = alive * dt_year)
        res.incidence_kpy[ti] = 1_000 * np.count_nonzero(ti_infctd == ti) / (np.count_nonzero(self.sim.people.alive) * dty)
        res.deaths_ppy[ti] = res.new_deaths[ti] / (np.count_nonzero(self.sim.people.alive) * dty)

        # Detectable 15+: count symptomatic + (CXR sensitivity * asymptomatic) for screening
        res['n_detectable_15+'][ti] = np.dot(
            self.sim.people.age >= 15,
            (self.state == TBSL.SYMPTOMATIC) + self.pars.cxr_asymp_sens * (self.state == TBSL.ASYMPTOMATIC)
        )

        return

    def finalize_results(self):
        """
        Compute cumulative series from new-event series after the run.

        Called once when the simulation ends. Fills cumulative series by cumsum
        of the corresponding new_* arrays:

        - cum_deaths, cum_deaths_15+ from new_deaths, new_deaths_15+
        - cum_active, cum_active_15+ from new_active, new_active_15+

        So cum_*[t] = sum of new_* from start up to time t.
        """
        super().finalize_results()
        res = self.results
        res['cum_deaths'] = np.cumsum(res['new_deaths'])
        res['cum_deaths_15+'] = np.cumsum(res['new_deaths_15+'])
        res['cum_active'] = np.cumsum(res['new_active'])
        res['cum_active_15+'] = np.cumsum(res['new_active_15+'])
        return

    def plot(self):
        """
        Plot all result time series on one figure.

        - Plots every key in :attr:`results` except ``timevec``.
        - Uses ``timevec`` for the x-axis.

        Returns
        -------
        matplotlib.figure.Figure
        """
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

    Extends :class:`TB_LSHTM` by inserting an ACUTE state between infection and
    the usual INFECTION (latent) state. New infections enter ACUTE first, then
    transition to INFECTION at rate :attr:`pars.rate_acute_latent`. Acute cases are
    infectious with relative transmissibility :attr:`pars.trans_acute` (α alpha). All other
    states and transitions match :class:`TB_LSHTM`.

    State flow (difference from base)
    ---------------------------------
    - Transmission → ACUTE (not INFECTION).
    - From ACUTE the only spontaneous transition is ACUTE → INFECTION at rate_acute_latent.
    - Once in INFECTION, flow is as in the base (CLEARED, NON_INFECTIOUS, ASYMPTOMATIC, etc.).
    - Reinfection of CLEARED/RECOVERED/TREATED again leads to ACUTE.
    - When :meth:`start_treatment` is called, both ACUTE and INFECTION (latent)
      are cleared; active states go to TREATMENT as in the base.
    """

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the acute-variant model; adds ACUTE-state parameters.

        Parameters
        ----------
        pars : dict, optional
            Override parameters (e.g. ``rate_acute_latent``, ``trans_acute``).
        **kwargs
            Passed to base.
        """
        super().__init__(pars=pars, **kwargs)

        # ACUTE is a brief infectious state before latent (INFECTION); reinfection leads to ACUTE
        self.define_pars(
            rate_acute_latent=ss.years(ss.expon(1/4.0)),  # ACUTE → INFECTION (per year)
            trans_acute=0.9,                              # α alpha: rel. transmissibility acute vs symptomatic
        )
        self.update_pars(pars, **kwargs)
        return

    @property
    def infectious(self):
        """
        Boolean array: True for agents who can transmit TB.

        Includes ACUTE in addition to ASYMPTOMATIC and SYMPTOMATIC.
        """
        return (self.state == TBSL.ACUTE) | (self.state == TBSL.ASYMPTOMATIC) | (self.state == TBSL.SYMPTOMATIC)

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for newly infected agents; they enter ACUTE (not INFECTION).

        Skips TB_LSHTM.set_prognoses (which would set state=INFECTION) and calls
        ss.Infection.set_prognoses directly, then sets ACUTE state and schedules
        ACUTE → INFECTION transition.
        """
        super(TB_LSHTM, self).set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti

        self.state[uids] = TBSL.ACUTE

        # Single possible transition: ACUTE → INFECTION
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.INFECTION: self.pars.rate_acute_latent,
        })

        return

    def step(self):
        """
        Advance TB state; same order of operations as base :meth:`TB_LSHTM.step`.

        Advance TB state; same order of operations as base :meth:`TB_LSHTM.step`. Overrides:

        - Treat ACUTE as infectious (infected-flag and rel_trans).
        - Add ACUTE → INFECTION transition.
        - Reinfection leads to ACUTE (handled in set_prognoses).

        Calls super(TB_LSHTM, self).step() to use this class's step body
        without going through a subclass step().
        """
        super(TB_LSHTM, self).step()
        ti = self.ti

        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return

        # Record new active (transition into ASYMPTOMATIC) from state_next before overwriting
        # Exclude SYMPTOMATIC → ASYMPTOMATIC regression; only first entry to active disease counts
        new_asymp_uids = uids[
            (self.state_next[uids] == TBSL.ASYMPTOMATIC) &
            (self.state[uids] != TBSL.SYMPTOMATIC)
        ]
        self.results['new_active'][ti] = len(new_asymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_asymp_uids] >= 15)

        # Infected flag: moving *to* ACUTE means still infected; moving to CLEARED/RECOVERED/TREATED means not
        new_inf_uids = uids[self.state_next[uids] == TBSL.ACUTE]
        self.infected[new_inf_uids] = True
        new_clr_uids = uids[np.isin(self.state_next[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])]
        self.infected[new_clr_uids] = False

        self.state[uids] = self.state_next[uids]
        self.ti_next[uids] = np.inf
        self.on_treatment[uids] = (self.state[uids] == TBSL.TREATMENT)

        self.susceptible[uids] = np.isin(self.state[uids], [TBSL.CLEARED, TBSL.RECOVERED, TBSL.TREATED])

        new_death_uids = uids[self.state_next[uids] == TBSL.DEAD]
        self.sim.people.request_death(new_death_uids)
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        self.rel_sus[uids] = 1
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.rr_rec
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rr_treat

        # rel_trans: ACUTE (α alpha) and ASYMPTOMATIC (κ kappa) both infectious
        self.rel_trans[uids] = 1
        self.rel_trans[uids[self.state[uids] == TBSL.ACUTE]] = self.pars.trans_acute
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.trans_asymp

        # ACUTE → INFECTION (single possible transition from acute)
        u = uids[self.state[uids] == TBSL.ACUTE]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.INFECTION: self.pars.rate_acute_latent,
        })

        # INFECTION → CLEARED, NON_INFECTIOUS, or ASYMPTOMATIC (same as base)
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.inf_cle,
            TBSL.NON_INFECTIOUS: make_scaled_rate(self.pars.inf_non, lambda uids: self.rr_activation[uids]),
            TBSL.ASYMPTOMATIC: make_scaled_rate(self.pars.inf_asy, lambda uids: self.rr_activation[uids]),
        })

        # NON_INFECTIOUS → RECOVERED or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.NON_INFECTIOUS]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: make_scaled_rate(self.pars.non_rec, lambda uids: self.rr_clearance[uids]),
            TBSL.ASYMPTOMATIC: self.pars.non_asy,
        })

        # ASYMPTOMATIC → NON_INFECTIOUS or SYMPTOMATIC
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.NON_INFECTIOUS: self.pars.asy_non,
            TBSL.SYMPTOMATIC: self.pars.asy_sym
        })

        # SYMPTOMATIC → ASYMPTOMATIC, TREATMENT, or TB DEATH
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.sym_asy,
            TBSL.TREATMENT: self.pars.sym_treat,
            TBSL.DEAD: make_scaled_rate(self.pars.sym_dead, lambda uids: self.rr_death[uids]),
        })

        # TREATMENT → SYMPTOMATIC (failure) or TREATED (completion)
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.fail_rate,
            TBSL.TREATED: self.pars.complete_rate
        })

        # Reset rate multipliers for all agents that transitioned this step
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

        # CLEARED, RECOVERED, TREATED: next change is reinfection → ACUTE (via transmission)
        return

    def start_treatment(self, uids):
        """
        Move specified agents onto treatment (or clear); ACUTE/INFECTION → CLEARED.

        Same as base but latent includes both ACUTE and INFECTION:

        - ACUTE or INFECTION (latent): cleared without treatment (→ CLEARED).
        - Active (NON_INFECTIOUS, ASYMPTOMATIC, SYMPTOMATIC): moved to TREATMENT.
        """
        if len(uids) == 0:
            return 0

        # ACUTE or INFECTION (latent): clear infection
        u = uids[(self.state[uids] == TBSL.ACUTE) | (self.state[uids] == TBSL.INFECTION)]
        self.state_next[u] = TBSL.CLEARED
        self.ti_next[u] = self.ti

        # Active TB: put on treatment
        u = uids[np.isin(self.state[uids], [TBSL.NON_INFECTIOUS, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti]+= np.count_nonzero(self.sim.people.age[u] >= 15)

        return