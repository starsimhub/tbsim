"""TB natural history model. State definitions and transition diagram are in the API docs (tbsim.tb)."""

from enum import IntEnum

import numpy as np
import starsim as ss
from .plots import plot as _tbsim_plot
from .resistance.strains import StrainRegistry
from .resistance.profile import StrainProfile
from .resistance.resolvers import ProgressionResolver, AcquisitionResolver


__all__ = ['TB', 'MultiStrainTB', 'TBS', 'get_tb', 'choice2d']


class TBS(IntEnum):
    """
    TB state labels.

    - Each agent is in exactly one of these states.
    - Transitions are driven by exponential rates in `TB`.
    """
    SUSCEPTIBLE     = -1    # Never infected (agents who clear/recover/treat remain in their last state, not here)
    INFECTION       = 0     # Latent infection (not yet active TB)
    CLEARED         = 1     # Post-infection: cleared latent, recovered from non-infectious, or completed treatment
    NON_INFECTIOUS  = 2     # Non-infectious TB (early/smear-negative)
    ASYMPTOMATIC    = 4     # Active TB, asymptomatic (infectious)
    SYMPTOMATIC     = 5     # Active TB, symptomatic (infectious)
    TREATMENT       = 6     # On TB treatment
    DEAD            = 8     # Dead (TB-caused via sym_dead; general mortality via step_die also sets this)
    REMOVED         = 10    # Removed from the active population (e.g. emigration)

    @staticmethod
    def active_tb_states():
        """States representing active TB disease (non-infectious, asymptomatic, symptomatic)."""
        return [TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC, TBS.SYMPTOMATIC]

    @staticmethod
    def care_seeking_eligible():
        """States eligible for care-seeking: only SYMPTOMATIC.
        Only individuals with clinical symptoms (cough, fever, night sweats, etc.)
        recognise their illness and seek healthcare."""
        return np.array([TBS.SYMPTOMATIC])

    @staticmethod
    def terminal_states():
        """States that should no longer participate in transmission or care flows."""
        return [TBS.DEAD, TBS.REMOVED]


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
    Reinfectable state (`TBS.CLEARED`) uses per-agent `rr_reinfection`, set on entry from each
    pathway (`rr_reinfection_cleared`, `rr_reinfection_rec`, `rr_reinfection_treat`). Per-agent modifiers
    ``rr_activation``, ``rr_clearance``, ``rr_death`` scale selected rates.

    Args (pars):
        *Transmission and reinfection*

        - ``init_prev``:   Initial seed infections (prevalence).
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
        """Initialize with default natural history parameters; override via ``pars``.

        Args:
            pars (dict): Natural history parameter overrides.
        """
        if 'strains' in kwargs:
            raise TypeError('Use MultiStrainTB for strain-aware TB simulations.')
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        self.define_pars(
            init_prev=ss.bernoulli(0.05),       # Initial seed infections (prevalence)
            beta=ss.permonth(0.2),              # Transmission rate per month
            trans_asymp=0.82,                   # κ kappa: rel. transmissibility asymptomatic vs symptomatic
            rr_reinfection_rec=0.21,            # π pi: RR reinfection after NON_INFECTIOUS → CLEARED
            rr_reinfection_treat=3.15,          # ρ rho: RR reinfection after TREATMENT → CLEARED
            rr_reinfection_cleared=1.0,         # RR reinfection after INFECTION → CLEARED (latent cleared); also applies to agents cleared via TPT sterilization
            dur_reinfection_protection=None,    # Distribution of protection duration; None = never wanes
            # --- From INFECTION (latent) ---
            inf_cle=ss.peryear(1.90),            # Clear infection (no active TB)
            inf_non=ss.peryear(0.16),            # Progress to non-infectious TB
            inf_asy=ss.peryear(0.06),            # Progress to asymptomatic active TB
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

        self._rng_inf = ss.random(name='tb_rng_inf')
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
            ss.FloatArr('ti_asymp', default=np.nan),
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

    @property
    def infectious(self):
        """
        Boolean array: True for agents who can transmit TB.

        In this model only ASYMPTOMATIC and SYMPTOMATIC states are infectious.
        Used by the base `starsim.Infection` for transmission.
        """
        return (self.state == TBS.ASYMPTOMATIC) | (self.state == TBS.SYMPTOMATIC)

    def set_prognoses(self, uids, sources=None):
        """
        Set prognoses for newly infected agents (called when transmission occurs).
        """
        super().set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti
        self.state[uids] = TBS.INFECTION
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
        2. **Reset RR multipliers** for all agents (interventions set fresh each step).
        3. **Evaluate transitions**: for each state group, evaluate competing-risk
           transitions and apply immediately. Agents may cascade through multiple
           states in one step.
        4. **Bookkeeping**: update flags, modifiers, deaths, results.
        """
        super().step()

        # --- Evaluate transitions (each mutates self.state in place) ---
        # For transitions that lead to CLEARED, we snapshot the pre-transition CLEARED mask
        # and compare after to identify agents newly entering CLEARED from each source state,
        # so we can assign the correct pathway-specific rr_reinfection to each new entrant.

        u = ss.uids(self.state == TBS.INFECTION)
        if len(u):
            self.transition(u, to={
                TBS.CLEARED:        self.pars.inf_cle,
                TBS.NON_INFECTIOUS: self.pars.inf_non * self.rr_activation[u],
                TBS.ASYMPTOMATIC:   self.pars.inf_asy * self.rr_activation[u],
            }, rng=self._rng_inf)
            newly_cleared = u[self.state[u] == TBS.CLEARED]
            self.rr_reinfection[newly_cleared] = self.pars.rr_reinfection_cleared
            if self.pars.dur_reinfection_protection is not None and len(newly_cleared):
                self.ti_rr_reinfection_wane[newly_cleared] = self.ti + self.pars.dur_reinfection_protection.rvs(newly_cleared)

        u = ss.uids(self.state == TBS.NON_INFECTIOUS)
        if len(u):
            self.transition(u, to={
                TBS.CLEARED:      self.pars.non_rec * self.rr_clearance[u],
                TBS.ASYMPTOMATIC: self.pars.non_asy,
            }, rng=self._rng_non)
            newly_cleared = u[self.state[u] == TBS.CLEARED]
            self.rr_reinfection[newly_cleared] = self.pars.rr_reinfection_rec
            if self.pars.dur_reinfection_protection is not None and len(newly_cleared):
                self.ti_rr_reinfection_wane[newly_cleared] = self.ti + self.pars.dur_reinfection_protection.rvs(newly_cleared)

        u = ss.uids(self.state == TBS.ASYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBS.NON_INFECTIOUS: self.pars.asy_non,
                TBS.SYMPTOMATIC:    self.pars.asy_sym,
            }, rng=self._rng_asy)

        u = ss.uids(self.state == TBS.SYMPTOMATIC)
        if len(u):
            self.transition(u, to={
                TBS.ASYMPTOMATIC: self.pars.sym_asy,
                TBS.DEAD:         self.pars.sym_dead * self.rr_death[u],
            }, rng=self._rng_sym)

        # NOTE: TREATMENT outcomes (success → CLEARED, failure → SYMPTOMATIC) are
        # handled by TxDelivery, not the natural history. Agents in TREATMENT state
        # without a TxDelivery intervention will remain in TREATMENT indefinitely.

        # --- Bookkeep from current state ---

        self.infected[:] = ~np.isin(self.state,
            [TBS.SUSCEPTIBLE, TBS.CLEARED, *TBS.terminal_states()])
        self.susceptible[:] = np.isin(self.state,
            [TBS.SUSCEPTIBLE, TBS.CLEARED])
        self.on_treatment[:] = (self.state == TBS.TREATMENT)

        # TB deaths
        dead = ss.uids((self.state == TBS.DEAD) & self.sim.people.alive)
        self.sim.people.request_death(dead)
        self.results['new_deaths'][self.ti] = len(dead)
        self.results['new_deaths_15+'][self.ti] = np.count_nonzero(self.sim.people.age[dead] >= 15)

        # Reset rr_* (interventions set fresh values next step)
        self.rr_activation[:] = 1
        self.rr_clearance[:] = 1
        self.rr_death[:] = 1

        # rel_sus / rel_trans
        # rel_sus is reset to 1 for all agents first; other modules can then *= their own factors
        self.rel_sus[:] = 1
        cleared = ss.uids(self.state == TBS.CLEARED)
        if self.pars.dur_reinfection_protection is not None and len(cleared):
            # Waning: agents whose protection period has elapsed revert to full susceptibility
            waned = cleared[self.ti >= self.ti_rr_reinfection_wane[cleared]]
            self.rr_reinfection[waned] = 1.0
            self.ti_rr_reinfection_wane[waned] = np.inf
        self.rel_sus[cleared] *= self.rr_reinfection[cleared]
        self.rel_trans[:] = 1
        self.rel_trans[self.state == TBS.ASYMPTOMATIC] = self.pars.trans_asymp

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
        infectious = self.infectious
        n_alive = self.sim.people.alive.count()
        new_asymp = self.ti_asymp == ti

        in_state = {}
        for state in TBS:
            in_state[state] = self.state == state
            res[f'n_{state.name}'][ti] = in_state[state].count()
            res[f'n_{state.name}_15+'][ti] = (age15 & in_state[state]).count()

        res.n_infectious[ti] = infectious.count()
        res['n_infectious_15+'][ti] = (infectious & age15).count()
        res.prevalence_active[ti] = res.n_infectious[ti] / n_alive if n_alive else 0
        res.incidence_kpy[ti] = 1_000 * (self.ti_infected == ti).count() / (n_alive * dty) if n_alive else 0
        res.deaths_ppy[ti] = res.new_deaths[ti] / (n_alive * dty) if n_alive else 0

        # New active: agents whose ti_asymp == this step
        res['new_active'][ti] = new_asymp.count()
        res['new_active_15+'][ti] = (new_asymp & age15).count()
        res['n_detectable_15+'][ti] = (age15 * (in_state[TBS.SYMPTOMATIC] + self.pars.cxr_asymp_sens*in_state[TBS.ASYMPTOMATIC])).sum()
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


class MultiStrainTB(TB):
    """TB natural history with a multi-strain drug-resistance overlay."""

    _strain_kw = ('progression_mode', 'p_multi', 'p_random_acquisition',
                  'alpha_super', 'alpha_act')

    def __init__(self, pars=None, strains=None, **kwargs):
        """Initialize the strain-aware TB model.

        Args:
            pars (dict): Natural history parameter overrides.
            strains (list/StrainRegistry): Strain configuration.
            progression_mode (str): Progression bottleneck mode. Default ``'bottleneck'``.
            p_multi (float): Multi-strain retention probability at activation. Default 1.0.
            p_random_acquisition (dict/None): Per-drug random acquisition probabilities.
            alpha_super (float/None): Superinfection susceptibility in ``INFECTION``.
            alpha_act (dict/None): Per-state superinfection factors for active disease.
        """
        if strains is None:
            raise ValueError('MultiStrainTB requires a strain configuration.')
        strain_kwargs = {}
        for key in self._strain_kw:
            if key in kwargs:
                strain_kwargs[key] = kwargs.pop(key)
        kwargs.setdefault('name', 'tb')
        super().__init__(pars=pars, **kwargs)
        self._init_strains(strains, **strain_kwargs)
        return

    def _init_strains(self, strains, **kwargs):
        """Initialize strain registry, per-agent strain states, and resolvers."""
        if isinstance(strains, StrainRegistry):
            registry = strains
        else:
            registry = StrainRegistry(list(strains))
        self._strain_registry = registry
        self.strain_profile = StrainProfile(registry)
        self._rng_strain_pick = ss.random(name='tb_rng_strain_pick')
        self._rng_strain_init = ss.random(name='tb_rng_strain_init')
        self._n_duplicate_blocked_this_step = 0
        self._progression_resolver = ProgressionResolver(
            mode=kwargs.pop('progression_mode', 'bottleneck'),
            p_multi=kwargs.pop('p_multi', 1.0),
        )
        self._acquisition_resolver = AcquisitionResolver(
            p_random=kwargs.pop('p_random_acquisition', None),
        )
        self._alpha_super_input = kwargs.pop('alpha_super', None)
        self._alpha_act_input = dict(kwargs.pop('alpha_act', {}) or {})
        self._finalize_alpha_defaults()
        self.define_states(*self.strain_profile.state_defs())
        self.strain_profile.attach(self)
        return

    def _finalize_alpha_defaults(self):
        """Set ``self._alpha_super`` and ``self._alpha_act`` from stored inputs."""
        rr_rec = float(self.pars.rr_reinfection_rec)
        self._alpha_super = float(
            self._alpha_super_input if self._alpha_super_input is not None else rr_rec
        )
        self._alpha_act = dict(self._alpha_act_input)
        self._alpha_act.setdefault('non_infectious', self._alpha_super)
        self._alpha_act.setdefault('asymptomatic',   0.0)
        self._alpha_act.setdefault('symptomatic',    0.0)
        return

    def infect(self):
        """Run transmission with alpha-scaled superinfection targets."""
        per_state_alpha = (
            (TBS.INFECTION,      self._alpha_super),
            (TBS.NON_INFECTIOUS, self._alpha_act['non_infectious']),
            (TBS.ASYMPTOMATIC,   self._alpha_act['asymptomatic']),
            (TBS.SYMPTOMATIC,    self._alpha_act['symptomatic']),
        )
        chunks_uids = []
        chunks_alpha = []
        for tbs_val, alpha in per_state_alpha:
            if alpha <= 0:
                continue
            state_uids = (self.state == tbs_val).uids
            if len(state_uids) == 0:
                continue
            chunks_uids.append(state_uids)
            chunks_alpha.append(np.full(len(state_uids), float(alpha)))

        if not chunks_uids:
            return super().infect()

        elig_uids = ss.uids.concatenate(chunks_uids)
        alphas    = np.concatenate(chunks_alpha)
        orig_sus     = np.asarray(self.susceptible[elig_uids], dtype=bool).copy()
        orig_rel_sus = np.asarray(self.rel_sus[elig_uids], dtype=float).copy()
        try:
            self.susceptible[elig_uids] = True
            self.rel_sus[elig_uids] = orig_rel_sus * alphas
            return super().infect()
        finally:
            self.susceptible[elig_uids] = orig_sus
            self.rel_sus[elig_uids] = orig_rel_sus

    def set_prognoses(self, uids, sources=None):
        """Set prognoses for new infections and assign transmitted strains."""
        super(TB, self).set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        susceptible_uids = self.susceptible.uids
        cleared_uids     = (self.state == TBS.CLEARED).uids
        new_uids = uids.intersect(susceptible_uids.union(cleared_uids))

        if len(new_uids):
            self.susceptible[new_uids] = False
            self.infected[new_uids] = True
            self.ever_infected[new_uids] = True
            self.ti_infected[new_uids] = self.ti
            self.state[new_uids] = TBS.INFECTION

        self._assign_transmitted_strains(uids, sources)
        return

    def transition(self, uids, to, rng):
        """Apply TB transition, then apply strain progression hooks."""
        if len(uids) == 0:
            return super().transition(uids, to, rng)

        from_state = int(self.state[uids[0]])
        out = super().transition(uids, to, rng)

        if from_state == int(TBS.INFECTION):
            newly_cleared = uids[self.state[uids] == TBS.CLEARED]
            if len(newly_cleared):
                self.strain_profile.clear_all(newly_cleared)
            progressing = uids[np.isin(self.state[uids], [TBS.NON_INFECTIOUS, TBS.ASYMPTOMATIC])]
            self._apply_strain_progression(progressing, activating_only=False)
        elif from_state == int(TBS.NON_INFECTIOUS):
            newly_cleared = uids[self.state[uids] == TBS.CLEARED]
            if len(newly_cleared):
                self.strain_profile.clear_all(newly_cleared)
            progressing = uids[self.state[uids] == TBS.ASYMPTOMATIC]
            self._apply_strain_progression(progressing, activating_only=True)

        return out

    def step(self):
        """Reset strain counters, then advance the TB state machine."""
        self._n_duplicate_blocked_this_step = 0
        return super().step()

    def step_die(self, uids):
        """Apply TB death handling and clear carried strains."""
        out = super().step_die(uids)
        if len(uids):
            self.strain_profile.clear_all(uids)
        return out

    def _apply_strain_progression(self, uids, activating_only):
        """Apply strain progression rules after latent/active transitions.

        Args:
            uids            (ss.uids): Agents that just transitioned out of
                ``INFECTION`` or ``NON_INFECTIOUS``.
            activating_only (bool): If ``True``, run the activation bottleneck
                on all ``uids`` and skip random acquisition. If ``False``,
                run random acquisition on all ``uids`` and run the bottleneck
                only on agents now in ``ASYMPTOMATIC``.
        """
        if len(uids) == 0:
            return
        if activating_only:
            activating = uids
        else:
            activating = uids[self.state[uids] == TBS.ASYMPTOMATIC]
        if self._progression_resolver is not None and len(activating):
            self._progression_resolver.resolve(self.strain_profile, activating)
        if self._acquisition_resolver is not None and not activating_only:
            self._acquisition_resolver.random_acquisition(self.strain_profile, uids)
        return

    def _assign_transmitted_strains(self, uids, sources):
        """Assign strain identity to newly infected ``uids``."""
        profile = self.strain_profile
        registry = self._strain_registry
        n = len(uids)
        picks = np.full(n, -1, dtype=int)

        if sources is not None and not np.isscalar(sources):
            try:
                src_arr = np.asarray(sources, dtype=int)
            except (TypeError, ValueError):
                src_arr = None
            if src_arr is not None and src_arr.ndim == 1 and src_arr.size == n:
                real = src_arr >= 0
                if real.any():
                    src_uids = ss.uids(src_arr[real])
                    src_picks = profile.sample_transmitted_strain(
                        src_uids, self._rng_strain_pick,
                    )
                    picks[real] = src_picks

        need_fallback = picks < 0
        if need_fallback.any():
            init_prev = registry.init_prev
            if init_prev.sum() > 0:
                p = init_prev / init_prev.sum()
                u = np.asarray(
                    self._rng_strain_init.rvs(int(need_fallback.sum())), dtype=float
                )
                cdf = np.cumsum(p)
                fallback_picks = (u[:, None] < cdf).argmax(axis=1)
            else:
                fallback_picks = np.zeros(int(need_fallback.sum()), dtype=int)
            picks[need_fallback] = fallback_picks

        n_blocked = 0
        for idx in range(registry.n):
            sel = picks == idx
            if not sel.any():
                continue
            target = uids[sel]
            arr = getattr(self, profile.names[idx])
            already = np.asarray(arr[target], dtype=bool)
            n_blocked += int(already.sum())
            new = target[~already]
            if len(new):
                arr[new] = True
        self._n_duplicate_blocked_this_step += n_blocked
        return

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