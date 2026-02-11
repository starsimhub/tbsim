"""
LSHTM-style tuberculosis (TB) compartmental models for Starsim.

This module provides individual-based TB models with states and transitions
inspired by LSHTM (London School of Hygiene & Tropical Medicine) formulations.
Two variants are available: :class:`TB_LSHTM` (latent → active progression) and
:class:`TB_LSHTM_Acute` (adds an acute infectious state immediately after
infection). State labels are defined in :class:`TBSL`.

State flow (TB_LSHTM):

- Susceptible → [transmission] → INFECTION (latent) → CLEARED | UNCONFIRMED | ASYMPTOMATIC
- From UNCONFIRMED: RECOVERED | ASYMPTOMATIC
- From ASYMPTOMATIC: UNCONFIRMED | SYMPTOMATIC
- From SYMPTOMATIC: ASYMPTOMATIC | TREATMENT | DEAD
- From TREATMENT: SYMPTOMATIC | TREATED
- CLEARED, RECOVERED, TREATED are susceptible to reinfection (modifiers pi, rho)
"""

import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd

from enum import IntEnum

__all__ = ['TB_LSHTM', 'TB_LSHTM_Acute', 'TBSL', 'TBSL']


class TBSL(IntEnum):
    """
    TB state labels for the LSHTM model.

    - Each agent is in exactly one of these states.
    - Transitions are driven by exponential rates in :class:`TB_LSHTM`
      (and :class:`TB_LSHTM_Acute`).
    """
    SUSCEPTIBLE  = -1    # No TB; never infected or cleared/recovered/treated
    INFECTION    = 0     # Latent infection (not yet active TB)
    CLEARED      = 1     # Cleared infection without developing active TB
    UNCONFIRMED  = 2     # Unconfirmed TB (early/smear-negative type state)
    RECOVERED    = 3     # Recovered from unconfirmed TB (susceptible to reinfection)
    ASYMPTOMATIC = 4     # Active TB, asymptomatic (infectious)
    SYMPTOMATIC  = 5     # Active TB, symptomatic (infectious)
    TREATMENT    = 6     # On TB treatment
    TREATED      = 7     # Completed treatment (susceptible to reinfection)
    DEAD         = 8     # TB-related death
    ACUTE        = 9     # Acute infection immediately after exposure (TB_LSHTM_Acute only)


class TB_LSHTM(ss.Infection):
    """
    LSHTM-style compartmental tuberculosis (TB) model.

    Implements a stochastic, individual-based TB model with states defined by the
    :class:`TBSL` enum. After infection, agents progress through latent infection
    (INFECTION), then may clear (CLEARED), develop unconfirmed TB (UNCONFIRMED), or
    progress to asymptomatic (ASYMPTOMATIC) or symptomatic (SYMPTOMATIC) active TB.
    Symptomatic cases may recover to asymptomatic, start treatment (TREATMENT), or
    die from TB (DEAD). Treated agents complete as TREATED. Cleared, recovered, and
    treated agents are susceptible to reinfection (with relative risks pi and rho).

    Transmission is driven by :attr:`pars.beta`; infectious states are ASYMPTOMATIC
    and SYMPTOMATIC (with relative transmissibility kappa for asymptomatic). All
    progression and mortality rates are defined as exponential waiting times (per
    year). Use :meth:`start_treatment` to move eligible agents onto treatment.

    State flow (high level)
    -----------------------
    - Susceptible agents (CLEARED, RECOVERED, TREATED, or never infected) are
      infected by contact; the base class calls :meth:`set_prognoses` for them.
    - Newly infected agents enter INFECTION and get a *scheduled* transition
      (state_next, ti_next) to CLEARED, UNCONFIRMED, or ASYMPTOMATIC (competing exponentials).
    - Each time step, :meth:`step` finds agents with ti >= ti_next, applies the
      transition, records outcomes, updates susceptibility/transmissibility, then
      schedules the *next* transition from the new state.
    - Reinfection is another transmission event into INFECTION (rel_sus modifies
      risk for RECOVERED/TREATED).
    - Treatment is triggered externally via :meth:`start_treatment`, which sets
      state_next and ti_next so the change takes effect at the next step.

    Subclasses :class:`starsim.Infection`; integrate with a :class:`starsim.Sim`
    and a population (e.g. :class:`starsim.People`) to run simulations.
    """

    def __init__(self, pars=None, **kwargs):
        """
        Initialize the LSHTM TB model parameters and state arrays.

        Parameters
        ----------
        pars : dict, optional
            Override or extend default parameters (e.g. ``beta``, ``kappa``,
            progression rates). Rates are per-year exponentials unless noted.
        **kwargs
            Passed to base; ``name`` and ``label`` are popped for the infection module.

        Notes
        -----
        Default parameters include:

        - Transmission: beta; relative risks for reinfection (pi, rho) and
          asymptomatic transmission (kappa)
        - Exponential rates for all state transitions: infcle, infunc, infasy,
          uncrec, uncasy, asyunc, asysym, symasy, theta, delta, phi, mutb, mu
        """
        super().__init__(name=kwargs.pop('name', None), label=kwargs.pop('label', None))

        # --- Transmission and reinfection ---
        self.define_pars(
            init_prev=ss.bernoulli(0.01),       # Initial seed infections (prevalence)
            beta=ss.peryear(0.25),              # Transmission rate per year (per-contact prob equivalent)
            kappa=0.82,                         # Relative transmission from asymptomatic vs symptomatic
            pi=0.21,                            # Relative risk of reinfection after recovery (unconfirmed)
            rho=3.15,                           # Relative risk of reinfection after treatment completion
            # --- From INFECTION (latent) ---
            infcle=ss.years(ss.expon(1/1.90)),  # Clear infection (no active TB)
            infunc=ss.years(ss.expon(1/0.16)), # Progress to unconfirmed TB
            infasy=ss.years(ss.expon(1/0.06)),  # Progress to asymptomatic active TB
            # --- From UNCONFIRMED ---
            uncrec=ss.years(ss.expon(1/0.18)), # Recover (→ RECOVERED)
            uncasy=ss.years(ss.expon(1/0.25)), # Progress to asymptomatic
            # --- From ASYMPTOMATIC ---
            asyunc=ss.years(ss.expon(1/1.66)),  # Revert to unconfirmed
            asysym=ss.years(ss.expon(1/0.88)), # Progress to symptomatic
            # --- From SYMPTOMATIC ---
            symasy=ss.years(ss.expon(1/0.54)), # Recover to asymptomatic
            theta=ss.years(ss.expon(1/0.46)),  # Start treatment
            mutb=ss.years(ss.expon(1/0.34)),   # TB-specific mortality
            # --- From TREATMENT ---
            phi=ss.years(ss.expon(1/0.63)),     # Treatment failure (back to symptomatic)
            delta=ss.years(ss.expon(1/2.00)),   # Treatment completion (→ TREATED)
            # --- Background ---
            mu=ss.years(ss.expon(1/0.014)),    # Background mortality (per year)
            cxr_asymp_sens=1.0,                 # CXR sensitivity for screening asymptomatic (0–1)
        )
        self.update_pars(pars, **kwargs)

        # Per-agent state: redefine base Infection states and add TB-specific ones
        self.define_states(
            ss.BoolState('susceptible', default=True),
            ss.BoolState('infected'),
            ss.FloatArr('rel_sus', default=1.0),
            ss.FloatArr('rel_trans', default=1.0),
            ss.FloatArr('ti_infected', default=-np.inf),
            ss.FloatArr('state', default=TBSL.SUSCEPTIBLE),       # Current TB state
            ss.FloatArr('state_next', default=TBSL.INFECTION),   # Scheduled next state
            ss.FloatArr('ti_next', default=np.inf),              # Time of next transition
            ss.BoolState('on_treatment', default=False),
            ss.BoolState('ever_infected', default=False),
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
        - Schedule the first competing transition: CLEARED, UNCONFIRMED, or ASYMPTOMATIC
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

        # Schedule first transition: latent → cleared, unconfirmed, or asymptomatic
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
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
            Mapping from TBSL state -> rate (e.g. ``ss.expon``-based). Keys are
            possible next states; values are callables that return a sample of
            waiting time (e.g. ``rate.rvs(uids)``).

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
            ti_state[idx, :] = ti + rate.rvs(uids)

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
        p = self.pars
        ti = self.ti

        # Agents whose scheduled transition time has arrived (ti_next <= ti)
        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return

        # --- Record outcomes from state_next *before* we overwrite state ---
        new_asymp_uids = uids[self.state_next[uids] == TBSL.ASYMPTOMATIC]
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

        # rel_sus: multiplier for susceptibility when base computes transmission; RECOVERED/TREATED at higher risk
        self.rel_sus[uids] = 1
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.pi
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rho

        # rel_trans: multiplier for infectiousness; ASYMPTOMATIC < SYMPTOMATIC (kappa typically < 1)
        self.rel_trans[uids] = 1
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.kappa

        # --- Schedule next transition: only states with spontaneous exits get a new (state_next, ti_next) ---
        # INFECTION → CLEARED, UNCONFIRMED, or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
        })

        # UNCONFIRMED → RECOVERED or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.UNCONFIRMED]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: self.pars.uncrec,
            TBSL.ASYMPTOMATIC: self.pars.uncasy
        })

        # ASYMPTOMATIC → UNCONFIRMED or SYMPTOMATIC
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.UNCONFIRMED: self.pars.asyunc,
            TBSL.SYMPTOMATIC: self.pars.asysym
        })

        # SYMPTOMATIC → ASYMPTOMATIC, TREATMENT, or TB DEATH
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.symasy,
            TBSL.TREATMENT: self.pars.theta,
            TBSL.DEAD: self.pars.mutb
        })

        # TREATMENT → SYMPTOMATIC (failure) or TREATED (completion)
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.phi,
            TBSL.TREATED: self.pars.delta
        })

        # CLEARED, RECOVERED, TREATED: no scheduled transition; reinfection via transmission only

        return

    def start_treatment(self, uids):
        """
        Move specified agents onto TB treatment (or clear latent infection).

        Called by interventions (e.g. screening/case-finding) when an agent is
        identified for treatment. We set state_next and ti_next to the *current*
        time so the change takes effect on the next :meth:`step` (no extra delay).

        - Latent (INFECTION): cleared without treatment (→ CLEARED).
        - Active (UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC): moved to TREATMENT.
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

        # Active TB (unconfirmed, asymptomatic, symptomatic): put on treatment
        u = uids[np.isin(self.state[uids], [TBSL.UNCONFIRMED, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[u] >= 15)

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

            ss.Result('n_detectable_15+', dtype=int, label='Symptomatic plus cxr_asymp_sens * Asymptomatic (15+)'),
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
    transition to INFECTION at rate :attr:`pars.acuinf`. Acute cases are
    infectious with relative transmissibility :attr:`pars.alpha`. All other
    states and transitions match :class:`TB_LSHTM`.

    State flow (difference from base)
    ---------------------------------
    - Transmission → ACUTE (not INFECTION).
    - From ACUTE the only spontaneous transition is ACUTE → INFECTION.
    - Once in INFECTION, flow is as in the base (CLEARED, UNCONFIRMED, ASYMPTOMATIC, etc.).
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
            Override parameters (e.g. ``acuinf``, ``alpha``).
        **kwargs
            Passed to base.
        """
        super().__init__(pars=pars, **kwargs)

        # ACUTE is a brief infectious state before latent (INFECTION); reinfection leads to ACUTE
        self.define_pars(
            acuinf=ss.years(ss.expon(1/4.0)),  # Rate: ACUTE → INFECTION (per year)
            alpha=0.9,                          # Relative transmission from acute vs symptomatic
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

        Same as base except initial state is ACUTE and first transition is
        ACUTE → INFECTION at rate acuinf.
        """
        super().set_prognoses(uids, sources)
        if len(uids) == 0:
            return

        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ever_infected[uids] = True
        self.ti_infected[uids] = self.ti

        self.state[uids] = TBSL.ACUTE

        # Single possible transition: ACUTE → INFECTION
        self.state_next[uids], self.ti_next[uids] = self.transition(uids, to={
            TBSL.INFECTION: self.pars.acuinf,
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
        p = self.pars
        ti = self.ti

        uids = ss.uids(ti >= self.ti_next)
        if len(uids) == 0:
            return

        # Record new active (transition into ASYMPTOMATIC) from state_next before overwriting
        new_asymp_uids = uids[self.state_next[uids] == TBSL.ASYMPTOMATIC]
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
        self.rel_sus[uids[self.state[uids] == TBSL.RECOVERED]] = self.pars.pi
        self.rel_sus[uids[self.state[uids] == TBSL.TREATED]] = self.pars.rho

        # rel_trans: ACUTE (alpha) and ASYMPTOMATIC (kappa) both infectious
        self.rel_trans[uids] = 1
        self.rel_trans[uids[self.state[uids] == TBSL.ACUTE]] = self.pars.alpha
        self.rel_trans[uids[self.state[uids] == TBSL.ASYMPTOMATIC]] = self.pars.kappa

        # ACUTE → INFECTION (single possible transition from acute)
        u = uids[self.state[uids] == TBSL.ACUTE]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.INFECTION: self.pars.acuinf,
        })

        # INFECTION → CLEARED, UNCONFIRMED, or ASYMPTOMATIC (same as base)
        u = uids[self.state[uids] == TBSL.INFECTION]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.CLEARED: self.pars.infcle,
            TBSL.UNCONFIRMED: self.pars.infunc,
            TBSL.ASYMPTOMATIC: self.pars.infasy
        })

        # UNCONFIRMED → RECOVERED or ASYMPTOMATIC
        u = uids[self.state[uids] == TBSL.UNCONFIRMED]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.RECOVERED: self.pars.uncrec,
            TBSL.ASYMPTOMATIC: self.pars.uncasy
        })

        # ASYMPTOMATIC → UNCONFIRMED or SYMPTOMATIC
        u = uids[self.state[uids] == TBSL.ASYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.UNCONFIRMED: self.pars.asyunc,
            TBSL.SYMPTOMATIC: self.pars.asysym
        })

        # SYMPTOMATIC → ASYMPTOMATIC, TREATMENT, or TB DEATH
        u = uids[self.state[uids] == TBSL.SYMPTOMATIC]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.ASYMPTOMATIC: self.pars.symasy,
            TBSL.TREATMENT: self.pars.theta,
            TBSL.DEAD: self.pars.mutb
        })

        # TREATMENT → SYMPTOMATIC (failure) or TREATED (completion)
        u = uids[self.state[uids] == TBSL.TREATMENT]
        self.state_next[u], self.ti_next[u] = self.transition(u, to={
            TBSL.SYMPTOMATIC: self.pars.phi,
            TBSL.TREATED: self.pars.delta
        })

        # CLEARED, RECOVERED, TREATED: next change is reinfection → ACUTE (via transmission)

        return

    def start_treatment(self, uids):
        """
        Move specified agents onto treatment (or clear); ACUTE/INFECTION → CLEARED.

        Same as base but latent includes both ACUTE and INFECTION:

        - ACUTE or INFECTION (latent): cleared without treatment (→ CLEARED).
        - Active (UNCONFIRMED, ASYMPTOMATIC, SYMPTOMATIC): moved to TREATMENT.
        """
        if len(uids) == 0:
            return 0

        # ACUTE or INFECTION (latent): clear infection
        u = uids[(self.state[uids] == TBSL.ACUTE) | (self.state[uids] == TBSL.INFECTION)]
        self.state_next[u] = TBSL.CLEARED
        self.ti_next[u] = self.ti

        # Active TB: put on treatment
        u = uids[np.isin(self.state[uids], [TBSL.UNCONFIRMED, TBSL.ASYMPTOMATIC, TBSL.SYMPTOMATIC])]
        self.state_next[u] = TBSL.TREATMENT
        self.ti_next[u] = self.ti

        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[u] >= 15)

        return