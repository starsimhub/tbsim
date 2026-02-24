"""Simplified HIV disease model and TB-HIV connector for co-infection simulations."""

from enum import IntEnum
import numpy as np
import starsim as ss
import tbsim

__all__ = ['HIVState', 'HIV', 'HivInterventions', 'TB_HIV_Connector']

class HIVState(IntEnum):
    """ HIV states: uninfected through advanced disease. """
    ATRISK = 0   # Uninfected
    ACUTE  = 1   # Newly infected (early stage)
    LATENT = 2   # Chronic infection
    AIDS   = 3   # Advanced disease




class HIV(ss.Disease):
    """
    A simplified agent-based HIV disease model for use with the Starsim framework.

    This model tracks HIV state progression through ACUTE, LATENT, and AIDS phases,
    influenced by whether the agent is receiving ART (antiretroviral therapy).

    Args:
        - init_prev: Initial probability of infection (ACUTE).
        - init_onart: Probability of being on ART at initialization (if infected).
        - acute_to_latent: Daily transition probability from ACUTE to LATENT.
        - latent_to_aids: Daily transition probability from LATENT to AIDS.
        - aids_to_dead: Daily transition probability from AIDS to DEAD (unused).
        - art_progression_factor: Multiplier applied to progression probabilities for agents on ART.

    States:
        - state: HIV progression state (ATRISK, ACUTE, LATENT, AIDS, DEAD).
        - on_ART: Boolean indicating whether agent is on ART.

    Results:
        - hiv_prevalence: Proportion of total agents with HIV.
        - infected: Total number of HIV-positive agents.
        - on_art: Number of agents on ART.
        - atrisk, acute, latent, aids: Percent of population in each state.
        - n_active: Total number of agents in ACUTE, LATENT, or AIDS states.

    Example
    -------
    ::

        import starsim as ss
        from tbsim.comorbidities.hiv import HIV

        sim = ss.Sim(diseases=HIV(), pars=dict(start='2000', stop='2020'))
        sim.run()
    """
    def __init__(self, pars=None, **kwargs):
        """Initialize with default HIV progression parameters; override via ``pars``."""
        super().__init__(**kwargs)
        # Define progression parameters (using a time step in weeks).
        self.define_pars(
            init_prev = ss.bernoulli(p=0.00),  # Initial prevalence of HIV
            init_onart = ss.bernoulli(p=0.00),  # Initial probability of being on ART (if infected).
            art_progression_factor  = 0.1, # Multiplier to reduce progression rates if on ART.
            acute_to_latent       = ss.perday(1/(7*12)), # 1-np.exp(-1/8),  # 8 weeks
            latent_to_aids        = ss.perday(1/(365*8)), # 1-np.exp(-1/416), # 416 weeks
        )
        self.update_pars(pars, **kwargs)

        # Define extra attributes for Agents of this disease.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK),  # Column name to store HIV state.
            ss.BoolArr('on_ART', default=False),            # Column name to store Whether agent is on ART.
        )

        self.dist_acute_to_latent = ss.bernoulli(p=self.p_acute_to_latent)
        self.dist_latent_to_aids  = ss.bernoulli(p=self.p_latent_to_aids)

        return

    @staticmethod
    def p_acute_to_latent(self, sim, uids):
        """ Calculate probability of HIV ACUTE → LATENT transition. """
        acute_to_latent = self.pars.acute_to_latent.to_prob()
        art_factor = self.pars['art_progression_factor']
        art_multiplier = np.where(self.on_ART[uids], art_factor, 1.0)
        return acute_to_latent * art_multiplier

    @staticmethod
    def p_latent_to_aids(self, sim, uids):
        """ Calculate probability of HIV LATENT → AIDS transition. """
        latent_to_aids = self.pars.latent_to_aids.to_prob()
        art_factor = self.pars['art_progression_factor']
        art_multiplier = np.where(self.on_ART[uids], art_factor, 1.0)
        return latent_to_aids * art_multiplier

    def set_prognoses(self):
        """ Assign initial HIV infection and ART status (called at t=0). """
        uids = self.sim.people.auids

        if len(self.state[self.state == HIVState.ACUTE]) == 0:
            initial_infected = self.pars.init_prev.filter(uids)
            self.state[initial_infected] = HIVState.ACUTE

        current = self.state[uids].copy()
        if len(self.on_ART[self.on_ART == True]) == 0:
            infected = uids[current == HIVState.ACUTE]
            initial_onart = self.pars.init_onart.filter(infected)
            self.on_ART[initial_onart] = True

        return

    def step(self):
        """ Progress HIV states; ART reduces transition probabilities. """
        if self.sim.ti == 0:
            # Set initial prognoses for all agents
            self.set_prognoses()
            return

        uids = self.sim.people.auids
        current = self.state[uids].copy()

        # HIV ACUTE → LATENT:
        hiv_ids = uids[current == HIVState.ACUTE]
        self.state[self.dist_acute_to_latent.filter(hiv_ids)] = HIVState.LATENT

        # HIV LATENT → AIDS:
        latent_ids = uids[current == HIVState.LATENT]
        self.state[self.dist_latent_to_aids.filter(latent_ids)] = HIVState.AIDS

        return

    def init_results(self):
        """Define HIV result channels (prevalence, state counts, ART coverage)."""
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float, label='Prevalence (% Infected)'),
            ss.Result(name='infected', dtype=int, label='Infected'),
            ss.Result(name='on_art', dtype=float, label='On ART'),
            ss.Result(name='atrisk', dtype=float, label='% ATRISK (Alive)'),
            ss.Result(name='acute', dtype=float, label='% ACUTE'),
            ss.Result(name='latent', dtype=float, label='% LATENT'),
            ss.Result(name='aids', dtype=float, label='% AIDS'),
        )

        return

    def update_results(self):
        """Record HIV state distribution and ART counts for the current timestep."""
        super().update_results()
        ti = self.sim.ti
        uids = self.sim.people.auids
        n_alive = np.count_nonzero(self.sim.people.alive)
        res = self.results
        n = len(uids)

        states = self.state[uids]
        if n_alive > 0:
            res.hiv_prevalence[ti] = np.count_nonzero(np.isin(self.state, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS])) / n_alive
        else:
            res.hiv_prevalence[ti] = 0.0
        res.infected[ti] = np.count_nonzero(np.isin(self.state, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS]))
        res.atrisk[ti]     = np.count_nonzero(self.state == HIVState.ATRISK)/n_alive
        res.acute[ti]      = np.count_nonzero(self.state == HIVState.ACUTE)/n_alive
        res.latent[ti]     = np.count_nonzero(self.state == HIVState.LATENT)/n_alive
        res.aids[ti]       = np.count_nonzero(self.state == HIVState.AIDS)/n_alive
        res.on_art[ti]     = np.count_nonzero(self.on_ART == True)

        return


class HivInterventions(ss.Intervention):
    """
    Adjust HIV prevalence and/or ART coverage in a simulated population at each
    timestep within a date window.
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with target prevalence and ART coverage parameters."""
        super().__init__(**kwargs)
        self.define_pars(
            use_prevalence=True,    # Whether to adjust HIV prevalence each step
            use_art=True,           # Whether to adjust ART coverage each step
            prevalence=0.20,        # Target prevalence of HIV in the population (20%)
            percent_on_ART=0.775,   # % infected individuals on ART - https://pmc.ncbi.nlm.nih.gov/articles/PMC6772052/
            min_age=0,              # Minimum age for prevalence (in years)
            max_age=200,            # Maximum age for prevalence (in years)
            start=ss.date('2000-01-01'),
            stop=ss.date('2035-12-31'),
        )
        self.update_pars(pars, **kwargs)

        return

    def step(self):
        """Adjust HIV prevalence and ART coverage to match targets each timestep."""
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return
        if self.pars.use_prevalence:
            self._apply_prevalence()
        if self.pars.use_art:
            self._apply_onart()

        return

    def _apply_prevalence(self):
        """Add or remove acute infections to match the target HIV prevalence."""
        self.hiv = self.sim.diseases.hiv
        alive = len(self.sim.people.alive)

        target_prev = self.pars.prevalence(self.sim) if callable(self.pars.prevalence) else self.pars.prevalence
        expected_infectious = int(np.round(alive * target_prev))
        infectious_uids = ((self.hiv.state == HIVState.ACUTE) | (self.hiv.state == HIVState.LATENT) | (self.hiv.state == HIVState.AIDS)).uids
        n_current = len(infectious_uids)
        delta = expected_infectious - n_current
        min_age = self.pars.min_age
        max_age = self.pars.max_age

        if delta > 0:
            # Not enough infections → add more
            at_risk_uids = (self.hiv.state == HIVState.ATRISK).uids
            is_within_age_range = ((self.sim.people.age[at_risk_uids] >= min_age)
                                 & (self.sim.people.age[at_risk_uids] <= (max_age if max_age is not None else np.inf)))

            eligible = at_risk_uids[is_within_age_range]

            if delta > len(eligible):
                ss.warn(msg=f"Not enough eligible people to infect. Expected: {delta}, Available: {len(eligible)}")


            n_to_add = min(delta, len(eligible))
            if n_to_add:
                dist = ss.randint(0, len(eligible), strict=False).init()
                chosen = eligible[dist(n_to_add)]
                self.hiv.state[chosen] = HIVState.ACUTE

        elif delta < 0:
            # Too many infections → revert some ACUTE cases to ATRISK
            acute_uids = (self.hiv.state == HIVState.ACUTE).uids

            if -delta > len(acute_uids):
                ss.warn(msg=f"Not enough acute cases to revert. Expected: {-delta}, Available: {len(acute_uids)}")


            n_to_remove = min(-delta, len(acute_uids))
            if n_to_remove:
                dist = ss.randint(0, len(acute_uids), strict=False).init()
                chosen = acute_uids[dist(n_to_remove)]
                self.hiv.state[chosen] = HIVState.ATRISK

        return

    def _apply_onart(self):
        """Add or remove agents from ART to match the target coverage."""
        self.hiv = self.sim.diseases.hiv

        alive = len(self.sim.people.alive)
        target_prev = self.pars.prevalence(self.sim) if callable(self.pars.prevalence) else self.pars.prevalence
        expected_on_art = int(np.round(alive * self.pars.percent_on_ART * target_prev))

        current_on_art = (self.hiv.on_ART == True).uids
        n_current = len(current_on_art)
        delta = expected_on_art - n_current

        if delta > 0:
            # Need to add people to ART
            candidates = ((~self.hiv.on_ART)
                          & (self.hiv.state != HIVState.ATRISK)).uids
            if delta > len(candidates):
                # Raise a warning
                ss.warn(msg=f"Not enough candidates for ART. Expected: {delta}, Available: {len(candidates)}")


            n_to_add = min(delta, len(candidates))
            if n_to_add:
                dist = ss.randint(0, len(candidates), strict=False).init()
                chosen = candidates[dist(n_to_add)]
                self.hiv.on_ART[chosen] = True

        elif delta < 0:
            # Need to remove people from ART
            if -delta > len(current_on_art):
                ss.warn(msg=f"Not enough people on ART to remove. Expected: {-delta}, Available: {len(current_on_art)}")

            n_to_remove = min(-delta, len(current_on_art))
            if n_to_remove:
                dist = ss.randint(0, len(current_on_art), strict=False).init()
                chosen = current_on_art[dist(n_to_remove)]
                self.hiv.on_ART[chosen] = False

        return

class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV: multiplies TB activation risk by HIV-state-dependent
    relative risk factors (ACUTE: 1.2202, LATENT: 1.9001, AIDS: 2.5955).

    Example
    -------
    ::

        import starsim as ss
        import tbsim
        from tbsim.comorbidities.hiv import HIV, HivInterventions, TB_HIV_Connector

        tb   = tbsim.TB_LSHTM(name='tb')
        hiv  = HIV(name='hiv')
        conn = TB_HIV_Connector()
        intv = HivInterventions()
        sim  = ss.Sim(diseases=[tb, hiv], connectors=conn, interventions=intv,
                      pars=dict(start='2000', stop='2020'))
        sim.run()
    """

    def __init__(self, pars=None, **kwargs):
        """Initialize with HIV-state-dependent TB risk multipliers."""
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_hiv_rr_func       = self.compute_tb_hiv_risk_rr,
            acute_multiplier     = 1.2202,   # TODO: citation needed
            latent_multiplier    = 1.9001,   # TODO: citation needed
            aids_multiplier      = 2.5955,   # TODO: citation needed
        )
        self.update_pars(pars=pars, **kwargs)
        self.state_multipliers = {
            HIVState.ACUTE: self.pars.acute_multiplier,
            HIVState.LATENT: self.pars.latent_multiplier,
            HIVState.AIDS:   self.pars.aids_multiplier,

        }

        return

    @staticmethod
    def compute_tb_hiv_risk_rr(self, tb, hiv, uids, base_factor=1.0):
        """ Compute per-agent TB relative risk multiplier based on HIV state. """
        states = hiv.state[uids]

        # Initialize multipliers with 1.0
        rr = np.ones_like(uids, dtype=float)

        # Define risk multipliers by HIV state
        state_multipliers = self.state_multipliers

        # Apply state-specific risk multipliers
        for state, mult in state_multipliers.items():
            rr[states == state] = mult

        return rr * base_factor

    def step(self):
        """ Apply HIV-based risk multipliers to TB activation rates. """
        tb = tbsim.get_tb(self.sim)
        hiv = self.sim.diseases['hiv']
        uids_tb = tb.infected.uids
        rr = self.pars.tb_hiv_rr_func(self, tb, hiv, uids_tb)
        tb.rr_activation[uids_tb] *= rr

        return
