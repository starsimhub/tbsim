#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'HIV', 'HivInterventions', 'HivInterventionMode', 'TB_HIV_Connector']

# Define HIV states as an enumeration.
class HIVState(IntEnum):
    """
    Enum representing the possible HIV states an agent can be in.

    States:
        - ATRISK: Agent is HIV-negative but at risk.
        - ACUTE: Recently infected with HIV.
        - LATENT: Chronic HIV infection.
        - AIDS: Advanced stage of HIV infection.
    """
    ATRISK = 0   # Uninfected
    ACUTE  = 1    # Newly infected (early state)
    LATENT = 2    # Chronic infection
    AIDS   = 3    # Advanced disease
    def __str__(self):
        return {0: 'ATRISK', 1: 'ACUTE', 2: 'LATENT', 3: 'AIDS'}[self.value]
    def __repr__(self):
        return {0: 'ATRISK', 1: 'ACUTE', 2: 'LATENT', 3: 'AIDS'}[self.value]




class HIV(ss.Disease):
    """
    A simplified agent-based HIV disease model for use with the Starsim framework.

    This model tracks HIV state progression through ACUTE, LATENT, and AIDS phases,
    influenced by whether the agent is receiving ART (antiretroviral therapy).

    Key Features:
        - Initial infection and ART status are assigned during the first timestep,
          unless a high-level intervention labeled 'hivinterventions' is present.
        - Disease progression is stochastic and modified by ART presence.
        - ART reduces the probability of progression from ACUTE → LATENT and LATENT → AIDS.
        - AIDS → DEAD transition is defined but not applied in this model.

    Parameters:
        - init_prev: Initial probability of infection (ACUTE).
        - init_onart: Probability of being on ART at initialization (if infected).
        - acute_to_latent: Daily transition probability from ACUTE to LATENT.
        - latent_to_aids: Daily transition probability from LATENT to AIDS.
        - aids_to_dead: Daily transition probability from AIDS to DEAD (unused).
        - art_progression_factor: Multiplier applied to progression probabilities for agents on ART.

    States:
        - state: HIV progression state (ATRISK, ACUTE, LATENT, AIDS, DEAD).
        - on_ART: Boolean indicating whether agent is on ART.

    Results Tracked:
        - hiv_prevalence: Proportion of total agents with HIV.
        - infected: Total number of HIV-positive agents.
        - on_art: Number of agents on ART.
        - atrisk, acute, latent, aids: Percent of population in each state.
        - n_active: Total number of agents in ACUTE, LATENT, or AIDS states.
    """
    def __init__(self, pars=None, **kwargs):
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

    def set_prognoses(self ):
        """
        Initialize HIV infection and ART status for agents in the simulation.

        This method is called at the beginning of the simulation (time index 0)
        to assign initial disease states and treatment (ART) status.

        Behavior:
            - If a high-level intervention labeled 'hivinterventions' is present in the simulation,
            initialization is skipped entirely, assuming that the intervention will handle infection
            and ART assignments dynamically at the appropriate time.
            - If no HIV states are currently set to ACUTE, a subset of agents is randomly assigned to
            ACUTE based on the `init_prev` parameter.
            - If no agents are currently on ART, a subset of those in the ACUTE state is randomly
            assigned to be on ART based on the `init_onart` parameter.

        Notes:
            - This check ensures the model does not reinitialize infected or ART states
            if they've already been set or will be handled externally.
            - It also avoids reapplying ART if ART status was assigned previously.

        Returns:
            None
        """
        if hasattr(self.sim, 'interventions'):
            import tbsim as mtb
            for i in self.sim.interventions:
                if i=='hivinterventions':           # Check if the intervention label is present among the interventions
                    print('HIV intervention present, skipping initialization.')
                    return

        uids = self.sim.people.auids
        if len(self.state[self.state == HIVState.ACUTE])==0:
            initial_infected= self.pars.init_prev.filter(uids)
            self.state[initial_infected] = HIVState.ACUTE

        current = self.state[uids].copy()
        if len(self.on_ART[self.on_ART == True])==0:
            # apply ART only to those who are in the ACUTE state
            infected =uids[current == HIVState.ACUTE]
            initial_onart = self.pars.init_onart.filter(infected)
            self.on_ART[initial_onart] = True
        return

    def step(self):
        """
        Update state transitions based solely on state and ART.
        If an agent is on ART, progression probabilities are reduced.
        """
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


    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float, label='Prevalence (% Infected)'),
            ss.Result(name='infected', dtype=int, label='Infected'),
            ss.Result(name='on_art', dtype=float, label='On ART'),
            ss.Result(name='atrisk', dtype=float, label='% ATRISK (Alive)'),
            ss.Result(name='acute', dtype=float, label='% ACUTE'),
            ss.Result(name='latent', dtype=float, label='% LATENT'),
            ss.Result(name='aids', dtype=float, label='% AIDS'),
            ss.Result(name='n_active', dtype=int, label='Active (Combined)'),
        )

    def update_results(self):
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
        res.n_active[ti]   = np.count_nonzero(np.isin(self.state, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS]))
        res.on_art[ti]     = np.count_nonzero(self.on_ART == True)

        # if n_alive > 0:
        #     res.hiv_prevalence[ti] = res.n_active[ti] / n_alive


class HivInterventionMode():
    """
    HivInterventionMode is an enumeration class that defines the modes of HIV interventions in a simulation.
    The modes include:
        - INFECTION: Adjusts the number of individuals infected with HIV to match a target prevalence.
        - ONART: Adjusts the number of individuals on antiretroviral therapy (ART) to match a target proportion.
        - BOTH: Applies both prevalence and OnART adjustments sequentially.
    """
    PREVALENCE = 'prevalence'
    ONART = 'onart'
    BOTH = 'both'

class HivInterventions(ss.Intervention):
    """
    HivInterventions is a class that models HIV-related interventions in a simulation.
    It allows for the adjustment of HIV prevalence and the proportion of individuals on ART (antiretroviral therapy)
    within a simulated population. The class supports three modes of operation: 'prevalence', 'onart', and 'both'.
    Attributes:
        pars (dict): Parameters for the intervention, including mode, prevalence, percent_on_ART, min_age, and max_age.
        handlers (dict): A dispatch table mapping modes ('prevalence', 'onart', 'both') to their respective handler methods.
    Methods:
        __init__(pars, **kwargs):
            Initializes the HivInterventions object with the given parameters and sets up the dispatch table.
        step():
            Executes the intervention logic based on the specified mode ('prevalence', 'onart', or 'both').
        _apply_prevalence():
            Adjusts the number of individuals infected with HIV to match the target prevalence.
            Adds or removes infections as necessary, ensuring the changes respect age constraints.
        _apply_onart():
            Adjusts the number of individuals OnART to match the target proportion.
            Adds or removes individuals from OnART as necessary.
        _apply_both():
            Applies both the prevalence and OnART adjustments sequentially.
    Raises:
        ValueError: If an unsupported mode is specified in the parameters.
    Warnings:
        - If there are not enough eligible individuals to infect when increasing prevalence.
        - If there are not enough acute cases to revert when decreasing prevalence.
        - If there are not enough candidates to add to OnART when increasing OnART coverage.
        - If there are not enough individuals on OnART to remove when decreasing OnART coverage.
    """

    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            mode='both',            # 'prevalence', 'onart', or 'both'
            prevalence=0.20,        # Default: target prevalence of HIV in the population (20%)
            percent_on_ART=0.775,   # Default: % infected individuals on ART - https://pmc.ncbi.nlm.nih.gov/articles/PMC6772052/
            min_age=0,          # Minimum age for prevalence (in years)
            max_age=200,            # Maximum age for prevalence (in years), None for no upper limit
            start=ss.date('2000-01-01'),
            stop=ss.date('2035-12-31'),
        )
        self.update_pars(pars, **kwargs)

        # Dispatch table to route logic by mode
        self.handlers = {
            'prevalence': self._apply_prevalence,
            'onart': self._apply_onart,
            'both': self._apply_both,
        }

    def step(self):
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return
        handler = self.handlers.get(self.pars.mode)
        if handler is None:
            raise ValueError(f"Unsupported mode: {self.pars.mode}, please specify 'prevalence', 'onart', or 'both' ")
        handler()

    def _apply_prevalence(self):
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

    def _apply_onart(self):
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
                # raie a warning
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

    def _apply_both(self):
        self._apply_prevalence()
        self._apply_onart()


class TB_HIV_Connector(ss.Connector):
    """
    Connector between TB and HIV.

    This connector uses the HIV state ( ACUTE, LATENT, AIDS)
    from the HIV disease model to modify TB progression parameters.

    Adjustments:
      - TB-infected individuals have increased:
          - Risk of progression from latent TB to presymptomatic TB (via `rr_activation`)

    State multipliers:
      - ACUTE:  1.5
      - LATENT: 2.0
      - AIDS:   3.0
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(label='TB-HIV')
        self.define_pars(
            tb_hiv_rr_func      = self.compute_tb_hiv_risk_rr,
            acute_multiplier     = 1.2202,
            latent_multiplier    = 1.9001,
            aids_multiplier      = 2.5955,
        )
        self.update_pars(pars=pars, **kwargs)
        self.state_multipliers = {
            HIVState.ACUTE: self.pars.acute_multiplier,
            HIVState.LATENT: self.pars.latent_multiplier,
            HIVState.AIDS:   self.pars.aids_multiplier,

        }

    @staticmethod
    def compute_tb_hiv_risk_rr(self, tb, hiv, uids, base_factor=1.0):
        """
        Computes the relative risk (RR) multiplier for TB progression and mortality
        based on HIV state and ART (antiretroviral therapy) status.
        Parameters:
            tb (object): The TB model object (not directly used in this function).
            hiv (object): The HIV model object containing state and ART status information.
            uids (array-like): Array of unique identifiers for individuals.
            base_factor (float, optional): A base multiplier applied to the computed RR.
                                           Defaults to 1.0.
        Returns:
            numpy.ndarray: An array of relative risk multipliers for the given individuals.

        Notes:
            - The function initializes the RR multipliers to 1.0 for all individuals.
            - It applies state-specific multipliers based on the individual's HIV state.
            - The final RR is scaled by the `base_factor` parameter.
        """
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
        """
        This is where the actual modification of TB parameters occurs.
        """
        tb = self.sim.diseases['tb']
        hiv = self.sim.diseases['hiv']

        # Adjust TB progression and death risks for infected individuals
        uids_tb = tb.infected.uids
        rr = self.pars.tb_hiv_rr_func(self, tb, hiv, uids_tb)
        # print(f"TB-HIV Connector: Adjusting TB progression risk for {len(uids_tb)} individuals. {rr}")

        tb.rr_activation[uids_tb] *= rr
        # print(tb.rr_activation[uids_tb])
        return
