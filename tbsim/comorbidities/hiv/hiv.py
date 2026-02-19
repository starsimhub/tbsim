#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'HIV']

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

        