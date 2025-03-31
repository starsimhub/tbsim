#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'ACUTE']

# Define HIV states as an enumeration.
class HIVState(IntEnum):
    ATRISK = -1   # Uninfected
    ACUTE    = 0    # Newly infected (early state)
    LATENT = 1    # Chronic infection
    AIDS   = 2    # Advanced disease
    DEAD   = 3    # Dead from HIV

class HIV(ss.Disease):
    """
    A simplified HIV disease model that tracks only the agent's state and ART status.
    
    Transitions are stochastic with expected durations:
      - ATRISK → HIV: 4 weeks
      - HIV → LATENT: 8 weeks
      - LATENT → AIDS: 416 weeks (≈8 years)
      - AIDS → DEAD: 104 weeks (≈2 years)
      
    If an agent is on ART, progression probabilities are reduced by a factor
      art_progression_factor (default 0.5), thus prolonging the duration in that state.
      
    Parameters:
      - init_prev: Initial prevalence of infection.
      - p_ACUTE_to_LATENT: Baseline probability per step for HIV → LATENT (1 - exp(-dt/8)).
      - p_LATENT_to_AIDS: Baseline probability per step for LATENT → AIDS (1 - exp(-dt/416)).
      - p_AIDS_to_DEAD: Baseline probability per step for AIDS → DEAD (1 - exp(-dt/104)).
      - art_progression_factor: Multiplier to reduce progression rates if on ART.
      
    References:
      - Expert Report to the Infected Blood Inquiry: HIV (Jan 2020) and related guidelines.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define progression parameters (using a time step in weeks).
        self.define_pars(
            init_prev           = 0.01,
            # Baseline transition probabilities computed using an exponential waiting time:
            # p = 1 - exp(-dt/mean_duration), with dt assumed to be 1 week.
            p_ACUTE_to_LATENT     = 1-np.exp(-1/8),    # ~0.117
            p_LATENT_to_AIDS    = 1-np.exp(-1/416),  # ~0.0024
            p_AIDS_to_DEAD      = 1-np.exp(-1/104),  # ~0.0096
            art_progression_factor  = 0.25,            # Halve the progression probability if on ART.
            on_ART                  = 0.25,           # Default ART status for agents (True = on ART)
        )
        self.update_pars(pars, **kwargs)
        
        # Define extra attributes for Agents of this disease.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK),  # State: ATRISK, HIV, LATENT, AIDS, DEAD.
            ss.BoolArr('on_ART', default=False),#self.pars.on_ART),             # Whether agent is on ART.
        )
        return
    
    def set_initial_states(self, sim):
        """
        Initialize the HIV state.
          - With probability init_prev, agents become HIV (newly infected).
          - Otherwise, they remain ATRISK.
        """
        uids = sim.people.auids
        n = len(uids)
        infected = np.random.rand(n) < self.pars.init_prev
        
        # Set Infected individuals to ACUTE, others to ATRISK.
        self.state[uids[infected]] = HIVState.ACUTE
        self.state[uids[~infected]] = HIVState.ATRISK
        # Set ART status for a % of agents.
        self.on_ART[uids] = np.random.rand(n) < self.pars.on_ART  # Randomly assign ART status.
        
    def step(self):
        """
        Update state transitions based solely on state and ART.
        If an agent is on ART, progression probabilities are reduced.
        """
        if self.ti == 0:  # Skip the first step (initialization)
            self.set_initial_states(self.sim)
            return
        
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0  # dt in weeks (default=1)
        uids = self.sim.people.auids
        current = self.state[uids].copy()
        
        # ART factor for progression:
        art_factor = self.pars['art_progression_factor']
        
        # ATRISK → HIV: (applied only to ATRISK; ART does not apply here)
        p_atrisk_to_HIV = 1 - np.exp(-dt/4)
        atrisk_ids = uids[current == HIVState.ATRISK]
        if atrisk_ids.size > 0:
            rand_vals = np.random.rand(atrisk_ids.size)
            to_HIV = rand_vals < p_atrisk_to_HIV
            self.state[atrisk_ids[to_HIV]] = HIVState.ACUTE
        
        # HIV → LATENT:
        p_ACUTE_to_LATENT = 1 - np.exp(-dt/8)
        hiv_ids = uids[current == HIVState.ACUTE]
        if hiv_ids.size > 0:
            # Apply ART factor if on ART.
            effective_p = np.array([p_ACUTE_to_LATENT * (art_factor if self.on_ART[uid] else 1.0)
                                      for uid in hiv_ids])
            rand_vals = np.random.rand(hiv_ids.size)
            to_latent = rand_vals < effective_p
            self.state[hiv_ids[to_latent]] = HIVState.LATENT
        
        # LATENT → AIDS:
        p_LATENT_to_AIDS = 1 - np.exp(-dt/416)
        latent_ids = uids[current == HIVState.LATENT]
        if latent_ids.size > 0:
            effective_p = np.array([p_LATENT_to_AIDS * (art_factor if self.on_ART[uid] else 1.0)
                                      for uid in latent_ids])
            rand_vals = np.random.rand(latent_ids.size)
            to_aids = rand_vals < effective_p
            self.state[latent_ids[to_aids]] = HIVState.AIDS
        
        # AIDS → DEAD:
        p_AIDS_to_DEAD = 1 - np.exp(-dt/104)
        aids_ids = uids[current == HIVState.AIDS]
        if aids_ids.size > 0:
            effective_p = np.array([p_AIDS_to_DEAD * (art_factor if self.on_ART[uid] else 1.0)
                                      for uid in aids_ids])
            rand_vals = np.random.rand(aids_ids.size)
            to_dead = rand_vals < effective_p
            self.state[aids_ids[to_dead]] = HIVState.DEAD
    
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float, label='Prevalence (Infected)'),
            ss.Result(name='atrisk', dtype=float, label='ATRISK'),
            ss.Result(name='acute', dtype=float, label='ACUTE'),
            ss.Result(name='latent', dtype=float, label='LATENT'),
            ss.Result(name='aids', dtype=float, label='AIDS'),
            ss.Result(name='dead', dtype=float, label='DEAD'),
            ss.Result(name='on_art', dtype=float, label='On ART'),
        )
    
    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        uids = self.sim.people.auids
        n = len(uids)
        states = self.state[uids]
        infected = (states != HIVState.ATRISK) & (states != HIVState.DEAD)
        self.results.hiv_prevalence[ti] = infected.sum()/n
        self.results.atrisk[ti] = np.sum(states == HIVState.ATRISK)/n
        self.results.acute[ti] = np.sum(states == HIVState.ACUTE)/n
        self.results.latent[ti] = np.sum(states == HIVState.LATENT)/n
        self.results.aids[ti] = np.sum(states == HIVState.AIDS)/n
        self.results.dead[ti] = np.sum(states == HIVState.DEAD)/n
        # Calculate proportion of agents on ART.
        self.results.on_art[ti] = self.on_ART[uids].mean()
    
    def set_ART(self, uids, on_ART=True):
        """Set the ART status for specified agents."""
        self.on_ARTs[uids] = on_ART
