#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'HIV']

# Define HIV states as an enumeration.
class HIVState(IntEnum):
    ATRISK = -1   # Uninfected
    ACUTE    = 0    # Newly infected
    LATENT = 1    # Chronic infection
    AIDS   = 2    # Advanced disease
    DEAD   = 3    # Death due to HIV

class HIV(ss.Disease):
    """
    A simplified HIV disease model that tracks only the agent's state.
    
    Transitions (assuming a simulation time step of dt weeks, default dt=1):
      - ATRISK → HIV: Expected duration 4 weeks  (p = 1 - exp(-dt/4))
      - HIV → LATENT: Expected duration 8 weeks   (p = 1 - exp(-dt/8))
      - LATENT → AIDS: Expected duration 416 weeks (≈8 years; p = 1 - exp(-dt/416))
      - AIDS → DEAD: Expected duration 104 weeks   (≈2 years; p = 1 - exp(-dt/104))
      
    Agents are moved stochastically from one state to the next.
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define progression parameters.
        self.define_pars(
            init_prev          = 0.01,  # Proportion initially infected (ATRISK becomes HIV)
        )
        self.update_pars(pars, **kwargs)
        
        # Only one state: state.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK)
        )
    
    def set_initial_states(self, sim):
        """
        Initialize the HIV state:
          - With probability init_prev, agents become HIV (newly infected).
          - Otherwise, they remain ATRISK.
        """
        uids = sim.people.auids
        n = len(uids)
        infected = np.random.rand(n) < self.pars['init_prev']
        self.state[uids[infected]] = HIVState.ACUTE
        self.state[uids[~infected]] = HIVState.ATRISK
    
    def step(self):
        """
        Update the state transitions based on expected durations.
        """
        # Use simulation time step in weeks (default=1 if not provided)
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0
        uids = self.sim.people.auids
        current = self.state[uids].copy()
        
        # ATRISK -> HIV (mean = 4 weeks)
        p_atrisk_to_HIV = 1 - np.exp(-dt/4)
        atrisk_ids = uids[current == HIVState.ATRISK]
        if atrisk_ids.size > 0:
            if np.any(np.random.rand(atrisk_ids.size) < p_atrisk_to_HIV):
                transition = np.random.rand(atrisk_ids.size) < p_atrisk_to_HIV
                self.state[atrisk_ids[transition]] = HIVState.ACUTE
        
        # HIV -> LATENT (mean = 8 weeks)
        p_HIV_to_LATENT = 1 - np.exp(-dt/8)
        hiv_ids = uids[current == HIVState.ACUTE]
        if hiv_ids.size > 0:
            transition = np.random.rand(hiv_ids.size) < p_HIV_to_LATENT
            self.state[hiv_ids[transition]] = HIVState.LATENT
        
        # LATENT -> AIDS (mean = 416 weeks)
        p_LATENT_to_AIDS = 1 - np.exp(-dt/416)
        latent_ids = uids[current == HIVState.LATENT]
        if latent_ids.size > 0:
            transition = np.random.rand(latent_ids.size) < p_LATENT_to_AIDS
            self.state[latent_ids[transition]] = HIVState.AIDS
        
        # AIDS -> DEAD (mean = 104 weeks)
        p_AIDS_to_DEAD = 1 - np.exp(-dt/104)
        aids_ids = uids[current == HIVState.AIDS]
        if aids_ids.size > 0:
            transition = np.random.rand(aids_ids.size) < p_AIDS_to_DEAD
            self.state[aids_ids[transition]] = HIVState.DEAD
    
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float, label='Prevalence (Infected)'),
            ss.Result(name='atrisk', dtype=float, label='ATRISK'),
            ss.Result(name='acute', dtype=float, label='ACUTE'),
            ss.Result(name='latent', dtype=float, label='LATENT'),
            ss.Result(name='aids', dtype=float, label='AIDS'),
            ss.Result(name='dead', dtype=float, label='DEAD'),
        )
    
    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        uids = self.sim.people.auids
        n = len(uids)
        states = self.state[uids]
        infected = (states != HIVState.ATRISK) & (states != HIVState.DEAD)
        self.results.hiv_prevalence[ti] = infected.sum() / n
        self.results.atrisk[ti] = np.sum(states == HIVState.ATRISK) / n
        self.results.acute[ti] = np.sum(states == HIVState.ACUTE) / n
        self.results.latent[ti] = np.sum(states == HIVState.LATENT) / n
        self.results.aids[ti] = np.sum(states == HIVState.AIDS) / n
        self.results.dead[ti] = np.sum(states == HIVState.DEAD) / n
    
    def set_ART(self, uids, on_art=True):
        # ART is not used in this simplified model.
        pass
