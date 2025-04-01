#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'ACUTE']

# Define HIV states as an enumeration.
class HIVState(IntEnum):
    ATRISK = -1   # Uninfected
    ACUTE  = 0    # Newly infected (early state)
    LATENT = 1    # Chronic infection
    AIDS   = 2    # Advanced disease
    DEAD   = 3    # Dead from HIV

class HIV(ss.Disease):
    """
    A simplified HIV disease model that tracks only the agent's state and ART status.
    
    Transitions are stochastic with expected durations:
      - ATRISK → HIV: 4 weeks
      - ACUTE → LATENT: 8 weeks
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
      - https://www.infectedbloodinquiry.org.uk/sites/default/files/documents/HIV%20Report%20with%20Addendum.pdf 
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define progression parameters (using a time step in weeks).
        self.define_pars(
            init_prev               = ss.bernoulli(p=0.25 ), # Initial prevalence of HIV
            # Baseline transition probabilities computed using an exponential waiting time:
            # p = 1 - exp(-dt/mean_duration), with dt assumed to be 1 week.
            p_ATRISK_to_ACUTE       = 0,  # If more than 0, then it will generate new infections
            p_ACUTE_to_HIV          = 1-np.exp(-1/4),    # ~0.25
            p_ACUTE_to_LATENT       = 1-np.exp(-1/8),    # ~0.117
            p_LATENT_to_AIDS        = 1-np.exp(-1/416),  # ~0.0024
            p_AIDS_to_DEAD          = 1-np.exp(-1/104),  # ~0.0096
            art_progression_factor  = 0.5,               # Default ART factor (0.5 = 50% reduction)
            on_ART                  = 0.25,              # Default percentage of people on ART
        )
        self.update_pars(pars, **kwargs)
        
        # Define extra attributes for Agents of this disease.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK),          
            ss.BoolArr('on_ART', default=False),          # Whether agent is on ART.
        )
        return
        
    def seed_infections(self):
        """
        Initialize the HIV state.
          - With probability init_prev, agents become HIV (newly infected).
          - Otherwise, they remain ATRISK.
        """
        
        uids = ss.uids(self.state == HIVState.ATRISK)  
        infected, noinfected = self.pars.init_prev.filter(uids, both=True)  # Get the initial infection status.
        n = len(infected)
        self.state[infected] = HIVState.ACUTE
        self.state[noinfected] = HIVState.ATRISK
        self.on_ART[infected] = np.random.rand(n) < self.pars.on_ART  # Randomly assign ART status.

    def step(self):
        """
        Update state transitions based solely on state and ART.
        If an agent is on ART, progression probabilities are reduced.
        """
        if self.ti == 0:  # Skip the first step (initialization)
            self.seed_infections()
            return
        
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0  # dt in weeks (default=1)
        uids = self.sim.people.auids
        current = self.state[uids].copy()
        
        # ART factor for progression:
        art_factor = self.pars['art_progression_factor']

        if self.pars.p_ATRISK_to_ACUTE>0:
            # ATRISK → HIV: (applied only to ATRISK; ART does not apply here)
            p_atrisk_to_HIV = self.pars.p_ATRISK_to_ACUTE
            atrisk_ids = uids[current == HIVState.ATRISK]
            rand_vals = np.random.rand(atrisk_ids.size)
            self.state[atrisk_ids[rand_vals<p_atrisk_to_HIV]] = HIVState.ACUTE
        
        # HIV → LATENT:
        p_ACUTE_to_LATENT = self.pars.p_ACUTE_to_LATENT
        hiv_ids = uids[current == HIVState.ACUTE]
        art_multiplier = np.where(self.on_ART[hiv_ids], art_factor, 1.0) # Apply ART factor
        effective_p = p_ACUTE_to_LATENT*art_multiplier
        rand_vals = np.random.rand(hiv_ids.size)
        self.state[hiv_ids[rand_vals < effective_p]] = HIVState.LATENT
        
        # LATENT → AIDS:
        p_LATENT_to_AIDS = self.pars.p_LATENT_to_AIDS
        latent_ids = uids[current == HIVState.LATENT]
        art_multiplier = np.where(self.on_ART[latent_ids], art_factor, 1.0)  # Apply ART factor
        effective_p = p_LATENT_to_AIDS*art_multiplier
        rand_vals = np.random.rand(latent_ids.size)
        self.state[latent_ids[rand_vals<effective_p]] = HIVState.AIDS
        
        # AIDS → DEAD:
        p_AIDS_to_DEAD  =self.pars.p_AIDS_to_DEAD
        aids_ids = uids[current == HIVState.AIDS]
        art_multiplier = np.where(self.on_ART[aids_ids], art_factor, 1.0)   # Apply ART factor
        effective_p = p_AIDS_to_DEAD*art_multiplier
        rand_vals = np.random.rand(aids_ids.size)   
        self.state[aids_ids[rand_vals<effective_p]] = HIVState.DEAD

    
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
        infected = (states != HIVState.ATRISK) & (states != HIVState.DEAD)
        # res.hiv_prevalence[ti] = infected.sum()/n
        res.atrisk[ti]     = np.count_nonzero(self.state == HIVState.ATRISK) #np.sum(states == HIVState.ATRISK)/n
        res.acute[ti]      = np.count_nonzero(self.state == HIVState.ACUTE) 
        res.latent[ti]     = np.count_nonzero(self.state == HIVState.LATENT) 
        res.aids[ti]       = np.count_nonzero(self.state == HIVState.AIDS) 
        res.dead[ti]       = np.count_nonzero(self.state == HIVState.DEAD) 
        res.n_active[ti]   = np.count_nonzero(np.isin(self.state, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS]))
        # Calculate proportion of agents on ART.
        res.on_art[ti]     = np.count_nonzero(self.on_ART == True)
        
        if n_alive > 0:
            res.hiv_prevalence[ti] = res.n_active[ti] / n_alive 

    
    def set_ART(self, uids, on_ART=True):
        """Set the ART status for specified agents."""
        self.on_ARTs[uids]          = on_ART
