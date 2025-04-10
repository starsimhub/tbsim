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
    
class HivFunctions():
    @staticmethod
    def seeding(sim, uids, minimum_age=15, max_age=49, p_value=0.10):
        p = np.zeros(len(uids))
        is_atrisk = sim.people.hiv.states[uids] == HIVState.ATRISK
        is_within_age_range = (sim.people.age[uids] >= minimum_age) & (sim.people.age[uids] <= (max_age if max_age is not None else np.inf))
        eligible_uids = uids[is_atrisk & is_within_age_range]
        if len(eligible_uids) == 0:  return p
        p[eligible_uids] = p_value
        return p
    
    def equilibrate_infections(self):

        alive = len(self.sim.people.alive)
        expected_infectious = int(np.round(alive * self.pars.prevalence))

        # Current infectious agents
        infectious_uids = ((self.state == HIVState.ACUTE) | (self.state == HIVState.LATENT)).uids
        n_current = len(infectious_uids)

        # Calculate delta
        delta = expected_infectious - n_current

        if delta > 0:
            # Not enough infections → add more
            at_risk_uids = (self.state == HIVState.ATRISK).uids
            n_to_add = min(delta, len(at_risk_uids))

            if n_to_add > 0:
                dist = ss.randint(low=0, high=len(at_risk_uids), strict=False).init()
                new_infectious = at_risk_uids[dist(n_to_add)]
                self.state[new_infectious] = HIVState.ACUTE

        elif delta < 0:
            # Too many infections → revert some ACUTE cases to ATRISK
            acute_uids = (self.state == HIVState.ACUTE).uids
            n_to_remove = min(-delta, len(acute_uids))  # Convert to positive, cap at available

            if n_to_remove > 0:
                dist = ss.randint(low=0, high=len(acute_uids), strict=False).init()
                to_revert = acute_uids[dist(n_to_remove)]
                self.state[to_revert] = HIVState.ATRISK       
        return
    
    def equilibrate_ART(self):
        alive = len(self.sim.people.alive)
        expected_onart = int(np.round(alive * self.pars.percent_on_ART * self.pars.prevalence))
        
        current_onart_uids = ((self.on_ART == True)).uids
        n_current = len(current_onart_uids)
        
        delta = expected_onart - n_current

        if delta > 0:
            # Need to add people to ART
            candidates = ((
                (~self.on_ART) &
                (self.state != HIVState.DEAD) &
                (self.state != HIVState.ATRISK)
            )).uids
            n_to_add = delta

            if n_to_add > 0:
                dist = ss.randint(low=0, high=len(candidates), strict=False).init()
                selected_uids = candidates[dist(n_to_add)]
                self.on_ART[selected_uids] = True

        elif delta < 0:
            # Need to remove some from ART
            n_to_remove = -delta

            if n_to_remove > 0:
                dist = ss.randint(low=0, high=n_current, strict=False).init()
                selected_uids = current_onart_uids[dist(n_to_remove)]
                self.on_ART[selected_uids] = False

        return

         
class HIV(ss.Disease):
    """
    A simplified HIV disease model that tracks only the agent's state and ART status.
    
    Transitions are stochastic with expected durations:
      - ATRISK → ACUTE HIVS: 4 weeks
      - ACUTE → LATENT: 8 weeks
      - LATENT → AIDS: 416 weeks (≈8 years)
      - AIDS → DEAD: 104 weeks (≈2 years)
      
    If an agent is on ART, progression probabilities are reduced by a factor
      art_progression_factor (default 0.5), thus prolonging the duration in that state.
      
    Parameters:
      - init_prev: Initial prevalence of infection.
      - ACUTE_to_LATENT: Baseline probability per step for HIV → LATENT (1 - exp(-dt/8)).
      - LATENT_to_AIDS: Baseline probability per step for LATENT → AIDS (1 - exp(-dt/416)).
      - AIDS_to_DEAD: Baseline probability per step for AIDS → DEAD (1 - exp(-dt/104)).
      - art_progression_factor: Multiplier to reduce progression rates if on ART.
      
    References:
      - Expert Report to the Infected Blood Inquiry: HIV (Jan 2020) and related guidelines.
      - https://www.infectedbloodinquiry.org.uk/sites/default/files/documents/HIV%20Report%20with%20Addendum.pdf 
    """
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define progression parameters (using a time step in weeks).
        # Define progression parameters (using a time step in weeks).
        self.define_pars(
            # init_prev               = ss.bernoulli(p=0.25 ), # Initial prevalence of HIV
            # init_prev              = ss.bernoulli(p=0.20), # Initial prevalence of HIV (can be a float for fixed probability)
            prevalence              = 0.20, # Prevalence to maintain along the simulation
            # Baseline transition probabilities computed using an exponential waiting time:
            # p = 1 - exp(-dt/mean_duration), with dt assumed to be 1 week.
            
            ATRISK_to_ACUTE       = ss.perday(0.25),  # 1-np.exp(-1/4),  # 4 weeks
            ACUTE_to_LATENT       = ss.perday(0.117), # 1-np.exp(-1/8),  # 8 weeks
            LATENT_to_AIDS        = ss.perday(0.0024), # 1-np.exp(-1/416), # 416 weeks
            AIDS_to_DEAD          = ss.perday(0.0096), # 1-np.exp(-1/104), # 104 weeks
            
            art_progression_factor  = 0.1, # Multiplier to reduce progression rates if on ART.
            percent_on_ART                  = 0.5, # Probability of being on ART (if infected).
        )
        self.update_pars(pars, **kwargs)
        
        # Define extra attributes for Agents of this disease.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK),          
            ss.BoolArr('on_ART', default=False),          # Whether agent is on ART.
        )
        return
        

    def step(self):
        """
        Update state transitions based solely on state and ART.
        If an agent is on ART, progression probabilities are reduced.
        """
        HivFunctions.equilibrate_infections(self)
        HivFunctions.equilibrate_ART(self)
        
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0  # dt in weeks (default=1)
        uids = self.sim.people.auids
        current = self.state[uids].copy()
        
        # ART factor for progression:
        art_factor = self.pars['art_progression_factor']

        if self.pars.ATRISK_to_ACUTE>0:
            # ATRISK → HIV: (applied only to ATRISK; ART does not apply here)
            p_atrisk_to_HIV = self.pars.ATRISK_to_ACUTE
            atrisk_ids = uids[current == HIVState.ATRISK]
            rand_vals = np.random.rand(atrisk_ids.size)
            self.state[atrisk_ids[rand_vals<p_atrisk_to_HIV]] = HIVState.ACUTE
        
        # HIV → LATENT:
        ACUTE_to_LATENT = self.pars.ACUTE_to_LATENT
        hiv_ids = uids[current == HIVState.ACUTE]
        art_multiplier = np.where(self.on_ART[hiv_ids], art_factor, 1.0) # Apply ART factor
        effective_p = ACUTE_to_LATENT*art_multiplier
        rand_vals = np.random.rand(hiv_ids.size)
        self.state[hiv_ids[rand_vals < effective_p]] = HIVState.LATENT
        
        # LATENT → AIDS:
        LATENT_to_AIDS = self.pars.LATENT_to_AIDS
        latent_ids = uids[current == HIVState.LATENT]
        art_multiplier = np.where(self.on_ART[latent_ids], art_factor, 1.0)  # Apply ART factor
        effective_p = LATENT_to_AIDS*art_multiplier
        rand_vals = np.random.rand(latent_ids.size)
        self.state[latent_ids[rand_vals<effective_p]] = HIVState.AIDS

    
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float, label='Prevalence (Infected)'),
            ss.Result(name='atrisk', dtype=float, label='ATRISK'),
            ss.Result(name='acute', dtype=float, label='ACUTE'),
            ss.Result(name='latent', dtype=float, label='LATENT'),
            ss.Result(name='aids', dtype=float, label='AIDS'),
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
        res.n_active[ti]   = np.count_nonzero(np.isin(self.state, [HIVState.ACUTE, HIVState.LATENT, HIVState.AIDS]))
        # Calculate proportion of agents on ART.
        res.on_art[ti]     = np.count_nonzero(self.on_ART == True)
        
        if n_alive > 0:
            res.hiv_prevalence[ti] = res.n_active[ti] / n_alive 
   
