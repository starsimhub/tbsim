#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum

__all__ = ['HIVState', 'HIV']

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
      - ATRISK → ACUTE HIVS: 4 weeks
      - ACUTE → LATENT: 8 weeks
      - LATENT → AIDS: 416 weeks (≈8 years)
      
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
            init_prev = ss.bernoulli(p=0.20),  # Initial prevalence of HIV
            init_onart = ss.bernoulli(p=0.50),  # Initial probability of being on ART (if infected).
            ACUTE_to_LATENT       = ss.perday(1/(7*12)), # 1-np.exp(-1/8),  # 8 weeks
            LATENT_to_AIDS        = ss.perday(1/(365*8)), # 1-np.exp(-1/416), # 416 weeks
            AIDS_to_DEAD          = ss.perday(1/(365*2)), # 1-np.exp(-1/104), # 104 weeks
            art_progression_factor  = 0.1, # Multiplier to reduce progression rates if on ART.
        )
        self.update_pars(pars, **kwargs)
        
        # Define extra attributes for Agents of this disease.
        self.define_states(
            ss.FloatArr('state', default=HIVState.ATRISK),          
            ss.BoolArr('on_ART', default=False),          # Whether agent is on ART.
        )
        return
        
    def set_prognoses(self ):
        # This protects against re-initialization or in case there is an 
        # intervention handling HIV infections and ART before this disease
        # is initialized. It also captures the case where the intervention only requests
        # not both (infections and art) are selected
        
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
        
        dt = self.sim.t.dt if hasattr(self.sim.t, 'dt') else 1.0  # dt in weeks (default=1)
        uids = self.sim.people.auids
        current = self.state[uids].copy()
        
        # ART factor for progression:
        art_factor = self.pars['art_progression_factor']

       
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
 
 
class HivInterventions(ss.Intervention):
    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            mode='both',  # 'infection', 'art', or 'both'
            prevalence=0.20,
            percent_on_ART=0.80,
            minimum_age=0,
            max_age=200,
        )
        self.update_pars(pars, **kwargs)

        # Dispatch table to route logic by mode
        self.handlers = {
            'infection': self._apply_infection,
            'art': self._apply_art,
            'both': self._apply_both,
        }
        
    def step(self):
        # logic for infections
        minimum_age = self.pars.minimum_age
        max_age = self.pars.max_age
        
        alive = len(self.sim.people.alive)
        
        # check if self.pars.prevalence is callable
        if callable(self.pars.prevalence):
            target_prevalence = self.pars.prevalence(self.sim)
        else:
            target_prevalence = self.pars.prevalence
        expected_infectious = int(np.round(alive * target_prevalence))

        
        handler = self.handlers.get(self.pars.mode)
        if handler is None:
            raise ValueError(f"Unsupported mode: {self.pars.mode}, please specify 'infection', 'art', or 'both' ")
        handler()

    def _apply_infection(self):
        self.hiv = self.sim.diseases.hiv
        alive = len(self.sim.people.alive)
        target_prev = self.pars.prevalence(self.sim) if callable(self.pars.prevalence) else self.pars.prevalence
        expected_infectious = int(np.round(alive * target_prev))

        infectious_uids = ((self.hiv.state == HIVState.ACUTE) | (self.hiv.state == HIVState.LATENT) | (self.hiv.state == HIVState.AIDS)).uids
        n_current = len(infectious_uids)
        delta = expected_infectious - n_current

        if delta > 0:
            at_risk_uids = (self.hiv.state == HIVState.ATRISK).uids
            ages = self.sim.people.age[at_risk_uids]
            in_age_range = (ages >= self.pars.minimum_age) & (ages <= (self.pars.max_age or np.inf))
            eligible = at_risk_uids[in_age_range]

            n_to_add = min(delta, len(eligible))
            if n_to_add:
                dist = ss.randint(0, len(eligible), strict=False).init()
                chosen = eligible[dist(n_to_add)]
                self.hiv.state[chosen] = HIVState.ACUTE

        elif delta < 0:
            acute_uids = (self.hiv.state == HIVState.ACUTE).uids
            n_to_remove = min(-delta, len(acute_uids))
            if n_to_remove:
                dist = ss.randint(0, len(acute_uids), strict=False).init()
                chosen = acute_uids[dist(n_to_remove)]
                self.hiv.state[chosen] = HIVState.ATRISK

    def _apply_art(self):
        self.hiv = self.sim.diseases.hiv
        
        alive = len(self.sim.people.alive)
        target_prev = self.pars.prevalence(self.sim) if callable(self.pars.prevalence) else self.pars.prevalence
        expected_on_art = int(np.round(alive * self.pars.percent_on_ART * target_prev))

        current_on_art = (self.hiv.on_ART == True).uids
        n_current = len(current_on_art)
        delta = expected_on_art - n_current

        if delta > 0:
            candidates = ((~self.hiv.on_ART) & (self.hiv.state != HIVState.DEAD) & (self.hiv.state != HIVState.ATRISK)).uids
            n_to_add = min(delta, len(candidates))
            if n_to_add:
                dist = ss.randint(0, len(candidates), strict=False).init()
                chosen = candidates[dist(n_to_add)]
                self.hiv.on_ART[chosen] = True

        elif delta < 0:
            n_to_remove = min(-delta, len(current_on_art))
            if n_to_remove:
                dist = ss.randint(0, len(current_on_art), strict=False).init()
                chosen = current_on_art[dist(n_to_remove)]
                self.hiv.on_ART[chosen] = False

    def _apply_both(self):
        self._apply_infection()
        self._apply_art()
