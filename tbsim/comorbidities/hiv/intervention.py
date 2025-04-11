import starsim as ss 
import numpy as np
from tbsim import HIVState

__all__ = ['HivInterventions']

class HivInterventions(ss.Intervention):
    """
    HivInterventions is a class that models HIV-related interventions in a simulation. 
    It allows for the adjustment of HIV prevalence and the proportion of individuals on ART (antiretroviral therapy) 
    within a simulated population. The class supports three modes of operation: 'infection', 'art', and 'both'.
    Attributes:
        pars (dict): Parameters for the intervention, including mode, prevalence, percent_on_ART, minimum_age, and max_age.
        handlers (dict): A dispatch table mapping modes ('infection', 'art', 'both') to their respective handler methods.
    Methods:
        __init__(pars, **kwargs):
            Initializes the HivInterventions object with the given parameters and sets up the dispatch table.
        step():
            Executes the intervention logic based on the specified mode ('infection', 'art', or 'both').
        _apply_infection():
            Adjusts the number of individuals infected with HIV to match the target prevalence. 
            Adds or removes infections as necessary, ensuring the changes respect age constraints.
        _apply_art():
            Adjusts the number of individuals on ART to match the target proportion. 
            Adds or removes individuals from ART as necessary.
        _apply_both():
            Applies both the infection and ART adjustments sequentially.
    Raises:
        ValueError: If an unsupported mode is specified in the parameters.
    Warnings:
        - If there are not enough eligible individuals to infect when increasing prevalence.
        - If there are not enough acute cases to revert when decreasing prevalence.
        - If there are not enough candidates to add to ART when increasing ART coverage.
        - If there are not enough individuals on ART to remove when decreasing ART coverage.
    """
    
    def __init__(self, pars, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            mode='both',  # 'infection', 'art', or 'both'
            prevalence=0.20,
            percent_on_ART=0.80,
            minimum_age=0,
            max_age=200,
            start=ss.date('2000-01-01'),
            stop=ss.date('2035-12-31'),
        )
        self.update_pars(pars, **kwargs)

        # Dispatch table to route logic by mode
        self.handlers = {
            'infection': self._apply_infection,
            'art': self._apply_art,
            'both': self._apply_both,
        }
        
    def step(self):
        t = self.sim.now
        if t < self.pars.start or t > self.pars.stop:
            return
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
        minimum_age = self.pars.minimum_age
        max_age = self.pars.max_age
        
        if delta > 0:
            # Not enough infections → add more
            at_risk_uids = (self.hiv.state == HIVState.ATRISK).uids
            is_within_age_range = ((self.sim.people.age[at_risk_uids] >= minimum_age) 
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

    def _apply_art(self):
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
                          & (self.hiv.state != HIVState.DEAD) 
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
        self._apply_infection()
        self._apply_art()
