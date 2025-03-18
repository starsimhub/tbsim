#!/usr/bin/env python3
import numpy as np
import starsim as ss
from enum import IntEnum
__all__ = ['HIVStage', 'HIV']
# Define HIV stages as an enumeration for clarity.
class HIVStage(IntEnum):
    ACUTE  = 0
    LATENT = 1
    AIDS   = 2

class HIV(ss.Disease):
    """
    A modular HIV disease model that tracks:
      - infection status (infected)
      - disease stage (ACUTE, LATENT, AIDS)
      - CD4 count
      - Viral load
      - ART status (on_ART)
      
    The model is divided into several independent parts:
      A) Initialization (infected status, CD4/viral load, stage assignment)
      B) Drift functions for CD4 and viral load (and associated scales)
      C) A step() function that orchestrates disease progression
      D) A results module that tracks prevalence, mean CD4, and mean viral load
      
    Parameters (defined via self.define_pars):
      - init_prev: initial prevalence of infection.
      - transmission_rate: base transmission rate (if you add transmission logic).
      - cd4_decline_rate: average decline in CD4 per time step (if not on ART).
      - cd4_recovery_rate: average increase in CD4 when on ART.
      - vl_rise_rate: average viral load increase per step (if not on ART).
      - vl_fall_rate: average viral load decrease per step (if on ART).

    References:
      - Expert Report to the Infected Blood Inquiry: HIV (Jan 2020) and related guidelines.
    """
    
    def __init__(self, pars=None, **kwargs):
        super().__init__(**kwargs)
        
        # Define model parameters.
        self.define_pars(
            init_prev         = 0.01,
            transmission_rate = 0.0001,
            cd4_decline_rate  = 5.0,
            cd4_recovery_rate = 10.0,
            vl_rise_rate      = 1000.0,
            vl_fall_rate      = 3000.0,
        )
        self.update_pars(pars, **kwargs)
        
        # Define disease-specific states.
        self.define_states(
            ss.BoolArr('infected',   default=False),
            ss.FloatArr('stage',       default=-1),  # will hold HIVStage values
            ss.BoolArr('on_ART',     default=False),
            ss.FloatArr('cd4_count', default=ss.constant(800, name='cd4_count')),
            ss.FloatArr('viral_load',default=ss.constant(0,   name='viral_load')),
        )
        
        # Create random-walk updaters for CD4 and viral load.
        self.cd4_rand = ss.normal(loc=self.cd4_drift, scale=self.cd4_scale)
        self.vl_rand  = ss.normal(loc=self.vl_drift,  scale=self.vl_scale)
    
    ###########################################################################
    # A) INITIALIZATION METHODS
    ###########################################################################
    def set_initial_states(self, sim):
        """Initialize infected status, CD4/viral load values, and disease stage."""
        self.init_infections(sim)
        self.init_cd4_and_vl(sim)
        self.init_stage(sim)
    
    def init_infections(self, sim):
        """Randomly assign infected status based on init_prev."""
        uids = sim.people.auids
        n = len(uids)
        infected_draw = np.random.rand(n) < self.pars['init_prev']
        self.infected[uids] = infected_draw
    
    def init_cd4_and_vl(self, sim):
        """
        For infected individuals, set initial CD4 count and viral load.
        For demonstration, we assume:
          - CD4 count uniformly between 400 and 900
          - Viral load uniformly between 10,000 and 50,000
        Uninfected individuals remain at defaults.
        """
        uids = sim.people.auids
        inf = self.infected[uids]
        if inf.any():
            self.cd4_count[uids[inf]] = np.random.uniform(400, 900, size=inf.sum())
            self.viral_load[uids[inf]] = np.random.uniform(1e4, 5e4, size=inf.sum())
    
    def init_stage(self, sim):
        """
        Initialize disease stage for infected individuals.
        For demonstration, assign half as ACUTE and half as LATENT.
        """
        uids = sim.people.auids
        inf = self.infected[uids]
        if inf.any():
            infected_idxs = np.where(inf)[0]
            # 50% chance to be ACUTE
            acute_mask = np.random.rand(inf.sum()) < 0.5
            self.stage[uids[infected_idxs[acute_mask]]] = HIVStage.ACUTE
            self.stage[uids[infected_idxs[~acute_mask]]] = HIVStage.LATENT
    
    ###########################################################################
    # B) DISEASE PROGRESSION: DRIFT FUNCTIONS & SCALE
    ###########################################################################
    @staticmethod
    def cd4_drift(self, sim, uids):
        """
        Compute the change in CD4 count.
          - Infected & not on ART: negative drift (decline)
          - Infected & on ART: positive drift (recovery)
          - Uninfected: no change
        """
        loc = np.zeros(len(uids))
        inf = self.infected[uids]
        if inf.any():
            on_art = self.on_ART[uids[inf]]
            # Default: decline in CD4
            drift_vals = np.full(inf.sum(), -self.pars['cd4_decline_rate'])
            drift_vals[on_art] = self.pars['cd4_recovery_rate']
            loc[inf] = drift_vals
        return loc
    
    @staticmethod
    def cd4_scale(self, sim, uids):
        """Return constant standard deviation for CD4 drift."""
        return np.full(len(uids), 3.0)
    
    @staticmethod
    def vl_drift(self, sim, uids):
        """
        Compute the change in viral load.
          - Infected & not on ART: viral load increases.
          - Infected & on ART: viral load decreases.
          - Uninfected: no change.
        """
        loc = np.zeros(len(uids))
        inf = self.infected[uids]
        if inf.any():
            on_art = self.on_ART[uids[inf]]
            drift_vals = np.full(inf.sum(), self.pars['vl_rise_rate'])
            drift_vals[on_art] = -self.pars['vl_fall_rate']
            loc[inf] = drift_vals
        return loc
    
    @staticmethod
    def vl_scale(self, sim, uids):
        """Return constant standard deviation for viral load drift."""
        return np.full(len(uids), 500.0)
    
    ###########################################################################
    # C) STEP AND DISEASE PROGRESSION
    ###########################################################################
    def step(self):
        """Orchestrate a single time step in disease progression."""
        self.apply_disease_progression()
    
    def apply_disease_progression(self):
        """Update CD4 count and viral load for infected individuals, then update stage."""
        uids = self.sim.people.auids
        inf = self.infected[uids]
        if inf.any():
            self.cd4_count[uids[inf]] += self.cd4_rand(uids[inf])
            self.viral_load[uids[inf]] += self.vl_rand(uids[inf])
            
            # Clip values to avoid negatives.
            self.cd4_count[uids[inf]] = np.clip(self.cd4_count[uids[inf]], 0, None)
            self.viral_load[uids[inf]] = np.clip(self.viral_load[uids[inf]], 0, None)
            
            # Update disease stage (for example: if CD4 < 200, transition to AIDS)
            self.transition_stages(uids[inf])
    
    def transition_stages(self, iuids):
        """
        A simple transition rule:
          - If an infected agent’s CD4 count falls below 200, mark the stage as AIDS.
        Additional transitions (e.g., from ACUTE to LATENT) could be added here.
        """
        cd4_vals = self.cd4_count[iuids]
        stage = self.stage[iuids]
        new_aids = cd4_vals < 200
        stage[new_aids] = HIVStage.AIDS
        self.stage[iuids] = stage
    
    ###########################################################################
    # D) RESULTS TRACKING
    ###########################################################################
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result(name='hiv_prevalence', dtype=float),
            ss.Result(name='mean_cd4', dtype=float),
            ss.Result(name='mean_vl',  dtype=float),
        )
    
    def update_results(self):
        super().update_results()
        ti = self.sim.ti
        uids = self.sim.people.auids
        n = len(uids)
        inf = self.infected[uids]
        
        self.results.hiv_prevalence[ti] = inf.sum() / n
        if inf.any():
            self.results.mean_cd4[ti] = self.cd4_count[uids[inf]].mean()
            self.results.mean_vl[ti]  = self.viral_load[uids[inf]].mean()
        else:
            self.results.mean_cd4[ti] = 0.0
            self.results.mean_vl[ti]  = 0.0

    ###########################################################################
    # E) HELPER METHODS
    ###########################################################################
    def set_ART(self, uids, on_art=True):
        """Switch the ART status for specified agents."""
        self.on_ART[uids] = on_art

###############################################################################
# EXAMPLE USAGE: DUMMY SIMULATION ENVIRONMENT
###############################################################################
if __name__ == "__main__":
    # Create a minimal dummy simulation environment for demonstration.
    class DummySim:
        def __init__(self, n_agents=1000):
            self.pars = {'n_agents': n_agents}
            self.ti = 0  # time index
            # Create a dummy people object with agent unique ids.
            self.people = type("People", (), {})()
            self.people.auids = np.arange(n_agents)
    
    # Instantiate the dummy simulation.
    sim = DummySim(n_agents=1000)
    
    # Create an instance of the HIV model.
    model = HIV()
    model.sim = sim  # attach the simulation reference to the model
    
    # Initialize the disease states.
    model.set_initial_states(sim)
    
    # Run a single time step.
    model.step()
    
    # Initialize and update results.
    model.init_results()
    model.update_results()
    
    # Print results.
    print("HIV Prevalence:", model.results.hiv_prevalence[sim.ti])
    print("Mean CD4:", model.results.mean_cd4[sim.ti])
    print("Mean Viral Load:", model.results.mean_vl[sim.ti])
