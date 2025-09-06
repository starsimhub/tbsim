"""
TB State Machine Implementation

This module implements a comprehensive state machine for TB disease progression that:
- Treats states as objects with encapsulated behavior
- Handles what can be triggered from each state
- Knows when to progress to different states
- Manages transition order and rates
- Provides a centralized state machine manager

The state machine is designed to be flexible, extensible, and maintainable while
integrating seamlessly with the existing Starsim framework.
"""

import numpy as np
import starsim as ss
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import IntEnum
from dataclasses import dataclass
import logging

from .tb import TBS


__all__ = [
    'TBState', 'TBStateMachine', 'TBStateManager', 'TBTransition',
    'LatentSlowState', 'LatentFastState', 'ActivePresympState', 
    'ActiveSmposState', 'ActiveSmnegState', 'ActiveExptbState',
    'ClearState', 'DeadState', 'ProtectedState'
]


@dataclass
class TBTransition:
    """
    Represents a transition between TB states with associated conditions and rates.
    
    Attributes:
        target_state: The state to transition to
        rate: The transition rate (per day/year)
        condition: Optional condition function that must be true for transition
        probability_func: Function to calculate transition probability
        priority: Order of execution (lower numbers execute first)
    """
    target_state: 'TBState'
    rate: ss.TimePar
    condition: Optional[Callable] = None
    probability_func: Optional[Callable] = None
    priority: int = 0
    
    def can_transition(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """
        Check if transition is possible for given UIDs.
        
        Args:
            uids: Array of individual IDs to check
            sim: Simulation object
            module: TB module instance
            
        Returns:
            Boolean array indicating which individuals can transition
        """
        if self.condition is None:
            return np.ones(len(uids), dtype=bool)
        return self.condition(uids, sim, module)
    
    def calculate_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """
        Calculate transition probability for given UIDs.
        
        Args:
            uids: Array of individual IDs
            sim: Simulation object
            module: TB module instance
            
        Returns:
            Array of transition probabilities
        """
        if self.probability_func is not None:
            return self.probability_func(uids, sim, module)
        
        # Default probability calculation from rate
        rate_val = self.rate.rate
        unit = self.rate.unit
        rate = ss.per(rate_val, unit=unit)
        return rate.to_prob()


class TBState(ABC):
    """
    Abstract base class for TB disease states.
    
    Each state encapsulates its behavior, transitions, and properties.
    States are responsible for:
    - Defining what transitions are possible
    - Calculating transition probabilities
    - Managing state-specific properties
    - Handling entry and exit actions
    """
    
    def __init__(self, state_id: TBS, name: str, description: str = ""):
        self.state_id = state_id
        self.name = name
        self.description = description
        self.transitions: List[TBTransition] = []
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        """Check if individuals in this state are infectious."""
        pass
    
    @abstractmethod
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        """Get relative transmission rate for individuals in this state."""
        pass
    
    def add_transition(self, transition: TBTransition):
        """Add a possible transition from this state."""
        self.transitions.append(transition)
        # Sort by priority
        self.transitions.sort(key=lambda t: t.priority)
    
    def get_possible_transitions(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> List[Tuple[TBTransition, np.ndarray]]:
        """
        Get all possible transitions for given UIDs.
        
        Returns:
            List of (transition, valid_uids) tuples
        """
        possible_transitions = []
        for transition in self.transitions:
            valid_uids = uids[transition.can_transition(uids, sim, module)]
            if len(valid_uids) > 0:
                possible_transitions.append((transition, valid_uids))
        return possible_transitions
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Called when individuals enter this state."""
        pass
    
    def on_exit(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Called when individuals exit this state."""
        pass
    
    def update_state_properties(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Update state-specific properties for individuals in this state."""
        pass


class LatentSlowState(TBState):
    """Latent TB state with slow progression to active disease."""
    
    def __init__(self):
        super().__init__(TBS.LATENT_SLOW, "Latent Slow", "Latent TB with slow progression")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids))
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Set susceptibility for slow progressors."""
        module.susceptible[uids] = True  # Slow progressors remain susceptible
        module.latent_tb_state[uids] = TBS.LATENT_SLOW


class LatentFastState(TBState):
    """Latent TB state with fast progression to active disease."""
    
    def __init__(self):
        super().__init__(TBS.LATENT_FAST, "Latent Fast", "Latent TB with fast progression")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids))
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Set non-susceptibility for fast progressors."""
        module.susceptible[uids] = False  # Fast progressors become non-susceptible
        module.latent_tb_state[uids] = TBS.LATENT_FAST


class ActivePresympState(TBState):
    """Active TB in pre-symptomatic phase."""
    
    def __init__(self):
        super().__init__(TBS.ACTIVE_PRESYMP, "Active Pre-symptomatic", "Active TB in pre-symptomatic phase")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.ones(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.full(len(uids), module.pars.rel_trans_presymp)
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Set non-susceptibility and track timing."""
        module.susceptible[uids] = False
        module.ti_presymp[uids] = sim.ti
        module.ti_cur[uids] = sim.ti


class ActiveSmposState(TBState):
    """Active TB, smear positive (most infectious)."""
    
    def __init__(self):
        super().__init__(TBS.ACTIVE_SMPOS, "Active Smear Positive", "Active TB, smear positive")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.ones(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        base_rate = np.full(len(uids), module.pars.rel_trans_smpos)
        # Apply treatment effect if on treatment
        treatment_mask = module.on_treatment[uids]
        base_rate[treatment_mask] *= module.pars.rel_trans_treatment
        return base_rate
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Track timing and set active state."""
        module.ti_active[uids] = sim.ti
        module.ti_cur[uids] = sim.ti
        module.active_tb_state[uids] = TBS.ACTIVE_SMPOS


class ActiveSmnegState(TBState):
    """Active TB, smear negative (moderately infectious)."""
    
    def __init__(self):
        super().__init__(TBS.ACTIVE_SMNEG, "Active Smear Negative", "Active TB, smear negative")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.ones(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        base_rate = np.full(len(uids), module.pars.rel_trans_smneg)
        # Apply treatment effect if on treatment
        treatment_mask = module.on_treatment[uids]
        base_rate[treatment_mask] *= module.pars.rel_trans_treatment
        return base_rate
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Track timing and set active state."""
        module.ti_active[uids] = sim.ti
        module.ti_cur[uids] = sim.ti
        module.active_tb_state[uids] = TBS.ACTIVE_SMNEG


class ActiveExptbState(TBState):
    """Active TB, extra-pulmonary (least infectious)."""
    
    def __init__(self):
        super().__init__(TBS.ACTIVE_EXPTB, "Active Extra-pulmonary", "Active TB, extra-pulmonary")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.ones(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        base_rate = np.full(len(uids), module.pars.rel_trans_exptb)
        # Apply treatment effect if on treatment
        treatment_mask = module.on_treatment[uids]
        base_rate[treatment_mask] *= module.pars.rel_trans_treatment
        return base_rate
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Track timing and set active state."""
        module.ti_active[uids] = sim.ti
        module.ti_cur[uids] = sim.ti
        module.active_tb_state[uids] = TBS.ACTIVE_EXPTB


class ClearState(TBState):
    """Cleared TB state (returned to susceptible)."""
    
    def __init__(self):
        super().__init__(TBS.NONE, "Clear", "Cleared TB infection")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids))
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Reset all TB-related states."""
        module.susceptible[uids] = True
        module.infected[uids] = False
        module.on_treatment[uids] = False
        module.active_tb_state[uids] = TBS.NONE
        module.ti_presymp[uids] = np.nan
        module.ti_active[uids] = np.nan
        module.ti_cur[uids] = sim.ti


class DeadState(TBState):
    """Death from TB."""
    
    def __init__(self):
        super().__init__(TBS.DEAD, "Dead", "Death from TB")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids))
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Request death from simulation framework."""
        sim.people.request_death(uids)
        module.ti_cur[uids] = sim.ti


class ProtectedState(TBState):
    """Protected from TB (e.g., BCG vaccination)."""
    
    def __init__(self):
        super().__init__(TBS.PROTECTED, "Protected", "Protected from TB")
    
    def is_infectious(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids), dtype=bool)
    
    def get_transmission_rate(self, uids: np.ndarray, module: Any) -> np.ndarray:
        return np.zeros(len(uids))
    
    def on_entry(self, uids: np.ndarray, sim: 'ss.Sim', module: Any):
        """Set non-susceptibility for protected individuals."""
        module.susceptible[uids] = False


class TBStateMachine:
    """
    Core state machine that manages TB state transitions.
    
    This class orchestrates all state transitions, ensuring proper order
    and handling of transition probabilities and conditions.
    """
    
    def __init__(self):
        self.states: Dict[TBS, TBState] = {}
        self.logger = logging.getLogger(f"{__name__}.TBStateMachine")
        self._initialize_states()
        self._setup_transitions()
    
    def _initialize_states(self):
        """Initialize all TB states."""
        self.states = {
            TBS.LATENT_SLOW: LatentSlowState(),
            TBS.LATENT_FAST: LatentFastState(),
            TBS.ACTIVE_PRESYMP: ActivePresympState(),
            TBS.ACTIVE_SMPOS: ActiveSmposState(),
            TBS.ACTIVE_SMNEG: ActiveSmnegState(),
            TBS.ACTIVE_EXPTB: ActiveExptbState(),
            TBS.NONE: ClearState(),
            TBS.DEAD: DeadState(),
            TBS.PROTECTED: ProtectedState(),
        }
    
    def _setup_transitions(self):
        """Set up all possible state transitions with their rates and conditions."""
        
        # Latent Slow -> Pre-symptomatic
        latent_slow = self.states[TBS.LATENT_SLOW]
        latent_slow.add_transition(TBTransition(
            target_state=self.states[TBS.ACTIVE_PRESYMP],
            rate=ss.perday(3e-5),  # rate_LS_to_presym
            probability_func=self._latent_to_presym_probability,
            priority=1
        ))
        
        # Latent Fast -> Pre-symptomatic
        latent_fast = self.states[TBS.LATENT_FAST]
        latent_fast.add_transition(TBTransition(
            target_state=self.states[TBS.ACTIVE_PRESYMP],
            rate=ss.perday(6e-3),  # rate_LF_to_presym
            probability_func=self._latent_to_presym_probability,
            priority=1
        ))
        
        # Pre-symptomatic -> Clear (treatment only)
        presymp = self.states[TBS.ACTIVE_PRESYMP]
        presymp.add_transition(TBTransition(
            target_state=self.states[TBS.NONE],
            rate=ss.peryear(6),  # rate_treatment_to_clear
            condition=self._on_treatment_condition,
            probability_func=self._presym_to_clear_probability,
            priority=1
        ))
        
        # Pre-symptomatic -> Active (any active state)
        presymp.add_transition(TBTransition(
            target_state=None,  # Will be determined by active_tb_state
            rate=ss.perday(3e-2),  # rate_presym_to_active
            probability_func=self._presym_to_active_probability,
            priority=2
        ))
        
        # Active states -> Clear
        for active_state in [TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB]:
            state = self.states[active_state]
            state.add_transition(TBTransition(
                target_state=self.states[TBS.NONE],
                rate=ss.perday(2.4e-4),  # rate_active_to_clear
                probability_func=self._active_to_clear_probability,
                priority=1
            ))
        
        # Active states -> Death
        smpos = self.states[TBS.ACTIVE_SMPOS]
        smpos.add_transition(TBTransition(
            target_state=self.states[TBS.DEAD],
            rate=ss.perday(4.5e-4),  # rate_smpos_to_dead
            condition=self._not_on_treatment_condition,
            probability_func=self._active_to_death_probability,
            priority=2
        ))
        
        smneg = self.states[TBS.ACTIVE_SMNEG]
        smneg.add_transition(TBTransition(
            target_state=self.states[TBS.DEAD],
            rate=ss.perday(0.3 * 4.5e-4),  # rate_smneg_to_dead
            condition=self._not_on_treatment_condition,
            probability_func=self._active_to_death_probability,
            priority=2
        ))
        
        exptb = self.states[TBS.ACTIVE_EXPTB]
        exptb.add_transition(TBTransition(
            target_state=self.states[TBS.DEAD],
            rate=ss.perday(0.15 * 4.5e-4),  # rate_exptb_to_dead
            condition=self._not_on_treatment_condition,
            probability_func=self._active_to_death_probability,
            priority=2
        ))
    
    # Condition functions
    def _on_treatment_condition(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        return module.on_treatment[uids]
    
    def _not_on_treatment_condition(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        return ~module.on_treatment[uids]
    
    # Probability calculation functions
    def _latent_to_presym_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """Calculate latent to pre-symptomatic transition probability."""
        # Get base rates and ensure consistent units
        unit = module.pars.rate_LS_to_presym.unit
        ls_rate_val = module.pars.rate_LS_to_presym.rate
        lf_rate_val = ss.per(module.pars.rate_LF_to_presym, unit).rate
        
        # Initialize rate array with slow rate for all individuals
        ratevals_arr = np.full(len(uids), ls_rate_val)
        
        # Apply fast rate to individuals in latent fast state
        fast_arr = module.state[uids] == TBS.LATENT_FAST
        ratevals_arr[fast_arr] = lf_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= module.rr_activation[uids]
        
        # Convert to Starsim rate object and calculate probability
        rate = ss.per(ratevals_arr, unit=unit)
        return rate.to_prob()
    
    def _presym_to_clear_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """Calculate pre-symptomatic to clear transition probability."""
        base_rate = module.pars.rate_treatment_to_clear.rate
        unit = module.pars.rate_treatment_to_clear.unit
        
        # Create rate array - zero for untreated, treatment rate for treated
        ratevals_arr = np.zeros(len(uids))
        ratevals_arr[module.on_treatment[uids]] = base_rate
        
        rate = ss.per(ratevals_arr, unit=unit)
        return rate.to_prob()
    
    def _presym_to_active_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """Calculate pre-symptomatic to active transition probability."""
        rate_val = module.pars.rate_presym_to_active.rate
        unit = module.pars.rate_presym_to_active.unit
        
        ratevals_arr = np.full(len(uids), rate_val)
        rate = ss.per(ratevals_arr, unit=unit)
        return rate.to_prob()
    
    def _active_to_clear_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """Calculate active to clear transition probability."""
        base_rate_val = module.pars.rate_active_to_clear.rate
        treatment_rate_val = module.pars.rate_treatment_to_clear.rate
        unit = module.pars.rate_active_to_clear.unit
        
        # Create rate array with base rate for all individuals
        ratevals_arr = np.full(len(uids), base_rate_val)
        
        # Apply treatment rate to those on treatment
        ratevals_arr[module.on_treatment[uids]] = treatment_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= module.rr_clearance[uids]
        
        rate = ss.per(ratevals_arr, unit=unit)
        return rate.to_prob()
    
    def _active_to_death_probability(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> np.ndarray:
        """Calculate active to death transition probability."""
        # Get death rates and unit
        smpos_rate_val = module.pars.rate_smpos_to_dead.rate
        smneg_rate_val = module.pars.rate_smneg_to_dead.rate
        exptb_rate_val = module.pars.rate_exptb_to_dead.rate
        unit = module.pars.rate_exptb_to_dead.unit
        
        # Create rate array with extra-pulmonary rate as default
        ratevals_arr = np.full(len(uids), exptb_rate_val)
        
        # Apply appropriate rates based on active TB type
        ratevals_arr[module.state[uids] == TBS.ACTIVE_SMPOS] = smpos_rate_val
        ratevals_arr[module.state[uids] == TBS.ACTIVE_SMNEG] = smneg_rate_val
        
        # Apply individual relative risk modifiers
        ratevals_arr *= module.rr_death[uids]
        
        rate = ss.per(ratevals_arr, unit=unit)
        return rate.to_prob()
    
    def get_state(self, state_id: TBS) -> TBState:
        """Get state object by ID."""
        return self.states[state_id]
    
    def process_transitions(self, uids: np.ndarray, sim: 'ss.Sim', module: Any) -> Dict[TBS, np.ndarray]:
        """
        Process all possible state transitions for given UIDs.
        
        Args:
            uids: Array of individual IDs to process
            sim: Simulation object
            module: TB module instance
            
        Returns:
            Dictionary mapping target states to arrays of UIDs transitioning to them
        """
        transitions_made = {}
        
        # Group UIDs by current state
        state_groups = {}
        for uid in uids:
            current_state = module.state[uid]
            if current_state not in state_groups:
                state_groups[current_state] = []
            state_groups[current_state].append(uid)
        
        # Process transitions for each state group
        for current_state_id, state_uids in state_groups.items():
            if current_state_id not in self.states:
                continue
                
            state_uids = np.array(state_uids)
            current_state = self.states[current_state_id]
            
            # Get possible transitions
            possible_transitions = current_state.get_possible_transitions(state_uids, sim, module)
            
            # Process transitions in priority order
            for transition, valid_uids in possible_transitions:
                if len(valid_uids) == 0:
                    continue
                
                # Calculate transition probabilities
                probs = transition.calculate_probability(valid_uids, sim, module)
                
                # Apply random transitions
                transition_mask = np.random.random(len(probs)) < probs
                transitioning_uids = valid_uids[transition_mask]
                
                if len(transitioning_uids) > 0:
                    # Handle special case for pre-symptomatic to active
                    if transition.target_state is None:
                        # Determine target state based on active_tb_state
                        target_states = module.active_tb_state[transitioning_uids]
                        for target_state_id in np.unique(target_states):
                            mask = target_states == target_state_id
                            target_uids = transitioning_uids[mask]
                            if target_state_id not in transitions_made:
                                transitions_made[target_state_id] = []
                            transitions_made[target_state_id].extend(target_uids)
                    else:
                        target_state_id = transition.target_state.state_id
                        if target_state_id not in transitions_made:
                            transitions_made[target_state_id] = []
                        transitions_made[target_state_id].extend(transitioning_uids)
        
        # Convert lists to arrays
        for state_id in transitions_made:
            transitions_made[state_id] = np.array(transitions_made[state_id])
        
        return transitions_made


class TBStateManager:
    """
    High-level manager for TB state machine operations.
    
    This class provides the main interface for integrating the state machine
    with the TB module and Starsim framework.
    """
    
    def __init__(self):
        self.state_machine = TBStateMachine()
        self.logger = logging.getLogger(f"{__name__}.TBStateManager")
    
    def initialize_states(self, module: Any):
        """Initialize state machine with TB module."""
        # Set up initial state transitions based on module parameters
        pass
    
    def process_time_step(self, module: Any) -> Dict[str, Any]:
        """
        Process one time step of state transitions.
        
        Args:
            module: TB module instance
            
        Returns:
            Dictionary with transition results and statistics
        """
        sim = module.sim
        ti = sim.ti
        
        # Get all living individuals
        living_uids = module.sim.people.auids
        
        # Process all state transitions
        transitions_made = self.state_machine.process_transitions(living_uids, sim, module)
        
        # Apply state transitions
        results = {
            'new_active': 0,
            'new_active_15+': 0,
            'new_deaths': 0,
            'new_deaths_15+': 0,
            'new_clear': 0,
            'transitions_by_state': {}
        }
        
        for target_state_id, transitioning_uids in transitions_made.items():
            if len(transitioning_uids) == 0:
                continue
            
            # Get target state object
            target_state = self.state_machine.get_state(target_state_id)
            
            # Call exit actions for current states
            current_states = module.state[transitioning_uids]
            for current_state_id in np.unique(current_states):
                current_state = self.state_machine.get_state(current_state_id)
                current_uids = transitioning_uids[current_states == current_state_id]
                current_state.on_exit(current_uids, sim, module)
            
            # Update state
            module.state[transitioning_uids] = target_state_id
            
            # Call entry actions
            target_state.on_entry(transitioning_uids, sim, module)
            
            # Update results
            results['transitions_by_state'][target_state.name] = len(transitioning_uids)
            
            # Track specific outcomes
            if target_state_id in TBS.all_active():
                results['new_active'] += len(transitioning_uids)
                results['new_active_15+'] += np.count_nonzero(sim.people.age[transitioning_uids] >= 15)
            elif target_state_id == TBS.DEAD:
                results['new_deaths'] += len(transitioning_uids)
                results['new_deaths_15+'] += np.count_nonzero(sim.people.age[transitioning_uids] >= 15)
            elif target_state_id == TBS.NONE:
                results['new_clear'] += len(transitioning_uids)
        
        # Update transmission rates
        self._update_transmission_rates(module)
        
        # Reset risk modifiers for next time step
        self._reset_risk_modifiers(module)
        
        return results
    
    def _update_transmission_rates(self, module: Any):
        """Update relative transmission rates based on current states."""
        # Reset all relative transmission rates
        module.rel_trans[:] = 1.0
        
        # Update transmission rates for each state
        for state_id, state_obj in self.state_machine.states.items():
            uids = module.state == state_id
            if uids.any():
                transmission_rates = state_obj.get_transmission_rate(uids.uids, module)
                module.rel_trans[uids.uids] *= transmission_rates
        
        # Apply transmission heterogeneity
        infectious_uids = module.infectious.uids
        if len(infectious_uids) > 0:
            module.rel_trans[infectious_uids] *= module.reltrans_het[infectious_uids]
    
    def _reset_risk_modifiers(self, module: Any):
        """Reset individual risk modifiers for next time step."""
        uids = module.sim.people.auids
        module.rr_activation[uids] = 1.0
        module.rr_clearance[uids] = 1.0
        module.rr_death[uids] = 1.0
    
    def get_state_statistics(self, module: Any) -> Dict[str, int]:
        """Get current state distribution statistics."""
        stats = {}
        for state_id, state_obj in self.state_machine.states.items():
            count = np.count_nonzero(module.state == state_id)
            stats[state_obj.name] = count
        return stats
    
    def get_transition_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get transition matrix showing all possible transitions and their rates."""
        matrix = {}
        for state_id, state_obj in self.state_machine.states.items():
            for transition in state_obj.transitions:
                if transition.target_state is not None:
                    key = (state_obj.name, transition.target_state.name)
                    matrix[key] = transition.rate.rate
        return matrix
