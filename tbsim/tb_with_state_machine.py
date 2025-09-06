"""
TB Disease Model with Integrated State Machine

This module provides an enhanced TB disease model that integrates the comprehensive
state machine implementation. It maintains full compatibility with the existing
Starsim framework while providing improved state management capabilities.

The integration demonstrates how the state machine can be used to:
- Encapsulate state behavior and transitions
- Provide cleaner separation of concerns
- Enable easier testing and validation
- Support more complex state logic
"""

import numpy as np
import starsim as ss
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any, Optional

from .tb import TB, TBS
from .state_machine import TBStateManager, TBStateMachine

__all__ = ['TBWithStateMachine']


class TBWithStateMachine(TB):
    """
    Enhanced TB disease model with integrated state machine.
    
    This class extends the original TB class to use the comprehensive state machine
    for managing disease progression. It maintains full compatibility with the
    existing Starsim framework while providing improved state management.
    
    Key Features:
    - Encapsulated state behavior and transitions
    - Centralized state machine management
    - Improved separation of concerns
    - Enhanced testing and validation capabilities
    - Backward compatibility with existing TB model
    
    The state machine handles all state transitions automatically, while this class
    provides the interface to the Starsim framework and maintains result tracking.
    """
    
    def __init__(self, pars=None, use_state_machine: bool = True, **kwargs):
        """
        Initialize TB model with optional state machine integration.
        
        Args:
            pars: Dictionary of parameters to override defaults
            use_state_machine: Whether to use the state machine for transitions
            **kwargs: Additional keyword arguments for parameters
        """
        super().__init__(pars, **kwargs)
        
        self.use_state_machine = use_state_machine
        self.state_manager: Optional[TBStateManager] = None
        
        if self.use_state_machine:
            self.state_manager = TBStateManager()
            self._initialize_state_machine()
    
    def _initialize_state_machine(self):
        """Initialize the state machine with TB module parameters."""
        if self.state_manager is None:
            return
        
        # The state machine is already initialized with default transitions
        # Additional customization can be done here if needed
        if hasattr(self, 'logger'):
            self.logger.info("State machine initialized for TB model")
    
    def step(self):
        """
        Execute one simulation time step for TB disease progression.
        
        This method uses the state machine to handle all state transitions
        automatically, providing cleaner separation of concerns and better
        maintainability compared to the original manual transition approach.
        
        **State Machine Integration:**
        - All state transitions are handled by the state machine
        - Transition probabilities and conditions are encapsulated in state objects
        - State-specific behavior is managed by individual state classes
        - Results are tracked and returned for analysis
        
        **Backward Compatibility:**
        - Maintains the same interface as the original TB class
        - Results structure is identical to the original implementation
        - Can be used as a drop-in replacement for the original TB class
        """
        if not self.use_state_machine:
            # Use original implementation
            super().step()
            return
        
        # Use state machine for all transitions (this replaces the original transition logic)
        # First, handle SIR model updates (infection, demographics, etc.)
        self._step_sir_updates()
        
        # Then, use state machine for TB-specific transitions
        self._step_with_state_machine()
    
    def _step_sir_updates(self):
        """Handle SIR model updates (infection, demographics, etc.) without TB transitions."""
        # Call the parent class (ss.Infection) step method to handle SIR updates
        # but skip the TB-specific transitions that are in the TB class
        super(TB, self).step()
    
    def _step_original(self):
        """Original step implementation for backward compatibility."""
        # This is the original step() method from the TB class
        # Included here for reference and fallback functionality
        p = self.pars
        ti = self.ti

        # Latent --> active pre-symptomatic
        latent_uids = (((self.state == TBS.LATENT_SLOW) | (self.state == TBS.LATENT_FAST))).uids
        if len(latent_uids):
            probs = self.p_latent_to_presym(self.sim, latent_uids)
            transition_mask = np.random.random(len(probs)) < probs
            new_presymp_uids = latent_uids[transition_mask]
        else:
            new_presymp_uids = np.array([], dtype=int)
        if len(new_presymp_uids):
            self.state[new_presymp_uids] = TBS.ACTIVE_PRESYMP
            self.ti_cur[new_presymp_uids] = ti
            self.ti_presymp[new_presymp_uids] = ti
            self.susceptible[new_presymp_uids] = False
        self.results['new_active'][ti] = len(new_presymp_uids)
        self.results['new_active_15+'][ti] = np.count_nonzero(self.sim.people.age[new_presymp_uids] >= 15)

        # Pre-symptomatic --> Active or Clear
        presym_uids = (self.state == TBS.ACTIVE_PRESYMP).uids
        new_clear_presymp_uids = np.array([], dtype=int)
        new_active_uids = np.array([], dtype=int)
        if len(presym_uids):
            # Pre-symptomatic --> Clear (if on treatment)
            clear_probs = self.p_presym_to_clear(self.sim, presym_uids)
            clear_mask = np.random.random(len(clear_probs)) < clear_probs
            new_clear_presymp_uids = presym_uids[clear_mask]

            # Pre-symptomatic --> Active
            active_probs = self.p_presym_to_active(self.sim, presym_uids)
            active_mask = np.random.random(len(active_probs)) < active_probs
            new_active_uids = presym_uids[active_mask]
            if len(new_active_uids):
                active_state = self.active_tb_state[new_active_uids] 
                self.state[new_active_uids] = active_state
                self.ti_cur[new_active_uids] = ti
                self.ti_active[new_active_uids] = ti

        # Active --> Susceptible via natural recovery or treatment (clear)
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMNEG) | (self.state == TBS.ACTIVE_EXPTB))).uids
        if len(active_uids):
            clear_probs = self.p_active_to_clear(self.sim, active_uids)
            clear_mask = np.random.random(len(clear_probs)) < clear_probs
            new_clear_active_uids = active_uids[clear_mask]
        else:
            new_clear_active_uids = np.array([], dtype=int)
        new_clear_uids = ss.uids.cat(new_clear_presymp_uids, new_clear_active_uids)
        if len(new_clear_uids):
            self.susceptible[new_clear_uids] = True
            self.infected[new_clear_uids] = False
            self.state[new_clear_uids] = TBS.NONE
            self.ti_cur[new_clear_uids] = ti
            self.active_tb_state[new_clear_uids] = TBS.NONE
            self.ti_presymp[new_clear_uids] = np.nan
            self.ti_active[new_clear_uids] = np.nan
            self.on_treatment[new_clear_uids] = False

        # Active --> Death
        active_uids = (((self.state == TBS.ACTIVE_SMPOS) | (self.state == TBS.ACTIVE_SMNEG) | (self.state == TBS.ACTIVE_EXPTB))).uids
        if len(active_uids):
            death_probs = self.p_active_to_death(self.sim, active_uids)
            death_mask = np.random.random(len(death_probs)) < death_probs
            new_death_uids = active_uids[death_mask]
        else:
            new_death_uids = np.array([], dtype=int)
        if len(new_death_uids):
            self.sim.people.request_death(new_death_uids)
            self.state[new_death_uids] = TBS.DEAD
            self.ti_cur[new_death_uids] = ti
        self.results['new_deaths'][ti] = len(new_death_uids)
        self.results['new_deaths_15+'][ti] = np.count_nonzero(self.sim.people.age[new_death_uids] >= 15)

        # Update transmission rates based on current states
        self._update_transmission_rates_original()

        # Reset relative rates for the next time step
        uids = self.sim.people.auids
        self.rr_activation[uids] = 1
        self.rr_clearance[uids] = 1
        self.rr_death[uids] = 1

    def _step_with_state_machine(self):
        """
        Execute time step using the state machine.
        
        This method delegates all state transition logic to the state machine,
        providing cleaner separation of concerns and better maintainability.
        """
        if self.state_manager is None:
            raise RuntimeError("State manager not initialized")
        
        # Process all state transitions using the state machine
        transition_results = self.state_manager.process_time_step(self)
        
        # Update results with transition outcomes
        ti = self.ti
        self.results['new_active'][ti] = transition_results['new_active']
        self.results['new_active_15+'][ti] = transition_results['new_active_15+']
        self.results['new_deaths'][ti] = transition_results['new_deaths']
        self.results['new_deaths_15+'][ti] = transition_results['new_deaths_15+']
        
        # Log transition statistics
        if hasattr(self, 'logger'):
            self.logger.debug(f"Time step {ti}: {transition_results['transitions_by_state']}")
    
    def _update_transmission_rates_original(self):
        """Original transmission rate update method."""
        p = self.pars
        
        # Reset all relative transmission rates
        self.rel_trans[:] = 1

        state_reltrans = [
            (TBS.ACTIVE_PRESYMP, p.rel_trans_presymp),
            (TBS.ACTIVE_EXPTB, p.rel_trans_exptb),
            (TBS.ACTIVE_SMPOS, p.rel_trans_smpos),
            (TBS.ACTIVE_SMNEG, p.rel_trans_smneg),
        ]

        for state, reltrans in state_reltrans:
            uids = self.state == state
            self.rel_trans[uids] *= reltrans

        # Apply transmission heterogeneity
        uids = self.infectious
        self.rel_trans[uids] *= self.reltrans_het[uids]

        # Treatment reduces transmissibility
        uids = self.on_treatment
        self.rel_trans[uids] *= self.pars.rel_trans_treatment
    
    def start_treatment(self, uids):
        """
        Start treatment for individuals with active TB.
        
        This method integrates with the state machine to ensure that treatment
        effects are properly applied to state transitions and transmission rates.
        """
        if len(uids) == 0:
            return 0

        rst = self.state[uids]
        is_active = np.isin(rst, [TBS.ACTIVE_PRESYMP, TBS.ACTIVE_SMPOS, TBS.ACTIVE_SMNEG, TBS.ACTIVE_EXPTB])
        tx_uids = uids[is_active]

        # Track notifications for individuals 15+
        self.results['new_notifications_15+'][self.ti] = np.count_nonzero(self.sim.people.age[tx_uids] >= 15)

        if len(tx_uids) == 0:
            return 0

        # Mark individuals as being on treatment
        self.on_treatment[tx_uids] = True

        # Adjust death and clearance rates for those starting treatment
        self.rr_death[tx_uids] = 0  # People on treatment have zero death rate

        # Reduce transmission rates for people on treatment
        self.rel_trans[tx_uids] *= self.pars.rel_trans_treatment

        return len(tx_uids)
    
    def get_state_statistics(self) -> Dict[str, int]:
        """
        Get current state distribution statistics.
        
        Returns:
            Dictionary mapping state names to counts
        """
        if self.state_manager is not None:
            return self.state_manager.get_state_statistics(self)
        else:
            # Fallback to manual calculation
            stats = {}
            for state_id in TBS.all():
                state_name = f"State_{state_id}"
                count = np.count_nonzero(self.state == state_id)
                stats[state_name] = count
            return stats
    
    def get_transition_matrix(self) -> Dict[tuple, float]:
        """
        Get transition matrix showing all possible transitions and their rates.
        
        Returns:
            Dictionary mapping (from_state, to_state) tuples to transition rates
        """
        if self.state_manager is not None:
            return self.state_manager.get_transition_matrix()
        else:
            # Fallback to empty matrix
            return {}
    
    def validate_state_machine(self) -> Dict[str, Any]:
        """
        Validate the state machine configuration and current state.
        
        Returns:
            Dictionary with validation results
        """
        if self.state_manager is None:
            return {"error": "State machine not initialized"}
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "state_distribution": self.get_state_statistics(),
            "transition_matrix": self.get_transition_matrix()
        }
        
        # Check for invalid states
        valid_states = set(TBS.all())
        current_states = set(self.state)
        invalid_states = current_states - valid_states
        
        if invalid_states:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Invalid states found: {invalid_states}")
        
        # Check for state consistency
        # (Add more validation logic as needed)
        
        return validation_results
    
    def plot_state_transitions(self, figsize=(12, 8)):
        """
        Create a visualization of the state transition matrix.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            matplotlib figure object
        """
        if self.state_manager is None:
            raise RuntimeError("State machine not initialized")
        
        transition_matrix = self.get_transition_matrix()
        
        if not transition_matrix:
            # Create empty plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No transitions available", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Extract states and create matrix
        states = sorted(set([state for transition in transition_matrix.keys() 
                           for state in transition]))
        n_states = len(states)
        
        # Create transition matrix
        matrix = np.zeros((n_states, n_states))
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        for (from_state, to_state), rate in transition_matrix.items():
            if from_state in state_to_idx and to_state in state_to_idx:
                i, j = state_to_idx[from_state], state_to_idx[to_state]
                matrix[i, j] = rate
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(states, rotation=45, ha='right')
        ax.set_yticklabels(states)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transition Rate')
        
        # Add text annotations
        for i in range(n_states):
            for j in range(n_states):
                if matrix[i, j] > 0:
                    text = ax.text(j, i, f'{matrix[i, j]:.2e}',
                                 ha="center", va="center", color="black")
        
        ax.set_title('TB State Transition Matrix')
        ax.set_xlabel('Target State')
        ax.set_ylabel('Source State')
        
        plt.tight_layout()
        return fig
    
    def export_state_machine_config(self) -> Dict[str, Any]:
        """
        Export the current state machine configuration.
        
        Returns:
            Dictionary containing the complete state machine configuration
        """
        if self.state_manager is None:
            return {"error": "State machine not initialized"}
        
        config = {
            "states": {},
            "transitions": {},
            "parameters": {}
        }
        
        # Export state information
        for state_id, state_obj in self.state_manager.state_machine.states.items():
            config["states"][state_obj.name] = {
                "id": int(state_id),
                "description": state_obj.description,
                "is_infectious": state_obj.is_infectious.__name__,
                "get_transmission_rate": state_obj.get_transmission_rate.__name__
            }
        
        # Export transition information
        for state_id, state_obj in self.state_manager.state_machine.states.items():
            for transition in state_obj.transitions:
                if transition.target_state is not None:
                    key = f"{state_obj.name} -> {transition.target_state.name}"
                    config["transitions"][key] = {
                        "rate": transition.rate.rate,
                        "unit": transition.rate.unit,
                        "priority": transition.priority,
                        "condition": transition.condition.__name__ if transition.condition else None
                    }
        
        # Export relevant parameters
        relevant_params = [
            'rate_LS_to_presym', 'rate_LF_to_presym', 'rate_presym_to_active',
            'rate_active_to_clear', 'rate_treatment_to_clear',
            'rate_smpos_to_dead', 'rate_smneg_to_dead', 'rate_exptb_to_dead',
            'rel_trans_presymp', 'rel_trans_smpos', 'rel_trans_smneg', 'rel_trans_exptb',
            'rel_trans_treatment'
        ]
        
        for param in relevant_params:
            if hasattr(self.pars, param):
                config["parameters"][param] = getattr(self.pars, param)
        
        return config
