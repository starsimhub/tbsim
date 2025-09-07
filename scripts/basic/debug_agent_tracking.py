#!/usr/bin/env python3
"""
Debug script to test agent tracking functionality
"""

import tbsim as mtb
import starsim as ss
import numpy as np
import sys
import os

# Add the parent directory to the path to import the state machine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tbsim.tb_with_state_machine import TBWithStateMachine


def main():
    print("Debug Agent Tracking")
    print("=" * 40)
    
    # Create a simple simulation
    pop = ss.People(n_agents=100)
    networks = ss.RandomNet()
    
    # Create TB module with state machine
    tb = TBWithStateMachine(
        use_state_machine=True,
        init_prev=0.1,  # 10% initial prevalence to ensure some infections
        beta=ss.peryear(1.0),  # High transmission rate
    )
    
    # Create simulation
    sim = ss.Sim(
        people=pop,
        networks=networks,
        diseases=tb,
        start='1940-01-01',
        stop='1945-01-01',  # 5 years
        verbose=True
    )
    
    print(f"Initial state distribution:")
    initial_states = tb.state
    unique_states, counts = np.unique(initial_states, return_counts=True)
    for state_id, count in zip(unique_states, counts):
        state_name = tb.state_machine.get_state(state_id).name if hasattr(tb, 'state_machine') else f"State {state_id}"
        print(f"  {state_name}: {count}")
    
    print(f"\nRunning simulation...")
    sim.run()
    
    print(f"\nFinal state distribution:")
    final_states = tb.state
    unique_states, counts = np.unique(final_states, return_counts=True)
    for state_id, count in zip(unique_states, counts):
        state_name = tb.state_machine.get_state(state_id).name if hasattr(tb, 'state_machine') else f"State {state_id}"
        print(f"  {state_name}: {count}")
    
    # Check state manager
    if tb.state_manager:
        print(f"\nState Manager Status:")
        print(f"  Agent histories: {len(tb.state_manager.agent_histories)}")
        print(f"  Transition log: {len(tb.state_manager.transition_log)}")
        print(f"  Current time: {tb.state_manager.current_time}")
        
        # Check if agents were initialized
        living_uids = sim.people.auids
        print(f"  Living agents: {len(living_uids)}")
        print(f"  First 5 UIDs: {living_uids[:5]}")
        
        # Show some agent histories
        if tb.state_manager.agent_histories:
            print(f"\nSample agent histories:")
            for uid, history in list(tb.state_manager.agent_histories.items())[:3]:
                print(f"  Agent {uid}: {len(history)} entries")
                for entry in history[:3]:  # Show first 3 entries
                    print(f"    {entry}")
        else:
            print("  No agent histories recorded")
            
        # Check if any transitions were processed
        if hasattr(tb.state_manager, 'state_machine'):
            print(f"\nState Machine Status:")
            print(f"  States defined: {len(tb.state_manager.state_machine.states)}")
            for state_id, state_obj in tb.state_manager.state_machine.states.items():
                print(f"    {state_id}: {state_obj.name}")
    else:
        print("  State manager not initialized")


if __name__ == '__main__':
    main()
