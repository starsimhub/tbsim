#!/usr/bin/env python3
"""
Simple test to verify agent tracking is working
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
    print("Simple Agent Tracking Test")
    print("=" * 40)
    
    # Create a simple simulation
    pop = ss.People(n_agents=50)
    networks = ss.RandomNet()
    
    # Create TB module with state machine
    tb = TBWithStateMachine(
        use_state_machine=True,
        init_prev=0.2,  # 20% initial prevalence
        beta=ss.peryear(2.0),  # High transmission rate
    )
    
    # Create simulation
    sim = ss.Sim(
        people=pop,
        networks=networks,
        diseases=tb,
        start='1940-01-01',
        stop='1942-01-01',  # 2 years
        verbose=False
    )
    
    print(f"Running simulation...")
    
    # Add debug output to see if step method is called
    original_step = tb.step
    step_count = 0
    
    def debug_step():
        nonlocal step_count
        step_count += 1
        print(f"  Step {step_count} called")
        return original_step()
    
    tb.step = debug_step
    
    sim.run()
    
    print(f"  Total steps called: {step_count}")
    
    # Check state manager directly
    print(f"\nDirect State Manager Check:")
    print(f"  State manager exists: {tb.state_manager is not None}")
    if tb.state_manager:
        print(f"  Agent histories: {len(tb.state_manager.agent_histories)}")
        print(f"  Transition log: {len(tb.state_manager.transition_log)}")
        print(f"  Current time: {tb.state_manager.current_time}")
        print(f"  State manager ID: {id(tb.state_manager)}")
        
        # Check if the state manager has been called
        print(f"  State machine states: {len(tb.state_manager.state_machine.states)}")
        
        # Show some agent histories
        if tb.state_manager.agent_histories:
            print(f"\nSample agent histories:")
            for uid, history in list(tb.state_manager.agent_histories.items())[:3]:
                print(f"  Agent {uid}: {len(history)} entries")
                for entry in history:
                    print(f"    {entry}")
        else:
            print("  No agent histories found")
    
    # Check if the state manager methods work
    print(f"\nState Manager Methods Test:")
    if tb.state_manager:
        try:
            all_histories = tb.state_manager.get_all_agent_histories()
            print(f"  get_all_agent_histories(): {len(all_histories)} agents")
            
            transition_log = tb.state_manager.get_transition_log()
            print(f"  get_transition_log(): {len(transition_log)} transitions")
            
            progression_stats = tb.state_manager.get_state_progression_stats()
            print(f"  get_state_progression_stats(): {len(progression_stats)} states")
            
        except Exception as e:
            print(f"  Error calling state manager methods: {e}")


if __name__ == '__main__':
    main()
