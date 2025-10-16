#!/usr/bin/env python3
"""
Demo script for testing the TB Simulation Web App
This script demonstrates various features and can be used to test execution.
"""

import time
import sys
import random

def main():
    """Main demo function that simulates a TB simulation script."""
    print("TB Simulation Demo Script")
    print("=" * 40)
    print("This is a demo script to test the web app functionality.")
    print()
    
    # Simulate initialization
    print("Initializing TB simulation...")
    time.sleep(1)
    print("✓ Population parameters loaded")
    time.sleep(0.5)
    print("✓ Disease parameters configured")
    time.sleep(0.5)
    print("✓ Network structures initialized")
    time.sleep(0.5)
    print("✓ Interventions configured")
    print()
    
    # Simulate simulation steps
    print("Running simulation...")
    for year in range(2020, 2030):
        print(f"Processing year {year}...")
        
        # Simulate some computation
        time.sleep(0.3)
        
        # Generate some realistic-looking output
        prevalence = random.uniform(0.5, 2.0)
        cases = random.randint(1000, 5000)
        deaths = random.randint(50, 200)
        
        print(f"  - TB Prevalence: {prevalence:.3f}%")
        print(f"  - New Cases: {cases:,}")
        print(f"  - Deaths: {deaths:,}")
        print()
    
    # Simulate results processing
    print("Processing results...")
    time.sleep(1)
    print("✓ Results calculated")
    time.sleep(0.5)
    print("✓ Plots generated")
    time.sleep(0.5)
    print("✓ Report created")
    print()
    
    # Final summary
    print("Simulation completed successfully!")
    print("=" * 40)
    print("Summary:")
    print("- Total simulation time: 10 years")
    print("- Final prevalence: 1.234%")
    print("- Total cases: 25,678")
    print("- Total deaths: 1,234")
    print("- Output files: results_2024.pdf, data_2024.csv")
    print()
    print("Demo script execution finished.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

