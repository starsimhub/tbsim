#!/usr/bin/env python3
"""
Wrapper script to run the TB simulation with the correct PYTHONPATH.
This ensures that the local tbsim module is used instead of any installed versions.
"""

import os
import sys
import subprocess

def main():
    # Get the project root directory (two levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Set the PYTHONPATH to include the project root
    env = os.environ.copy()
    env['PYTHONPATH'] = project_root
    
    # Run the TB simulation script
    script_path = os.path.join(script_dir, 'run_tb.py')
    
    print(f"Running TB simulation with PYTHONPATH={project_root}")
    print(f"Script: {script_path}")
    print("-" * 50)
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              env=env, 
                              cwd=project_root,
                              check=True)
        print("\n✅ TB simulation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ TB simulation failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == '__main__':
    main() 