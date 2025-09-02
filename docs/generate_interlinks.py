#!/usr/bin/env python3
"""
Script to generate Quarto interlinks for the TBSim documentation.
This script handles the interlinks generation process and provides
detailed logging and error handling.
"""

import subprocess
import sys
import os
from pathlib import Path

def generate_interlinks():
    """Generate Quarto interlinks database."""
    print("Starting Quarto interlinks generation...")
    
    # Check if we're in the docs directory
    if not Path("_quarto.yml").exists():
        print("Error: _quarto.yml not found. Make sure you're in the docs directory.")
        sys.exit(1)
    
    # Check if Quarto is available
    try:
        result = subprocess.run(['quarto', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"Quarto version: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error checking Quarto version: {e}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Quarto not found in PATH")
        sys.exit(1)
    
    # Generate interlinks
    try:
        print("Generating interlinks database...")
        result = subprocess.run(['quarto', 'interlinks', '--generate'], 
                              capture_output=True, text=True, check=True)
        print("Interlinks generated successfully!")
        if result.stdout.strip():
            print("Output:", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"Error generating interlinks: {e}")
        print(f"stderr: {e.stderr}")
        print("Attempting to continue without interlinks...")
        return False
    
    return True

if __name__ == "__main__":
    success = generate_interlinks()
    if not success:
        print("Warning: Interlinks generation failed, but continuing with build...")
    sys.exit(0 if success else 1)
