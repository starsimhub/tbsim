#!/usr/bin/env python3
"""
Binder configuration verification script.
This script is called by GitHub Actions workflows to verify Binder setup.
"""

import os
import sys

def check_binder_files():
    """Check that all required Binder files exist."""
    print("=== Checking Binder Configuration Files ===")
    
    required_files = [
        'binder/requirements.txt',
        'binder/postBuild',
        'binder/runtime.txt'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_readme_badge():
    """Check that README has the Binder badge."""
    print("\n=== Checking README Binder Badge ===")
    
    try:
        with open('README.md', 'r') as f:
            content = f.read()
        
        if 'mybinder.org' in content:
            print("✓ Binder badge found in README")
            return True
        else:
            print("✗ Binder badge not found in README")
            return False
            
    except Exception as e:
        print(f"✗ Error reading README.md: {e}")
        return False

def check_tutorial_files():
    """Check that tutorial files exist."""
    print("\n=== Checking Tutorial Files ===")
    
    tutorial_files = [
        'docs/tutorials/tb_interventions_tutorial.ipynb',
        'docs/tutorials/tbhiv_comorbidity.ipynb',
        'docs/tutorials/tuberculosis_sim.ipynb'
    ]
    
    all_exist = True
    
    for file_path in tutorial_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    """Main verification function."""
    print("Binder Configuration Verification")
    print("=" * 50)
    
    success = True
    
    if not check_binder_files():
        success = False
    
    if not check_readme_badge():
        success = False
    
    if not check_tutorial_files():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 All Binder checks passed!")
        print("\nBinder Links:")
        print("- Main (uses repo default branch): https://mybinder.org/v2/gh/starsimhub/tbsim/HEAD")
        print("- Direct to tutorial: https://mybinder.org/v2/gh/starsimhub/tbsim/HEAD?filepath=docs%2Ftutorials%2Ftb_interventions_tutorial.ipynb")
        sys.exit(0)
    else:
        print("***** Some Binder checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 