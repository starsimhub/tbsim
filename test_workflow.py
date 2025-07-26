#!/usr/bin/env python3
"""
Local test script to verify workflow configuration.
This simulates the checks that the GitHub Actions workflow will perform.
"""

import os
import sys
import subprocess

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✓ {description}: {filepath} ({size} bytes)")
        return True
    else:
        print(f"✗ {description}: {filepath} - MISSING")
        return False

def check_binder_config():
    """Check Binder configuration files."""
    print("\n=== Checking Binder Configuration ===")
    
    success = True
    
    success &= check_file_exists("binder/requirements.txt", "Binder requirements")
    success &= check_file_exists("binder/postBuild", "Binder postBuild script")
    success &= check_file_exists("binder/runtime.txt", "Binder runtime")
    
    # Check README for Binder badge
    try:
        with open("README.md", "r") as f:
            content = f.read()
        if "mybinder.org" in content:
            print("✓ Binder badge found in README")
        else:
            print("✗ Binder badge not found in README")
            success = False
    except Exception as e:
        print(f"✗ Error reading README.md: {e}")
        success = False
    
    return success

def check_tutorial_files():
    """Check tutorial files exist."""
    print("\n=== Checking Tutorial Files ===")
    
    success = True
    
    success &= check_file_exists("scripts/run_tb_interventions.py", "Tutorial source script")
    success &= check_file_exists("docs/tutorials/tb_interventions_tutorial.ipynb", "Tutorial notebook")
    
    return success

def check_workflow_files():
    """Check workflow files exist."""
    print("\n=== Checking Workflow Files ===")
    
    success = True
    
    success &= check_file_exists(".github/workflows/deploy-docs.yml", "Main docs workflow")
    success &= check_file_exists(".github/workflows/test-docs.yml", "Test docs workflow")
    
    return success

def test_imports():
    """Test that key modules can be imported."""
    print("\n=== Testing Imports ===")
    
    success = True
    
    try:
        import tbsim
        print("✓ tbsim imported")
    except Exception as e:
        print(f"✗ tbsim import failed: {e}")
        success = False
    
    try:
        import tbsim.interventions.bcg
        print("✓ tbsim.interventions.bcg imported")
    except Exception as e:
        print(f"✗ tbsim.interventions.bcg import failed: {e}")
        success = False
    
    try:
        import tbsim.interventions.tpt
        print("✓ tbsim.interventions.tpt imported")
    except Exception as e:
        print(f"✗ tbsim.interventions.tpt import failed: {e}")
        success = False
    
    try:
        import tbsim.interventions.beta
        print("✓ tbsim.interventions.beta imported")
    except Exception as e:
        print(f"✗ tbsim.interventions.beta import failed: {e}")
        success = False
    
    return success

def test_tutorial_functions():
    """Test tutorial functions can be imported."""
    print("\n=== Testing Tutorial Functions ===")
    
    success = True
    
    try:
        sys.path.append('.')
        from scripts.run_tb_interventions import build_sim, get_scenarios
        print("✓ Tutorial functions imported")
        
        # Test scenario creation
        scenarios = get_scenarios()
        print(f"✓ {len(scenarios)} scenarios created")
        
    except Exception as e:
        print(f"✗ Tutorial function test failed: {e}")
        success = False
    
    return success

def main():
    """Run all checks."""
    print("Workflow Configuration Test")
    print("=" * 50)
    
    all_success = True
    
    all_success &= check_binder_config()
    all_success &= check_tutorial_files()
    all_success &= check_workflow_files()
    all_success &= test_imports()
    all_success &= test_tutorial_functions()
    
    print("\n" + "=" * 50)
    
    if all_success:
        print("🎉 All checks passed! Workflow is ready.")
        print("\nAvailable workflows:")
        print("- deploy-docs.yml: Full documentation build and deployment")
        print("- test-docs.yml: Quick documentation testing")
        print("\nBinder links:")
        print("- Main: https://mybinder.org/v2/gh/starsimhub/tbsim/main")
        print("- Direct: https://mybinder.org/v2/gh/starsimhub/tbsim/main?filepath=docs%2Ftutorials%2Ftb_interventions_tutorial.ipynb")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 