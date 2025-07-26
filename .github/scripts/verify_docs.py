#!/usr/bin/env python3
"""
Documentation verification script.
This script is called by GitHub Actions workflows to verify documentation setup.
"""

import os
import sys
import subprocess

def check_docs_structure():
    """Check that documentation structure is correct."""
    print("=== Checking Documentation Structure ===")
    
    required_dirs = [
        'docs',
        'docs/tutorials',
        'docs/api'
    ]
    
    required_files = [
        'docs/conf.py',
        'docs/index.rst',
        'docs/tutorials.rst',
        'docs/requirements.txt'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            all_good = False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_path} ({size} bytes)")
        else:
            print(f"✗ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_sphinx_packages():
    """Check that Sphinx packages are available."""
    print("\n=== Checking Sphinx Packages ===")
    
    required_packages = [
        'sphinx',
        'nbsphinx',
        'myst_parser',
        'sphinx_rtd_theme'
    ]
    
    all_available = True
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not available")
            all_available = False
    
    return all_available

def check_tutorial_integration():
    """Check that tutorials are properly integrated."""
    print("\n=== Checking Tutorial Integration ===")
    
    # Check tutorials.rst includes the new tutorial
    try:
        with open('docs/tutorials.rst', 'r') as f:
            content = f.read()
        
        if 'tb_interventions_tutorial.ipynb' in content:
            print("✓ TB interventions tutorial included in tutorials.rst")
        else:
            print("✗ TB interventions tutorial not in tutorials.rst")
            return False
            
    except Exception as e:
        print(f"✗ Error reading tutorials.rst: {e}")
        return False
    
    return True

def main():
    """Main verification function."""
    print("Documentation Setup Verification")
    print("=" * 50)
    
    success = True
    
    if not check_docs_structure():
        success = False
    
    if not check_sphinx_packages():
        success = False
    
    if not check_tutorial_integration():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 All documentation checks passed!")
        print("\nDocumentation is ready for building.")
        sys.exit(0)
    else:
        print("❌ Some documentation checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 