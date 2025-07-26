#!/usr/bin/env python3
"""
Documentation build verification script.
This script is called by GitHub Actions workflows to verify build output.
"""

import os
import sys

def check_build_output():
    """Check that documentation was built successfully."""
    print("=== Verifying Build Output ===")
    
    # Check if we're in the docs directory
    if not os.path.exists('_build/html'):
        print("✗ _build/html directory not found")
        return False
    
    # Check for index.html
    if not os.path.exists('_build/html/index.html'):
        print("✗ index.html not found in build output")
        return False
    
    print("✓ index.html found")
    
    # Check for tutorials
    tutorials_dir = '_build/html/tutorials'
    if not os.path.exists(tutorials_dir):
        print("✗ tutorials directory not found in build output")
        return False
    
    print("✓ tutorials directory found")
    
    # Check for specific tutorial files
    tutorial_files = [
        'tb_interventions_tutorial.html',
        'tb_interventions_tutorial.ipynb',
        'tbhiv_comorbidity.html',
        'tuberculosis_sim.html'
    ]
    
    for file_name in tutorial_files:
        file_path = os.path.join(tutorials_dir, file_name)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ {file_name} ({size} bytes)")
        else:
            print(f"✗ {file_name} not found")
            return False
    
    return True

def check_package_versions():
    """Check that required packages are available."""
    print("\n=== Checking Package Versions ===")
    
    try:
        import nbsphinx
        print(f"✓ nbsphinx version: {nbsphinx.__version__}")
    except ImportError:
        print("✗ nbsphinx not available")
        return False
    
    try:
        import myst_parser
        print(f"✓ myst_parser version: {myst_parser.__version__}")
    except ImportError:
        print("✗ myst_parser not available")
        return False
    
    return True

def main():
    """Main verification function."""
    print("Documentation Build Verification")
    print("=" * 50)
    
    success = True
    
    if not check_package_versions():
        success = False
    
    if not check_build_output():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 Build verification passed!")
        print("Documentation is ready for deployment.")
        sys.exit(0)
    else:
        print("❌ Build verification failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 