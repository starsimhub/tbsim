#!/usr/bin/env python3
"""
Test script to verify Quarto documentation builds correctly.
"""

import subprocess
import sys
import os
from pathlib import Path

def test_quarto_installation():
    """Test if Quarto is installed and accessible."""
    try:
        result = subprocess.run(['quarto', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✓ Quarto version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ Quarto not found. Please install Quarto: https://quarto.org/docs/get-started/")
        return False

def test_quarto_config():
    """Test if _quarto.yml is valid."""
    quarto_config = Path("_quarto.yml")
    if not quarto_config.exists():
        print("✗ _quarto.yml not found")
        return False
    
    try:
        result = subprocess.run(['quarto', 'check'], 
                              capture_output=True, text=True, check=True)
        print("✓ Quarto configuration is valid")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Quarto configuration error: {e.stderr}")
        return False

def test_quarto_build():
    """Test if documentation builds successfully."""
    try:
        # Clean previous build
        site_dir = Path("_site")
        if site_dir.exists():
            import shutil
            shutil.rmtree(site_dir)
        
        # Build documentation
        result = subprocess.run(['quarto', 'render', '--to', 'html'], 
                              capture_output=True, text=True, check=True)
        print("✓ Documentation built successfully")
        
        # Check if output files exist
        if Path("_site/index.html").exists():
            print("✓ index.html generated")
        else:
            print("✗ index.html not found")
            return False
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Build failed: {e.stderr}")
        return False

def test_assets():
    """Test if required assets exist."""
    required_files = [
        "assets/styles.css",
        "assets/styles-light.scss", 
        "assets/styles-dark.scss"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def test_content_files():
    """Test if main content files exist."""
    required_files = [
        "index.qmd",
        "tutorials.qmd",
        "user_guide.qmd"
    ]
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path} exists")
        else:
            print(f"✗ {file_path} missing")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Testing Quarto documentation setup...\n")
    
    tests = [
        ("Quarto Installation", test_quarto_installation),
        ("Quarto Configuration", test_quarto_config),
        ("Required Assets", test_assets),
        ("Content Files", test_content_files),
        ("Documentation Build", test_quarto_build),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Documentation is ready.")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
