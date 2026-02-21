#!/usr/bin/env python3
"""
SFT Validation Test Runner

This script runs all validation tests in the SFT directory to ensure
BCG intervention functionality and effectiveness.

Usage:
    python tests/SFT/run_validation.py
"""

import sys
import os
import subprocess
import time

def run_validation_test(test_name, test_file):
    """Run a single validation test"""
    print(f"\n{'='*60}")
    print(f"Running {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ {test_name} PASSED ({duration:.1f}s)")
            print("Output:")
            print(result.stdout)
            return True
        else:
            print(f"❌ {test_name} FAILED ({duration:.1f}s)")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"💥 {test_name} ERROR: {e}")
        return False

def main():
    """Run all SFT validation tests"""
    print("🧪 SFT Validation Test Suite")
    print("=" * 60)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define validation tests
    tests = [
        ("BCG Intervention Effectiveness", "validation_bcg_effectiveness.py"),
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for test_name, test_file in tests:
        test_path = os.path.join(script_dir, test_file)
        
        if not os.path.exists(test_path):
            print(f"⚠️  Test file not found: {test_path}")
            failed += 1
            continue
            
        if run_validation_test(test_name, test_path):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("SFT VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All SFT validation tests PASSED!")
        return 0
    else:
        print(f"\n⚠️  {failed} SFT validation test(s) FAILED!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
