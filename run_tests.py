#!/usr/bin/env python3
"""
Test runner for SAI-Benchmark test suite.

This script runs the unit tests and provides a summary of results.
Can be used with or without pytest installed.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def run_with_pytest():
    """Run tests using pytest."""
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit",
        "-v",
        "--tb=short",
        "--maxfail=1",
        "-ra"
    ]
    
    print("Running tests with pytest...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 80)
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def run_without_pytest():
    """Run tests using unittest as fallback."""
    import unittest
    
    print("Running tests with unittest (pytest not available)...")
    print("-" * 80)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = 'tests/unit'
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def check_test_structure():
    """Verify test structure is set up correctly."""
    issues = []
    
    # Check test directories exist
    if not Path("tests").exists():
        issues.append("Missing 'tests' directory")
    
    if not Path("tests/unit").exists():
        issues.append("Missing 'tests/unit' directory")
    
    # Check for test files
    test_files = list(Path("tests/unit").glob("test_*.py"))
    if not test_files:
        issues.append("No test files found in tests/unit/")
    else:
        print(f"Found {len(test_files)} test files:")
        for tf in test_files:
            print(f"  - {tf.name}")
    
    # Check for conftest.py
    if not Path("tests/conftest.py").exists():
        issues.append("Missing tests/conftest.py (pytest fixtures)")
    
    if issues:
        print("\nTest structure issues found:")
        for issue in issues:
            print(f"  ❌ {issue}")
        return False
    
    print("\n✅ Test structure looks good!")
    return True


def main():
    """Main test runner."""
    print("SAI-Benchmark Test Runner")
    print("=" * 80)
    
    # Check test structure
    if not check_test_structure():
        print("\nPlease fix the test structure issues before running tests.")
        return 1
    
    print()
    
    # Try to run with pytest first
    try:
        import pytest
        success = run_with_pytest()
    except ImportError:
        print("⚠️  pytest not installed. Install with: pip install -r requirements.txt")
        print("Falling back to unittest...\n")
        success = run_without_pytest()
    
    if success:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())