#!/usr/bin/env python3

"""
Test runner for NCCL Bad Node Finder tests
"""

import unittest
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_tests():
    """Run all tests with coverage if available"""
    
    # Try to run with coverage
    try:
        import coverage
        
        # Start coverage
        cov = coverage.Coverage()
        cov.start()
        
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        # Stop coverage and report
        cov.stop()
        cov.save()
        
        print("\n" + "="*50)
        print("COVERAGE REPORT")
        print("="*50)
        cov.report(show_missing=True)
        
        # Generate HTML report
        try:
            cov.html_report(directory='htmlcov')
            print(f"\nHTML coverage report generated in: htmlcov/index.html")
        except Exception as e:
            print(f"Could not generate HTML report: {e}")
            
        return result.wasSuccessful()
        
    except ImportError:
        print("Coverage not available, running tests without coverage...")
        
        # Run tests without coverage
        loader = unittest.TestLoader()
        suite = loader.discover('.', pattern='test_*.py')
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)