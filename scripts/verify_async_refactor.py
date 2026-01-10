#!/usr/bin/env python3
"""
Quick verification script to test async HTTP functionality.
This doesn't require a running server - just tests that the imports and structure work.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import asyncio
        print("  ✓ asyncio imported")
    except ImportError as e:
        print(f"  ✗ asyncio failed: {e}")
        return False
    
    try:
        import httpx
        print(f"  ✓ httpx imported (version {httpx.__version__})")
    except ImportError as e:
        print(f"  ✗ httpx failed: {e}")
        print("  → Run: pip install httpx")
        return False
    
    try:
        from slimgest.web.test_client import (
            send_batch_to_server,
            process_single_pdf_async,
            test_health_check_async,
        )
        print("  ✓ async functions imported from test_client")
    except ImportError as e:
        print(f"  ✗ test_client imports failed: {e}")
        return False
    
    return True


def test_async_structure():
    """Test that async functions have correct signatures."""
    print("\nTesting async function signatures...")
    
    import asyncio
    import inspect
    from slimgest.web.test_client import (
        send_batch_to_server,
        process_single_pdf_async,
        test_health_check_async,
    )
    
    # Check if functions are coroutines
    tests = [
        (send_batch_to_server, "send_batch_to_server"),
        (process_single_pdf_async, "process_single_pdf_async"),
        (test_health_check_async, "test_health_check_async"),
    ]
    
    for func, name in tests:
        if asyncio.iscoroutinefunction(func):
            print(f"  ✓ {name} is async")
        else:
            print(f"  ✗ {name} is not async")
            return False
    
    return True


def test_sync_wrappers():
    """Test that sync wrapper functions exist."""
    print("\nTesting sync wrapper functions...")
    
    try:
        from slimgest.web.test_client import (
            process_single_pdf,
            test_health_check,
        )
        print("  ✓ Sync wrappers exist")
        
        import inspect
        if not inspect.iscoroutinefunction(process_single_pdf):
            print("  ✓ process_single_pdf is sync wrapper")
        else:
            print("  ✗ process_single_pdf should be sync")
            return False
        
        if not inspect.iscoroutinefunction(test_health_check):
            print("  ✓ test_health_check is sync wrapper")
        else:
            print("  ✗ test_health_check should be sync")
            return False
        
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import sync wrappers: {e}")
        return False


def main():
    print("=" * 60)
    print("Async HTTP Refactoring - Verification Script")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Async structure
    if all_passed and not test_async_structure():
        all_passed = False
    
    # Test 3: Sync wrappers
    if all_passed and not test_sync_wrappers():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nThe async HTTP refactoring is properly structured.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -e .")
        print("  2. Start the server: python -m slimgest.web")
        print("  3. Test with: python examples/zero_io_example.py ./images/")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
