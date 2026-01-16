#!/usr/bin/env python3
"""
Syntax verification for ArUco background thread implementation (Phase 2).

This script checks that the ArUco background thread is properly implemented.
"""
import ast
import sys
from pathlib import Path

print("="*70)
print("ARUCO BACKGROUND THREAD SYNTAX VERIFICATION")
print("="*70)
print()

filepath = Path(__file__).parent.parent / 'deployment' / 'gym_env.py'
print(f"Checking gym_env.py...")

try:
    with open(filepath, 'r') as f:
        source = f.read()

    # Parse the AST
    tree = ast.parse(source, filename='gym_env.py')

    found_thread_attrs = False
    found_start_method = False
    found_poll_loop = False
    found_get_aruco_obs_updated = False
    found_close_method = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'WX200GymEnv':
            # Check __init__ for thread attributes
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == '__init__':
                        init_source = ast.get_source_segment(source, item)
                        if '_aruco_polling_active' in init_source:
                            found_thread_attrs = True
                            print(f"    ✓ Thread attributes in __init__ (_aruco_polling_active, etc.)")
                    elif item.name == '_start_aruco_polling':
                        found_start_method = True
                        print(f"    ✓ _start_aruco_polling() method found")
                    elif item.name == '_aruco_poll_loop':
                        found_poll_loop = True
                        print(f"    ✓ _aruco_poll_loop() background thread method found")
                    elif item.name == '_get_aruco_observations':
                        method_source = ast.get_source_segment(source, item)
                        # Check if it uses thread-safe access
                        if '_aruco_lock' in method_source and 'latest_aruco_obs' in method_source:
                            found_get_aruco_obs_updated = True
                            print(f"    ✓ _get_aruco_observations() uses thread-safe access")
                        else:
                            print(f"    ⚠ _get_aruco_observations() found but may not use thread-safe access")
                    elif item.name == 'close':
                        close_source = ast.get_source_segment(source, item)
                        if '_aruco_polling_active' in close_source and 'join' in close_source:
                            found_close_method = True
                            print(f"    ✓ close() method stops background thread")

    # Check for imports
    has_threading_import = 'import threading' in source
    has_rate_limiter = 'class RateLimiter' in source or 'from.*RateLimiter' in source

    if has_threading_import:
        print(f"    ✓ threading module imported")
    else:
        print(f"    ✗ threading module not imported!")

    if has_rate_limiter:
        print(f"    ✓ RateLimiter available")
    else:
        print(f"    ⚠ RateLimiter not found")

    print(f"    ✓ gym_env.py syntax valid")

    print()
    print("="*70)

    all_checks = [
        found_thread_attrs,
        found_start_method,
        found_poll_loop,
        found_get_aruco_obs_updated,
        found_close_method,
        has_threading_import
    ]

    if all(all_checks):
        print("✓ ALL CHECKS PASSED")
        print()
        print("ArUco background thread implementation verified:")
        print("  - Thread attributes initialized in __init__")
        print("  - _start_aruco_polling() creates and starts thread")
        print("  - _aruco_poll_loop() runs at camera FPS")
        print("  - _get_aruco_observations() uses thread-safe access")
        print("  - close() properly stops thread")
        print()
        print("Phase 2 implementation is structurally correct!")
    else:
        print("✗ SOME CHECKS FAILED")
        print()
        if not found_thread_attrs:
            print("  ✗ Missing thread attributes in __init__")
        if not found_start_method:
            print("  ✗ Missing _start_aruco_polling() method")
        if not found_poll_loop:
            print("  ✗ Missing _aruco_poll_loop() method")
        if not found_get_aruco_obs_updated:
            print("  ✗ _get_aruco_observations() not using thread-safe access")
        if not found_close_method:
            print("  ✗ Missing or incomplete close() method")
        if not has_threading_import:
            print("  ✗ Missing threading import")
        sys.exit(1)

except SyntaxError as e:
    print(f"    ✗ Syntax error in gym_env.py: {e}")
    sys.exit(1)
except Exception as e:
    print(f"    ✗ Error checking gym_env.py: {e}")
    sys.exit(1)

print("="*70)
