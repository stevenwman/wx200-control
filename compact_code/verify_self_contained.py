#!/usr/bin/env python3
"""
Verify that compact_code/ is self-contained and doesn't import from parent directory.
"""
import sys
import os
from pathlib import Path

# Get the compact_code directory
compact_code_dir = Path(__file__).parent.absolute()
parent_dir = compact_code_dir.parent

print("="*70)
print("COMPACT_CODE SELF-CONTAINMENT VERIFICATION")
print("="*70)
print(f"compact_code/ path: {compact_code_dir}")
print(f"parent path: {parent_dir}")
print()

# Check sys.path to ensure we're not accidentally importing from parent
print("Python import path (sys.path):")
for i, p in enumerate(sys.path[:5]):
    if p:
        print(f"  [{i}] {p}")
print()

# Test critical imports
test_modules = [
    ('robot_control.robot_config', 'robot_config'),
    ('fix_gstreamer_env', None),
    ('system_checks', 'run_system_checks'),
    ('wx200_robot_teleop_control', 'TeleopControl'),
    ('camera', 'Camera'),
    ('loop_rate_limiters', 'RateLimiter'),
]

all_passed = True

for module_name, attr_name in test_modules:
    try:
        module = __import__(module_name, fromlist=[attr_name] if attr_name else [])

        # Check if module is loaded from compact_code directory
        if hasattr(module, '__file__') and module.__file__:
            module_path = Path(module.__file__).resolve()

            # Check if it's within compact_code
            try:
                relative = module_path.relative_to(compact_code_dir)
                print(f"✓ {module_name:40s} -> {relative}")
            except ValueError:
                # Module is not within compact_code - check if it's from parent (bad) or site-packages (ok)
                try:
                    # If it's in parent directory (but not site-packages), that's an error
                    module_path.relative_to(parent_dir)
                    # Check if it's actually in site-packages or conda env (which is OK)
                    if 'site-packages' in str(module_path) or 'anaconda3' in str(module_path):
                        print(f"✓ {module_name:40s} (external dependency)")
                    else:
                        print(f"✗ {module_name:40s} -> {module_path}")
                        print(f"  ERROR: Module loaded from parent directory (not self-contained)")
                        all_passed = False
                except ValueError:
                    # Module is from somewhere else entirely (site-packages, system, etc.) - that's fine
                    print(f"✓ {module_name:40s} (external dependency)")
        else:
            print(f"✓ {module_name:40s} (built-in or no __file__)")
    except ImportError as e:
        print(f"✗ {module_name:40s} IMPORT FAILED: {e}")
        all_passed = False

print()
print("="*70)

if all_passed:
    print("✓ ALL CHECKS PASSED - compact_code/ is self-contained!")
    print()
    print("You can safely run scripts from compact_code/ without")
    print("accidentally importing from the parent directory.")
else:
    print("✗ SOME CHECKS FAILED - compact_code/ imports from parent!")
    print()
    print("This means running scripts from compact_code/ might use")
    print("modules from the parent directory instead of local ones.")
    sys.exit(1)

print("="*70)
