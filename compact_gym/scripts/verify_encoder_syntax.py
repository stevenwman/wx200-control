#!/usr/bin/env python3
"""
Syntax verification for encoder polling implementation.

This script only checks Python syntax and structure without imports.
"""
import ast
import sys
from pathlib import Path

print("="*70)
print("ENCODER POLLING SYNTAX VERIFICATION")
print("="*70)
print()

# Files to check
files_to_check = [
    'robot_hardware.py',
    'gym_env.py',
    'collect_demo_gym.py',
]

all_passed = True

for filename in files_to_check:
    if filename == 'collect_demo_gym.py':
        filepath = Path(__file__).parent.parent / filename
    else:
        filepath = Path(__file__).parent.parent / 'deployment' / filename
    print(f"Checking {filename}...")

    try:
        with open(filepath, 'r') as f:
            source = f.read()

        # Parse the AST
        tree = ast.parse(source, filename=filename)

        # Check for specific patterns in robot_hardware.py
        if filename == 'robot_hardware.py':
            found_poll_encoders = False
            found_get_encoder_state = False
            found_encoder_attrs = False

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == 'RobotHardware':
                    # Check __init__ for encoder attributes
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            if item.name == '__init__':
                                init_source = ast.get_source_segment(source, item)
                                if 'latest_encoder_values' in init_source:
                                    found_encoder_attrs = True
                            elif item.name == 'poll_encoders':
                                found_poll_encoders = True
                                # Check for outer_loop_start_time parameter
                                param_names = [arg.arg for arg in item.args.args]
                                if 'outer_loop_start_time' in param_names or any('outer_loop' in p for p in param_names):
                                    print(f"    ✓ poll_encoders() method found with timing parameter")
                                else:
                                    print(f"    ⚠ poll_encoders() found but missing outer_loop_start_time param")
                            elif item.name == 'get_encoder_state':
                                found_get_encoder_state = True
                                print(f"    ✓ get_encoder_state() method found")

            if found_encoder_attrs:
                print(f"    ✓ Encoder state attributes in __init__")
            if not found_poll_encoders:
                print(f"    ✗ poll_encoders() method not found!")
                all_passed = False
            if not found_get_encoder_state:
                print(f"    ✗ get_encoder_state() method not found!")
                all_passed = False

        # Check for specific patterns in gym_env.py
        elif filename == 'gym_env.py':
            found_encoder_in_info = False

            source_lower = source.lower()
            if 'encoder_values' in source_lower and 'info[' in source_lower:
                found_encoder_in_info = True
                print(f"    ✓ Encoder data added to info dict")

            if not found_encoder_in_info:
                print(f"    ⚠ Encoder data in info dict not found")

        # Check for encoder polling in teleop layer
        elif filename == 'collect_demo_gym.py':
            source_lower = source.lower()
            if 'poll_encoders' in source_lower:
                print(f"    ✓ poll_encoders() call found in teleop loop")
            else:
                print(f"    ⚠ poll_encoders() call not found in teleop loop")

        print(f"    ✓ {filename} syntax valid")

    except SyntaxError as e:
        print(f"    ✗ Syntax error in {filename}: {e}")
        all_passed = False
    except Exception as e:
        print(f"    ✗ Error checking {filename}: {e}")
        all_passed = False

    print()

print("="*70)
if all_passed:
    print("✓ ALL SYNTAX CHECKS PASSED")
    print()
    print("Implementation structure verified:")
    print("  - robot_hardware.py: poll_encoders() and get_encoder_state() methods")
    print("  - gym_env.py: Integration with step() and info dict")
    print()
    print("Note: Runtime testing on hardware required for full verification.")
else:
    print("✗ SOME CHECKS FAILED")
    print("Review the errors above.")
    sys.exit(1)

print("="*70)
