#!/usr/bin/env python3
"""
Verification script for encoder polling implementation.

This script verifies that the encoder polling code is correctly implemented
without requiring hardware or full environment initialization.
"""
import sys
import inspect
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*70)
print("ENCODER POLLING IMPLEMENTATION VERIFICATION")
print("="*70)
print()

# Test 1: Import robot_hardware module
print("[1/5] Testing robot_hardware module import...")
try:
    from deployment import robot_hardware
    print("    ✓ robot_hardware module imported successfully")
except ImportError as e:
    print(f"    ✗ Failed to import robot_hardware: {e}")
    sys.exit(1)

# Test 2: Check RobotHardware class has required attributes
print("\n[2/5] Checking RobotHardware class attributes...")
from deployment.robot_hardware import RobotHardware

required_attrs = [
    'latest_encoder_values',
    'latest_joint_angles_from_encoders',
    'latest_ee_pose_from_encoders',
    'last_poll_timestamp',
    'encoder_poll_times',
    'encoder_poll_intervals',
    'encoder_poll_count',
    '_skipped_reads_count'
]

hw = RobotHardware()
missing_attrs = []
for attr in required_attrs:
    if not hasattr(hw, attr):
        missing_attrs.append(attr)
    else:
        print(f"    ✓ {attr}")

if missing_attrs:
    print(f"    ✗ Missing attributes: {missing_attrs}")
    sys.exit(1)

# Test 3: Check RobotHardware has required methods
print("\n[3/5] Checking RobotHardware class methods...")
required_methods = ['poll_encoders', 'get_encoder_state']

for method in required_methods:
    if not hasattr(hw, method):
        print(f"    ✗ Missing method: {method}")
        sys.exit(1)
    else:
        print(f"    ✓ {method}")

# Test 4: Check poll_encoders signature
print("\n[4/5] Checking poll_encoders() signature...")
sig = inspect.signature(hw.poll_encoders)
params = list(sig.parameters.keys())
if 'outer_loop_start_time' in params:
    print(f"    ✓ poll_encoders has outer_loop_start_time parameter")
else:
    print(f"    ✗ poll_encoders missing outer_loop_start_time parameter")
    sys.exit(1)

# Test 5: Check get_encoder_state return structure
print("\n[5/5] Checking get_encoder_state() return structure...")
try:
    state = hw.get_encoder_state()
    required_keys = ['encoder_values', 'joint_angles', 'ee_pose']
    missing_keys = []
    for key in required_keys:
        if key not in state:
            missing_keys.append(key)
        else:
            print(f"    ✓ '{key}' in return dict")

    if missing_keys:
        print(f"    ✗ Missing keys: {missing_keys}")
        sys.exit(1)
except Exception as e:
    print(f"    ✗ get_encoder_state() failed: {e}")
    sys.exit(1)

print()
print("="*70)
print("✓ ALL VERIFICATION CHECKS PASSED")
print("="*70)
print()
print("Summary:")
print("  - RobotHardware class has all required encoder polling attributes")
print("  - poll_encoders() method exists with correct signature")
print("  - get_encoder_state() method returns correct structure")
print()
print("The encoder polling implementation is structurally correct.")
print("Hardware testing required to verify runtime behavior.")
print()
print("="*70)
