#!/usr/bin/env python3
"""
Validation script for collected demos.

Checks that NPZ files have all expected fields with correct shapes
and validates critical data quality issues.
"""
import numpy as np
import glob
import os
import sys

def validate_demo(npz_path):
    """Validate a single demo NPZ file."""
    print(f"Validating: {npz_path}")
    print("=" * 70)

    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"✗ Failed to load NPZ: {e}")
        return False

    # File info
    file_size_mb = os.path.getsize(npz_path) / 1024 / 1024
    print(f"File size: {file_size_mb:.2f} MB")
    print()

    # Check all expected fields
    expected_fields = [
        'timestamp', 'state', 'encoder_values', 'ee_pose_encoder',
        'action', 'augmented_actions', 'ee_pose_target',
        'object_pose', 'object_visible',
        'aruco_ee_in_world', 'aruco_object_in_world',
        'aruco_ee_in_object', 'aruco_object_in_ee', 'aruco_visibility',
        'camera_frame'
    ]
    optional_fields = ['action_normalized', 'metadata']

    print("Field Check:")
    all_fields_present = True
    for field in expected_fields:
        if field in data:
            shape = data[field].shape
            dtype = data[field].dtype
            print(f"  ✓ {field:25s} shape={shape} dtype={dtype}")
        else:
            print(f"  ✗ {field:25s} MISSING!")
            all_fields_present = False

    if not all_fields_present:
        print("\n✗ Validation FAILED: Missing fields")
        return False

    # Optional fields (warn only)
    for field in optional_fields:
        if field in data:
            shape = data[field].shape if hasattr(data[field], 'shape') else 'scalar'
            dtype = getattr(data[field], 'dtype', type(data[field]))
            print(f"  ✓ {field:25s} shape={shape} dtype={dtype} (optional)")
        else:
            print(f"  ⚠ {field:25s} MISSING (optional)")

    print()

    # Critical validations
    print("Critical Checks:")
    all_checks_passed = True

    # 1. ee_pose_target should NOT be all zeros
    ee_target_nonzero = not np.allclose(data['ee_pose_target'], 0)
    status = '✓' if ee_target_nonzero else '✗'
    print(f"  {status} ee_pose_target not all zeros: {ee_target_nonzero}")
    if not ee_target_nonzero:
        all_checks_passed = False
        print(f"     ERROR: ee_pose_target is all zeros! IK target not being recorded.")

    # 2. Encoder values shape
    encoder_ok = data['encoder_values'].shape[1] == 7
    status = '✓' if encoder_ok else '✗'
    print(f"  {status} encoder_values has 7 motors: {data['encoder_values'].shape}")
    if not encoder_ok:
        all_checks_passed = False

    # 3. Camera frame shape (1/4 resolution by default)
    expected_frame_h = 270  # 1080 / 4
    expected_frame_w = 480  # 1920 / 4
    frame_shape = data['camera_frame'].shape[1:]
    frame_ok = frame_shape == (expected_frame_h, expected_frame_w, 3)
    status = '✓' if frame_ok else '⚠'
    print(f"  {status} camera_frame shape: {data['camera_frame'].shape}")
    if not frame_ok:
        print(f"     WARNING: Expected ({expected_frame_h}, {expected_frame_w}, 3), got {frame_shape}")
        print(f"              (May be OK if frame_downscale_factor was changed)")

    # 4. Trajectory length and frequency
    traj_len = len(data['timestamp'])
    if traj_len > 1:
        duration = data['timestamp'][-1] - data['timestamp'][0]
        freq = traj_len / duration if duration > 0 else 0
        print(f"  ✓ Trajectory: {traj_len} steps, {duration:.2f}s, {freq:.1f} Hz")

        # Check if frequency is close to expected (10 Hz)
        if abs(freq - 10.0) > 1.0:
            print(f"     WARNING: Frequency {freq:.1f} Hz differs from expected 10 Hz")
    else:
        print(f"  ⚠ Trajectory too short: {traj_len} steps")
        all_checks_passed = False

    # 5. Check for NaN values
    has_nan = False
    for field in ['ee_pose_target', 'encoder_values', 'aruco_object_in_world']:
        if np.any(np.isnan(data[field])):
            print(f"  ✗ {field} contains NaN values!")
            has_nan = True
            all_checks_passed = False

    if not has_nan:
        print(f"  ✓ No NaN values in critical fields")

    # 6. ArUco visibility stats
    visibility = data['aruco_visibility']
    # Order is [world, object, ee]
    world_visible_pct = np.mean(visibility[:, 0]) * 100
    obj_visible_pct = np.mean(visibility[:, 1]) * 100
    ee_visible_pct = np.mean(visibility[:, 2]) * 100

    print()
    print("ArUco Visibility:")
    print(f"  World marker:  {world_visible_pct:5.1f}% visible")
    print(f"  Object marker: {obj_visible_pct:5.1f}% visible")
    print(f"  EE marker:     {ee_visible_pct:5.1f}% visible")

    if obj_visible_pct < 50:
        print(f"  ⚠ Object marker visibility low ({obj_visible_pct:.1f}%)")

    print()
    print("=" * 70)

    if all_checks_passed and all_fields_present:
        print("✓ VALIDATION PASSED")
        return True
    else:
        print("✗ VALIDATION FAILED")
        return False


def main():
    """Validate most recent demo or all demos."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate collected demo NPZ files")
    parser.add_argument('--all', action='store_true', help='Validate all demos in data/gym_demos/')
    parser.add_argument('--file', type=str, help='Validate specific NPZ file')

    args = parser.parse_args()

    if args.file:
        # Validate specific file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        success = validate_demo(args.file)
        sys.exit(0 if success else 1)

    # Find demo files
    demo_files = sorted(glob.glob("data/gym_demos/demo_*.npz"))

    if not demo_files:
        print("No demo files found in data/gym_demos/")
        print("Have you collected any demos yet?")
        sys.exit(1)

    if args.all:
        # Validate all demos
        print(f"Found {len(demo_files)} demo files")
        print()

        results = []
        for demo_file in demo_files:
            success = validate_demo(demo_file)
            results.append((demo_file, success))
            print()

        # Summary
        passed = sum(1 for _, s in results if s)
        failed = len(results) - passed

        print("=" * 70)
        print(f"SUMMARY: {passed}/{len(results)} demos passed validation")
        if failed > 0:
            print("\nFailed demos:")
            for path, success in results:
                if not success:
                    print(f"  ✗ {path}")

        sys.exit(0 if failed == 0 else 1)
    else:
        # Validate most recent demo
        most_recent = demo_files[-1]
        print(f"Validating most recent demo ({len(demo_files)} total)")
        print()
        success = validate_demo(most_recent)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
