#!/usr/bin/env python3
"""
Test script to verify encoder polling functionality in compact_gym.

This script tests:
1. Encoder polling is working
2. Encoder state is being updated
3. Performance metrics are within acceptable ranges

REQUIREMENTS:
- Robot hardware must be connected
- Python environment with: gymnasium, mujoco, numpy, scipy, cv2
- Run from compact_gym directory or set PYTHONPATH appropriately
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from deployment.gym_env import WX200GymEnv
    from deployment.robot_config import robot_config
except ImportError as e:
    print("="*70)
    print("ERROR: Missing dependencies")
    print("="*70)
    print(f"\n{e}\n")
    print("This test requires a full Python environment with:")
    print("  - gymnasium")
    print("  - mujoco")
    print("  - opencv-python (cv2)")
    print("  - scipy")
    print("  - numpy")
    print()
    print("For syntax verification only, run:")
    print("  python verify_encoder_syntax.py")
    print()
    print("="*70)
    sys.exit(1)

def test_encoder_polling():
    print("="*70)
    print("ENCODER POLLING TEST")
    print("="*70)
    print()

    # Create environment
    print("[1/4] Creating environment...")
    env = WX200GymEnv(
        max_episode_length=100,
        control_frequency=robot_config.control_frequency,
        enable_aruco=False,  # Disable ArUco for simpler test
        show_video=False
    )

    try:
        # Reset environment (initializes hardware)
        print("[2/4] Resetting environment (initializing hardware)...")
        obs, info = env.reset()
        print("    Initial observation shape:", obs.shape)

        # Run a few steps with no-op action to test encoder polling
        # Action space is [-1, 1] where -1 = gripper_open, 0 = halfway, 1 = gripper_closed
        # To maintain open position (no movement), use -1.0 for gripper
        print("[3/4] Running 20 steps with encoder polling (no movement)...")
        noop_action = np.zeros(7)
        noop_action[6] = -1.0  # Keep gripper open (no movement)

        encoder_poll_counts = []
        encoder_states = []

        for i in range(20):
            step_start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(noop_action)
            step_time = time.perf_counter() - step_start

            # Check encoder data in info
            if 'encoder_values' in info and info['encoder_values'] is not None:
                encoder_states.append(info['encoder_values'])
                encoder_poll_counts.append(env.robot_hardware.encoder_poll_count)

                if i % 5 == 0:
                    print(f"    Step {i:2d}: encoder_poll_count={env.robot_hardware.encoder_poll_count}, "
                          f"step_time={step_time*1000:.1f}ms")
            else:
                print(f"    Step {i:2d}: WARNING - No encoder data in info!")

        # Print statistics
        print()
        print("[4/4] Encoder Polling Statistics:")
        print("="*70)

        if len(encoder_poll_counts) > 0:
            print(f"✓ Total encoder polls: {encoder_poll_counts[-1]}")
            print(f"✓ Encoder states collected: {len(encoder_states)}")

            if len(env.robot_hardware.encoder_poll_times) > 0:
                avg_poll_time = np.mean(env.robot_hardware.encoder_poll_times) * 1000
                max_poll_time = np.max(env.robot_hardware.encoder_poll_times) * 1000
                print(f"✓ Average encoder read time: {avg_poll_time:.1f}ms")
                print(f"✓ Max encoder read time: {max_poll_time:.1f}ms")

                if avg_poll_time < 15.0:
                    print("  → Performance: GOOD (avg < 15ms)")
                else:
                    print("  → Performance: WARNING (avg >= 15ms)")

            if len(env.robot_hardware.encoder_poll_intervals) > 1:
                avg_interval = np.mean(env.robot_hardware.encoder_poll_intervals)
                avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                print(f"✓ Average polling frequency: {avg_freq:.1f}Hz")
                print(f"  (Target: {robot_config.control_frequency}Hz)")

            if env.robot_hardware._skipped_reads_count > 0:
                print(f"⚠ Skipped encoder reads: {env.robot_hardware._skipped_reads_count}")
            else:
                print(f"✓ No skipped reads")

            # Check encoder value consistency
            if len(encoder_states) > 1:
                print()
                print("Encoder Value Changes:")
                first_encoders = encoder_states[0]
                last_encoders = encoder_states[-1]

                if isinstance(first_encoders, dict) and isinstance(last_encoders, dict):
                    for motor_id in robot_config.motor_ids:
                        if motor_id in first_encoders and motor_id in last_encoders:
                            delta = abs(last_encoders[motor_id] - first_encoders[motor_id])
                            print(f"  Motor {motor_id}: {first_encoders[motor_id]:4d} → {last_encoders[motor_id]:4d} (Δ={delta:3d})")
        else:
            print("✗ ERROR: No encoder data was collected!")
            print("  This indicates encoder polling is not working correctly.")

        print()
        print("="*70)
        print("TEST COMPLETE")
        print("="*70)

    finally:
        # Cleanup - ALWAYS runs even if there's an error
        print()
        print("Cleaning up and shutting down robot...")
        env.close()
        print("✓ Robot shutdown complete")

if __name__ == "__main__":
    try:
        test_encoder_polling()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
