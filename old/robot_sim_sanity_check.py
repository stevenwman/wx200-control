"""
Sanity check tool to verify robot state matches simulation state.

This tool helps verify:
1. Joint-to-motor translation is correct (forward: joint angles → encoder positions)
2. Motor-to-joint translation is correct (inverse: encoder positions → joint angles)
3. Robot encoder readings match simulation expectations
4. Calibration offsets for joint 1 (motors 2 & 3) are correct

Use this before running actual robot control to build confidence.
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import mink
from robot_joint_to_motor import (
    JointToMotorTranslator, 
    encoder_to_joint_angle, 
    ENCODER_CENTER
)
from robot_driver import RobotDriver
from dynamixel_sdk import *

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
ADDR_PRESENT_POSITION = 132


def read_robot_encoders(portHandler, packetHandler):
    """
    Read current encoder positions from all motors.
    
    Returns:
        dict: {motor_id: encoder_position}
    """
    encoder_positions = {}
    for motor_id in MOTOR_IDS:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, motor_id, ADDR_PRESENT_POSITION
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            encoder_positions[motor_id] = dxl_present_position
        else:
            print(f"Warning: Failed to read motor {motor_id}")
            encoder_positions[motor_id] = None
    return encoder_positions


def encoders_to_joint_angles(encoder_positions, translator):
    """
    Convert encoder positions to joint angles (inverse mapping).
    
    Args:
        encoder_positions: dict {motor_id: encoder_position}
        translator: JointToMotorTranslator instance
    
    Returns:
        np.ndarray: Joint angles in radians [q0, q1, q2, q3, q4]
    """
    joint_angles = np.zeros(5)
    
    # Joint 0 (base-1_z) <- Motor 1
    # IMPORTANT: Robot model has this joint FLIPPED
    if 1 in encoder_positions and encoder_positions[1] is not None:
        # NEGATE because robot model is flipped
        joint_angles[0] = -encoder_to_joint_angle(encoder_positions[1])
    
    # Joint 1 (link1-2_x) <- Motors 2 and 3 (average, accounting for flip)
    # IMPORTANT: Robot model has this joint FLIPPED, so we negate the result
    if 2 in encoder_positions and 3 in encoder_positions:
        if encoder_positions[2] is not None and encoder_positions[3] is not None:
            # Motor 2: direct reading
            motor2_angle = encoder_to_joint_angle(encoder_positions[2] - translator.joint1_motor2_offset)
            
            # Motor 3: flipped reading (invert the relationship)
            motor3_encoder_flipped = 2 * ENCODER_CENTER - (encoder_positions[3] - translator.joint1_motor3_offset)
            motor3_angle = encoder_to_joint_angle(motor3_encoder_flipped)
            
            # Average the two, then NEGATE because robot model is flipped
            joint_angles[1] = -(motor2_angle + motor3_angle) / 2.0
    
    # Joint 2 (link2-3_x) <- Motor 4
    if 4 in encoder_positions and encoder_positions[4] is not None:
        joint_angles[2] = encoder_to_joint_angle(encoder_positions[4])
    
    # Joint 3 (link3-4_x) <- Motor 5
    if 5 in encoder_positions and encoder_positions[5] is not None:
        joint_angles[3] = encoder_to_joint_angle(encoder_positions[5])
    
    # Joint 4 (link4-5_y) <- Motor 6
    # IMPORTANT: Robot model has this joint FLIPPED
    if 6 in encoder_positions and encoder_positions[6] is not None:
        # NEGATE because robot model is flipped
        joint_angles[4] = -encoder_to_joint_angle(encoder_positions[6])
    
    return joint_angles


def print_comparison_table(robot_encoders, robot_joint_angles, sim_joint_angles, 
                          sim_to_motor_expected, joint_names):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("ROBOT vs SIMULATION STATE COMPARISON")
    print("="*80)
    
    print("\n--- ENCODER POSITIONS ---")
    print(f"{'Motor ID':<10} {'Robot Encoder':<15} {'Sim Expected':<15} {'Difference':<15}")
    print("-" * 60)
    for motor_id in MOTOR_IDS:
        robot_enc = robot_encoders.get(motor_id, "N/A")
        sim_enc = sim_to_motor_expected.get(motor_id, "N/A")
        if robot_enc != "N/A" and sim_enc != "N/A":
            diff = abs(robot_enc - sim_enc)
            print(f"{motor_id:<10} {robot_enc:<15} {sim_enc:<15} {diff:<15}")
        else:
            print(f"{motor_id:<10} {robot_enc:<15} {sim_enc:<15} {'N/A':<15}")
    
    print("\n--- JOINT ANGLES (radians) ---")
    print(f"{'Joint':<10} {'Robot (from encoders)':<25} {'Sim (from model)':<25} {'Difference (rad)':<20}")
    print("-" * 80)
    for i, joint_name in enumerate(joint_names):
        robot_angle = robot_joint_angles[i]
        sim_angle = sim_joint_angles[i]
        diff = abs(robot_angle - sim_angle)
        diff_deg = np.degrees(diff)
        print(f"{joint_name:<10} {robot_angle:<25.4f} {sim_angle:<25.4f} {diff:.4f} ({diff_deg:.2f}°)")
    
    print("\n" + "="*80)


def main():
    print("Robot-Simulation Sanity Check Tool")
    print("="*60)
    print("This tool will:")
    print("1. Read current encoder positions from robot")
    print("2. Convert to joint angles (inverse mapping)")
    print("3. Compare with simulation state")
    print("4. Show forward mapping (sim joints → expected encoders)")
    print("\nNOTE: For best results, move robot to sim home position first.")
    print("      Large differences are expected if robot is not at home.")
    print("="*60)
    
    # Load simulation model
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # Initialize translator (with default offsets - you may need to adjust)
    translator = JointToMotorTranslator(
        joint1_motor2_offset=0,  # TODO: Calibrate
        joint1_motor3_offset=0   # TODO: Calibrate
    )
    
    # Connect to robot
    robot_driver = RobotDriver()
    
    try:
        robot_driver.connect()
        print("\nConnected to robot. Reading encoder positions...")
        
        # Read current robot encoder positions
        robot_encoders = read_robot_encoders(
            robot_driver.portHandler, 
            robot_driver.packetHandler
        )
        
        print(f"\nRobot encoder positions:")
        for motor_id, enc_pos in robot_encoders.items():
            if enc_pos is not None:
                print(f"  Motor {motor_id}: {enc_pos}")
            else:
                print(f"  Motor {motor_id}: Failed to read")
        
        # Convert robot encoders to joint angles (inverse mapping)
        robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
        
        print(f"\nRobot joint angles (from encoders):")
        joint_names = ["base-1_z", "link1-2_x", "link2-3_x", "link3-4_x", "link4-5_y"]
        for i, joint_name in enumerate(joint_names):
            print(f"  {joint_name}: {robot_joint_angles[i]:.4f} rad ({np.degrees(robot_joint_angles[i]):.2f}°)")
        
        # Get simulation state
        mujoco.mj_forward(model, data)
        sim_joint_angles = data.qpos[:5].copy()  # First 5 joints
        
        print(f"\nSimulation joint angles (from model):")
        for i, joint_name in enumerate(joint_names):
            print(f"  {joint_name}: {sim_joint_angles[i]:.4f} rad ({np.degrees(sim_joint_angles[i]):.2f}°)")
        
        # Forward mapping: sim joints → expected motor positions
        sim_to_motor_expected = translator.joint_commands_to_motor_positions(
            joint_angles_rad=sim_joint_angles,
            gripper_position=data.qpos[5] if len(data.qpos) > 5 else -0.01
        )
        
        print(f"\nExpected motor positions (from sim joints):")
        for motor_id, enc_pos in sorted(sim_to_motor_expected.items()):
            print(f"  Motor {motor_id}: {enc_pos}")
        
        # Print comparison table
        print_comparison_table(
            robot_encoders,
            robot_joint_angles,
            sim_joint_angles,
            sim_to_motor_expected,
            joint_names
        )
        
        # Check for large discrepancies
        print("\n--- SANITY CHECK RESULTS ---")
        max_joint_error = 0
        max_joint_error_name = ""
        total_error = 0
        for i, joint_name in enumerate(joint_names):
            error = abs(robot_joint_angles[i] - sim_joint_angles[i])
            total_error += error
            if error > max_joint_error:
                max_joint_error = error
                max_joint_error_name = joint_name
        
        # Determine if robot is likely at home position
        is_at_home = max_joint_error < 0.2  # ~11.5 degrees tolerance
        
        if is_at_home:
            print("✓ Robot appears to be at sim home position")
            if max_joint_error < 0.1:  # ~5.7 degrees
                print("✓ Joint angles match well (< 0.1 rad difference)")
            else:
                print(f"⚠ Small differences detected. Max error in {max_joint_error_name}: {max_joint_error:.4f} rad ({np.degrees(max_joint_error):.2f}°)")
                print("  This is acceptable for home position")
        else:
            print(f"⚠ Robot is NOT at sim home position")
            print(f"  Max difference in {max_joint_error_name}: {max_joint_error:.4f} rad ({np.degrees(max_joint_error):.2f}°)")
            print(f"  Total joint error: {total_error:.4f} rad ({np.degrees(total_error):.2f}°)")
            print("\n  To get accurate comparison:")
            print("  1. Move robot to sim home position (use go_home.py or manually)")
            print("  2. Run this tool again")
            print("  3. Then you can verify the translation layer is correct")
        
        # Special check for joint 1 (motors 2 & 3)
        if 2 in robot_encoders and 3 in robot_encoders:
            if robot_encoders[2] is not None and robot_encoders[3] is not None:
                motor2_enc = robot_encoders[2]
                motor3_enc = robot_encoders[3]
                
                # For joint 1 at 0, motors 2 and 3 should be symmetric around center
                # (accounting for offsets)
                motor2_relative = motor2_enc - translator.joint1_motor2_offset
                motor3_relative = motor3_enc - translator.joint1_motor3_offset
                expected_symmetry = 2 * ENCODER_CENTER - motor2_relative
                symmetry_error = abs(motor3_relative - expected_symmetry)
                
                print(f"\n--- Joint 1 (Motors 2 & 3) Check ---")
                print(f"Motor 2 encoder: {motor2_enc} (relative to offset: {motor2_relative})")
                print(f"Motor 3 encoder: {motor3_enc} (relative to offset: {motor3_relative})")
                print(f"Expected symmetry: {expected_symmetry}")
                print(f"Symmetry error: {symmetry_error} encoder units (~{symmetry_error * 360 / 4095:.2f}°)")
                
                if symmetry_error < 50:  # ~4.4 degrees
                    print("✓ Motors 2 & 3 are properly symmetric")
                else:
                    print(f"⚠ Warning: Motors 2 & 3 symmetry error is large")
                    print(f"  Current offsets: motor2_offset={translator.joint1_motor2_offset}, motor3_offset={translator.joint1_motor3_offset}")
                    print(f"  You may need to calibrate these offsets")
                    print(f"  Suggested: Set robot to joint 1 = 0, then:")
                    print(f"    joint1_motor2_offset = {motor2_enc} - {ENCODER_CENTER}")
                    print(f"    joint1_motor3_offset = {motor3_enc} - {ENCODER_CENTER}")
        
        # Additional insights
        print("\n--- TRANSLATION LAYER VERIFICATION ---")
        print("Forward mapping (sim → motors):")
        print("  This shows what encoder positions the translation layer would send")
        print("  for the current sim joint angles.")
        print("\nInverse mapping (motors → joints):")
        print("  This shows what joint angles the translation layer would compute")
        print("  from the current robot encoder positions.")
        print("\nIf both mappings are consistent (forward then inverse = identity),")
        print("the translation layer is working correctly.")
        
        # Test round-trip: sim joints → motors → joints
        print("\n--- ROUND-TRIP TEST ---")
        print("Testing: sim joints → motors → joints (should recover sim joints)")
        recovered_joints = encoders_to_joint_angles(sim_to_motor_expected, translator)
        round_trip_errors = np.abs(recovered_joints - sim_joint_angles)
        max_round_trip_error = np.max(round_trip_errors)
        
        if max_round_trip_error < 0.01:  # Very small error
            print("✓ Round-trip test PASSED: Translation layer is consistent")
            print(f"  Max round-trip error: {max_round_trip_error:.6f} rad")
        else:
            print(f"⚠ Round-trip test: Max error {max_round_trip_error:.6f} rad")
            print("  This might indicate an issue with the translation logic")
            for i, joint_name in enumerate(joint_names):
                if round_trip_errors[i] > 0.01:
                    print(f"    {joint_name}: {round_trip_errors[i]:.6f} rad")
        
        print("\n" + "="*80)
        print("Sanity check complete!")
        if is_at_home:
            print("✓ Robot is at home. Translation layer verified.")
            print("  You can proceed with robot control.")
        else:
            print("⚠ Robot is not at home. Move to home position and run again")
            print("  to verify the translation layer.")
        print("="*80)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot_driver.disconnect()


if __name__ == "__main__":
    main()
