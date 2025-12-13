"""
Live synchronization tool: Robot encoders → Simulation visualization.

This tool reads encoder positions from the real robot and updates
the MuJoCo simulation in real-time, so you can see the simulation
match the robot as you manipulate it manually.

Great for verifying the translation layer is working correctly!
"""
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from robot_joint_to_motor import JointToMotorTranslator, encoder_to_joint_angle, ENCODER_CENTER, ENCODER_MAX
from robot_driver import RobotDriver
from dynamixel_sdk import *

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

MOTOR_IDS = [1, 2, 3, 4, 5, 6, 7]
ADDR_PRESENT_POSITION = 132
ADDR_TORQUE_ENABLE = 64

# Update frequency
UPDATE_FREQUENCY = 50.0  # Hz


def read_robot_encoders(portHandler, packetHandler):
    """Read current encoder positions from all motors."""
    encoder_positions = {}
    for motor_id in MOTOR_IDS:
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(
            portHandler, motor_id, ADDR_PRESENT_POSITION
        )
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            encoder_positions[motor_id] = dxl_present_position
        else:
            encoder_positions[motor_id] = None
    return encoder_positions


def joint_angles_to_expected_encoders(joint_angles, gripper_position, translator):
    """
    Convert joint angles to expected encoder positions (forward mapping).
    This is the inverse of encoders_to_joint_angles - used for verification.
    
    Args:
        joint_angles: np.ndarray of joint angles in radians [q0, q1, q2, q3, q4]
        gripper_position: Gripper position in meters (gripper_l position, -0.026 to 0.0)
        translator: JointToMotorTranslator instance
    
    Returns:
        dict: {motor_id: expected_encoder_position} for all 7 motors
    """
    from robot_joint_to_motor import joint_angle_to_encoder, ENCODER_CENTER, ENCODER_MAX
    
    expected_encoders = {}
    
    # Joint 0 (base-1_z) -> Motor 1 (FLIPPED)
    if len(joint_angles) > 0:
        joint0_angle = -joint_angles[0]  # Flip
        expected_encoders[1] = joint_angle_to_encoder(joint0_angle)
    
    # Joint 1 (link1-2_x) -> Motors 2 and 3 (FLIPPED, opposing)
    if len(joint_angles) > 1:
        joint1_angle = -joint_angles[1]  # Flip
        encoder_joint1 = joint_angle_to_encoder(joint1_angle)
        # Motor 2: direct
        expected_encoders[2] = int(np.clip(encoder_joint1 + translator.joint1_motor2_offset, 0, ENCODER_MAX))
        # Motor 3: flipped
        motor3_base = ENCODER_CENTER - (encoder_joint1 - ENCODER_CENTER)
        expected_encoders[3] = int(np.clip(motor3_base + translator.joint1_motor3_offset, 0, ENCODER_MAX))
    
    # Joint 2 (link2-3_x) -> Motor 4
    if len(joint_angles) > 2:
        expected_encoders[4] = joint_angle_to_encoder(joint_angles[2])
    
    # Joint 3 (link3-4_x) -> Motor 5
    if len(joint_angles) > 3:
        expected_encoders[5] = joint_angle_to_encoder(joint_angles[3])
    
    # Joint 4 (link4-5_y) -> Motor 6 (FLIPPED)
    if len(joint_angles) > 4:
        joint4_angle = -joint_angles[4]  # Flip
        expected_encoders[6] = joint_angle_to_encoder(joint4_angle)
    
    # Gripper -> Motor 7
    if gripper_position is not None:
        GRIPPER_ENCODER_MIN = 1559  # Closed position
        GRIPPER_ENCODER_MAX = 2776  # Open position
        GRIPPER_ENCODER_RANGE = GRIPPER_ENCODER_MAX - GRIPPER_ENCODER_MIN  # 1217
        sim_gripper_range = 0.026
        # Normalize sim position: -0.026 -> 0, 0.0 -> 1
        sim_normalized = (gripper_position + sim_gripper_range) / sim_gripper_range  # 0 to 1
        # Map to encoder: sim_normalized=0 (open) -> 2776, sim_normalized=1 (closed) -> 1559
        encoder = GRIPPER_ENCODER_MIN + (1.0 - sim_normalized) * GRIPPER_ENCODER_RANGE
        expected_encoders[7] = int(encoder)
    
    return expected_encoders


def encoders_to_joint_angles(encoder_positions, translator):
    """
    Convert encoder positions to joint angles (inverse mapping).
    
    Args:
        encoder_positions: dict {motor_id: encoder_position}
        translator: JointToMotorTranslator instance
    
    Returns:
        np.ndarray: Joint angles in radians [q0, q1, q2, q3, q4, gripper]
    """
    joint_angles = np.zeros(6)  # 5 joints + gripper
    
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
            motor2_enc_relative = encoder_positions[2] - translator.joint1_motor2_offset
            motor2_angle = encoder_to_joint_angle(motor2_enc_relative)
            
            # Motor 3: flipped reading
            motor3_enc_relative = encoder_positions[3] - translator.joint1_motor3_offset
            motor3_enc_flipped = 2 * ENCODER_CENTER - motor3_enc_relative
            motor3_angle = encoder_to_joint_angle(motor3_enc_flipped)
            
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
    
    # Gripper <- Motor 7 (approximate conversion)
    # IMPORTANT: 
    # 1. Gripper motor is FLIPPED
    # 2. Real robot gripper encoder range: [1559 (closed), 2776 (open)]
    #    Measured actual range, not centered around 2048
    if 7 in encoder_positions and encoder_positions[7] is not None:
        # Real robot encoder range
        GRIPPER_ENCODER_MIN = 1559  # Closed position
        GRIPPER_ENCODER_MAX = 2776  # Open position
        GRIPPER_ENCODER_RANGE = GRIPPER_ENCODER_MAX - GRIPPER_ENCODER_MIN  # 1217
        
        # Clamp encoder to valid range
        encoder = max(GRIPPER_ENCODER_MIN, min(GRIPPER_ENCODER_MAX, encoder_positions[7]))
        
        # Normalize encoder to [0, 1] where 0 = closed, 1 = open
        normalized = (encoder - GRIPPER_ENCODER_MIN) / GRIPPER_ENCODER_RANGE  # 0 to 1
        
        # Map to sim range: [0, 1] -> sim position [0.0, -0.026]
        # 0 (closed, encoder=1559) -> sim 0.0 (closed)
        # 1 (open, encoder=2776) -> sim -0.026 (open)
        sim_gripper_range = 0.026
        joint_angles[5] = -sim_gripper_range * normalized  # 0.0 to -0.026 (closed to open)
    
    return joint_angles


def main():
    print("Robot-Simulation Live Sync Tool")
    print("="*60)
    print("This tool will:")
    print("1. Read encoder positions from robot in real-time")
    print("2. Convert to joint angles")
    print("3. Update MuJoCo simulation to match robot pose")
    print("4. Display live visualization")
    print("\nManually move the robot and watch the simulation follow!")
    print("Press ESC in the viewer to exit")
    print("="*60)
    
    # Load simulation model
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    
    # Initialize translator
    translator = JointToMotorTranslator(
        joint1_motor2_offset=0,  # TODO: Calibrate if needed
        joint1_motor3_offset=0   # TODO: Calibrate if needed
    )
    
    # Connect to robot
    robot_driver = RobotDriver()
    
    try:
        robot_driver.connect()
        
        # Disable torque so robot can be moved manually
        # Note: connect() enables torque, so we disable it here
        print("\nDisabling torque on all motors (robot will be limp)...")
        robot_driver.disable_torque_all()
        print("✓ Torque disabled. Robot is now limp - you can move it manually.")
        
        print("\nStarting live sync...")
        print("Move the robot manually and watch the simulation follow!")
        print("Press ESC in the viewer to exit")
        
        # Initialize viewer
        with mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            
            # Control loop
            rate = RateLimiter(frequency=UPDATE_FREQUENCY, warn=False)
            
            print_count = 0
            print_interval = 50  # Print status every N frames
            
            while viewer.is_running():
                dt = rate.dt
                
                # Read robot encoders
                robot_encoders = read_robot_encoders(
                    robot_driver.portHandler,
                    robot_driver.packetHandler
                )
                
                # Convert to joint angles
                robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
                
                # Update simulation with robot joint angles
                data.qpos[:5] = robot_joint_angles[:5]  # Update first 5 joints
                
                # Update gripper: sync gripper_l and gripper_r like in sim (equality constraint)
                # gripper_l: -0.026 (open) to 0.0 (closed)
                # gripper_r: 0.0 (open) to 0.026 (closed) - opposite (gripper_r = -gripper_l)
                gripper_l_pos = robot_joint_angles[5]  # From motor 7
                gripper_r_pos = -gripper_l_pos  # Sync: gripper_r = -gripper_l (like sim equality constraint)
                
                # Find gripper joint IDs
                gripper_l_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_l")
                gripper_r_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_r")
                
                if gripper_l_joint_id >= 0:
                    data.qpos[gripper_l_joint_id] = gripper_l_pos
                if gripper_r_joint_id >= 0:
                    data.qpos[gripper_r_joint_id] = gripper_r_pos
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
                
                # Update viewer
                viewer.sync()
                
                # Calculate expected encoders from current joint angles
                expected_encoders = joint_angles_to_expected_encoders(
                    robot_joint_angles[:5],  # First 5 joints
                    robot_joint_angles[5],   # Gripper position
                    translator
                )
                
                # Print detailed comparison every frame
                print("\n" + "="*80)
                print("JOINT POSITIONS (from simulation kinematics):")
                joint_names = ["base-1_z", "link1-2_x", "link2-3_x", "link3-4_x", "link4-5_y", "gripper_l"]
                for i, name in enumerate(joint_names):
                    if i < 5:
                        print(f"  {name:15s}: {robot_joint_angles[i]:8.4f} rad ({np.degrees(robot_joint_angles[i]):7.2f}°)")
                    else:
                        print(f"  {name:15s}: {robot_joint_angles[i]:8.4f} m")
                
                print("\nEXPECTED vs ACTUAL ENCODER VALUES:")
                print(f"{'Motor':<8} {'Expected':<12} {'Actual':<12} {'Error':<12} {'Error %':<10}")
                print("-" * 60)
                for motor_id in MOTOR_IDS:
                    expected = expected_encoders.get(motor_id, None)
                    actual = robot_encoders.get(motor_id, None)
                    
                    if expected is not None and actual is not None:
                        error = actual - expected
                        error_pct = (error / ENCODER_MAX) * 100 if ENCODER_MAX > 0 else 0
                        status = "✓" if abs(error) < 10 else "⚠"  # Warning if error > 10 encoder units
                        print(f"{'M' + str(motor_id):<8} {expected:<12} {actual:<12} {error:<12} {error_pct:6.2f}% {status}")
                    elif expected is not None:
                        print(f"{'M' + str(motor_id):<8} {expected:<12} {'N/A':<12} {'N/A':<12} {'N/A':<10}")
                    elif actual is not None:
                        print(f"{'M' + str(motor_id):<8} {'N/A':<12} {actual:<12} {'N/A':<12} {'N/A':<10}")
                
                # Special check for opposing motors (2 and 3)
                if 2 in expected_encoders and 3 in expected_encoders:
                    if 2 in robot_encoders and 3 in robot_encoders:
                        motor2_expected = expected_encoders[2]
                        motor3_expected = expected_encoders[3]
                        motor2_actual = robot_encoders[2]
                        motor3_actual = robot_encoders[3]
                        
                        # Check if they're properly opposing
                        expected_opposition = abs(motor2_expected - (2 * ENCODER_CENTER - motor3_expected))
                        actual_opposition = abs(motor2_actual - (2 * ENCODER_CENTER - motor3_actual))
                        
                        print(f"\nOPPOSING MOTOR CHECK (Motors 2 & 3):")
                        print(f"  Expected opposition error: {expected_opposition:.1f} encoder units")
                        print(f"  Actual opposition error: {actual_opposition:.1f} encoder units")
                        if actual_opposition > 50:
                            print(f"  ⚠ WARNING: Large opposition error! Motors may be fighting each other.")
                        else:
                            print(f"  ✓ Motors are properly opposing.")
                
                rate.sleep()
        
        print("\nLive sync stopped")
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected. Stopping...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        robot_driver.disconnect()
        print("Robot driver disconnected")


if __name__ == "__main__":
    main()
