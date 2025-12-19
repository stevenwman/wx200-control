"""
Robot startup sequence utilities.

Provides safe startup sequence for WX200 robot, including:
- Computing home positions (from config or sim keyframe)
- Moving to safe intermediate pose
- Moving to startup home position
- Syncing MuJoCo simulation to actual robot state
"""
import time
import numpy as np
import mujoco
import mink
from .robot_config import robot_config
from .robot_joint_to_motor import JointToMotorTranslator, sync_robot_to_mujoco


def get_home_motor_positions(translator, home_qpos=None):
    """
    Get home motor positions from config or compute from sim keyframe.
    
    Args:
        translator: JointToMotorTranslator instance
        home_qpos: Optional qpos from sim keyframe (if None, uses config)
    
    Returns:
        dict: {motor_id: encoder_position} for home pose
    """
    if robot_config.startup_home_positions is not None:
        home_motor_positions = {
            motor_id: pos for motor_id, pos in zip(robot_config.motor_ids, robot_config.startup_home_positions)
        }
        print(f"Using configured home positions: {robot_config.startup_home_positions}")
    else:
        if home_qpos is None:
            raise ValueError("home_qpos required when startup_home_positions is None")
        # Compute from sim keyframe
        home_joint_angles = home_qpos[:5]
        home_gripper_pos = robot_config.gripper_open_pos
        
        home_motor_positions = translator.joint_commands_to_motor_positions(
            joint_angles_rad=home_joint_angles,
            gripper_position=home_gripper_pos
        )
        print(f"Computed home positions from sim keyframe: {home_motor_positions}")
    
    return home_motor_positions


def startup_sequence(robot_driver, translator, model, data, configuration, home_qpos=None):
    """
    Execute safe startup sequence: move to reasonable home, then startup home, and sync MuJoCo.
    
    Args:
        robot_driver: RobotDriver instance
        translator: JointToMotorTranslator instance
        model: MuJoCo model
        data: MuJoCo data
        configuration: mink.Configuration to update
        home_qpos: Optional qpos from sim keyframe (if None, uses config home positions)
    
    Returns:
        tuple: (robot_joint_angles, actual_position, actual_orientation_quat_wxyz, home_motor_positions)
            - robot_joint_angles: np.ndarray of joint angles [q0-q4, gripper]
            - actual_position: [x, y, z] end-effector position
            - actual_orientation_quat_wxyz: [w, x, y, z] end-effector orientation
            - home_motor_positions: dict {motor_id: encoder_position} for home pose
    """
    # Get home motor positions
    home_motor_positions = get_home_motor_positions(translator, home_qpos)
    translator.set_home_encoders(home_motor_positions)
    
    print("\n" + "="*60)
    print("STARTUP SEQUENCE")
    print("="*60)
    
    # Step 1: Move to reasonable home position first (safe intermediate pose)
    print("\nStep 1: Moving to reasonable home position...")
    reasonable_home_positions = {
        motor_id: pos for motor_id, pos in zip(robot_config.motor_ids, robot_config.reasonable_home_pose)
        if pos != -1  # Skip motors with -1 (not used in this pose)
    }
    robot_driver.send_motor_positions(reasonable_home_positions, velocity_limit=robot_config.velocity_limit)
    print("Waiting for motors to reach reasonable home position...")
    time.sleep(3.0)
    
    # Step 2: Move to configured startup home position
    print(f"\nStep 2: Moving to startup home position...")
    robot_driver.move_to_home(home_motor_positions, velocity_limit=robot_config.velocity_limit)
    
    # Step 3: Read actual robot position and sync MuJoCo
    print("\nStep 3: Reading actual robot position and syncing simulation...")
    print("Waiting for encoders to settle...")
    time.sleep(0.1)
    
    # Read encoders and sync to MuJoCo
    robot_encoders = robot_driver.read_all_encoders(max_retries=5, retry_delay=0.2)
    robot_joint_angles, actual_position, actual_orientation_quat_wxyz = sync_robot_to_mujoco(
        robot_encoders, translator, model, data, configuration
    )
    
    print(f"✓ Synced to actual robot position: {actual_position}")
    
    return robot_joint_angles, actual_position, actual_orientation_quat_wxyz, home_motor_positions
