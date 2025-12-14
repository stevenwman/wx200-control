"""
WX200 real robot control with SpaceMouse and trajectory recording.

This script records trajectories for BC (Behavioral Cloning) policy training.

Recorded data:
- Timestamp (starting at t=0)
- State: Robot joint positions from model (configuration.q, not raw encoder values)
  - 5 arm joints + gripper position
- Action: Desired delta xyz and rpy (velocity commands sent to IK)
  - velocity_world: [vx, vy, vz] in m/s
  - angular_velocity_world: [wx, wy, wz] in rad/s

Flow:
1. Startup: Move robot to sim keyframe home position
2. Main loop:
   - Read SpaceMouse → Update pose controller → Solve IK → Send to robot
   - Record trajectory data (if --record enabled)
3. Shutdown: Execute safe exit sequence and save trajectory

Usage:
    python wx200_real_robot_spacemouse_control_record.py --record [--output OUTPUT_FILE]
    
    --record: Enable trajectory recording (required)
    --output: Output file path for trajectory (default: trajectory_YYYYMMDD_HHMMSS.npz)

Example:
    # Record a trajectory with default filename
    python wx200_real_robot_spacemouse_control_record.py --record
    
    # Record with custom filename
    python wx200_real_robot_spacemouse_control_record.py --record --output my_demo.npz
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
import argparse
from datetime import datetime
from spacemouse.spacemouse_driver import SpaceMouseDriver
from robot_control.robot_controller import RobotController
from robot_control.robot_joint_to_motor import JointToMotorTranslator
from robot_control.robot_driver import RobotDriver
from robot_control.robot_shutdown import shutdown_sequence, reboot_motors
from robot_control.robot_config import robot_config
from robot_control.robot_joint_to_motor import encoders_to_joint_angles

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"


def get_sim_home_pose(model):
    """Get the home pose from sim keyframe."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    
    qpos = data.qpos.copy()
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    position = data.site(site_id).xpos.copy()
    current_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    current_site_rot = R.from_matrix(current_site_xmat)
    current_site_quat = current_site_rot.as_quat()
    orientation_quat_wxyz = np.array([
        current_site_quat[3], 
        current_site_quat[0], 
        current_site_quat[1], 
        current_site_quat[2]
    ])
    
    return qpos, position, orientation_quat_wxyz


def save_trajectory(trajectory, output_path):
    """
    Save trajectory data to NPZ file.
    
    Args:
        trajectory: List of dicts with keys: 'timestamp', 'state', 'action'
        output_path: Path to save the trajectory file
    """
    if not trajectory:
        print("Warning: No trajectory data to save")
        return
    
    # Convert to numpy arrays for efficient storage
    timestamps = np.array([t['timestamp'] for t in trajectory])
    states = np.array([t['state'] for t in trajectory])  # Shape: (N, 6) - 5 joints + gripper
    actions = np.array([t['action'] for t in trajectory])  # Shape: (N, 6) - [vx,vy,vz,wx,wy,wz]
    
    # Save as NPZ (numpy compressed format)
    np.savez_compressed(
        output_path,
        timestamps=timestamps,
        states=states,
        actions=actions,
        metadata={
            'num_samples': len(trajectory),
            'control_frequency': robot_config.control_frequency,
            'duration_seconds': timestamps[-1] if len(timestamps) > 0 else 0.0,
            'state_dim': 6,  # 5 joints + gripper
            'action_dim': 6,  # 3 linear + 3 angular velocities
            'state_labels': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'gripper'],
            'action_labels': ['vx', 'vy', 'vz', 'wx', 'wy', 'wz'],
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"\n✓ Trajectory saved to: {output_path}")
    print(f"  - {len(trajectory)} samples")
    print(f"  - Duration: {timestamps[-1]:.2f} seconds")
    print(f"  - Frequency: {len(trajectory) / timestamps[-1]:.2f} Hz" if timestamps[-1] > 0 else "")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='WX200 Real Robot Control with SpaceMouse and Trajectory Recording')
    parser.add_argument('--record', action='store_true', required=True,
                       help='Enable trajectory recording (required)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path for trajectory (default: trajectory_YYYYMMDD_HHMMSS.npz)')
    args = parser.parse_args()
    
    ENABLE_RECORDING = args.record
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"trajectory_{timestamp_str}.npz"
    
    print("WX200 Real Robot Control with SpaceMouse - Trajectory Recording")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    print("- Trajectory recording enabled")
    print(f"- Output file: {args.output}")
    print("- Safe startup and shutdown sequences")
    print("="*60)
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"\nSim home pose - EE position: {home_position}")
    
    robot_driver = RobotDriver()
    
    try:
        print("\nConnecting to robot...")
        robot_driver.connect()
        
        translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        # Use config home positions if specified, otherwise compute from sim keyframe
        if robot_config.startup_home_positions is not None:
            # Use configured home positions directly
            home_motor_positions = {
                motor_id: pos for motor_id, pos in zip(robot_config.motor_ids, robot_config.startup_home_positions)
            }
            print(f"Using configured home positions: {robot_config.startup_home_positions}")
        else:
            # Compute from sim keyframe (original behavior)
            home_joint_angles = home_qpos[:5]
            home_gripper_pos = robot_config.gripper_open_pos
            
            home_motor_positions = translator.joint_commands_to_motor_positions(
                joint_angles_rad=home_joint_angles,
                gripper_position=home_gripper_pos
            )
            print(f"Computed home positions from sim keyframe: {home_motor_positions}")
        
        translator.set_home_encoders([
            home_motor_positions.get(1, 2048),
            home_motor_positions.get(2, 2048),
            home_motor_positions.get(3, 2048),
            home_motor_positions.get(4, 2048),
            home_motor_positions.get(5, 2048),
            home_motor_positions.get(6, 2048),
            home_motor_positions.get(7, 2048),
        ])
        
        print("\n" + "="*60)
        print("STARTUP SEQUENCE")
        print("="*60)
        
        # Step 1: Move all motors to center position (2048) first
        print("\nStep 1: Moving all motors to center position (2048)...")
        center_positions = {motor_id: 2048 for motor_id in robot_config.motor_ids}
        robot_driver.send_motor_positions(center_positions, velocity_limit=robot_config.velocity_limit)
        print("Waiting for motors to reach center position...")
        time.sleep(3.0)  # Wait for movement to complete
        
        # Step 2: Move to configured home position
        print(f"\nStep 2: Moving to home position...")
        print(f"Home motor positions: {home_motor_positions}")
        robot_driver.move_to_home(home_motor_positions, velocity_limit=robot_config.velocity_limit)
        
        # Step 3: Read actual robot position and sync MuJoCo
        print("\nStep 3: Reading actual robot position and syncing simulation...")
        print("Waiting for encoders to settle...")
        time.sleep(2.0)  # Wait longer for encoders to settle after movement
        
        # Try reading encoders with retries
        print("Reading encoder positions (with retries)...")
        robot_encoders = robot_driver.read_all_encoders(max_retries=5, retry_delay=0.2)
        
        # Check if any encoders were successfully read
        successful_reads = sum(1 for v in robot_encoders.values() if v is not None)
        print(f"Successfully read {successful_reads}/{len(robot_encoders)} encoders")
        
        if successful_reads == 0:
            print("\n⚠️  ERROR: Failed to read any encoder positions!")
            print("   This could indicate a communication problem with the robot.")
            print("   Aborting to prevent robot damage...")
            raise RuntimeError("Encoder reading failed - all encoders returned None")
        
        if successful_reads < len(robot_encoders):
            print(f"⚠️  WARNING: Only {successful_reads}/{len(robot_encoders)} encoders read successfully")
            print(f"   Missing encoders: {[mid for mid, val in robot_encoders.items() if val is None]}")
        
        print(f"Encoder values: {robot_encoders}")
        
        robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
        print(f"Converted joint angles: {robot_joint_angles[:5]}")
        
        # Safety check: abort if joint angles are all zero (encoder reading failed)
        if np.allclose(robot_joint_angles[:5], 0, atol=1e-6):
            print("\n⚠️  ERROR: Joint angle conversion resulted in all zeros!")
            print("   This will cause the robot to jump to 2048 (center position)")
            print("   Aborting to prevent robot damage...")
            raise RuntimeError("Joint angle conversion failed - robot_joint_angles are all zero")
        print(f"DEBUG: data.qpos shape: {data.qpos.shape}, len: {len(data.qpos)}")
        print(f"DEBUG: data.qpos before update: {data.qpos[:10]}")
        
        # Update MuJoCo data with actual robot position
        data.qpos[:5] = robot_joint_angles[:5]  # 5 arm joints
        if len(data.qpos) > 5:
            data.qpos[5] = robot_joint_angles[5]  # Gripper
        
        print(f"DEBUG: data.qpos after setting: {data.qpos[:10]}")
        
        # Compute forward kinematics
        mujoco.mj_forward(model, data)
        
        # CRITICAL: Update mink Configuration from MuJoCo data
        # This properly syncs the configuration object with the actual robot state
        print(f"DEBUG: configuration.q before update: {configuration.q[:5]}")
        configuration.update(data.qpos)
        print(f"DEBUG: configuration.q after update: {configuration.q[:5]}")
        
        # Get actual end-effector pose from robot
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        actual_position = data.site(site_id).xpos.copy()
        actual_site_xmat = data.site(site_id).xmat.reshape(3, 3)
        actual_site_rot = R.from_matrix(actual_site_xmat)
        actual_site_quat = actual_site_rot.as_quat()
        actual_orientation_quat_wxyz = np.array([
            actual_site_quat[3], 
            actual_site_quat[0], 
            actual_site_quat[1], 
            actual_site_quat[2]
        ])
        
        print(f"✓ Synced to actual robot position: {actual_position}")
        print(f"  Robot joint angles (from encoders): {robot_joint_angles[:5]}")
        print(f"  data.qpos after update: {data.qpos[:5]}")
        print(f"  Configuration.q after update: {configuration.q[:5]}")
        print(f"  Configuration gripper: {configuration.q[5] if len(configuration.q) > 5 else 'N/A'}")
        
        print("\n" + "="*60)
        print("✓ Robot is now at home position")
        print("Ready for SpaceMouse control!")
        print("RECORDING: Trajectory will be saved on exit")
        print("Press Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        spacemouse = SpaceMouseDriver(
            velocity_scale=robot_config.velocity_scale,
            angular_velocity_scale=robot_config.angular_velocity_scale
        )
        spacemouse.start()
        
        robot_controller = RobotController(
            model=model,
            initial_position=actual_position,
            initial_orientation_quat_wxyz=actual_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture target with actual robot configuration
        # This sets the posture task to prefer the current configuration
        robot_controller.initialize_posture_target(configuration)
        
        # Also reset the pose controller target to match actual robot pose
        # This ensures the target starts at the actual robot position, not a computed position
        robot_controller.reset_pose(actual_position, actual_orientation_quat_wxyz)
        
        # CRITICAL: Set the end-effector task target to match current configuration
        # This ensures IK doesn't try to move from current position
        current_target_pose = robot_controller.get_target_pose()
        robot_controller.end_effector_task.set_target(current_target_pose)
        
        # Initialize gripper position from actual robot state (not hardcoded)
        gripper_current_position = robot_joint_angles[5] if len(robot_joint_angles) > 5 else robot_config.gripper_open_pos
        
        # Final verification: Check configuration state before entering control loop
        print(f"\n[FINAL CHECK before control loop]")
        print(f"  Configuration.q: {configuration.q[:5]}")
        print(f"  data.qpos: {data.qpos[:5]}")
        print(f"  Pose controller target: {robot_controller.pose_controller.target_position}")
        
        # If configuration is still zero, something is wrong - abort
        if np.allclose(configuration.q[:5], 0, atol=1e-6):
            print("\n⚠️  ERROR: Configuration is still zero after sync!")
            print("   This will cause the robot to jump to 2048 (center position)")
            print("   Aborting to prevent robot damage...")
            raise RuntimeError("Configuration sync failed - configuration.q is still zero")
        
        # Verify configuration matches data by recomputing forward kinematics
        if len(data.qpos) >= len(configuration.q):
            data.qpos[:len(configuration.q)] = configuration.q
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        computed_ee_pos = data.site(site_id).xpos.copy()
        print(f"  Computed EE position from config: {computed_ee_pos}")
        print(f"  Actual EE position: {actual_position}")
        print(f"  Match: {np.allclose(computed_ee_pos, actual_position, atol=1e-3)}")
        
        
        control_rate = RateLimiter(frequency=robot_config.control_frequency, warn=False)
        control_loop_active = True
        
        # Initialize trajectory recording
        trajectory = []
        recording_start_time = None
        
        try:
            iteration_count = 0
            while control_loop_active:
                dt = control_rate.dt
                iteration_count += 1
                
                spacemouse.update()
                
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                # Debug: Check configuration before IK (first few iterations or when input detected)
                vel_magnitude = np.linalg.norm(velocity_world) + np.linalg.norm(angular_velocity_world)
                if iteration_count <= 10 or (vel_magnitude > 0.001 and iteration_count <= 30):
                    print(f"\n[DEBUG Iteration {iteration_count}]")
                    print(f"  Config BEFORE IK: {configuration.q[:5]}")
                    print(f"  Velocity command: vel={velocity_world}, omega={angular_velocity_world}")
                    print(f"  Target pose: {robot_controller.pose_controller.target_position}")
                
                # Store config before IK for comparison
                config_before = configuration.q[:5].copy() if iteration_count <= 10 or (vel_magnitude > 0.001 and iteration_count <= 30) else None
                
                # Update robot controller with velocity commands
                # This modifies configuration.q via IK solver
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                
                # Get joint commands from IK solution (configuration.q is updated by IK)
                joint_commands_rad = configuration.q[:5].copy()
                
                # Debug: Check configuration after IK
                if iteration_count <= 10 or (vel_magnitude > 0.001 and iteration_count <= 30):
                    print(f"  Config AFTER IK:  {joint_commands_rad}")
                    if config_before is not None:
                        config_change = np.linalg.norm(joint_commands_rad - config_before)
                        print(f"  Config change magnitude: {config_change:.6f} rad")
                    
                    # Check IK error
                    data.qpos[:len(configuration.q)] = configuration.q
                    mujoco.mj_forward(model, data)
                    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                    current_ee_pos = data.site(site_id).xpos.copy()
                    target_ee_pos = robot_controller.pose_controller.target_position
                    pos_error = np.linalg.norm(current_ee_pos - target_ee_pos)
                    print(f"  EE position error: {pos_error:.6f} m (current: {current_ee_pos}, target: {target_ee_pos})")
                    
                    motor_positions = translator.joint_commands_to_motor_positions(
                        joint_angles_rad=joint_commands_rad,
                        gripper_position=gripper_current_position
                    )
                    motor_list = [motor_positions.get(i, 0) for i in robot_config.motor_ids]
                    print(f"  Motor positions:  {motor_list}")
                    # Check if motors are being sent to 2048 (center)
                    if any(abs(m - 2048) < 10 for m in motor_list[:5]):
                        print(f"  ⚠️  WARNING: Some motors near 2048 (center position)!")
                
                # Incremental gripper control
                left_button_pressed, right_button_pressed = spacemouse.get_gripper_button_states()
                
                if left_button_pressed:
                    gripper_current_position -= robot_config.gripper_increment_rate
                    gripper_current_position = max(gripper_current_position, robot_config.gripper_open_pos)
                elif right_button_pressed:
                    gripper_current_position += robot_config.gripper_increment_rate
                    gripper_current_position = min(gripper_current_position, robot_config.gripper_closed_pos)
                
                gripper_target = gripper_current_position
                
                # Record trajectory data (state and action)
                if ENABLE_RECORDING:
                    # Initialize recording start time on first sample
                    if recording_start_time is None:
                        recording_start_time = time.perf_counter()
                    
                    # State: joint positions from model (configuration.q)
                    # 5 arm joints + gripper position
                    state = np.concatenate([
                        configuration.q[:5],  # 5 joint angles (radians)
                        np.array([gripper_target])  # Gripper position (meters)
                    ])
                    
                    # Action: velocity commands sent to IK
                    # delta xyz (linear velocity) + delta rpy (angular velocity)
                    action = np.concatenate([
                        velocity_world,  # [vx, vy, vz] in m/s
                        angular_velocity_world  # [wx, wy, wz] in rad/s
                    ])
                    
                    # Timestamp relative to recording start (t=0)
                    timestamp = time.perf_counter() - recording_start_time
                    
                    trajectory.append({
                        'timestamp': timestamp,
                        'state': state.copy(),
                        'action': action.copy()
                    })
                
                # Convert joint commands to motor positions
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                
                # Send motor commands to robot
                robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
                
                control_rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            control_loop_active = False
            time.sleep(0.3)
            
            # Save trajectory if recording was enabled
            if ENABLE_RECORDING and trajectory:
                output_path = Path(args.output)
                save_trajectory(trajectory, output_path)
            
            # Execute shutdown sequence
            try:
                shutdown_sequence(robot_driver, velocity_limit=robot_config.velocity_limit)
            except Exception as e:
                print(f"Error during shutdown sequence: {e}")
                import traceback
                traceback.print_exc()
            
            # Stop SpaceMouse
            spacemouse.stop()
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during initialization...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Reboot motors after shutdown
        reboot_motors(robot_driver)
        robot_driver.disconnect()


if __name__ == "__main__":
    main()
