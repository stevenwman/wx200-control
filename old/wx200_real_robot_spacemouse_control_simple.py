"""
WX200 real robot control with SpaceMouse.

Flow:
1. Startup: Move robot to sim keyframe home position
2. Main loop: Read SpaceMouse → Update pose controller → Solve IK → Send to robot
3. Shutdown: Execute safe exit sequence

Usage:
    python wx200_real_robot_spacemouse_control_simple.py [--profile]
    
    --profile: Enable control frequency profiling
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
import argparse
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


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='WX200 Real Robot Control with SpaceMouse')
    parser.add_argument('--profile', action='store_true',
                       help='Enable control frequency profiling')
    args = parser.parse_args()
    
    ENABLE_PROFILING = args.profile
    
    print("WX200 Real Robot Control with SpaceMouse")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    print("- Safe startup and shutdown sequences")
    if ENABLE_PROFILING:
        print("- Control frequency profiling enabled")
    print("="*60)
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"\nSim home pose - EE position: {home_position}")
    
    robot_driver = RobotDriver()
    
    # Enable profiling if requested
    if ENABLE_PROFILING:
        from utils.control_frequency_profiler import ControlFrequencyProfiler
        from robot_control.robot_driver_profiling import create_profiled_driver
        profiler = ControlFrequencyProfiler(stats_interval=100)
        robot_driver = create_profiled_driver(robot_driver, stats_interval=100)
    else:
        profiler = None
    
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
        
        robot_encoders = robot_driver.read_all_encoders()
        print(f"DEBUG: robot_encoders: {robot_encoders}")
        
        robot_joint_angles = encoders_to_joint_angles(robot_encoders, translator)
        print(f"DEBUG: robot_joint_angles: {robot_joint_angles}")
        
        # Safety check: abort if joint angles are all zero (encoder reading failed)
        if np.allclose(robot_joint_angles[:5], 0, atol=1e-6):
            print("\n⚠️  ERROR: Encoder reading failed - all joint angles are zero!")
            print("   This will cause the robot to jump to 2048 (center position)")
            print("   Aborting to prevent robot damage...")
            raise RuntimeError("Encoder reading failed - robot_joint_angles are all zero")
        
        # Update MuJoCo data with actual robot position
        data.qpos[:5] = robot_joint_angles[:5]  # 5 arm joints
        if len(data.qpos) > 5:
            data.qpos[5] = robot_joint_angles[5]  # Gripper
        
        # Compute forward kinematics
        mujoco.mj_forward(model, data)
        
        # CRITICAL: Update mink Configuration from MuJoCo data
        # This properly syncs the configuration object with the actual robot state
        configuration.update(data.qpos)
        
        # Verify configuration was updated correctly
        if np.allclose(configuration.q[:5], 0, atol=1e-6):
            print("\n⚠️  ERROR: Configuration is still zero after update!")
            print("   Aborting to prevent robot damage...")
            raise RuntimeError("Configuration update failed - configuration.q is still zero")
        
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
        
        print("\n" + "="*60)
        print("✓ Robot is now at home position")
        print("Ready for SpaceMouse control!")
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
        
        control_rate = RateLimiter(frequency=robot_config.control_frequency, warn=False)
        control_loop_active = True
        
        # Start profiling if enabled
        if profiler:
            profiler.start()
        
        try:
            while control_loop_active:
                if profiler:
                    profiler.before_loop()
                
                dt = control_rate.dt
                
                # Profile SpaceMouse update if profiling
                if profiler:
                    t0 = time.perf_counter()
                    spacemouse.update()
                    profiler.record_pipeline_step('spacemouse_update', time.perf_counter() - t0)
                else:
                    spacemouse.update()
                
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                # Profile IK solve if profiling
                if profiler:
                    t0 = time.perf_counter()
                    robot_controller.update_from_velocity_command(
                        velocity_world=velocity_world,
                        angular_velocity_world=angular_velocity_world,
                        dt=dt,
                        configuration=configuration
                    )
                    profiler.record_pipeline_step('ik_solve', time.perf_counter() - t0)
                else:
                    robot_controller.update_from_velocity_command(
                        velocity_world=velocity_world,
                        angular_velocity_world=angular_velocity_world,
                        dt=dt,
                        configuration=configuration
                    )
                
                joint_commands_rad = configuration.q[:5]
                
                # Incremental gripper control
                left_button_pressed, right_button_pressed = spacemouse.get_gripper_button_states()
                
                if left_button_pressed:
                    gripper_current_position -= robot_config.gripper_increment_rate
                    gripper_current_position = max(gripper_current_position, robot_config.gripper_open_pos)
                elif right_button_pressed:
                    gripper_current_position += robot_config.gripper_increment_rate
                    gripper_current_position = min(gripper_current_position, robot_config.gripper_closed_pos)
                
                gripper_target = gripper_current_position
                
                # Profile joint-to-motor conversion if profiling
                if profiler:
                    t0 = time.perf_counter()
                    motor_positions = translator.joint_commands_to_motor_positions(
                        joint_angles_rad=joint_commands_rad,
                        gripper_position=gripper_target
                    )
                    profiler.record_pipeline_step('joint_to_motor', time.perf_counter() - t0)
                else:
                    motor_positions = translator.joint_commands_to_motor_positions(
                        joint_angles_rad=joint_commands_rad,
                        gripper_position=gripper_target
                    )
                
                # Profile command send if profiling
                if profiler:
                    profiler.before_send()
                    robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
                    profiler.after_send()
                else:
                    robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
                
                if profiler:
                    profiler.after_loop()
                
                control_rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            control_loop_active = False
            time.sleep(0.3)
            
            # Stop profiling and print final stats if enabled
            if profiler:
                profiler.stop()
            
            # Execute shutdown sequence (homing motors) before stopping SpaceMouse
            # This is safe because motor commands don't interfere with SpaceMouse
            try:
                shutdown_sequence(robot_driver, velocity_limit=robot_config.velocity_limit)
            except Exception as e:
                print(f"Error during shutdown sequence: {e}")
                import traceback
                traceback.print_exc()
            
            # Stop SpaceMouse immediately after homing (before motor reboot)
            # This is safe because SpaceMouse doesn't interfere with motor operations
            spacemouse.stop()
    
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt detected during initialization...")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Reboot motors after shutdown (shutdown sequence already executed in inner finally)
        reboot_motors(robot_driver)
        robot_driver.disconnect()


if __name__ == "__main__":
    main()
