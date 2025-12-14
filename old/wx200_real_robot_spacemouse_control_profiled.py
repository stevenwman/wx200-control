"""
WX200 real robot control with SpaceMouse - PROFILED VERSION.

This is a profiled version of the main control script that measures actual
command send frequency and identifies bottlenecks.

Usage:
    python wx200_real_robot_spacemouse_control_profiled.py
"""
from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
import time
from spacemouse_driver import SpaceMouseDriver
from robot_controller import RobotController
from robot_joint_to_motor import JointToMotorTranslator
from robot_driver import RobotDriver
from robot_driver_profiling import create_profiled_driver
from robot_shutdown import shutdown_sequence, reboot_motors
from robot_config import robot_config
from control_frequency_profiler import ControlFrequencyProfiler

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
    print("WX200 Real Robot Control with SpaceMouse (PROFILED)")
    print("="*60)
    print("Features:")
    print("- SpaceMouse control")
    print("- Safe startup and shutdown sequences")
    print("- Control frequency profiling (measures actual send rate)")
    print("="*60)
    
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)
    configuration = mink.Configuration(model)
    
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"\nSim home pose - EE position: {home_position}")
    
    robot_driver = RobotDriver()
    
    # Enable detailed send_motor_positions profiling (toggleable)
    ENABLE_DRIVER_PROFILING = True  # Set to False to disable
    if ENABLE_DRIVER_PROFILING:
        robot_driver = create_profiled_driver(robot_driver, stats_interval=100)
    
    # Initialize profiler
    profiler = ControlFrequencyProfiler(stats_interval=100)
    
    try:
        print("\nConnecting to robot...")
        robot_driver.connect()
        
        translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        home_joint_angles = home_qpos[:5]
        home_gripper_pos = robot_config.gripper_open_pos
        
        home_motor_positions = translator.joint_commands_to_motor_positions(
            joint_angles_rad=home_joint_angles,
            gripper_position=home_gripper_pos
        )
        
        translator.set_home_encoders([
            home_motor_positions.get(1, 2048),
            home_motor_positions.get(2, 2048),
            home_motor_positions.get(3, 2048),
            home_motor_positions.get(4, 2048),
            home_motor_positions.get(5, 2048),
            home_motor_positions.get(6, 2048),
            home_motor_positions.get(7, 2048),
        ])
        
        print(f"Home motor positions: {home_motor_positions}")
        print("\n" + "="*60)
        print("STARTUP SEQUENCE: Moving robot to home position...")
        print("This may take a few seconds...")
        print("="*60)
        
        robot_driver.move_to_home(home_motor_positions, velocity_limit=robot_config.velocity_limit)
        
        print("\n" + "="*60)
        print("✓ Robot is now at home position (sim keyframe)")
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
            initial_position=home_position,
            initial_orientation_quat_wxyz=home_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        configuration.update(home_qpos)
        robot_controller.initialize_posture_target(configuration)
        
        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)
        
        control_rate = RateLimiter(frequency=robot_config.control_frequency, warn=False)
        control_loop_active = True
        gripper_current_position = robot_config.gripper_open_pos
        
        # Start profiling
        profiler.start()
        
        try:
            while control_loop_active:
                profiler.before_loop()
                
                dt = control_rate.dt
                
                # Profile SpaceMouse update
                t0 = time.perf_counter()
                spacemouse.update()
                profiler.record_pipeline_step('spacemouse_update', time.perf_counter() - t0)
                
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                # Profile IK solve
                t0 = time.perf_counter()
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                profiler.record_pipeline_step('ik_solve', time.perf_counter() - t0)
                
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
                
                # Profile joint-to-motor conversion
                t0 = time.perf_counter()
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                profiler.record_pipeline_step('joint_to_motor', time.perf_counter() - t0)
                
                # Profile command send (this is the critical measurement)
                # Break down send_motor_positions to find bottleneck
                profiler.before_send()
                t_send_start = time.perf_counter()
                
                # Check if velocity limit setting is the bottleneck
                t_vel_check = time.perf_counter()
                # This is inside send_motor_positions, but we can't easily profile it
                # without modifying robot_driver.py
                
                robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
                
                t_send_end = time.perf_counter()
                profiler.record_pipeline_step('send_commands', t_send_end - t_send_start)
                profiler.after_send()
                
                profiler.after_loop()
                control_rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            control_loop_active = False
            time.sleep(0.3)
            
            # Stop profiling and print final stats
            profiler.stop()
            
            # Execute shutdown sequence (homing motors) before stopping SpaceMouse
            try:
                shutdown_sequence(robot_driver, velocity_limit=robot_config.velocity_limit)
            except Exception as e:
                print(f"Error during shutdown sequence: {e}")
                import traceback
                traceback.print_exc()
            
            # Stop SpaceMouse immediately after homing (before motor reboot)
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
