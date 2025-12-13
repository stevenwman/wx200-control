"""
WX200 real robot control with SpaceMouse.

This version controls the actual hardware robot instead of simulation.
Uses the same clean architecture as the sim version, but replaces
simulation actuation with real Dynamixel motor commands.

Flow: SpaceMouse → Pose Control → IK → Joint-to-Motor Translation → Robot Driver
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

_HERE = Path(__file__).parent
_XML = _HERE / "wx200" / "scene.xml"

# SpaceMouse scaling
VELOCITY_SCALE = 0.5  # m/s per unit SpaceMouse input
ANGULAR_VELOCITY_SCALE = 0.5  # rad/s per unit SpaceMouse rotation input

# Gripper positions (in meters, same as sim)
GRIPPER_OPEN_POS = -0.026
GRIPPER_CLOSED_POS = 0.0

# Robot control parameters
VELOCITY_LIMIT = 30  # Speed limit for movements (0=Max, 30=Slow/Safe)
CONTROL_FREQUENCY = 50.0  # Control loop frequency (Hz) - lower than sim for safety


def get_sim_home_pose(model):
    """
    Get the home pose from sim keyframe.
    This is what we want the real robot to match.
    
    Returns:
        tuple: (qpos, position, orientation_quat_wxyz)
    """
    # Create temporary data to read keyframe
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    
    # Get joint positions from keyframe
    qpos = data.qpos.copy()
    
    # Get end-effector pose
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    position = data.site(site_id).xpos.copy()
    current_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    current_site_rot = R.from_matrix(current_site_xmat)
    current_site_quat = current_site_rot.as_quat()  # [x, y, z, w]
    orientation_quat_wxyz = np.array([
        current_site_quat[3], 
        current_site_quat[0], 
        current_site_quat[1], 
        current_site_quat[2]
    ])  # [w, x, y, z]
    
    return qpos, position, orientation_quat_wxyz


def main():
    # Load model for IK (we still need MuJoCo model for IK solving)
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    
    # Create Mink configuration
    configuration = mink.Configuration(model)
    
    # Get sim home pose
    home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(model)
    print(f"Sim home pose - qpos: {home_qpos}")
    print(f"Sim home pose - EE position: {home_position}")
    
    # Initialize robot driver
    robot_driver = RobotDriver()
    
    try:
        # Connect to robot
        robot_driver.connect()
        
        # Initialize joint-to-motor translator
        # NOTE: You'll need to calibrate these offsets based on your actual robot
        # joint1_motor2_offset and joint1_motor3_offset should be set so that when
        # joint 1 is at 0 radians, motors 2 and 3 are at their home positions
        translator = JointToMotorTranslator(
            joint1_motor2_offset=0,  # TODO: Calibrate this
            joint1_motor3_offset=0   # TODO: Calibrate this
        )
        
        # Convert sim home qpos to motor encoder positions
        # home_qpos is [q0, q1, q2, q3, q4, gripper] = [0, 0, 0, 0, 0, -0.01]
        home_joint_angles = home_qpos[:5]  # First 5 joints
        home_gripper_pos = home_qpos[5] if len(home_qpos) > 5 else -0.01
        
        home_motor_positions = translator.joint_commands_to_motor_positions(
            joint_angles_rad=home_joint_angles,
            gripper_position=home_gripper_pos
        )
        
        # Store home encoders for reference
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
        print("\nMoving robot to home position (sim keyframe)...")
        print("This may take a few seconds...")
        
        # Move robot to home position
        robot_driver.move_to_home(home_motor_positions, velocity_limit=VELOCITY_LIMIT)
        
        print("\n" + "="*60)
        print("Robot is now at home position (sim keyframe)")
        print("Ready for SpaceMouse control!")
        print("Press Ctrl+C to stop and disable torque")
        print("="*60 + "\n")
        
        # Initialize SpaceMouse driver
        spacemouse = SpaceMouseDriver(
            velocity_scale=VELOCITY_SCALE,
            angular_velocity_scale=ANGULAR_VELOCITY_SCALE
        )
        spacemouse.start()
        
        # Initialize robot controller with home pose
        robot_controller = RobotController(
            model=model,
            initial_position=home_position,
            initial_orientation_quat_wxyz=home_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture task target
        # Set configuration to home pose first
        configuration.update(home_qpos)
        robot_controller.initialize_posture_target(configuration)
        
        # Control loop
        rate = RateLimiter(frequency=CONTROL_FREQUENCY, warn=False)
        
        try:
            while True:
                dt = rate.dt
                
                # Update SpaceMouse input
                spacemouse.update()
                
                # Get velocity commands from SpaceMouse
                velocity_world = spacemouse.get_velocity_command()
                angular_velocity_world = spacemouse.get_angular_velocity_command()
                
                # Update robot controller with velocity commands
                robot_controller.update_from_velocity_command(
                    velocity_world=velocity_world,
                    angular_velocity_world=angular_velocity_world,
                    dt=dt,
                    configuration=configuration
                )
                
                # Get joint commands from controller (in radians)
                joint_commands_rad = robot_controller.get_joint_commands(configuration, num_joints=5)
                
                # Convert joint commands to motor positions
                gripper_target = spacemouse.get_gripper_target_position(
                    open_position=GRIPPER_OPEN_POS,
                    closed_position=GRIPPER_CLOSED_POS
                )
                
                motor_positions = translator.joint_commands_to_motor_positions(
                    joint_angles_rad=joint_commands_rad,
                    gripper_position=gripper_target
                )
                
                # Send motor commands to robot
                robot_driver.send_motor_positions(motor_positions, velocity_limit=VELOCITY_LIMIT)
                
                rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping control loop...")
        
        finally:
            # Cleanup
            spacemouse.stop()
            print("SpaceMouse stopped")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always disconnect and disable torque
        robot_driver.disconnect()
        print("Robot driver disconnected")


if __name__ == "__main__":
    main()
