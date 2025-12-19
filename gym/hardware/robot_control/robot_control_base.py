"""
Base robot control infrastructure shared between teleop and replay.

Provides common initialization, control loop structure, and shutdown logic.
"""
import time
import numpy as np
import mujoco
import mink
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from .robot_config import robot_config
from .robot_driver import RobotDriver
from .robot_controller import RobotController
from .robot_joint_to_motor import JointToMotorTranslator
from .robot_startup import startup_sequence
from .robot_shutdown import shutdown_sequence, reboot_motors
from loop_rate_limiters import RateLimiter


_XML = Path(__file__).parent.parent / "wx200" / "scene.xml"


def get_sim_home_pose(model):
    """Get the home pose from sim keyframe."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    home_position = data.site(site_id).xpos.copy()
    home_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    home_site_rot = R.from_matrix(home_site_xmat)
    home_site_quat = home_site_rot.as_quat()
    home_orientation_quat_wxyz = np.array([
        home_site_quat[3], 
        home_site_quat[0], 
        home_site_quat[1], 
        home_site_quat[2]
    ])
    
    qpos = data.qpos.copy()
    return qpos, home_position, home_orientation_quat_wxyz


class RobotControlBase:
    """
    Base class for robot control scripts (teleop, replay, etc.).
    
    Handles common initialization, control loop structure, and shutdown.
    Subclasses implement get_action() to provide velocity commands and gripper target.
    """
    
    def __init__(self, control_frequency=None):
        """
        Initialize base robot control infrastructure.
        
        Args:
            control_frequency: Control loop frequency (Hz). If None, uses robot_config.
        """
        self.control_frequency = control_frequency or robot_config.control_frequency
        self.model = None
        self.data = None
        self.configuration = None
        self.robot_driver = None
        self.translator = None
        self.robot_controller = None
        self.control_rate = None
        self.gripper_current_position = None
        
    def initialize(self):
        """Initialize MuJoCo model, robot driver, and controller."""
        print("\nInitializing robot control...")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        
        # Get sim home pose
        home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(self.model)
        print(f"Sim home pose - EE position: {home_position}")
        
        # Connect to robot
        self.robot_driver = RobotDriver()
        print("\nConnecting to robot...")
        self.robot_driver.connect()
        
        # Create translator
        self.translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        # Execute startup sequence
        robot_joint_angles, actual_position, actual_orientation_quat_wxyz, home_motor_positions = startup_sequence(
            self.robot_driver, self.translator, self.model, self.data, self.configuration, home_qpos=home_qpos
        )
        
        print(f"✓ Synced to actual robot position: {actual_position}")
        
        # Initialize robot controller
        self.robot_controller = RobotController(
            model=self.model,
            initial_position=actual_position,
            initial_orientation_quat_wxyz=actual_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture target and pose
        self.robot_controller.initialize_posture_target(self.configuration)
        self.robot_controller.reset_pose(actual_position, actual_orientation_quat_wxyz)
        current_target_pose = self.robot_controller.get_target_pose()
        self.robot_controller.end_effector_task.set_target(current_target_pose)
        
        # Initialize gripper position
        self.gripper_current_position = robot_joint_angles[5] if len(robot_joint_angles) > 5 else robot_config.gripper_open_pos
        
        # Create rate limiter
        self.control_rate = RateLimiter(frequency=self.control_frequency, warn=False)
        
        return actual_position, actual_orientation_quat_wxyz
    
    def get_action(self, dt):
        """
        Get action from input source (to be implemented by subclasses).
        
        Args:
            dt: Time step (seconds)
        
        Returns:
            tuple: (velocity_world, angular_velocity_world, gripper_target)
                - velocity_world: [vx, vy, vz] in m/s
                - angular_velocity_world: [wx, wy, wz] in rad/s
                - gripper_target: Gripper target position (meters)
        """
        raise NotImplementedError("Subclasses must implement get_action()")
    
    def _execute_control_step(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """
        Execute a single control step: update controller, solve IK, send to robot.
        
        This is the core control loop logic shared by all control modes.
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in m/s
            angular_velocity_world: [wx, wy, wz] angular velocity in rad/s
            gripper_target: Gripper target position (meters)
            dt: Time step (seconds)
        """
        # Update robot controller with velocity commands
        self.robot_controller.update_from_velocity_command(
            velocity_world=velocity_world,
            angular_velocity_world=angular_velocity_world,
            dt=dt,
            configuration=self.configuration
        )
        
        # Get joint commands from IK solution
        joint_commands_rad = self.configuration.q[:5].copy()
        
        # Convert joint commands to motor positions
        motor_positions = self.translator.joint_commands_to_motor_positions(
            joint_angles_rad=joint_commands_rad,
            gripper_position=gripper_target
        )
        
        # Send commands to robot
        self.robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """
        Hook called on each control loop iteration before executing control step.
        
        Subclasses can override to add custom behavior (e.g., recording, progress reporting).
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in m/s
            angular_velocity_world: [wx, wy, wz] angular velocity in rad/s
            gripper_target: Gripper target position (meters)
            dt: Time step (seconds)
        """
        pass
    
    def run_control_loop(self):
        """
        Main control loop - calls get_action() and executes robot commands.
        
        Subclasses can override on_control_loop_iteration() for custom behavior.
        """
        control_loop_active = True
        
        try:
            while control_loop_active:
                dt = self.control_rate.dt
                
                # Get action from input source (SpaceMouse, trajectory, NN, etc.)
                velocity_world, angular_velocity_world, gripper_target = self.get_action(dt)
                
                # Check for end condition (e.g., replay finished)
                if velocity_world is None:
                    break
                
                # Update gripper current position
                self.gripper_current_position = gripper_target
                
                # Hook for subclasses to add custom behavior
                self.on_control_loop_iteration(velocity_world, angular_velocity_world, gripper_target, dt)
                
                # Execute control step (update controller, solve IK, send to robot)
                self._execute_control_step(velocity_world, angular_velocity_world, gripper_target, dt)
                
                self.control_rate.sleep()
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping control loop...")
            control_loop_active = False
    
    def shutdown(self):
        """Execute shutdown sequence and cleanup."""
        try:
            shutdown_sequence(self.robot_driver, velocity_limit=robot_config.velocity_limit)
        except Exception as e:
            print(f"Error during shutdown sequence: {e}")
        
        try:
            reboot_motors(self.robot_driver)
        except Exception as e:
            print(f"Error rebooting motors: {e}")
        
        try:
            self.robot_driver.disconnect()
        except Exception as e:
            print(f"Error disconnecting: {e}")
    
    def run(self):
        """
        Main entry point - initialize, run control loop, and shutdown.
        
        Subclasses can override this for custom behavior.
        """
        try:
            self.initialize()
            self.on_ready()  # Hook for subclasses to do setup after initialization
            self.run_control_loop()
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Executing emergency shutdown...")
            try:
                shutdown_sequence(self.robot_driver, velocity_limit=robot_config.velocity_limit)
            except:
                pass
            try:
                self.robot_driver.disconnect()
            except:
                pass
            raise
        finally:
            self.shutdown()
    
    def on_ready(self):
        """
        Hook called after initialization, before control loop starts.
        
        Subclasses can override this to print messages, set up recording, etc.
        """
        pass
