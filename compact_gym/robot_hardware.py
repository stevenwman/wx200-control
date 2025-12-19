"""
Hardware interface for the Gym environment.

Manages robot lifecycle (init, shutdown) and execution of control commands,
but leaves the "looping" logic to the Gym environment's step() method.
"""
import time
import numpy as np
import mujoco
import mink
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from .robot_config import robot_config
from .robot_driver import RobotDriver
from .robot_kinematics import (
    RobotController, JointToMotorTranslator, sync_robot_to_mujoco
)

# Use original scene.xml location (it's in the repo root/wx200 usually, but here pointing to parent of original code)
# Adjusting path to point to the correct scene.xml location.
# Based on `compact_code/robot_control/robot_control_base.py`:
# _XML = Path(__file__).parent.parent / "wx200" / "scene.xml"
# compact_gym/ is at same level as compact_code/, so ../wx200 should work if "wx200" is in root.
# Wait, `compact_code` is in `openarm_control/`. `wx200` likely in `openarm_control/compact_code/wx200`?
# In `robot_control_base.py` (inside `compact_code/robot_control`), `parent.parent` is `compact_code`.
# config said `_XML = Path(__file__).parent.parent / "wx200" / "scene.xml"`
# So `compact_code/wx200/scene.xml`.
# My new file is `compact_gym/robot_hardware.py`.
# So I need `../compact_code/wx200/scene.xml`
_XML = Path(__file__).parent.parent / "compact_code" / "wx200" / "scene.xml"


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


class RobotHardware:
    """
    Manages the robot hardware connection and state for the Gym environment.
    Replaces RobotControlBase but without the internal control loop.
    """
    
    def __init__(self):
        self.model = None
        self.data = None
        self.configuration = None
        self.robot_driver = None
        self.translator = None
        self.robot_controller = None
        self.gripper_current_position = None
        self.initialized = False

    def initialize(self):
        """Initialize MuJoCo, RobotDriver, and RobotController."""
        if self.initialized:
            return
            
        print("\n[RobotHardware] Initializing...")
        
        # Load MuJoCo model
        if not _XML.exists():
            raise FileNotFoundError(f"Scene XML not found at {_XML}")
            
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        
        # Get sim home pose
        home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(self.model)
        
        # Connect to robot
        self.robot_driver = RobotDriver()
        self.robot_driver.connect()
        
        # Create translator
        self.translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        # Execute startup sequence (inline here to avoid circular/extra imports)
        actual_position, actual_orientation_quat_wxyz = self._run_startup_sequence(home_qpos)
        
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
        
        # Initialize gripper position tracking
        self.gripper_current_position = robot_config.gripper_open_pos
        
        self.initialized = True
        print("[RobotHardware] Initialization Complete.")
        
        return actual_position, actual_orientation_quat_wxyz

    def _run_startup_sequence(self, home_qpos):
        """Run the standard startup sequence (Reasonable Home -> Home -> Sync)."""
        # 0. Get Home Motor Positions
        if robot_config.startup_home_positions:
            home_motor_positions = {mid: pos for mid, pos in zip(robot_config.motor_ids, robot_config.startup_home_positions)}
        else:
             home_motor_positions = self.translator.joint_commands_to_motor_positions(
                joint_angles_rad=home_qpos[:5],
                gripper_position=robot_config.gripper_open_pos
            )
        self.translator.set_home_encoders(home_motor_positions)

        # 1. Reasonable Home
        print("[Startup] Moving to Reasonable Home...")
        reasonable_home_positions = {
            mid: pos for mid, pos in zip(robot_config.motor_ids, robot_config.reasonable_home_pose)
            if pos != -1
        }
        self.robot_driver.send_motor_positions(reasonable_home_positions, velocity_limit=robot_config.velocity_limit)
        time.sleep(3.0)
        
        # 2. Startup Home
        print("[Startup] Moving to Startup Home...")
        self.robot_driver.move_to_home(home_motor_positions, velocity_limit=robot_config.velocity_limit)
        
        # 3. Sync
        print("[Startup] Syncing...")
        time.sleep(0.1)
        robot_encoders = self.robot_driver.read_all_encoders(max_retries=5)
        _, actual_pos, actual_quat = sync_robot_to_mujoco(
            robot_encoders, self.translator, self.model, self.data, self.configuration
        )
        return actual_pos, actual_quat

    def execute_command(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """
        Execute a single control step.
        
        Args:
            velocity_world: [vx, vy, vz] (m/s)
            angular_velocity_world: [wx, wy, wz] (rad/s)
            gripper_target: gripper position (meters)
            dt: Time step (seconds)
        """
        if not self.initialized:
            raise RuntimeError("RobotHardware not initialized (call initialize() first)")

        # Update controller & Solve IK
        self.robot_controller.update_from_velocity_command(
            velocity_world=velocity_world,
            angular_velocity_world=angular_velocity_world,
            dt=dt,
            configuration=self.configuration
        )
        
        # Get Motor Commands
        joint_commands_rad = self.configuration.q[:5].copy()
        motor_positions = self.translator.joint_commands_to_motor_positions(
            joint_angles_rad=joint_commands_rad,
            gripper_position=gripper_target
        )
        
        # Send to Robot
        self.robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
        
        # Update local gripper state
        self.gripper_current_position = gripper_target

    def shutdown(self):
        """Safe shutdown sequence."""
        print("\n[RobotHardware] Shutting down...")
        if self.robot_driver and self.robot_driver.connected:
            try:
                # Use a fresh connection for shutdown sequence like original code?
                # For simplicity in compact_gym, we'll try to use existing driver first, 
                # but valid shutdown often requires re-opening port to ensure clean state.
                # We will implement a simplified version of shutdown_sequence here.
                
                # 1. Reasonable Home
                reasonable_pose = list(robot_config.reasonable_home_pose)
                if len(reasonable_pose) > 6: reasonable_pose[6] = robot_config.gripper_encoder_max # Open gripper
                
                self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, reasonable_pose) if pos != -1})
                time.sleep(1.0)
                
                # 2. Base Home
                base_pose = list(robot_config.base_home_pose)
                if len(base_pose) > 6: base_pose[6] = robot_config.gripper_encoder_max
                self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, base_pose) if pos != -1})
                time.sleep(1.0)
                
                # 3. Folded Home
                folded_pose = list(robot_config.folded_home_pose)
                if len(folded_pose) > 6: folded_pose[6] = robot_config.gripper_encoder_max
                self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, folded_pose) if pos != -1})
                time.sleep(1.0)
                
                # 4. Disable Torque
                self.robot_driver.disable_torque_all()
                self.robot_driver.disconnect()
                
            except Exception as e:
                print(f"Error during shutdown: {e}")
        
        self.initialized = False

    def _failsafe_move(self, motor_positions):
        """Helper to move motors with broad try/catch."""
        try:
            self.robot_driver.send_motor_positions(motor_positions, velocity_limit=30)
        except Exception as e:
            print(f"Move failed: {e}")
