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
# Scene XML is now local to compact_gym (self-contained)
_XML = Path(__file__).parent / "wx200" / "scene.xml"


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

        # Encoder polling state
        self.latest_encoder_values = None
        self.latest_joint_angles_from_encoders = None
        self.latest_ee_pose_from_encoders = None  # (position, quat_wxyz)
        self.last_poll_timestamp = None

        # Performance tracking for encoder polling
        self.encoder_poll_times = []
        self.encoder_poll_intervals = []
        self.encoder_poll_count = 0
        self._skipped_reads_count = 0

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

    def poll_encoders(self, outer_loop_start_time=None):
        """
        Poll encoder values from robot hardware using fast bulk read.

        Uses opportunistic read strategy: skip if insufficient time remaining
        in outer loop period to avoid blocking.

        Args:
            outer_loop_start_time: Optional timestamp of outer loop start for opportunistic reads
        """
        poll_start = time.perf_counter()

        # Opportunistic read strategy
        if outer_loop_start_time is not None:
            outer_loop_period = 1.0 / robot_config.control_frequency
            time_elapsed = poll_start - outer_loop_start_time
            time_remaining = outer_loop_period - time_elapsed

            # Safe margin: need at least 30ms to attempt read
            if time_remaining < 0.030:  # 30ms
                self._skipped_reads_count += 1
                # Only warn if skipping becomes frequent
                if self._skipped_reads_count % 100 == 0 and robot_config.warning_only_mode:
                    print(f"⚠️  Skipped {self._skipped_reads_count} encoder reads (insufficient time)")
                return

        # Track interval since last poll
        if self.last_poll_timestamp is not None:
            interval = poll_start - self.last_poll_timestamp
            if interval < 1.0:  # Only track reasonable intervals
                self.encoder_poll_intervals.append(interval)

        # Read encoders using bulk read
        encoder_values = self.robot_driver.read_all_encoders(
            max_retries=1, retry_delay=0.01, use_bulk_read=True
        )

        # Update encoder state
        if encoder_values:
            self.latest_encoder_values = encoder_values.copy()

            try:
                from .robot_kinematics import encoders_to_joint_angles
                robot_joint_angles = encoders_to_joint_angles(encoder_values, self.translator)
                self.latest_joint_angles_from_encoders = robot_joint_angles.copy()

                # Compute EE pose from encoders using temporary MuJoCo computation
                temp_data = mujoco.MjData(self.model)
                temp_data.qpos[:5] = robot_joint_angles[:5]
                if len(temp_data.qpos) > 5:
                    temp_data.qpos[5] = robot_joint_angles[5]
                mujoco.mj_forward(self.model, temp_data)
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                actual_position = temp_data.site(site_id).xpos.copy()
                actual_xmat = temp_data.site(site_id).xmat.reshape(3, 3)
                actual_quat = R.from_matrix(actual_xmat).as_quat()
                actual_orientation_quat_wxyz = np.array([actual_quat[3], actual_quat[0], actual_quat[1], actual_quat[2]])
                self.latest_ee_pose_from_encoders = (actual_position.copy(), actual_orientation_quat_wxyz.copy())

            except Exception as e:
                if robot_config.warning_only_mode:
                    # Only print warnings occasionally to avoid spam
                    if not hasattr(self, '_last_encoder_error_time'):
                        self._last_encoder_error_time = 0
                    now = time.perf_counter()
                    if now - self._last_encoder_error_time > 5.0:
                        print(f"⚠️  Warning: Failed to sync encoders to MuJoCo: {e}")
                        self._last_encoder_error_time = now

        # Track performance
        poll_duration = time.perf_counter() - poll_start
        self.encoder_poll_times.append(poll_duration)
        self.encoder_poll_count += 1
        self.last_poll_timestamp = poll_start

        # Print encoder poll stats periodically
        if (robot_config.encoder_poll_stats_interval > 0 and
            not robot_config.warning_only_mode and
            self.encoder_poll_count % robot_config.encoder_poll_stats_interval == 0):
            if len(self.encoder_poll_times) >= 50:
                avg_poll_time = np.mean(self.encoder_poll_times[-50:])
                if len(self.encoder_poll_intervals) >= 50:
                    avg_interval = np.mean(self.encoder_poll_intervals[-50:])
                    avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                    expected_freq = robot_config.control_frequency

                    print(f"[ENCODER POLL #{self.encoder_poll_count}] "
                          f"avg_read={avg_poll_time*1000:.1f}ms, "
                          f"avg_freq={avg_freq:.1f}Hz (target={expected_freq:.1f}Hz)")

        # Warn on excessive encoder read times
        if robot_config.warning_only_mode and len(self.encoder_poll_times) >= 50:
            recent_times = self.encoder_poll_times[-50:]
            avg_time = np.mean(recent_times)
            max_time = np.max(recent_times)
            # Warn if average > 15ms or max > 20ms (indicates potential issues)
            if avg_time > 0.015 or max_time > 0.020:
                if not hasattr(self, '_last_encoder_warning_time'):
                    self._last_encoder_warning_time = 0
                now = time.perf_counter()
                if now - self._last_encoder_warning_time > 5.0:
                    print(f"⚠️  ENCODER WARNING: Slow reads detected (avg={avg_time*1000:.1f}ms, max={max_time*1000:.1f}ms)")
                    self._last_encoder_warning_time = now

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

    def get_encoder_state(self):
        """
        Get the latest encoder state.

        Returns:
            dict with keys:
                - encoder_values: Raw encoder positions (dict {motor_id: position})
                - joint_angles: Joint angles from encoders (6D array)
                - ee_pose: End effector pose from encoders (position, quat_wxyz)
        """
        return {
            'encoder_values': self.latest_encoder_values.copy() if self.latest_encoder_values is not None else None,
            'joint_angles': self.latest_joint_angles_from_encoders.copy() if self.latest_joint_angles_from_encoders is not None else None,
            'ee_pose': (self.latest_ee_pose_from_encoders[0].copy(), self.latest_ee_pose_from_encoders[1].copy()) if self.latest_ee_pose_from_encoders is not None else None,
        }

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
                time.sleep(robot_config.move_delay)

                # 2. Base Home
                base_pose = list(robot_config.base_home_pose)
                if len(base_pose) > 6: base_pose[6] = robot_config.gripper_encoder_max
                self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, base_pose) if pos != -1})
                time.sleep(robot_config.move_delay)
                
                # 3. Folded Home
                folded_pose = list(robot_config.folded_home_pose)
                if len(folded_pose) > 6: folded_pose[6] = robot_config.gripper_encoder_max
                self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, folded_pose) if pos != -1})
                time.sleep(robot_config.move_delay)

                # 4. Extra delay before disabling torque (critical for preventing slam!)
                print("Waiting for movement to complete before disabling torque...")
                time.sleep(robot_config.move_delay)

                # 5. Disable Torque
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
