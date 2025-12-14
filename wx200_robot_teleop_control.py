"""
Teleop control module for WX200 robot using SpaceMouse input.

Provides TeleopControl class for real-time robot control with optional trajectory recording.
Can be imported as a module or used via the unified wx200_robot_control.py script.
"""
import time
import numpy as np
import mujoco
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation as R

from robot_control.robot_config import robot_config
from robot_control.robot_control_base import RobotControlBase
from robot_control.robot_startup import get_home_motor_positions
from spacemouse.spacemouse_driver import SpaceMouseDriver
from utils.robot_control_gui import SimpleControlGUI


# Export classes and functions for use as a module
__all__ = ['TeleopControl', 'save_trajectory']


def save_trajectory(trajectory, output_path):
    """Save trajectory data to NPZ file."""
    if not trajectory:
        print("Warning: No trajectory data to save")
        return
    
    timestamps = np.array([t['timestamp'] for t in trajectory])
    states = np.array([t['state'] for t in trajectory])
    actions = np.array([t['action'] for t in trajectory])
    
    # EE pose trajectory (debug data, not used by policy)
    ee_poses_debug = np.array([t['ee_pose_debug'] for t in trajectory])
    
    np.savez_compressed(
        output_path,
        timestamps=timestamps,
        states=states,
        actions=actions,
        ee_poses_debug=ee_poses_debug,  # Debug data for inspection
        metadata={
            'num_samples': len(trajectory),
            'control_frequency': robot_config.control_frequency,
            'duration_seconds': timestamps[-1] if len(timestamps) > 0 else 0.0,
            'state_dim': 6,  # 5 joints + gripper
            'action_dim': 7,  # 3 linear + 3 angular velocities + gripper target
            'ee_pose_debug_dim': 7,  # 3 position + 4 quaternion (wxyz)
            'state_labels': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'gripper'],
            'action_labels': ['vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'gripper_target'],
            'ee_pose_debug_labels': ['ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz'],
            'ee_pose_debug_note': 'End-effector pose trajectory for debugging/inspection only, NOT used by policy',
            'timestamp': datetime.now().isoformat()
        }
    )
    
    print(f"\n✓ Trajectory saved to: {output_path}")
    print(f"  - {len(trajectory)} samples")
    if len(timestamps) > 0 and timestamps[-1] > 0:
        print(f"  - Duration: {timestamps[-1]:.2f} seconds")
        print(f"  - Frequency: {len(trajectory) / timestamps[-1]:.2f} Hz")
    print(f"  - Includes EE pose trajectory (debug data)")


class TeleopControl(RobotControlBase):
    """Teleop control using SpaceMouse input with optional trajectory recording."""
    
    def __init__(self, enable_recording=False, output_path=None):
        super().__init__()
        self.enable_recording = enable_recording  # Whether recording is enabled (can be toggled)
        self.is_recording = False  # Whether currently recording (starts False, toggled by GUI)
        self.output_path = output_path
        self.spacemouse = None
        self.trajectory = []
        self.recording_start_time = None
        self.control_gui = None
        self.home_motor_positions = None
        self.trajectory_counter = 0  # Counter for multiple trajectory saves
    
    def on_ready(self):
        """Setup SpaceMouse, GUI, and print ready message."""
        print("\n" + "="*60)
        print("✓ Robot is now at home position")
        print("Ready for SpaceMouse control!")
        print("\nControl Options:")
        print("  - GUI window with 3 buttons (Home, Start Recording, Stop & Save)")
        if self.enable_recording:
            print("\nRECORDING: Recording capability enabled (use GUI to start/stop)")
        print("\nPress Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        self.spacemouse = SpaceMouseDriver(
            velocity_scale=robot_config.velocity_scale,
            angular_velocity_scale=robot_config.angular_velocity_scale
        )
        self.spacemouse.start()
        
        # Start GUI control
        self.control_gui = SimpleControlGUI()
        self.control_gui.start()
        
        if self.control_gui.is_available():
            print("✓ Control GUI opened")
        else:
            print("⚠️  GUI not available, using keyboard input only")
        
        # Get home motor positions
        from robot_control.robot_control_base import get_sim_home_pose
        home_qpos, _, _ = get_sim_home_pose(self.model)
        self.home_motor_positions = get_home_motor_positions(self.translator, home_qpos=home_qpos)
    
    def get_action(self, dt):
        """Get action from SpaceMouse input."""
        self.spacemouse.update()
        return (
            self.spacemouse.get_velocity_command(),
            self.spacemouse.get_angular_velocity_command(),
            self.spacemouse.get_gripper_command(self.gripper_current_position, dt)
        )
    
    def _get_current_ee_pose(self):
        """Get current end-effector pose from MuJoCo (for debugging/inspection only)."""
        # Sync MuJoCo data with current configuration
        self.data.qpos[:5] = self.configuration.q[:5]
        if len(self.data.qpos) > 5:
            self.data.qpos[5] = self.configuration.q[5] if len(self.configuration.q) > 5 else 0.0
        
        # Update forward kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get site pose
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        ee_position = self.data.site(site_id).xpos.copy()
        ee_xmat = self.data.site(site_id).xmat.reshape(3, 3)
        ee_rot = R.from_matrix(ee_xmat)
        ee_quat = ee_rot.as_quat()
        ee_orientation_quat_wxyz = np.array([
            ee_quat[3],  # w
            ee_quat[0],  # x
            ee_quat[1],  # y
            ee_quat[2]   # z
        ])
        
        return ee_position, ee_orientation_quat_wxyz
    
    def _handle_control_input(self):
        """Handle control commands from GUI."""
        if not self.control_gui or not self.control_gui.is_available():
            return
        
        command = self.control_gui.get_command()
        if command is None:
            return
        
        # Process only valid commands
        if command not in ['h', 'r', 's']:
            return
        
        try:
            if command == 'h':
                # Move to home position
                print("\n" + "="*60, flush=True)
                print("GUI COMMAND: 'h' - Moving to home position...", flush=True)
                print("="*60, flush=True)
                
                # Send home command
                self.robot_driver.send_motor_positions(
                    self.home_motor_positions, 
                    velocity_limit=robot_config.velocity_limit
                )
                
                # Wait for movement (commands will be checked in control loop)
                print("Moving to home...", flush=True)
                start_time = time.perf_counter()
                movement_duration = 3.0
                check_interval = 0.1  # Check every 100ms
                
                while time.perf_counter() - start_time < movement_duration:
                    # Check for new commands during movement
                    if self.control_gui:
                        new_cmd = self.control_gui.get_command()
                        if new_cmd and new_cmd in ['h', 'r', 's']:
                            # Queue command for next iteration
                            if self.control_gui.is_available():
                                with self.control_gui.lock:
                                    self.control_gui.command_queue.insert(0, new_cmd)
                            print("(Movement interrupted by new command)", flush=True)
                            break
                    time.sleep(check_interval)
                
                # Sync MuJoCo to actual robot position after moving
                time.sleep(0.1)
                robot_encoders = self.robot_driver.read_all_encoders(max_retries=5, retry_delay=0.2)
                from robot_control.robot_joint_to_motor import sync_robot_to_mujoco
                robot_joint_angles, actual_position, actual_orientation_quat_wxyz = sync_robot_to_mujoco(
                    robot_encoders, self.translator, self.model, self.data, self.configuration
                )
                
                # Update controller target pose
                self.robot_controller.reset_pose(actual_position, actual_orientation_quat_wxyz)
                current_target_pose = self.robot_controller.get_target_pose()
                self.robot_controller.end_effector_task.set_target(current_target_pose)
                
                print(f"✓ Robot moved to home position: {actual_position}", flush=True)
                print("="*60 + "\n", flush=True)
            
            elif command == 'r':
                # Start recording (reset timestep)
                if not self.enable_recording:
                    print("\n⚠️  Recording not enabled. Start script with --record flag.\n", flush=True)
                    return
                
                print("\n" + "="*60, flush=True)
                print("GUI COMMAND: 'r' - Starting new recording (resetting timestep)...", flush=True)
                print("="*60, flush=True)
                self.trajectory = []  # Clear previous trajectory
                self.recording_start_time = time.perf_counter()
                self.is_recording = True  # Set recording flag
                print("✓ Recording started (timestep reset to 0)", flush=True)
                print("="*60 + "\n", flush=True)
            
            elif command == 's':
                # Stop recording and save
                if not self.enable_recording:
                    print("\n⚠️  Recording not enabled. Start script with --record flag.\n", flush=True)
                    return
                
                if not self.is_recording:
                    print("\n⚠️  Not currently recording. Press 'r' to start recording first.\n", flush=True)
                    return
                
                if not self.trajectory:
                    print("\n⚠️  No trajectory data to save (recording was empty).\n", flush=True)
                    self.is_recording = False
                    self.recording_start_time = None
                    return
                
                print("\n" + "="*60, flush=True)
                print("GUI COMMAND: 's' - Stopping recording and saving trajectory...", flush=True)
                print("="*60, flush=True)
                
                # Generate filename with counter
                if self.output_path:
                    base_path = Path(self.output_path)
                    if self.trajectory_counter == 0:
                        # First save: use original path
                        save_path = base_path
                    else:
                        # Subsequent saves: add counter suffix
                        save_path = base_path.parent / f"{base_path.stem}_{self.trajectory_counter}{base_path.suffix}"
                else:
                    # Generate default path with counter
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if self.trajectory_counter == 0:
                        save_path = data_dir / f"trajectory_{timestamp}.npz"
                    else:
                        save_path = data_dir / f"trajectory_{timestamp}_{self.trajectory_counter}.npz"
                
                save_trajectory(self.trajectory, save_path)
                self.trajectory_counter += 1
                
                # Stop recording: clear trajectory and reset flags
                self.trajectory = []
                self.recording_start_time = None
                self.is_recording = False  # IMPORTANT: Stop recording
                
                print("="*60 + "\n", flush=True)
        except Exception as e:
            print(f"\n⚠️  Error handling GUI command '{command}': {e}\n", flush=True)
            import traceback
            traceback.print_exc()
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """Handle control input and record trajectory data if recording is active."""
        # Check for control commands from GUI
        self._handle_control_input()
        
        # Record trajectory data only if actively recording
        if self.is_recording:
            if self.recording_start_time is None:
                self.recording_start_time = time.perf_counter()
            
            # State: joint positions from model (used for policy observations)
            state = np.concatenate([
                self.configuration.q[:5],  # 5 joint angles (radians)
                np.array([gripper_target])  # Gripper position (meters)
            ])
            
            # Action: velocity commands + gripper target (used for policy actions)
            action = np.concatenate([
                velocity_world,  # [vx, vy, vz] in m/s
                angular_velocity_world,  # [wx, wy, wz] in rad/s
                np.array([gripper_target])  # Gripper target position (meters)
            ])
            
            # EE pose: position + orientation (for debugging/inspection only, NOT used by policy)
            ee_position, ee_orientation_quat_wxyz = self._get_current_ee_pose()
            ee_pose_debug = np.concatenate([
                ee_position,  # [x, y, z] in meters
                ee_orientation_quat_wxyz  # [w, x, y, z] quaternion
            ])
            
            timestamp = time.perf_counter() - self.recording_start_time
            self.trajectory.append({
                'timestamp': timestamp,
                'state': state.copy(),
                'action': action.copy(),
                'ee_pose_debug': ee_pose_debug.copy()  # Debug data, not used by policy
            })
    
    def run_control_loop(self):
        """Override to save trajectory and stop SpaceMouse/keyboard on exit."""
        try:
            super().run_control_loop()
        finally:
            # Save trajectory if recording was active and not already saved
            if self.is_recording and self.trajectory:
                # Generate filename for final save
                if self.output_path:
                    base_path = Path(self.output_path)
                    if self.trajectory_counter == 0:
                        save_path = base_path
                    else:
                        save_path = base_path.parent / f"{base_path.stem}_{self.trajectory_counter}{base_path.suffix}"
                else:
                    data_dir = Path("data")
                    data_dir.mkdir(exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    if self.trajectory_counter == 0:
                        save_path = data_dir / f"trajectory_{timestamp}.npz"
                    else:
                        save_path = data_dir / f"trajectory_{timestamp}_{self.trajectory_counter}.npz"
                
                print("\n" + "="*60)
                print("Saving final trajectory on exit...")
                print("="*60)
                save_trajectory(self.trajectory, save_path)
            
            # Stop GUI first (before SpaceMouse) to allow proper cleanup
            if self.control_gui:
                self.control_gui.stop()
            
            if self.spacemouse:
                self.spacemouse.stop()
    
    def shutdown(self):
        """Override to stop SpaceMouse and GUI before shutdown."""
        # Stop GUI first to allow proper tkinter cleanup
        if self.control_gui:
            self.control_gui.stop()
        if self.spacemouse:
            self.spacemouse.stop()
        super().shutdown()


