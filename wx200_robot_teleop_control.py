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
from spacemouse.spacemouse_driver import SpaceMouseDriver


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
        self.enable_recording = enable_recording
        self.output_path = output_path
        self.spacemouse = None
        self.trajectory = []
        self.recording_start_time = None
    
    def on_ready(self):
        """Setup SpaceMouse and print ready message."""
        print("\n" + "="*60)
        print("✓ Robot is now at home position")
        print("Ready for SpaceMouse control!")
        if self.enable_recording:
            print("RECORDING: Trajectory will be saved on exit")
        print("Press Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        self.spacemouse = SpaceMouseDriver(
            velocity_scale=robot_config.velocity_scale,
            angular_velocity_scale=robot_config.angular_velocity_scale
        )
        self.spacemouse.start()
    
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
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """Record trajectory data if recording is enabled."""
        if self.enable_recording:
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
        """Override to save trajectory and stop SpaceMouse on exit."""
        try:
            super().run_control_loop()
        finally:
            # Save trajectory if recording was enabled
            if self.enable_recording and self.trajectory:
                save_trajectory(self.trajectory, self.output_path)
            
            if self.spacemouse:
                self.spacemouse.stop()
    
    def shutdown(self):
        """Override to stop SpaceMouse before shutdown."""
        if self.spacemouse:
            self.spacemouse.stop()
        super().shutdown()


