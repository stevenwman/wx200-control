"""
Trajectory replay module for WX200 robot.

Provides ReplayControl class for replaying recorded trajectories on the real robot.
Can be imported as a module or used via the unified wx200_robot_control.py script.
"""
from pathlib import Path
import numpy as np
import time

from robot_control.robot_config import robot_config
from robot_control.robot_control_base import RobotControlBase


# Export classes and functions for use as a module
__all__ = ['ReplayControl', 'load_trajectory']


def load_trajectory(trajectory_path):
    """Load trajectory data from NPZ file."""
    data = np.load(trajectory_path, allow_pickle=True)
    
    trajectory = {
        'timestamps': data['timestamps'],
        'states': data['states'],
        'actions': data['actions'],
        'metadata': data['metadata'].item() if 'metadata' in data else {}
    }
    
    # Load EE pose debug data if present (for inspection, not used during replay)
    if 'ee_poses_debug' in data:
        trajectory['ee_poses_debug'] = data['ee_poses_debug']
    
    print(f"\n✓ Loaded trajectory from: {trajectory_path}")
    print(f"  - {len(trajectory['timestamps'])} samples")
    print(f"  - Duration: {trajectory['timestamps'][-1]:.2f} seconds")
    if 'control_frequency' in trajectory['metadata']:
        print(f"  - Recorded frequency: {trajectory['metadata']['control_frequency']:.1f} Hz")
    if 'ee_poses_debug' in trajectory:
        print(f"  - Includes EE pose trajectory (debug data, not used during replay)")
    
    return trajectory


class ReplayControl(RobotControlBase):
    """Replay control using recorded trajectory."""
    
    def __init__(self, trajectory, start_idx=0, end_idx=None):
        recorded_freq = trajectory['metadata'].get('control_frequency', robot_config.control_frequency)
        super().__init__(control_frequency=recorded_freq)
        
        self.trajectory = trajectory
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(trajectory['timestamps'])
        self.end_idx = min(self.end_idx, len(trajectory['timestamps']))
        self.current_idx = start_idx
        self.replay_start_time = None
        
        if start_idx >= self.end_idx:
            raise ValueError(f"Invalid range: start_index={start_idx}, end_index={self.end_idx}")
    
    def on_ready(self):
        """Print ready message and replay info."""
        print("\n" + "="*60)
        print("✓ Robot is now at home position")
        print("Ready to replay trajectory!")
        print("Press Ctrl+C to stop and execute shutdown sequence")
        print("="*60 + "\n")
        
        print(f"\nReplay range: indices {self.start_idx} to {self.end_idx-1} ({self.end_idx - self.start_idx} samples)")
        print(f"Estimated duration: {self.trajectory['timestamps'][self.end_idx-1] - self.trajectory['timestamps'][self.start_idx]:.2f} seconds")
        print(f"\nStarting replay at {self.control_frequency:.1f} Hz...")
        print("="*60 + "\n")
        
    def get_action(self, dt):
        """Get action from recorded trajectory."""
        if self.current_idx >= self.end_idx:
            return None, None, None
        
        action = self.trajectory['actions'][self.current_idx]
        return action[:3], action[3:6], action[6]  # [vx,vy,vz], [wx,wy,wz], gripper_target
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """Track replay progress and report status."""
        # Move to next sample
        self.current_idx += 1
        
        # Print progress every second
        if self.current_idx % int(self.control_frequency) == 0:
            elapsed = time.perf_counter() - self.replay_start_time
            progress = (self.current_idx - self.start_idx) / (self.end_idx - self.start_idx) * 100
            print(f"Replay progress: {self.current_idx}/{self.end_idx} ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
        
    def run_control_loop(self):
        """Override to initialize replay timing and handle completion."""
        self.replay_start_time = time.perf_counter()
        
        try:
            super().run_control_loop()
            print(f"\n✓ Replay complete! ({self.current_idx - self.start_idx} samples)")
        except KeyboardInterrupt:
            print(f"\n\nKeyboard interrupt detected. Stopped at index {self.current_idx}/{self.end_idx}")


