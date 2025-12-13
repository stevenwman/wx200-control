"""
SpaceMouse driver interface.

Provides a clean interface for SpaceMouse input that can be easily swapped
with other input sources (neural network, joystick, etc.).

Maintains:
- Current 6-vector velocity command [vx, vy, vz, wx, wy, wz] in world frame
- Gripper state (open/closed)
- Button state handling
"""
import multiprocessing as mp
import queue
import numpy as np
from spacemouse_reader import spacemouse_process


class SpaceMouseDriver:
    """
    Driver interface for SpaceMouse input.
    
    Provides a clean API to get current velocity commands and gripper state.
    Can be easily replaced with other input sources (NN, joystick, etc.).
    """
    
    def __init__(self, velocity_scale=0.5, angular_velocity_scale=0.5):
        """
        Initialize SpaceMouse driver.
        
        Args:
            velocity_scale: Scale factor for translation (m/s per unit input)
            angular_velocity_scale: Scale factor for rotation (rad/s per unit input)
        """
        self.velocity_scale = velocity_scale
        self.angular_velocity_scale = angular_velocity_scale
        
        # Current state
        self.velocity_world = np.zeros(3)  # [vx, vy, vz] in world frame (m/s)
        self.angular_velocity_world = np.zeros(3)  # [wx, wy, wz] in world frame (rad/s)
        
        # Internal: multiprocessing setup
        self.data_queue = mp.Queue(maxsize=5)
        self.spacemouse_proc = None
        self.left_button_prev = False
        self.right_button_prev = False
        
        # Deadzone thresholds
        self.velocity_deadzone = 0.001
        self.angular_velocity_deadzone = 0.01
    
    def start(self):
        """Start the SpaceMouse reader process."""
        self.spacemouse_proc = mp.Process(
            target=spacemouse_process,
            args=(self.data_queue, self.velocity_scale, self.angular_velocity_scale)
        )
        self.spacemouse_proc.start()
        print("SpaceMouse driver started")
        print("  Translation: Pushing forward = world +X, right = world +Y, up = world +Z")
        print("  Rotation: Roll/Pitch/Yaw = world frame angular velocity [wx, wy, wz]")
        print("  Gripper: Hold left button = open incrementally, Hold right button = close incrementally")
    
    def stop(self):
        """Stop the SpaceMouse reader process."""
        if self.spacemouse_proc is not None:
            self.spacemouse_proc.terminate()
            self.spacemouse_proc.join()
            self.spacemouse_proc = None
        print("SpaceMouse driver stopped")
    
    def update(self):
        """
        Update current state from SpaceMouse input.
        Should be called every control loop iteration.
        
        Returns:
            bool: True if new data was processed, False otherwise
        """
        # Reset velocities (prevents drift when no input)
        self.velocity_world = np.zeros(3)
        self.angular_velocity_world = np.zeros(3)
        
        data_updated = False
        
        # Process all available commands in queue
        while not self.data_queue.empty():
            try:
                twist_command = self.data_queue.get_nowait()
                data_updated = True
                
                # Extract translation as velocity in world frame
                vel_raw = twist_command['translation']
                vel_magnitude = np.linalg.norm(vel_raw)
                if vel_magnitude > self.velocity_deadzone:
                    self.velocity_world = vel_raw
                
                # Extract rotation as angular velocity in world frame
                # Based on spacemouse_reader.py: rotation_twist = [roll, yaw, pitch]
                # Map to world frame: [wx, wy, wz] = [pitch, -roll, yaw] (x and y flipped)
                rotation_twist = twist_command.get('rotation', np.zeros(3))
                omega_raw = np.array([
                    rotation_twist[2],   # pitch -> wx
                    -rotation_twist[0],  # -roll -> wy
                    rotation_twist[1]     # yaw -> wz
                ])
                
                # Negate to match CAD convention (opposite of SolidWorks behavior)
                omega_raw = -omega_raw
                
                omega_magnitude = np.linalg.norm(omega_raw)
                if omega_magnitude > self.angular_velocity_deadzone:
                    self.angular_velocity_world = omega_raw
                
                # Handle gripper buttons: left = open incrementally, right = close incrementally
                button_state = twist_command.get('button', [])
                
                # Left button (index 0) = open (when held)
                left_button = len(button_state) > 0 and button_state[0] == 1
                # Right button (index 1) = close (when held)
                right_button = len(button_state) > 1 and button_state[1] == 1
                
                self.left_button_prev = left_button
                self.right_button_prev = right_button
                
            except queue.Empty:
                break
        
        return data_updated
    
    def get_velocity_command(self):
        """
        Get current linear velocity command in world frame.
        
        Returns:
            np.ndarray: [vx, vy, vz] in m/s
        """
        return self.velocity_world.copy()
    
    def get_angular_velocity_command(self):
        """
        Get current angular velocity command in world frame.
        
        Returns:
            np.ndarray: [wx, wy, wz] in rad/s
        """
        return self.angular_velocity_world.copy()
    
    def get_6d_velocity_command(self):
        """
        Get current 6D velocity command in world frame.
        
        Returns:
            np.ndarray: [vx, vy, vz, wx, wy, wz] in m/s and rad/s
        """
        return np.concatenate([self.velocity_world, self.angular_velocity_world])
    
    def get_gripper_button_states(self):
        """
        Get current gripper button states for incremental control.
        
        Returns:
            tuple: (left_button_pressed, right_button_pressed)
                - left_button_pressed: True if left button is currently held
                - right_button_pressed: True if right button is currently held
        """
        return (self.left_button_prev, self.right_button_prev)
