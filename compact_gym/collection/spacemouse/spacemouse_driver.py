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
from .spacemouse_reader import spacemouse_process


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
        self.velocity_world = np.zeros(3)
        self.angular_velocity_world = np.zeros(3)
        
        # Internal: multiprocessing setup
        self.data_queue = mp.Queue(maxsize=5)
        self.spacemouse_proc = None
        self.stop_event = mp.Event()
        self.left_button_prev = False
        self.right_button_prev = False
        
        # Deadzone thresholds - prevents drift from noise
        # Import here to avoid circular dependencies
        from deployment.robot_config import robot_config
        self.velocity_deadzone = robot_config.velocity_deadzone
        self.angular_velocity_deadzone = robot_config.angular_velocity_deadzone
    
    def start(self):
        """Start the SpaceMouse reader process."""
        self.stop_event.clear()
        self.spacemouse_proc = mp.Process(
            target=spacemouse_process,
            args=(self.data_queue, self.velocity_scale, self.angular_velocity_scale, self.stop_event)
        )
        self.spacemouse_proc.start()
        print("SpaceMouse driver started")
        print("  Translation: Pushing forward = world +X, right = world +Y, up = world +Z")
        print("  Rotation: Roll/Pitch/Yaw = world frame angular velocity [wx, wy, wz]")
        print("  Gripper: Hold left button = open incrementally, Hold right button = close incrementally")
    
    def stop(self):
        """Stop the SpaceMouse reader process."""
        if self.spacemouse_proc is not None:
            self.stop_event.set()
            self.spacemouse_proc.join(timeout=1.0)
            if self.spacemouse_proc.is_alive():
                self.spacemouse_proc.terminate()
                self.spacemouse_proc.join(timeout=1.0)
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
        self.velocity_world.fill(0.0)
        self.angular_velocity_world.fill(0.0)
        
        # Process all queued commands to get the latest (drop older ones for responsiveness)
        latest_command = None
        while not self.data_queue.empty():
            try:
                latest_command = self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        if latest_command is None:
            return False
        
        # Extract translation as velocity in world frame
        vel_raw = latest_command['translation']
        vel_magnitude = np.linalg.norm(vel_raw)
        if vel_magnitude > self.velocity_deadzone:
            self.velocity_world = vel_raw
        
        # Map SpaceMouse rotation [roll, yaw, pitch] to world frame [wx, wy, wz]
        # Then negate to match CAD convention (opposite of SolidWorks)
        rotation_twist = latest_command.get('rotation', np.zeros(3))
        omega_raw = np.array([rotation_twist[2], -rotation_twist[0], rotation_twist[1]])
        omega_raw = -omega_raw
        
        omega_magnitude = np.linalg.norm(omega_raw)
        if omega_magnitude > self.angular_velocity_deadzone:
            self.angular_velocity_world = omega_raw
        
        # Handle gripper buttons
        button_state = latest_command.get('button', [])
        self.left_button_prev = len(button_state) > 0 and button_state[0] == 1
        self.right_button_prev = len(button_state) > 1 and button_state[1] == 1
        
        return True
    
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
    
    def get_gripper_button_states(self):
        """
        Get current gripper button states for incremental control.
        
        Returns:
            tuple: (left_button_pressed, right_button_pressed)
                - left_button_pressed: True if left button is currently held
                - right_button_pressed: True if right button is currently held
        """
        return (self.left_button_prev, self.right_button_prev)
    
    def get_gripper_command(self, current_gripper_position, dt):
        """
        Get gripper command from button states (incremental control).

        Converts button presses to a continuous gripper position command.
        This allows the same interface for both SpaceMouse and neural network inputs.

        Args:
            current_gripper_position: Current gripper position (meters)
            dt: Time step (seconds)

        Returns:
            float: Desired gripper position (meters) after applying button increments
        """
        from deployment.robot_config import robot_config
        
        left_button_pressed, right_button_pressed = self.get_gripper_button_states()
        gripper_position = current_gripper_position
        
        if left_button_pressed:
            gripper_position -= robot_config.gripper_increment_rate
            gripper_position = max(gripper_position, robot_config.gripper_open_pos)
        elif right_button_pressed:
            gripper_position += robot_config.gripper_increment_rate
            gripper_position = min(gripper_position, robot_config.gripper_closed_pos)
        
        return gripper_position
