"""
SpaceMouse reader process that outputs end-effector twist commands.
Runs in a separate process to avoid blocking the main simulation loop.
"""
import pyspacemouse
import multiprocessing as mp
import queue
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

class SpaceMouseReader:
    def __init__(self, data_queue, translation_scale=0.05, rotation_scale=0.5):
        """
        Initialize SpaceMouse reader.
        
        Args:
            data_queue: multiprocessing.Queue to put twist commands
            translation_scale: Scale factor for translation (m/s per unit input)
            rotation_scale: Scale factor for rotation (rad/s per unit input)
        """
        self.data_queue = data_queue
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.running = False
        self.button_state = []  # Store current button state
        
    def button_callback(self, state, buttons):
        """Callback for button state changes"""
        # buttons is a list where 1 = pressed, 0 = released
        self.button_state = buttons.copy() if buttons else []
    
    def start(self):
        """Start reading from spacemouse and putting commands in queue"""
        success = pyspacemouse.open(button_callback=self.button_callback)
        if not success:
            raise RuntimeError("Failed to open spacemouse")
        
        self.running = True
        self.button_state = []  # Initialize button state
        print("SpaceMouse reader started")
        
        while self.running:
            try:
                state = pyspacemouse.read()
                if state:
                    # Extract translation (xyz) - these are in range [-1, 1]
                    tx, ty, tz = state.x, state.y, state.z
                    
                    # Extract rotation (roll, pitch, yaw) - these are in range [-1, 1]
                    # Spacemouse convention (device frame):
                    # - Roll: rotation about X-axis (forward/backward tilt)
                    # - Pitch: rotation about Y-axis (left/right tilt)  
                    # - Yaw: rotation about Z-axis (twist)
                    roll, pitch, yaw = state.roll, state.pitch, state.yaw
                    
                    # Convert to twist command in world frame
                    # Spacemouse translation: x=forward, y=left, z=up
                    # Map directly to world frame: [x, y, z] -> [world_x, world_y, world_z]
                    translation_twist = np.array([tx, ty, tz]) * self.translation_scale
                    
                    # Spacemouse rotation: [roll, pitch, yaw] in device frame
                    # Map to world frame angular velocities: [wx, wy, wz]
                    # Spacemouse: roll=rotation about X, pitch=rotation about Y, yaw=rotation about Z
                    # 
                    # If spacemouse yaw (z twist) causes wrong world axis rotation, try remapping:
                    # Current: [roll, pitch, yaw] -> [wx, wy, wz]
                    # If yaw->wy issue: try [roll, yaw, pitch] -> [wx, wy, wz]
                    # If yaw->wx issue: try [yaw, pitch, roll] -> [wx, wy, wz]
                    # If pitch/yaw swapped: try [roll, yaw, pitch] -> [wx, wy, wz]
                    
                    # Map spacemouse rotations to world frame angular velocities [wx, wy, wz]
                    # Spacemouse: roll (x-axis), pitch (y-axis), yaw (z-axis)  
                    # User reports: yaw (z twist) causes world y rotation, but should cause world z rotation
                    # 
                    # If spacemouse pitch and yaw channels are swapped, swap them in input:
                    # [roll, yaw, pitch] -> [wx, wy, wz] means: roll->wx, yaw->wy, pitch->wz
                    # But we want yaw->wz, so try: [roll, pitch, yaw] -> [wx, wz, wy] (swap output wy/wz)
                    # OR: spacemouse might have pitch/yaw swapped, so: [roll, yaw, pitch] -> [wx, wy, wz]
                    
                    # Try swapping pitch and yaw in INPUT (most likely fix):
                    rotation_twist = np.array([roll, yaw, pitch]) * self.rotation_scale
                    
                    # If that doesn't work, try swapping output axes instead:
                    # rotation_raw = np.array([roll, pitch, yaw]) * self.rotation_scale
                    # rotation_twist = np.array([rotation_raw[0], rotation_raw[2], rotation_raw[1]])  # [wx, wz, wy]
                    
                    # Original mapping (for reference):
                    # rotation_twist = np.array([roll, pitch, yaw]) * self.rotation_scale  # [wx, wy, wz]
                    
                    # Get button state (updated via callback)
                    # buttons is a list where 1 = pressed, 0 = released
                    button_state = self.button_state.copy() if self.button_state else []
                    
                    # Create twist command (6D: [vx, vy, vz, wx, wy, wz] in world frame)
                    twist_command = {
                        'translation': translation_twist,  # [vx, vy, vz] in m/s (world frame)
                        'rotation': rotation_twist,        # [wx, wy, wz] in rad/s (world frame)
                        'button': button_state,            # Button states (list: 1=pressed, 0=released)
                        'timestamp': time.time()
                    }
                    
                    # Put in queue (non-blocking, drop if queue is full)
                    try:
                        self.data_queue.put_nowait(twist_command)
                    except queue.Full:
                        # Drop old command if queue is full
                        try:
                            self.data_queue.get_nowait()
                            self.data_queue.put_nowait(twist_command)
                        except queue.Empty:
                            pass
                
                time.sleep(0.001)  # ~1000 Hz reading rate
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error reading spacemouse: {e}")
                time.sleep(0.01)
        
        pyspacemouse.close()
        print("SpaceMouse reader stopped")
    
    def stop(self):
        """Stop the reader"""
        self.running = False


def spacemouse_process(data_queue, translation_scale, rotation_scale):
    """Process function for multiprocessing"""
    reader = SpaceMouseReader(data_queue, translation_scale, rotation_scale)
    try:
        reader.start()
    except KeyboardInterrupt:
        reader.stop()
