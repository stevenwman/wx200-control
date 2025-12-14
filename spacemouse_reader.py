"""
SpaceMouse reader process that outputs end-effector twist commands.
Runs in a separate process to avoid blocking the main simulation loop.
"""
import pyspacemouse
import multiprocessing as mp
import queue
import time
import numpy as np

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
        self.button_state = []
        
    def button_callback(self, state, buttons):
        """Callback for button state changes. buttons is a list where 1 = pressed, 0 = released."""
        self.button_state = buttons.copy() if buttons else []
    
    def start(self):
        """Start reading from spacemouse and putting commands in queue"""
        success = pyspacemouse.open(button_callback=self.button_callback)
        if not success:
            raise RuntimeError("Failed to open spacemouse")
        
        self.running = True
        self.button_state = []
        print("SpaceMouse reader started")
        
        while self.running:
            try:
                state = pyspacemouse.read()
                if state:
                    # Extract translation (xyz) - range [-1, 1]
                    tx, ty, tz = state.x, state.y, state.z
                    
                    # Extract rotation (roll, pitch, yaw) - range [-1, 1]
                    roll, pitch, yaw = state.roll, state.pitch, state.yaw
                    
                    # Convert to twist command
                    # Translation: SpaceMouse x=forward, y=left, z=up -> world frame
                    translation_twist = np.array([tx, ty, tz]) * self.translation_scale
                    
                    # Rotation: [roll, yaw, pitch] - mapping to world frame handled by driver
                    rotation_twist = np.array([roll, yaw, pitch]) * self.rotation_scale
                    
                    # Get button state (updated via callback)
                    button_state = self.button_state.copy() if self.button_state else []
                    
                    # Create twist command
                    twist_command = {
                        'translation': translation_twist,  # [vx, vy, vz] in m/s (world frame)
                        'rotation': rotation_twist,        # [roll, yaw, pitch] - mapped to [wx, wy, wz] by driver
                        'button': button_state,            # Button states (1=pressed, 0=released)
                        'timestamp': time.time()
                    }
                    
                    # Put in queue (non-blocking, drop oldest if queue is full)
                    # This ensures we always have the latest command available, which is
                    # important for responsive control. Older commands are dropped to prevent lag.
                    try:
                        self.data_queue.put_nowait(twist_command)
                    except queue.Full:
                        # Queue is full: drop oldest command and add new one
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
