"""
SpaceMouse reader process that outputs end-effector twist commands.
Runs in a separate process to avoid blocking the main simulation loop.
"""
import logging
import sys
import os
import pyspacemouse
import queue
import time
import numpy as np

# Suppress pyspacemouse INFO messages (e.g., "SpaceMouse Wireless found")
# Try multiple logger names that pyspacemouse might use
for logger_name in ['pyspacemouse', 'spacemouse', '__main__']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Also suppress stdout temporarily during pyspacemouse.open() if needed

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
    
    def start(self, stop_event=None):
        """Start reading from spacemouse and putting commands in queue"""
        # Suppress stdout during device discovery to hide "SpaceMouse Wireless found" messages
        original_stdout = sys.stdout
        devnull = open(os.devnull, 'w')
        try:
            sys.stdout = devnull
            success = pyspacemouse.open(button_callback=self.button_callback)
        finally:
            sys.stdout = original_stdout
            devnull.close()
        
        if not success:
            raise RuntimeError("Failed to open spacemouse")
        
        self.running = True
        self.button_state = []
        print("SpaceMouse reader started")
        
        while self.running and (stop_event is None or not stop_event.is_set()):
            try:
                state = pyspacemouse.read()
                if state:
                    # Extract translation and rotation (range [-1, 1])
                    translation_twist = np.array([state.x, state.y, state.z]) * self.translation_scale
                    rotation_twist = np.array([state.roll, state.yaw, state.pitch]) * self.rotation_scale
                    
                    # Create twist command (rotation mapping to world frame handled by driver)
                    twist_command = {
                        'translation': translation_twist,
                        'rotation': rotation_twist,
                        'button': self.button_state.copy() if self.button_state else [],
                        'timestamp': time.time()
                    }
                    
                    # Put in queue (drop oldest if full to ensure latest command is available)
                    try:
                        self.data_queue.put_nowait(twist_command)
                    except queue.Full:
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


def spacemouse_process(data_queue, translation_scale, rotation_scale, stop_event=None):
    """Process function for multiprocessing"""
    reader = SpaceMouseReader(data_queue, translation_scale, rotation_scale)
    try:
        reader.start(stop_event=stop_event)
    except KeyboardInterrupt:
        reader.stop()
