import pyspacemouse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

class SpaceMouseVisualizer:
    def __init__(self):
        # Initialize spacemouse
        success = pyspacemouse.open()
        if not success:
            raise RuntimeError("Failed to open spacemouse")
        
        # Enable interactive mode
        plt.ion()
        
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        self.setup_plot()
        
        # Initialize quiver objects (we'll update these, not recreate)
        self.translation_quiver = None
        self.rotation_quiver = None
        
        # Scale factors
        self.translation_scale = 0.7
        self.rotation_scale = 0.7
        
        # Show initial plot
        plt.show(block=False)
        plt.pause(0.1)
        
    def setup_plot(self):
        """Configure the 3D plot appearance"""
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel('X', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Y', fontsize=12, fontweight='bold')
        self.ax.set_zlabel('Z', fontsize=12, fontweight='bold')
        self.ax.set_title('SpaceMouse Twist Commands', fontsize=14, fontweight='bold', pad=20)
        
        # Add coordinate axes for reference
        axis_length = 0.3
        self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='gray', alpha=0.3, 
                       arrow_length_ratio=0.2, linewidth=1)
        self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='gray', alpha=0.3, 
                       arrow_length_ratio=0.2, linewidth=1)
        self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='gray', alpha=0.3, 
                       arrow_length_ratio=0.2, linewidth=1)
        
        # Add origin point
        self.ax.scatter([0], [0], [0], c='black', s=80, marker='o', alpha=0.6)
        
        # Add grid
        self.ax.grid(True, alpha=0.2)
        
        # Set equal aspect ratio
        self.ax.set_box_aspect([1,1,1])
        
        # Set background
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        
    def update_visualization(self, state):
        """Update the visualization with new spacemouse state"""
        # Extract translation (xyz)
        tx, ty, tz = state.x, state.y, state.z
        
        # Extract rotation (roll, pitch, yaw)
        roll, pitch, yaw = state.roll, state.pitch, state.yaw
        
        # Calculate scaled vectors
        translation_vec = np.array([tx, ty, tz]) * self.translation_scale
        rotation_vec = np.array([roll, pitch, yaw]) * self.rotation_scale
        
        translation_magnitude = np.linalg.norm(translation_vec)
        rotation_magnitude = np.linalg.norm(rotation_vec)
        
        # Update translation arrow
        if translation_magnitude > 0.01:
            if self.translation_quiver is None:
                # Create new quiver
                self.translation_quiver = self.ax.quiver(
                    0, 0, 0, translation_vec[0], translation_vec[1], translation_vec[2],
                    color='#2E86AB', arrow_length_ratio=0.25, linewidth=4, alpha=0.9
                )
            else:
                # Update existing quiver (much faster!)
                self.translation_quiver.remove()
                self.translation_quiver = self.ax.quiver(
                    0, 0, 0, translation_vec[0], translation_vec[1], translation_vec[2],
                    color='#2E86AB', arrow_length_ratio=0.25, linewidth=4, alpha=0.9
                )
        elif self.translation_quiver is not None:
            self.translation_quiver.remove()
            self.translation_quiver = None
        
        # Update rotation arrow
        if rotation_magnitude > 0.01:
            if self.rotation_quiver is None:
                # Create new quiver
                self.rotation_quiver = self.ax.quiver(
                    0, 0, 0, rotation_vec[0], rotation_vec[1], rotation_vec[2],
                    color='#A23B72', arrow_length_ratio=0.25, linewidth=4, alpha=0.9
                )
            else:
                # Update existing quiver (much faster!)
                self.rotation_quiver.remove()
                self.rotation_quiver = self.ax.quiver(
                    0, 0, 0, rotation_vec[0], rotation_vec[1], rotation_vec[2],
                    color='#A23B72', arrow_length_ratio=0.25, linewidth=4, alpha=0.9
                )
        elif self.rotation_quiver is not None:
            self.rotation_quiver.remove()
            self.rotation_quiver = None
        
        # Update text (remove old, add new - text is lightweight)
        if hasattr(self, 'info_text_obj'):
            try:
                self.info_text_obj.remove()
            except:
                pass
        
        info_text = f'Translation (Force):\n'
        info_text += f'  X: {tx:7.4f}  Y: {ty:7.4f}  Z: {tz:7.4f}\n'
        info_text += f'  Magnitude: {translation_magnitude:.4f}\n\n'
        info_text += f'Rotation (Torque):\n'
        info_text += f'  Roll: {roll:7.4f}  Pitch: {pitch:7.4f}  Yaw: {yaw:7.4f}\n'
        info_text += f'  Magnitude: {rotation_magnitude:.4f}'
        
        self.info_text_obj = self.ax.text2D(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            fontsize=9,
            fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
        )
        
        # Minimal update - just draw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def run(self):
        """Main loop to continuously update visualization"""
        print("SpaceMouse Visualizer Running (Optimized Matplotlib)...")
        print("Move your spacemouse to see the twist commands!")
        print("Press Ctrl+C to exit")
        
        try:
            while True:
                state = pyspacemouse.read()
                if state:
                    self.update_visualization(state)
                time.sleep(0.001)  # ~1000 Hz
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            pyspacemouse.close()
            plt.close()

if __name__ == "__main__":
    visualizer = SpaceMouseVisualizer()
    visualizer.run()
