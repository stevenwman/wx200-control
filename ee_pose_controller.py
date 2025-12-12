"""
End-effector pose controller that takes 6-vector velocity commands and produces mink-compatible targets.

The 6-vector input represents:
- [vx, vy, vz]: Linear velocity in world frame (m/s)
- [wx, wy, wz]: Angular velocity in world frame (rad/s)

This controller integrates these velocities to maintain a target pose (position + orientation)
that can be used with mink IK solvers.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


class EndEffectorPoseController:
    """
    Controller that takes world-frame velocity commands and produces target poses for IK.
    
    Maintains internal state for target position and orientation, and integrates
    velocity commands to update these targets.
    """
    
    def __init__(self, initial_position, initial_orientation_quat_wxyz):
        """
        Initialize the controller with an initial pose.
        
        Args:
            initial_position: [x, y, z] initial position in world frame (m)
            initial_orientation_quat_wxyz: [w, x, y, z] initial orientation quaternion
        """
        self.target_position = np.array(initial_position, dtype=np.float64).copy()
        self.target_quat_wxyz = np.array(initial_orientation_quat_wxyz, dtype=np.float64).copy()
        # Normalize quaternion
        self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)
    
    def update_from_velocity_command(self, velocity_world, angular_velocity_world, dt):
        """
        Update target pose by integrating velocity commands.
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in world frame (m/s)
            angular_velocity_world: [wx, wy, wz] angular velocity in world frame (rad/s)
            dt: Time step (s)
        
        Returns:
            None (updates internal state)
        """
        # Integrate linear velocity to update target position
        # target_pos = current_pos + velocity_world * dt
        self.target_position = self.target_position + velocity_world * dt
        
        # Integrate angular velocity to update target orientation in world frame
        # Use Rodrigues' formula to integrate angular velocity into quaternion
        # IMPORTANT: Since angular velocity is in world frame, we apply it via LEFT multiplication
        # (world-frame rotation applied to current orientation)
        omega_magnitude = np.linalg.norm(angular_velocity_world)
        if omega_magnitude > 1e-6:
            # Normalize angular velocity and compute rotation angle
            omega_normalized = angular_velocity_world / omega_magnitude
            theta = omega_magnitude * dt
            # Clamp theta to prevent numerical issues
            theta = np.clip(theta, 0, np.pi)
            
            # Convert current quaternion to rotation object
            # target_quat_wxyz is [w, x, y, z], R.from_quat expects [x, y, z, w]
            current_rot = R.from_quat([
                self.target_quat_wxyz[1], 
                self.target_quat_wxyz[2], 
                self.target_quat_wxyz[3], 
                self.target_quat_wxyz[0]
            ])  # [x, y, z, w]
            
            # Create rotation from axis-angle (Rodrigues' formula)
            # omega_normalized * theta is the rotation vector in world frame
            delta_rot = R.from_rotvec(omega_normalized * theta)
            
            # Compose rotations: LEFT multiplication for world-frame angular velocity
            # new_rot = delta_rot * current_rot (world-frame rotation applied to current orientation)
            # This ensures that rotating about +z world always rotates about +z world
            new_rot = delta_rot * current_rot
            
            # Convert back to quaternion [w, x, y, z]
            new_quat = new_rot.as_quat()  # [x, y, z, w]
            self.target_quat_wxyz = np.array([
                new_quat[3], 
                new_quat[0], 
                new_quat[1], 
                new_quat[2]
            ])  # [w, x, y, z]
            # Normalize quaternion
            self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)
    
    def get_target_pose_se3(self):
        """
        Get the current target pose as an mink.SE3 object.
        
        Returns:
            mink.SE3: Target pose with position and orientation
        """
        import mink
        # mink.SE3 expects [w, x, y, z, x, y, z] = [quat_wxyz, position_xyz]
        return mink.SE3(np.concatenate([self.target_quat_wxyz, self.target_position]))
    
    def get_target_position(self):
        """Get current target position."""
        return self.target_position.copy()
    
    def get_target_orientation_quat_wxyz(self):
        """Get current target orientation as [w, x, y, z] quaternion."""
        return self.target_quat_wxyz.copy()
    
    def reset_pose(self, position, orientation_quat_wxyz):
        """
        Reset the target pose to a new value.
        
        Args:
            position: [x, y, z] position in world frame (m)
            orientation_quat_wxyz: [w, x, y, z] orientation quaternion
        """
        self.target_position = np.array(position, dtype=np.float64).copy()
        self.target_quat_wxyz = np.array(orientation_quat_wxyz, dtype=np.float64).copy()
        # Normalize quaternion
        self.target_quat_wxyz = self.target_quat_wxyz / np.linalg.norm(self.target_quat_wxyz)
