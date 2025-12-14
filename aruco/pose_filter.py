"""
Pose filtering to smooth ArUco marker poses and reject outliers.
"""

import cv2
import numpy as np


class PoseFilter:
    """Advanced filter to smooth pose, handle Z-axis flipping, and reject outliers."""
    
    def __init__(self, alpha=0.7, max_translation_jump=0.1, max_rotation_jump=0.5,
                 max_consecutive_failures=3):
        """
        Initialize pose filter.
        
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing, more responsive.
            max_translation_jump: Maximum allowed translation change (meters) per frame.
            max_rotation_jump: Maximum allowed rotation change (radians) per frame.
            max_consecutive_failures: After N consecutive rejected poses, accept the next one.
        """
        self.alpha = alpha
        self.max_translation_jump = max_translation_jump
        self.max_rotation_jump = max_rotation_jump
        self.max_consecutive_failures = max_consecutive_failures
        self.last_rvec = None
        self.last_tvec = None
        self.last_z_axis = None
        self.last_rmat = None
        self.consecutive_failures = 0
    
    def filter_pose(self, rvec, tvec):
        """
        Filter pose with outlier rejection and Z-axis flip correction.
        
        Args:
            rvec: Rotation vector (3x1)
            tvec: Translation vector (3x1)
        
        Returns:
            (filtered_rvec, filtered_tvec): Filtered pose
        """
        # Ensure proper shape
        rvec = rvec.reshape(3, 1) if rvec.shape != (3, 1) else rvec
        tvec = tvec.reshape(3, 1) if tvec.shape != (3, 1) else tvec
        
        if self.last_rvec is None:
            # First frame - initialize
            # Validate initial pose
            if np.any(np.isnan(rvec)) or np.any(np.isnan(tvec)) or \
               np.any(np.isinf(rvec)) or np.any(np.isinf(tvec)):
                return None, None  # Invalid initial pose
            
            R, _ = cv2.Rodrigues(rvec)
            if R is None or np.any(np.isnan(R)) or np.any(np.isinf(R)):
                return None, None  # Invalid rotation matrix
            
            self.last_rvec = rvec.copy()
            self.last_tvec = tvec.copy()
            self.last_rmat = R
            self.last_z_axis = R[:, 2]
            return rvec, tvec
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        current_z_axis = R[:, 2]
        
        # Check for Z-axis flip
        z_dot = np.dot(self.last_z_axis, current_z_axis)
        if z_dot < -0.5:  # Significant flip detected
            # Flip Z-axis and X-axis to maintain right-handed coordinate system
            R_flip = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
            R = R @ R_flip
            # Re-orthonormalize
            u, _, vt = np.linalg.svd(R)
            R = u @ vt
            if np.linalg.det(R) < 0:
                u[:, -1] = -u[:, -1]
                R = u @ vt
            rvec, _ = cv2.Rodrigues(R)
            rvec = rvec.flatten()
            current_z_axis = R[:, 2]
        
        # Check for sudden jumps (outlier rejection)
        translation_jump = np.linalg.norm(tvec - self.last_tvec)
        rotation_jump = np.linalg.norm(rvec - self.last_rvec)
        
        if translation_jump > self.max_translation_jump or rotation_jump > self.max_rotation_jump:
            # Sudden jump detected - likely outlier
            self.consecutive_failures += 1
            if self.consecutive_failures < self.max_consecutive_failures:
                # Reject this pose, use last known good pose
                return self.last_rvec.copy(), self.last_tvec.copy()
            else:
                # Too many failures, accept it (maybe marker moved quickly)
                self.consecutive_failures = 0
        
        # Reset failure counter if pose is reasonable
        if translation_jump <= self.max_translation_jump and rotation_jump <= self.max_rotation_jump:
            self.consecutive_failures = 0
        
        # Exponential smoothing
        filtered_rvec = self.alpha * rvec + (1 - self.alpha) * self.last_rvec
        filtered_tvec = self.alpha * tvec + (1 - self.alpha) * self.last_tvec
        
        # Validate filtered pose - check for NaN/Inf
        if np.any(np.isnan(filtered_rvec)) or np.any(np.isnan(filtered_tvec)) or \
           np.any(np.isinf(filtered_rvec)) or np.any(np.isinf(filtered_tvec)):
            # Filter produced invalid values, return last known good pose
            return self.last_rvec.copy(), self.last_tvec.copy()
        
        # Update state
        self.last_rvec = filtered_rvec.copy()
        self.last_tvec = filtered_tvec.copy()
        self.last_rmat = R
        self.last_z_axis = current_z_axis
        
        return filtered_rvec, filtered_tvec
