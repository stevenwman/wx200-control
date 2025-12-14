"""
Teleop control with ArUco marker tracking.
Inherits from TeleopControl and adds camera support.
"""
import time
import cv2
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from wx200_robot_teleop_control import TeleopControl
from camera import Camera, is_gstreamer_available
from aruco_pose_estimator import ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix

class TeleopCameraControl(TeleopControl):
    """
    Teleop control that also records ArUco marker poses.
    
    Tracks:
    - ID 0: World/Base Frame
    - ID 2: Object
    - ID 3: Gripper (optional, logged if found)
    """
    
    def __init__(self, enable_recording=False, output_path=None, camera_id=1, width=1920, height=1080, fps=30):
        super().__init__(enable_recording=enable_recording, output_path=output_path)
        
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        
        self.camera = None
        self.estimator = None
        self.detector = None
        self.cam_matrix = None
        self.dist_coeffs = None
        
        # Latest detections
        self.latest_object_pose = None # (rvec, tvec)
        self.latest_world_pose = None  # (rvec, tvec)
        self.latest_gripper_pose = None # (rvec, tvec)
        
        # Visualization
        self.show_video = True
        
    def on_ready(self):
        """Initialize camera and estimator."""
        super().on_ready()
        
        print("\n" + "="*60)
        print("Initializing Camera & ArUco Estimator...")
        
        if not is_gstreamer_available():
            print("ℹ️  Note: GStreamer not found. Using OpenCV fallback.")
        
        try:
            # Camera factory handles GStreamer vs OpenCV selection
            self.camera = Camera(device=self.camera_id, width=self.width, height=self.height, fps=self.fps)
            self.camera.start()
            
            self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
            self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(self.width, self.height)
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            print(f"✓ Camera started: Device {self.camera_id} @ {self.width}x{self.height}")
            print("✓ ArUco Estimator ready (IDs: 0=World, 2=Object)")
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            self.camera = None
            
        print("="*60 + "\n")

    def _compute_relative_pose(self, r_ref, t_ref, r_tgt, t_tgt):
        """Compute target pose relative to reference frame. Returns (pos, quat_wxyz) or (zeros, zeros)."""
        if r_ref is None or t_ref is None or r_tgt is None or t_tgt is None:
            return np.zeros(3), np.array([1., 0., 0., 0.]) # Identity quaternion (w, x, y, z)
            
        # Get relative pose: Target in Reference
        r_rel, t_rel = self.estimator.get_relative_pose(r_ref, t_ref, r_tgt, t_tgt)
        
        # Convert rotation to quaternion (wxyz)
        R_rel, _ = cv2.Rodrigues(r_rel)
        quat_xyzw = R.from_matrix(R_rel).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return t_rel.flatten(), quat_wxyz

    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """Read camera, detect markers, and update trajectory."""
        
        # Initialize storage for all composite observations
        # Formats: 7D poses [x, y, z, qw, qx, qy, qz]
        obs = {
            'aruco_ee_in_world': np.zeros(7),      # ID 3 in ID 0
            'aruco_object_in_world': np.zeros(7),  # ID 2 in ID 0
            'aruco_ee_in_object': np.zeros(7),     # ID 3 in ID 2
            'aruco_object_in_ee': np.zeros(7),     # ID 2 in ID 3
            'aruco_visibility': np.zeros(3)        # [World(0), Object(2), Gripper(3)]
        }
        
        # Alias for backward compatibility (maps to object_in_world)
        object_pose_data = np.zeros(7)
        object_visible = 0.0
        
        if self.camera:
            ret, frame = self.camera.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = self.detector.detectMarkers(gray)
                
                # Process tags: 0=World, 2=Object, 3=Gripper
                r_world, t_world = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, 0)
                r_obj, t_obj = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, 2)
                r_ee, t_ee = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, 3)
                
                # Update visibility flags
                obs['aruco_visibility'][0] = 1.0 if r_world is not None else 0.0
                obs['aruco_visibility'][1] = 1.0 if r_obj is not None else 0.0
                obs['aruco_visibility'][2] = 1.0 if r_ee is not None else 0.0
                
                # Visualization
                if self.show_video:
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    if r_world is not None:
                        cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_world, t_world, MARKER_SIZE)
                    if r_obj is not None:
                        cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_obj, t_obj, MARKER_SIZE)
                    if r_ee is not None:
                        cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_ee, t_ee, MARKER_SIZE)
                    
                    # Resize for display
                    disp = cv2.resize(frame, (640, 360))
                    cv2.imshow('Robot Camera View', disp)
                    cv2.waitKey(1)

                # Compute Composite Observations
                
                # 1. EE (3) relative to World (0)
                pos, quat = self._compute_relative_pose(r_world, t_world, r_ee, t_ee)
                obs['aruco_ee_in_world'] = np.concatenate([pos, quat])

                # 2. Object (2) relative to World (0)
                pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
                obs['aruco_object_in_world'] = np.concatenate([pos, quat])
                
                # Backward compatibility
                if obs['aruco_visibility'][0] and obs['aruco_visibility'][1]:
                    object_pose_data = obs['aruco_object_in_world']
                    object_visible = 1.0

                # 3. EE (3) relative to Object (2)
                pos, quat = self._compute_relative_pose(r_obj, t_obj, r_ee, t_ee)
                obs['aruco_ee_in_object'] = np.concatenate([pos, quat])

                # 4. Object (2) relative to EE (3)
                pos, quat = self._compute_relative_pose(r_ee, t_ee, r_obj, t_obj)
                obs['aruco_object_in_ee'] = np.concatenate([pos, quat])
        
        # 2. Base Robot Recording (adds to self.trajectory)
        super().on_control_loop_iteration(velocity_world, angular_velocity_world, gripper_target, dt)
        
        # 3. Augment Trajectory with Composite Observations
        if self.is_recording and self.trajectory:
            current_step = self.trajectory[-1]
            
            # Store backward compatible fields
            current_step['object_pose'] = object_pose_data
            current_step['object_visible'] = np.array([object_visible])
            
            # Store new composite observations
            current_step['aruco_ee_in_world'] = obs['aruco_ee_in_world']
            current_step['aruco_object_in_world'] = obs['aruco_object_in_world']
            current_step['aruco_ee_in_object'] = obs['aruco_ee_in_object']
            current_step['aruco_object_in_ee'] = obs['aruco_object_in_ee']
            current_step['aruco_visibility'] = obs['aruco_visibility']

    def shutdown(self):
        """Cleanup camera and windows."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        super().shutdown()

def main():
    parser = argparse.ArgumentParser(description='WX200 Teleop with Camera')
    parser.add_argument('--record', action='store_true', help='Enable recording')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--camera-id', type=int, default=1, help='Camera device ID')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    
    args = parser.parse_args()
    
    # Create and run
    controller = TeleopCameraControl(
        enable_recording=args.record,
        output_path=args.output,
        camera_id=args.camera_id
    )
    
    if args.no_vis:
        controller.show_video = False
        
    controller.run()

if __name__ == "__main__":
    main()
