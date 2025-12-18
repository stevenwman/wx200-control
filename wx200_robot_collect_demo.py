"""
Teleop control with ArUco marker tracking.
Inherits from TeleopControl and adds camera support.
"""
import cv2
import numpy as np
import argparse
from scipy.spatial.transform import Rotation as R

from robot_control.robot_config import robot_config
from wx200_robot_teleop_control import TeleopControl
from camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix

# Shorter axes for visualization to reduce chance of going off-frame (and warnings)
AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale

class TeleopCameraControl(TeleopControl):
    """
    Teleop control that also records ArUco marker poses.
    
    Tracks:
    - ID 0: World/Base Frame
    - ID 2: Object
    - ID 3: Gripper (optional, logged if found)
    """
    
    def __init__(self, enable_recording=False, output_path=None,
                 camera_id=None, width=None, height=None, fps=None):
        super().__init__(enable_recording=enable_recording, output_path=output_path)
        
        # Use config defaults unless overridden explicitly
        self.camera_id = camera_id if camera_id is not None else robot_config.camera_id
        self.width = width if width is not None else robot_config.camera_width
        self.height = height if height is not None else robot_config.camera_height
        self.fps = fps if fps is not None else robot_config.camera_fps
        
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
        # Draw axes by default, but with reduced length (AXIS_LENGTH) to reduce warnings.
        self.show_axes = True  # Set to False if you want markers only.
        
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
                
                # Process tags using configured IDs
                r_world, t_world = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id
                )
                r_obj, t_obj = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id
                )
                r_ee, t_ee = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id
                )
                
                # Update visibility flags based on actual detections (not just held poses)
                world_visible = False
                object_visible = False
                ee_visible = False
                if ids is not None:
                    ids_arr = np.atleast_1d(ids).ravel()
                    world_visible = np.any(ids_arr == robot_config.aruco_world_id)
                    object_visible = np.any(ids_arr == robot_config.aruco_object_id)
                    ee_visible = np.any(ids_arr == robot_config.aruco_ee_id)

                obs['aruco_visibility'][0] = 1.0 if world_visible else 0.0
                obs['aruco_visibility'][1] = 1.0 if object_visible else 0.0
                obs['aruco_visibility'][2] = 1.0 if ee_visible else 0.0
                
                # Visualization
                if self.show_video:
                    if ids is not None:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    if self.show_axes:
                        if r_world is not None:
                            cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_world, t_world, AXIS_LENGTH)
                        if r_obj is not None:
                            cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_obj, t_obj, AXIS_LENGTH)
                        if r_ee is not None:
                            cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_ee, t_ee, AXIS_LENGTH)
                    
                    # Resize for display (configurable preview size)
                    disp = cv2.resize(
                        frame,
                        (robot_config.camera_width // 2, robot_config.camera_height // 2)
                    )
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
        
        # 3. Augment Trajectory with Composite Observations and Axis-Angle Actions
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

            # Convert angular velocity [wx, wy, wz] to per-step axis-angle 3-vector.
            # We encode the actual rotation this timestep:
            #   axis_angle_vec = omega * dt
            # so ||axis_angle_vec|| = rotation angle in this step, direction = rotation axis.
            angular_vel = angular_velocity_world
            if dt > 0.0:
                axis_angle_vec = angular_vel * dt
            else:
                axis_angle_vec = np.zeros(3)

            # Augmented actions: original action + axis-angle 3-vector
            augmented_actions = np.concatenate([
                velocity_world,          # [vx, vy, vz]
                angular_velocity_world,  # [wx, wy, wz] (original angular velocity)
                axis_angle_vec,          # 3D axis-angle vector
                [gripper_target]         # gripper
            ])
            current_step['augmented_actions'] = augmented_actions

    def shutdown(self):
        """Cleanup camera and windows."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        super().shutdown()

def main():
    parser = argparse.ArgumentParser(description='WX200 Teleop with Camera')
    # Note: Recording is controlled via the GUI (Start/Stop/Discard buttons).
    # This flag is kept only for CLI compatibility and is ignored.
    parser.add_argument('--record', action='store_true', help='(Deprecated) Recording is controlled via GUI')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--camera-id', type=int, default=None, help='Camera device ID (defaults to robot_config.camera_id)')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    
    args = parser.parse_args()
    
    # Create and run
    controller = TeleopCameraControl(
        # Always enable recording capability; GUI decides when to actually record.
        enable_recording=True,
        output_path=args.output,
        camera_id=args.camera_id
    )
    
    if args.no_vis:
        controller.show_video = False
        
    controller.run()

if __name__ == "__main__":
    main()
