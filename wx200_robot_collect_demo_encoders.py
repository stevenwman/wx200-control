"""
Teleop control with ArUco marker tracking and true encoder state polling.

This version polls actual robot encoder values at high frequency (50Hz+),
uses them to sync MuJoCo state, and records encoder values in the trajectory.
This ensures we're recording the true robot state rather than relying on MuJoCo estimates.
"""
import cv2
import numpy as np
import argparse
import time
import mujoco
import mink
from scipy.spatial.transform import Rotation as R

from robot_control.robot_config import robot_config
from wx200_robot_teleop_control import TeleopControl, save_trajectory
from camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix
from robot_control.robot_joint_to_motor import sync_robot_to_mujoco

# Shorter axes for visualization to reduce chance of going off-frame (and warnings)
AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale

class TeleopCameraControlEncoders(TeleopControl):
    """
    Teleop control that polls true encoder values and records them.
    
    Key differences from base TeleopCameraControl:
    - Polls actual robot encoders at 50Hz+ using GroupSyncRead
    - Uses encoder values to sync MuJoCo state before solving IK
    - Records encoder values in trajectory
    - Records joint angles derived from encoders (not MuJoCo estimates)
    
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
        
        # Latest encoder state (updated by polling)
        self.latest_encoder_values = {}  # {motor_id: encoder_position}
        self.latest_joint_angles_from_encoders = None  # np.array([q0-q4, gripper])
        self.latest_ee_pose_from_encoders = None  # (position, quat_wxyz)
        
        # Separate MuJoCo instance for encoder-based forward kinematics only
        # This is independent from the IK solver's MuJoCo instance
        self.encoder_fk_model = None
        self.encoder_fk_data = None
        self.encoder_fk_configuration = None
        self.latest_ee_pose_from_encoders_fk = None  # EE pose from encoder-only FK model
        
        # Performance tracking
        self.encoder_poll_times = []
        self.encoder_poll_intervals = []  # Time between polls
        self.encoder_poll_count = 0
        self.last_poll_timestamp = None
        
        # Visualization
        self.show_video = True
        # Draw axes by default, but with reduced length (AXIS_LENGTH) to reduce warnings.
        self.show_axes = True  # Set to False if you want markers only.
        
    def on_ready(self):
        """Initialize camera and estimator."""
        super().on_ready()
        
        print("\n" + "="*60)
        print("Initializing Camera & ArUco Estimator...")
        print(f"Encoder polling: At recording rate ({robot_config.control_frequency} Hz)")
        
        # Initialize separate MuJoCo instance for encoder-based forward kinematics
        from pathlib import Path
        from robot_control.robot_control_base import _XML
        
        self.encoder_fk_model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.encoder_fk_data = mujoco.MjData(self.encoder_fk_model)
        self.encoder_fk_configuration = mink.Configuration(self.encoder_fk_model)
        print("✓ Initialized separate MuJoCo instance for encoder-based FK")
        
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
    
    def _poll_encoders(self):
        """
        Poll encoder values from robot hardware using fast bulk read.
        
        Updates:
        - self.latest_encoder_values
        - self.latest_joint_angles_from_encoders
        - self.latest_ee_pose_from_encoders (via MuJoCo sync)
        """
        poll_start = time.perf_counter()
        
        # Track interval since last poll
        if self.last_poll_timestamp is not None:
            interval = poll_start - self.last_poll_timestamp
            self.encoder_poll_intervals.append(interval)
            interval_freq = 1.0 / interval if interval > 0 else 0
        else:
            interval = 0.0
            interval_freq = 0.0
        
        # Read encoders using bulk read (GroupSyncRead) for speed
        encoder_values = self.robot_driver.read_all_encoders(
            max_retries=1,  # Fast retry (bulk read should be reliable)
            retry_delay=0.01,
            use_bulk_read=True  # Use GroupSyncRead for speed
        )
        
        # Update latest encoder values
        self.latest_encoder_values = encoder_values.copy()
        
        # Convert encoders to joint angles and sync MuJoCo
        # NOTE: We sync MuJoCo state for accurate recording, but we do NOT reset the controller's
        # target pose. The controller continues with velocity-based control, and IK will use the
        # actual robot state (from encoders) as the starting point, which is correct.
        try:
            robot_joint_angles, actual_position, actual_orientation_quat_wxyz = sync_robot_to_mujoco(
                encoder_values, self.translator, self.model, self.data, self.configuration
            )
            
            self.latest_joint_angles_from_encoders = robot_joint_angles.copy()
            self.latest_ee_pose_from_encoders = (actual_position.copy(), actual_orientation_quat_wxyz.copy())
            
            # DO NOT reset controller pose - let velocity-based control continue normally
            # The IK solver will use self.configuration.q (which is now synced with actual robot state)
            # as the starting point, which is correct for solving IK from the true robot position.
            
            # Update separate FK-only MuJoCo instance with encoder values
            # This gives us the "ground truth" EE pose from encoders, independent of IK solver
            if self.encoder_fk_model is not None:
                # Convert encoders to joint angles (same as above)
                # Update FK-only model with encoder-based joint angles
                self.encoder_fk_data.qpos[:5] = robot_joint_angles[:5]
                if len(self.encoder_fk_data.qpos) > 5:
                    self.encoder_fk_data.qpos[5] = robot_joint_angles[5]
                
                # Compute forward kinematics
                mujoco.mj_forward(self.encoder_fk_model, self.encoder_fk_data)
                
                # Update configuration
                self.encoder_fk_configuration.update(self.encoder_fk_data.qpos)
                
                # Get EE pose from FK-only model
                site_id = mujoco.mj_name2id(self.encoder_fk_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                fk_position = self.encoder_fk_data.site(site_id).xpos.copy()
                fk_xmat = self.encoder_fk_data.site(site_id).xmat.reshape(3, 3)
                fk_quat = R.from_matrix(fk_xmat).as_quat()
                fk_orientation_quat_wxyz = np.array([fk_quat[3], fk_quat[0], fk_quat[1], fk_quat[2]])
                
                self.latest_ee_pose_from_encoders_fk = (fk_position.copy(), fk_orientation_quat_wxyz.copy())
            
        except Exception as e:
            # If sync fails, log warning but continue
            print(f"⚠️  Warning: Failed to sync encoders to MuJoCo: {e}")
            # Keep previous values
        
        # Track performance
        poll_duration = time.perf_counter() - poll_start
        self.encoder_poll_times.append(poll_duration)
        self.encoder_poll_count += 1
        self.last_poll_timestamp = poll_start
        
        # Debug output: print every poll for first 10, then every 10 polls, then every 50
        if self.encoder_poll_count <= 10:
            # Print every poll for first 10
            print(f"[ENCODER POLL #{self.encoder_poll_count}] "
                  f"read_time={poll_duration*1000:.2f}ms, "
                  f"interval={interval*1000:.2f}ms ({interval_freq:.1f}Hz), "
                  f"encoders={[encoder_values.get(mid, 'None') for mid in robot_config.motor_ids]}")
        elif self.encoder_poll_count % 10 == 0:
            # Print every 10 polls
            print(f"[ENCODER POLL #{self.encoder_poll_count}] "
                  f"read_time={poll_duration*1000:.2f}ms, "
                  f"interval={interval*1000:.2f}ms ({interval_freq:.1f}Hz)")
        
        # Print detailed performance stats periodically
        if self.encoder_poll_count % 50 == 0:
            if len(self.encoder_poll_times) >= 50:
                avg_poll_time = np.mean(self.encoder_poll_times[-50:])
                max_poll_time = np.max(self.encoder_poll_times[-50:])
                min_poll_time = np.min(self.encoder_poll_times[-50:])
                
                if len(self.encoder_poll_intervals) >= 50:
                    avg_interval = np.mean(self.encoder_poll_intervals[-50:])
                    min_interval = np.min(self.encoder_poll_intervals[-50:])
                    max_interval = np.max(self.encoder_poll_intervals[-50:])
                    avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                    min_freq = 1.0 / max_interval if max_interval > 0 else 0  # min interval = max freq
                    max_freq = 1.0 / min_interval if min_interval > 0 else 0  # max interval = min freq
                else:
                    avg_interval = 0.0
                    min_interval = 0.0
                    max_interval = 0.0
                    avg_freq = 0.0
                    min_freq = 0.0
                    max_freq = 0.0
                
                expected_freq = robot_config.control_frequency
                print(f"\n[ENCODER STATS (last 50 polls)]")
                print(f"  Read time: avg={avg_poll_time*1000:.2f}ms, min={min_poll_time*1000:.2f}ms, max={max_poll_time*1000:.2f}ms")
                if len(self.encoder_poll_intervals) >= 50:
                    print(f"  Interval: avg={avg_interval*1000:.2f}ms ({avg_freq:.1f}Hz), "
                          f"min={min_interval*1000:.2f}ms ({max_freq:.1f}Hz), "
                          f"max={max_interval*1000:.2f}ms ({min_freq:.1f}Hz)")
                else:
                    print(f"  Interval: (insufficient data, need {50 - len(self.encoder_poll_intervals)} more polls)")
                print(f"  Expected frequency: {expected_freq}Hz (control loop rate)")
                print()
    
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
        """Read camera, detect markers, poll encoders, and update trajectory."""
        
        # Handle GUI commands (from parent class)
        self._handle_control_input()
        
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
        
        # 2. Robot Recording with encoder-based state
        # Note: We override the base class recording to use encoder-based state instead of MuJoCo estimates
        if self.is_recording:
            if self.recording_start_time is None:
                self.recording_start_time = time.perf_counter()
            
            # IMPORTANT: Capture IK solver's target EE pose BEFORE syncing with encoders
            # This represents the target/commanded pose from the IK solver (what we're trying to achieve)
            # After we sync with encoders, this state will be overwritten
            ik_ee_position, ik_ee_orientation_quat_wxyz = self._get_current_ee_pose()
            ee_pose_target = np.concatenate([ik_ee_position, ik_ee_orientation_quat_wxyz])
            
            # NOW poll encoders and sync MuJoCo (this will overwrite self.configuration.q)
            # This ensures we record the true robot state, not MuJoCo estimates
            self._poll_encoders()
            
            # Use encoder-derived state instead of MuJoCo estimates
            if self.latest_joint_angles_from_encoders is not None:
                # State: joint angles from encoders (5 joints + gripper)
                state = self.latest_joint_angles_from_encoders.copy()
            else:
                # Fallback to MuJoCo if encoder sync failed
                state = np.concatenate([self.configuration.q[:5], [gripper_target]])
            
            # Action: velocity commands + gripper
            action = np.concatenate([velocity_world, angular_velocity_world, [gripper_target]])
            
            self.trajectory.append({
                'timestamp': time.perf_counter() - self.recording_start_time,
                'state': state.copy(),
                'action': action.copy(),
                'ee_pose_target': ee_pose_target.copy()
            })
            
            # Augment with encoder values and ArUco observations
            if self.trajectory:
                current_step = self.trajectory[-1]
                
                # Store encoder values (raw hardware readings)
                encoder_array = np.array([
                    self.latest_encoder_values.get(mid, 0) 
                    for mid in robot_config.motor_ids
                ])
                current_step['encoder_values'] = encoder_array
                
                # Store encoder-based FK EE pose (from separate FK-only MuJoCo instance)
                # This is the "ground truth" EE pose computed purely from encoder readings
                # Format: [x, y, z, qw, qx, qy, qz]
                if self.latest_ee_pose_from_encoders_fk is not None:
                    fk_position, fk_orientation = self.latest_ee_pose_from_encoders_fk
                    ee_pose_encoder = np.concatenate([fk_position, fk_orientation])
                    current_step['ee_pose_encoder'] = ee_pose_encoder.copy()
                else:
                    # Fallback: use zeros if FK not available
                    current_step['ee_pose_encoder'] = np.zeros(7)
                
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
        
        # Print encoder read statistics from robot driver
        if self.robot_driver:
            encoder_stats = self.robot_driver.get_encoder_read_stats()
            if encoder_stats['total_reads'] > 0:
                print(f"\n{'='*60}")
                print(f"ENCODER READ STATISTICS (Hardware Communication)")
                print(f"{'='*60}")
                print(f"  Total reads: {encoder_stats['total_reads']}")
                print(f"  Successful: {encoder_stats['successful_reads']} ({encoder_stats.get('success_rate', 0):.1f}%)")
                print(f"  Failed: {encoder_stats['failed_reads']} ({encoder_stats.get('failure_rate', 0):.1f}%)")
                print(f"  Partial reads: {encoder_stats['partial_reads']} (some motors succeeded)")
                print(f"  Timeouts: {encoder_stats['timeout_reads']}")
                if encoder_stats['failed_reads'] > 0:
                    print(f"  Timeout rate (of failures): {encoder_stats.get('timeout_rate_of_failures', 0):.1f}%")
                print(f"{'='*60}\n")
        
        # Print final encoder polling stats
        if self.encoder_poll_count > 0:
            avg_poll_time = np.mean(self.encoder_poll_times)
            max_poll_time = np.max(self.encoder_poll_times)
            min_poll_time = np.min(self.encoder_poll_times)
            
            if len(self.encoder_poll_intervals) > 0:
                avg_interval = np.mean(self.encoder_poll_intervals)
                min_interval = np.min(self.encoder_poll_intervals)
                max_interval = np.max(self.encoder_poll_intervals)
                avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                min_freq = 1.0 / max_interval if max_interval > 0 else 0
                max_freq = 1.0 / min_interval if min_interval > 0 else 0
            else:
                avg_interval = 0.0
                avg_freq = 0.0
                min_freq = 0.0
                max_freq = 0.0
            
            expected_freq = robot_config.control_frequency
            
            print(f"\n{'='*60}")
            print(f"ENCODER POLLING STATISTICS (Final)")
            print(f"{'='*60}")
            print(f"  Total polls: {self.encoder_poll_count}")
            print(f"  Read time: avg={avg_poll_time*1000:.2f}ms, min={min_poll_time*1000:.2f}ms, max={max_poll_time*1000:.2f}ms")
            if len(self.encoder_poll_intervals) > 0:
                print(f"  Poll interval: avg={avg_interval*1000:.2f}ms, min={min_interval*1000:.2f}ms, max={max_interval*1000:.2f}ms")
                print(f"  Actual frequency: avg={avg_freq:.1f}Hz, min={min_freq:.1f}Hz, max={max_freq:.1f}Hz")
            print(f"  Expected frequency: {expected_freq}Hz (control loop rate)")
            if avg_freq > 0 and expected_freq > 0:
                efficiency = (avg_freq / expected_freq) * 100
                print(f"  Efficiency: {efficiency:.1f}% of expected")
            print(f"{'='*60}\n")
        
        super().shutdown()

def main():
    parser = argparse.ArgumentParser(description='WX200 Teleop with Camera and Encoder Polling')
    parser.add_argument('--record', action='store_true', help='(Deprecated) Recording is controlled via GUI')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--camera-id', type=int, default=None, help='Camera device ID (defaults to robot_config.camera_id)')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    args = parser.parse_args()
    
    # Create and run
    controller = TeleopCameraControlEncoders(
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
