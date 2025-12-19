"""
Gym environment for WX200 robot (Compact Version).

Uses RobotHardware for direct control and OpenCV/ArUco for observations.
"""
import time
import cv2
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from .robot_config import robot_config
from .robot_hardware import RobotHardware
from .camera import Camera, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix
from .profiling import ArUcoProfiler

AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale

class WX200GymEnv(gym.Env):
    """
    Gym environment for WX200 robot with low-dimensional state observations.
    """
    # Class-level variable to track which instance has hardware authority
    _hardware_authority = None
    
    def __init__(self, max_episode_length=1000, control_frequency=None, 
                 camera_id=None, width=None, height=None, fps=None, 
                 enable_aruco=True, show_video=False, show_axes=True):
        self.max_episode_length = max_episode_length
        self.control_frequency = control_frequency or robot_config.control_frequency
        self.dt = 1.0 / self.control_frequency
        self.enable_aruco = enable_aruco
        self.show_video = show_video
        self.show_axes = show_axes
        self.min_step_interval = 1.0 / self.control_frequency
        
        self.has_authority = False
        self.robot_hardware = None # Instance of RobotHardware
        self._hardware_initialized = False
        
        # Camera/ArUco init
        self.camera = None
        self.last_frame = None
        self.estimator = None
        self.detector = None
        self.cam_matrix = None
        self.dist_coeffs = None
        self.camera_width = width if width is not None else robot_config.camera_width
        self.camera_height = height if height is not None else robot_config.camera_height
        self.aruco_obs_dict = {}
        self.prev_aruco_obs_dict = {}
        
        self.profiler = ArUcoProfiler() if robot_config.profiler_window_size > 0 else None
        
        # Spaces
        self.action_space = Box(
            low=np.array([-1.0]*7),
            high=np.array([1.0]*7),
            dtype=np.float32
        )
        
        # Obs: aruco_obj_in_world (7D) + robot_state (6D) + ee_pose_debug (7D) = 20D
        self.observation_space = Box(
            low=np.full(20, -np.inf, dtype=np.float32),
            high=np.full(20, np.inf, dtype=np.float32),
            dtype=np.float32
        )
        
        self.episode_step = 0
        self.last_step_time = None
    
    def _initialize_hardware(self, camera_id=None, width=None, height=None, fps=None):
        """Lazily initialize robot and camera hardware."""
        if self._hardware_initialized:
            return
            
        # Authority Check
        if WX200GymEnv._hardware_authority is None:
            WX200GymEnv._hardware_authority = self
            self.has_authority = True
        elif WX200GymEnv._hardware_authority == self:
            self.has_authority = True
        else:
            raise RuntimeError("Hardware authority already claimed by another instance.")
        
        try:
            print("[WX200GymEnv] Initializing hardware...")
            self.robot_hardware = RobotHardware()
            self.robot_hardware.initialize()
            
            if self.enable_aruco:
                self._setup_camera(camera_id, width, height, fps)
                
            self._hardware_initialized = True
        except Exception as e:
            if self.has_authority:
                WX200GymEnv._hardware_authority = None
                self.has_authority = False
            raise e

    def _setup_camera(self, camera_id=None, width=None, height=None, fps=None):
        if not self.has_authority: return
        
        camera_id = camera_id if camera_id is not None else robot_config.camera_id
        width = width if width is not None else robot_config.camera_width
        height = height if height is not None else robot_config.camera_height
        fps = fps if fps is not None else robot_config.camera_fps
        
        self.camera = Camera(device=camera_id, width=width, height=height, fps=fps)
        self.camera.start()
        
        self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
        self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(width, height)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        print(f"[WX200GymEnv] Camera started: {camera_id} @ {width}x{height}")

    def _compute_relative_pose(self, r_ref, t_ref, r_tgt, t_tgt):
        if r_ref is None or t_ref is None or r_tgt is None or t_tgt is None:
            return np.zeros(3), np.array([1., 0., 0., 0.])
        r_rel, t_rel = self.estimator.get_relative_pose(r_ref, t_ref, r_tgt, t_tgt)
        R_rel, _ = cv2.Rodrigues(r_rel)
        quat_xyzw = R.from_matrix(R_rel).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return t_rel.flatten(), quat_wxyz

    def _get_aruco_observations(self):
        # Init dict
        if self.prev_aruco_obs_dict:
            obs = {k: v.copy() for k, v in self.prev_aruco_obs_dict.items()}
        else:
            obs = {
                 'aruco_ee_in_world': np.zeros(7),
                 'aruco_object_in_world': np.zeros(7),
                 'aruco_visibility': np.zeros(3)
            }
            
        if not self.has_authority or not self.camera or not self.enable_aruco:
            return obs
        
        loop_start = time.perf_counter()
        ret, frame = self.camera.read()
        if not ret: return obs
        
        detect_start = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        detect_time = time.perf_counter() - detect_start
        
        pose_start = time.perf_counter()
        # Process tags
        r_world, t_world = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id)
        r_obj, t_obj = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id)
        # Note: We use physical EE marker if available, OR we can use kinematic EE.
        # But here we stick to markers as per original code.
        r_ee, t_ee = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id)
        
        # Visibility
        if ids is not None:
            ids_arr = np.atleast_1d(ids).ravel()
            obs['aruco_visibility'][0] = 1.0 if np.any(ids_arr == robot_config.aruco_world_id) else 0.0
            obs['aruco_visibility'][1] = 1.0 if np.any(ids_arr == robot_config.aruco_object_id) else 0.0
            # EE marker visibility (optional if we use kinematics for EE)
            
        # Draw markers on frame (always, for recording/rendering)
        if ids is not None: cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if self.show_axes:
            if r_world is not None: cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_world, t_world, AXIS_LENGTH)
            if r_obj is not None: cv2.drawFrameAxes(frame, self.cam_matrix, self.dist_coeffs, r_obj, t_obj, AXIS_LENGTH)
        
        self.last_frame = frame # Store annotated frame
        
        if self.show_video:
            cv2.imshow('Gym View', cv2.resize(frame, (self.camera_width//2, self.camera_height//2)))
            cv2.waitKey(1)
            
        # Compute Poses
        # Object in World
        if r_world is not None and r_obj is not None:
             pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
             obs['aruco_object_in_world'] = np.concatenate([pos, quat])
             
        self.prev_aruco_obs_dict = {k: v.copy() for k, v in obs.items()}
        
        pose_time = time.perf_counter() - pose_start
        total_time = time.perf_counter() - loop_start
        
        # Profiling
        if self.profiler:
            interval = None
            if self.profiler.last_poll_timestamp is not None:
                interval = loop_start - self.profiler.last_poll_timestamp
            self.profiler.last_poll_timestamp = loop_start
            
            self.profiler.record_poll_iteration(total_time, detect_time, pose_time, interval)
            
            # Print stats every 50 frames
            if self.profiler.poll_count % 50 == 0:
                self.profiler.print_periodic_stats()
                
        return obs

    def _get_observation(self):
        # 1. ArUco
        self.aruco_obs_dict = self._get_aruco_observations()
        
        # 2. Robot State & EE Pose
        if not self.has_authority or not self._hardware_initialized:
            robot_state = np.zeros(6)
            ee_pose_debug = np.zeros(7)
        else:
            robot_state = self.robot_hardware.configuration.q[:6] 
            # Get FK EE pose
            # Current Config is updated in execute_command, but we might want to sync with MuJoCo just in case?
            # Or just use current target? 
            # Better to use actual forward kinematics from current config
            # (Note: RobotHardware's execute_command updates configuration with INTEGRATION, 
            # but ideally we'd sync with actual encoders if we want "real" state.
            # However, for high frequency control, we often rely on command state or 
            # periodic sync. In `wx200_env_utils`, `_get_current_ee_pose` calls `mj_forward`.
            # We can do the same here using RobotHardware's model/data/config.
            
            self.robot_hardware.data.qpos[:5] = self.robot_hardware.configuration.q[:5]
            if len(self.robot_hardware.data.qpos) > 5:
                self.robot_hardware.data.qpos[5] = self.robot_hardware.configuration.q[5]
            mujoco.mj_forward(self.robot_hardware.model, self.robot_hardware.data)
            
            site_id = mujoco.mj_name2id(self.robot_hardware.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
            site = self.robot_hardware.data.site(site_id)
            quat = R.from_matrix(site.xmat.reshape(3,3)).as_quat() #[x,y,z,w]
            ee_pose_debug = np.concatenate([site.xpos, [quat[3], quat[0], quat[1], quat[2]]]) #[w,x,y,z]

        obs = np.concatenate([
            self.aruco_obs_dict['aruco_object_in_world'], # 7D
            robot_state, # 6D
            ee_pose_debug # 7D
        ])
        return obs.astype(np.float32)

    def _denormalize_action(self, action):
        # Denormalize logic (Copied from wx200_env_utils)
        action_low = np.array([
            -robot_config.velocity_scale, -robot_config.velocity_scale, -robot_config.velocity_scale,
            -robot_config.angular_velocity_scale, -robot_config.angular_velocity_scale, -robot_config.angular_velocity_scale,
            robot_config.gripper_open_pos,
        ])
        action_high = np.array([
            robot_config.velocity_scale, robot_config.velocity_scale, robot_config.velocity_scale,
            robot_config.angular_velocity_scale, robot_config.angular_velocity_scale, robot_config.angular_velocity_scale,
            robot_config.gripper_closed_pos,
        ])
        
        denormalized = (action + 1.0) / 2.0 * (action_high - action_low) + action_low
        return denormalized[:3], denormalized[3:6], denormalized[6]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        self.last_step_time = None
        self.prev_aruco_obs_dict = {}
        
        if not self._hardware_initialized:
            try: self._initialize_hardware()
            except RuntimeError as e:
                print(f"Hardware init failed: {e}")
                return self._get_observation(), {}
        
        if not self.has_authority: return self._get_observation(), {}
        
        # Reset Gripper Logic (Simplified version of wx200_env_utils)
        print("[WX200GymEnv] Resetting Gripper...")
        gripper_id = robot_config.motor_ids[-1]
        try:
            self.robot_hardware.robot_driver.reboot_motor(gripper_id)
            time.sleep(0.5)
            self.robot_hardware.robot_driver.send_motor_positions({gripper_id: robot_config.gripper_encoder_max})
            time.sleep(1.0)
        except Exception as e:
            print(f"Gripper reset warning: {e}")
            
        # Move Home
        print("[WX200GymEnv] Moving Home...")
        # Get home motor positions using RobotHardware's translator
        # We can implement a helper in RobotHardware or just do it here loosely since we have access
        # Better: call a method on RobotHardware if it existed, but we can compute it.
        # Actually RobotConfig has `startup_home_positions`
        if robot_config.startup_home_positions:
             home_pos = {mid: pos for mid, pos in zip(robot_config.motor_ids, robot_config.startup_home_positions)}
             # Force gripper open in home
             home_pos[gripper_id] = robot_config.gripper_encoder_max
             self.robot_hardware.robot_driver.move_to_home(home_pos, velocity_limit=robot_config.velocity_limit)
        
        # Sync
        robot_encoders = self.robot_hardware.robot_driver.read_all_encoders()
        # We need to sync physics
        from .robot_kinematics import sync_robot_to_mujoco
        sync_robot_to_mujoco(robot_encoders, self.robot_hardware.translator, 
                             self.robot_hardware.model, self.robot_hardware.data, self.robot_hardware.configuration)
        
        # Reset controller target
        # RobotHardware.initialize() does this, but we need to do it on reset too
        # We need access to `actual_position` from sync. sync_robot_to_mujoco returns it.
        # Let's call it properly.
        _, act_pos, act_quat = sync_robot_to_mujoco(robot_encoders, self.robot_hardware.translator, 
                             self.robot_hardware.model, self.robot_hardware.data, self.robot_hardware.configuration)
                             
        self.robot_hardware.robot_controller.reset_pose(act_pos, act_quat)
        self.robot_hardware.robot_controller.end_effector_task.set_target(
             self.robot_hardware.robot_controller.get_target_pose()
        )
        self.robot_hardware.gripper_current_position = robot_config.gripper_open_pos
        
        return self._get_observation(), {}

    def step(self, action):
        # Rate Limiting
        current_time = time.perf_counter()
        if self.last_step_time is not None:
             elapsed = current_time - self.last_step_time
             if elapsed < self.min_step_interval:
                 time.sleep(self.min_step_interval - elapsed)
        self.last_step_time = time.perf_counter()
        
        # Denormalize
        vel, ang_vel, grip = self._denormalize_action(action)
        
        # Execute
        if self.has_authority and self._hardware_initialized:
            self.robot_hardware.execute_command(vel, ang_vel, grip, self.dt)
            
        obs = self._get_observation()
        reward = 0.0
        terminated = False
        truncated = False
        
        self.episode_step += 1
        if self.episode_step >= self.max_episode_length:
            truncated = True
            
        return obs, reward, terminated, truncated, {}
