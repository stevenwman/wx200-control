"""
Gym environment for WX200 robot (Compact Version).

Uses RobotHardware for direct control and OpenCV/ArUco for observations.
"""
import time
from collections import deque
import threading
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


class RateLimiter:
    """Simple rate limiter for background threads."""

    def __init__(self, frequency, warn=False):
        self.period = 1.0 / frequency
        self.last_time = None
        self.warn = warn

    def sleep(self):
        """Sleep to maintain target frequency."""
        now = time.perf_counter()
        if self.last_time is not None:
            elapsed = now - self.last_time
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.warn and sleep_time < -0.01:  # Warn if more than 10ms late
                print(f"⚠️  Rate limiter: {-sleep_time*1000:.1f}ms late")
        self.last_time = time.perf_counter()

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

        # ArUco background thread (runs at camera FPS)
        self._aruco_polling_active = False
        self._aruco_poll_thread = None
        self._aruco_lock = threading.Lock()  # Protects latest_aruco_obs updates
        self._frame_lock = threading.Lock()  # Protects last_frame updates
        self.latest_aruco_obs = {
            'aruco_ee_in_world': np.zeros(7),
            'aruco_object_in_world': np.zeros(7),
            'aruco_ee_in_object': np.zeros(7),
            'aruco_object_in_ee': np.zeros(7),
            'aruco_visibility': np.zeros(3)
        }

        self.profiler = ArUcoProfiler() if robot_config.profiler_window_size > 0 else None
        self._step_timing_samples = deque()
        
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

        try:
            self.camera = Camera(device=camera_id, width=width, height=height, fps=fps)
            self.camera.start()

            self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
            self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(width, height)
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
            print(f"[WX200GymEnv] Camera started: {camera_id} @ {width}x{height}")

            # Start ArUco background polling thread at camera FPS
            self._start_aruco_polling()
        except Exception as e:
            print(f"⚠️  Camera initialization failed: {e}")
            print(f"⚠️  ArUco tracking will be disabled. Robot control will still work.")
            self.camera = None
            self.enable_aruco = False

    def _start_aruco_polling(self):
        """Start background thread for high-frequency ArUco polling at camera FPS."""
        if self.camera is None:
            return

        self._aruco_polling_active = True
        self._aruco_poll_thread = threading.Thread(target=self._aruco_poll_loop, daemon=True)
        self._aruco_poll_thread.start()
        print(f"✓ Started ArUco polling thread at {robot_config.camera_fps} Hz")

    def _aruco_poll_loop(self):
        """Background thread that polls camera and detects ArUco markers at camera FPS."""
        rate_limiter = RateLimiter(frequency=robot_config.camera_fps, warn=False)

        while self._aruco_polling_active and self.camera:
            loop_start = time.perf_counter()

            # Track interval since last poll
            interval = None
            if self.profiler and self.profiler.last_poll_timestamp is not None:
                interval = loop_start - self.profiler.last_poll_timestamp
                if interval >= 1.0:  # Skip large intervals (gaps)
                    interval = None

            # Read camera frame
            ret, frame = self.camera.read()
            if not ret:
                rate_limiter.sleep()
                continue

            # Convert to grayscale and detect markers
            detect_start = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            detect_time = time.perf_counter() - detect_start

            # Process tags
            pose_start = time.perf_counter()
            r_world, t_world = self.estimator.process_tag(
                corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id
            )
            r_obj, t_obj = self.estimator.process_tag(
                corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id
            )
            r_ee, t_ee = self.estimator.process_tag(
                corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id
            )

            # Update visibility flags
            world_visible = False
            object_visible = False
            ee_visible = False
            if ids is not None:
                ids_arr = np.atleast_1d(ids).ravel()
                world_visible = np.any(ids_arr == robot_config.aruco_world_id)
                object_visible = np.any(ids_arr == robot_config.aruco_object_id)
                ee_visible = np.any(ids_arr == robot_config.aruco_ee_id)

            # Compute relative poses
            obs = {
                'aruco_ee_in_world': np.zeros(7),
                'aruco_object_in_world': np.zeros(7),
                'aruco_ee_in_object': np.zeros(7),
                'aruco_object_in_ee': np.zeros(7),
                'aruco_visibility': np.zeros(3)
            }

            obs['aruco_visibility'][0] = 1.0 if world_visible else 0.0
            obs['aruco_visibility'][1] = 1.0 if object_visible else 0.0
            obs['aruco_visibility'][2] = 1.0 if ee_visible else 0.0

            # Compute relative poses
            pos, quat = self._compute_relative_pose(r_world, t_world, r_ee, t_ee)
            obs['aruco_ee_in_world'] = np.concatenate([pos, quat])

            pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
            obs['aruco_object_in_world'] = np.concatenate([pos, quat])

            pos, quat = self._compute_relative_pose(r_obj, t_obj, r_ee, t_ee)
            obs['aruco_ee_in_object'] = np.concatenate([pos, quat])

            pos, quat = self._compute_relative_pose(r_ee, t_ee, r_obj, t_obj)
            obs['aruco_object_in_ee'] = np.concatenate([pos, quat])
            pose_time = time.perf_counter() - pose_start

            # Update latest observations (thread-safe)
            with self._aruco_lock:
                self.latest_aruco_obs = {k: v.copy() for k, v in obs.items()}
            # Store latest frame for rendering (separate lock to reduce contention)
            with self._frame_lock:
                self.last_frame = frame.copy()

            # Visualization (in background thread to avoid conflicts)
            if self.show_video:
                vis_frame = frame.copy()
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids)
                if self.show_axes:
                    if r_world is not None:
                        cv2.drawFrameAxes(vis_frame, self.cam_matrix, self.dist_coeffs, r_world, t_world, AXIS_LENGTH)
                    if r_obj is not None:
                        cv2.drawFrameAxes(vis_frame, self.cam_matrix, self.dist_coeffs, r_obj, t_obj, AXIS_LENGTH)
                    if r_ee is not None:
                        cv2.drawFrameAxes(vis_frame, self.cam_matrix, self.dist_coeffs, r_ee, t_ee, AXIS_LENGTH)

                disp = cv2.resize(
                    vis_frame,
                    (robot_config.camera_width // 2, robot_config.camera_height // 2)
                )
                cv2.imshow('Robot Camera View', disp)
                cv2.waitKey(1)

            # Track total poll time and record in profiler
            if self.profiler:
                total_poll_time = time.perf_counter() - loop_start
                self.profiler.record_poll_iteration(
                    total_poll_time, detect_time, pose_time, interval
                )
                self.profiler.last_poll_timestamp = loop_start

                # Print periodic stats (every 300 polls = ~10 seconds at 30 Hz)
                if self.profiler.poll_count % 300 == 0:
                    self.profiler.print_periodic_stats()

            rate_limiter.sleep()

    def _compute_relative_pose(self, r_ref, t_ref, r_tgt, t_tgt):
        if r_ref is None or t_ref is None or r_tgt is None or t_tgt is None:
            return np.zeros(3), np.array([1., 0., 0., 0.])
        r_rel, t_rel = self.estimator.get_relative_pose(r_ref, t_ref, r_tgt, t_tgt)
        R_rel, _ = cv2.Rodrigues(r_rel)
        quat_xyzw = R.from_matrix(R_rel).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return t_rel.flatten(), quat_wxyz

    def _get_aruco_observations(self):
        """
        Get latest ArUco observations from background thread (thread-safe).

        The background thread polls at camera FPS (30 Hz), while this method
        reads the latest observations at control frequency (10 Hz).
        """
        if not self.has_authority or not self.camera or not self.enable_aruco:
            return {
                'aruco_ee_in_world': np.zeros(7),
                'aruco_object_in_world': np.zeros(7),
                'aruco_ee_in_object': np.zeros(7),
                'aruco_object_in_ee': np.zeros(7),
                'aruco_visibility': np.zeros(3)
            }

        # Read latest observations from background thread (thread-safe)
        with self._aruco_lock:
            obs = {k: v.copy() for k, v in self.latest_aruco_obs.items()}

        return obs

    def get_last_frame_copy(self):
        """Thread-safe access to the latest camera frame."""
        if not self.has_authority or not self.enable_aruco:
            return None
        with self._frame_lock:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()

    def poll_encoders(self, outer_loop_start_time=None):
        """Refresh encoder cache (outer-loop only)."""
        if self.robot_hardware and self._hardware_initialized:
            self.robot_hardware.poll_encoders(outer_loop_start_time=outer_loop_start_time)

    def reset_gripper(self):
        """Reboot/open the gripper using hardware layer."""
        if not self._hardware_initialized:
            try:
                self._initialize_hardware()
            except Exception as e:
                print(f"Hardware init failed: {e}")
                return False
        if not self.has_authority:
            return False
        self.robot_hardware.reset_gripper()
        return True

    def home(self):
        """Move to home pose and sync controller state."""
        if not self._hardware_initialized:
            try:
                self._initialize_hardware()
            except Exception as e:
                print(f"Hardware init failed: {e}")
                return False
        if not self.has_authority:
            return False
        self.robot_hardware.home()
        return True

    def emergency_stop(self):
        """Disable torque immediately via hardware layer."""
        if self.robot_hardware and self._hardware_initialized:
            self.robot_hardware.emergency_stop()

    def _get_observation(self):
        # 1. ArUco
        self.aruco_obs_dict = self._get_aruco_observations()

        # 2. Robot State & EE Pose
        if not self.has_authority or not self._hardware_initialized:
            robot_state = np.zeros(6)
            ee_pose_debug = np.zeros(7)
        else:
            # Try to use encoder state if available (more accurate than commanded state)
            encoder_state = self.robot_hardware.get_encoder_state()
            if encoder_state['joint_angles'] is not None:
                robot_state = encoder_state['joint_angles'][:6]
            else:
                # Fall back to commanded state from configuration
                robot_state = self.robot_hardware.configuration.q[:6]

            # Use encoder-based EE pose if available
            if encoder_state['ee_pose'] is not None:
                ee_position, ee_quat_wxyz = encoder_state['ee_pose']
                ee_pose_debug = np.concatenate([ee_position, ee_quat_wxyz])
            else:
                # Fall back to FK from commanded configuration
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
        
        if not self._hardware_initialized:
            try:
                self._initialize_hardware()
            except Exception as e:
                print(f"Hardware init failed: {e}")
                return self._get_observation(), {}
        
        if not self.has_authority: return self._get_observation(), {}
        
        # Reset gripper and move home via hardware layer
        print("[WX200GymEnv] Resetting Gripper...")
        self.robot_hardware.reset_gripper()

        print("[WX200GymEnv] Moving Home...")
        self.robot_hardware.home()
        
        return self._get_observation(), {}

    def step(self, action):
        # NO rate limiting here - let motor commands run as fast as possible (like compact_code's 120Hz inner loop)
        # The collector controls recording frequency, not the env
        step_start = time.perf_counter()

        # Denormalize
        t_denorm_start = time.perf_counter()
        vel, ang_vel, grip = self._denormalize_action(action)
        t_denorm = time.perf_counter() - t_denorm_start

        # Execute
        t_exec_start = time.perf_counter()
        if self.has_authority and self._hardware_initialized:
            self.robot_hardware.execute_command(vel, ang_vel, grip, self.dt)
        t_exec = time.perf_counter() - t_exec_start

        # NOTE: Encoder polling removed from here - collector does it at 10Hz instead
        # This allows motor commands to run at high frequency (~100Hz) without 10ms blocking
        t_encoder = 0.0

        # Get observation
        t_obs_start = time.perf_counter()
        obs = self._get_observation()
        t_obs = time.perf_counter() - t_obs_start

        reward = 0.0
        terminated = False
        truncated = False

        self.episode_step += 1
        if self.episode_step >= self.max_episode_length:
            truncated = True

        # Build info dict with encoder data and raw ArUco observations
        t_info_start = time.perf_counter()
        info = {}
        if self.has_authority and self._hardware_initialized:
            encoder_state = self.robot_hardware.get_encoder_state()
            info['encoder_values'] = encoder_state['encoder_values']
            info['qpos'] = encoder_state['joint_angles']
            info['ee_pose_fk'] = np.concatenate([
                encoder_state['ee_pose'][0], encoder_state['ee_pose'][1]
            ]) if encoder_state['ee_pose'] is not None else None
            info['raw_aruco'] = self.aruco_obs_dict.copy()
        t_info = time.perf_counter() - t_info_start

        # Store timing info for debugging (summary by default, more frequent if verbose)
        now = time.perf_counter()
        self._step_timing_samples.append((now, t_denorm, t_exec, t_encoder, t_obs, t_info))
        window_sec = robot_config.control_perf_window_sec
        if window_sec and window_sec > 0:
            cutoff = now - window_sec
            while self._step_timing_samples and self._step_timing_samples[0][0] < cutoff:
                self._step_timing_samples.popleft()

        interval = 50 if robot_config.verbose_profiling else robot_config.control_perf_stats_interval
        if interval and self.episode_step % interval == 0 and self._step_timing_samples:
            denorm_avg = np.mean([s[1] for s in self._step_timing_samples])
            exec_avg = np.mean([s[2] for s in self._step_timing_samples])
            enc_avg = np.mean([s[3] for s in self._step_timing_samples])
            obs_avg = np.mean([s[4] for s in self._step_timing_samples])
            info_avg = np.mean([s[5] for s in self._step_timing_samples])
            print(f"    [ENV.STEP] Avg~{window_sec:.1f}s Denorm={denorm_avg*1000:.1f}ms, "
                  f"Exec={exec_avg*1000:.1f}ms, Encoder={enc_avg*1000:.1f}ms, "
                  f"Obs={obs_avg*1000:.1f}ms, Info={info_avg*1000:.1f}ms")

        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources including background thread."""
        # Stop ArUco polling thread
        self._aruco_polling_active = False
        if self._aruco_poll_thread is not None:
            self._aruco_poll_thread.join(timeout=1.0)

        # Close camera
        if self.camera:
            if hasattr(self.camera, 'release'):
                self.camera.release()
            elif hasattr(self.camera, 'stop'):
                self.camera.stop()

        # Clean up hardware
        if self.robot_hardware:
            self.robot_hardware.shutdown()

        # Release hardware authority
        if self.has_authority:
            WX200GymEnv._hardware_authority = None
            self.has_authority = False

        cv2.destroyAllWindows()
