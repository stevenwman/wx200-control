"""
Gym environment utilities for WX200 robot.

Similar structure to robomimic_env_utils.py for compatibility with learning algorithms.
"""
import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from robot_control.robot_config import robot_config
from robot_control.robot_control_base import RobotControlBase, get_sim_home_pose
from robot_control.robot_joint_to_motor import sync_robot_to_mujoco, encoder_to_gripper_position
from robot_control.robot_startup import get_home_motor_positions
from camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix

AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale


class WX200GymEnvBase(gym.Env):
    """
    Base gym environment for WX200 robot with low-dimensional state observations.
    
    This is the core environment that handles robot control and ArUco tracking.
    """
    
    def __init__(self, max_episode_length=1000, control_frequency=None, 
                 camera_id=None, width=None, height=None, fps=None, 
                 enable_aruco=True, show_video=True, show_axes=True):
        """
        Initialize WX200 gym environment.
        
        Args:
            max_episode_length: Maximum steps per episode
            control_frequency: Control loop frequency (Hz). If None, uses robot_config.
            camera_id: Camera device ID (None uses config default)
            width: Camera width (None uses config default)
            height: Camera height (None uses config default)
            fps: Camera FPS (None uses config default)
            enable_aruco: Enable ArUco tracking (default True)
            show_video: Show camera preview window (default True)
            show_axes: Draw ArUco axes in preview (default True)
        """
        self.max_episode_length = max_episode_length
        self.control_frequency = control_frequency or robot_config.control_frequency
        self.dt = 1.0 / self.control_frequency
        self.enable_aruco = enable_aruco
        self.show_video = show_video
        self.show_axes = show_axes
        
        # Initialize robot control base
        self.robot_base = RobotControlBase(control_frequency=self.control_frequency)
        self.robot_base.initialize()
        
        # Setup camera and ArUco (optional)
        self.camera = None
        self.estimator = None
        self.detector = None
        self.cam_matrix = None
        self.dist_coeffs = None
        self.camera_width = width if width is not None else robot_config.camera_width
        self.camera_height = height if height is not None else robot_config.camera_height
        self.aruco_obs_dict = {}  # Store ArUco observations as dict
        self.prev_aruco_obs_dict = {}  # Store previous ArUco observations for lost tracking
        
        if self.enable_aruco:
            self._setup_camera(camera_id, width, height, fps)
        
        # Setup spaces
        self.action_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation: aruco_obj_in_world (7D) + aruco_obj_in_ee (7D) + aruco_ee_in_world (7D) + gripper (1D) = 22D
        obs_example = self._get_observation()
        self.observation_space = Box(
            low=np.full_like(obs_example, -np.inf),
            high=np.full_like(obs_example, np.inf),
            dtype=np.float32
        )
        
        self.episode_step = 0
        self.episode_return = 0.0
    
    def _setup_camera(self, camera_id=None, width=None, height=None, fps=None):
        """Initialize camera and ArUco detector."""
        try:
            camera_id = camera_id if camera_id is not None else robot_config.camera_id
            width = width if width is not None else robot_config.camera_width
            height = height if height is not None else robot_config.camera_height
            fps = fps if fps is not None else robot_config.camera_fps
            
            self.camera = Camera(device=camera_id, width=width, height=height, fps=fps)
            self.camera.start()
            
            self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
            self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(width, height)
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            print(f"✓ Camera started: Device {camera_id} @ {width}x{height}")
            print("✓ ArUco Estimator ready")
        except Exception as e:
            print(f"⚠️  Warning: Failed to initialize camera/ArUco: {e}")
            self.camera = None
            self.enable_aruco = False
    
    def _compute_relative_pose(self, r_ref, t_ref, r_tgt, t_tgt):
        """Compute target pose relative to reference frame. Returns (pos, quat_wxyz) or (zeros, zeros)."""
        if r_ref is None or t_ref is None or r_tgt is None or t_tgt is None:
            return np.zeros(3), np.array([1., 0., 0., 0.])
        
        r_rel, t_rel = self.estimator.get_relative_pose(r_ref, t_ref, r_tgt, t_tgt)
        R_rel, _ = cv2.Rodrigues(r_rel)
        quat_xyzw = R.from_matrix(R_rel).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return t_rel.flatten(), quat_wxyz
    
    def _get_aruco_observations_dict(self):
        """
        Get ArUco observations as a dictionary (same structure as collect demo).
        Preserves previous timestep values when tracking is lost.
        
        Returns:
            dict: ArUco observations with keys:
                - 'aruco_ee_in_world': 7D [x, y, z, qw, qx, qy, qz]
                - 'aruco_object_in_world': 7D
                - 'aruco_ee_in_object': 7D
                - 'aruco_object_in_ee': 7D
                - 'aruco_visibility': 3D [world, object, ee]
        """
        # Initialize with previous values if available, otherwise zeros
        if self.prev_aruco_obs_dict:
            obs = {k: v.copy() for k, v in self.prev_aruco_obs_dict.items()}
        else:
            obs = {
                'aruco_ee_in_world': np.zeros(7),
                'aruco_object_in_world': np.zeros(7),
                'aruco_ee_in_object': np.zeros(7),
                'aruco_object_in_ee': np.zeros(7),
                'aruco_visibility': np.zeros(3)
            }
        
        if not self.camera or not self.enable_aruco:
            return obs
        
        ret, frame = self.camera.read()
        if not ret:
            return obs
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        # Process tags using configured IDs (handles lost tracking internally)
        r_world, t_world = self.estimator.process_tag(
            corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id
        )
        r_obj, t_obj = self.estimator.process_tag(
            corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id
        )
        r_ee, t_ee = self.estimator.process_tag(
            corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id
        )
        
        # Update visibility flags based on actual detections
        if ids is not None:
            ids_arr = np.atleast_1d(ids).ravel()
            obs['aruco_visibility'][0] = 1.0 if np.any(ids_arr == robot_config.aruco_world_id) else 0.0
            obs['aruco_visibility'][1] = 1.0 if np.any(ids_arr == robot_config.aruco_object_id) else 0.0
            obs['aruco_visibility'][2] = 1.0 if np.any(ids_arr == robot_config.aruco_ee_id) else 0.0
        
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
            
            # Resize for display
            disp = cv2.resize(frame, (self.camera_width // 2, self.camera_height // 2))
            cv2.imshow('Robot Camera View', disp)
            cv2.waitKey(1)
        
        # Compute relative poses (only update if we have valid detections)
        # 1. EE (3) relative to World (0)
        if r_world is not None and r_ee is not None:
            pos, quat = self._compute_relative_pose(r_world, t_world, r_ee, t_ee)
            obs['aruco_ee_in_world'] = np.concatenate([pos, quat])
        # Otherwise keep previous value (already set above)
        
        # 2. Object (2) relative to World (0)
        if r_world is not None and r_obj is not None:
            pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
            obs['aruco_object_in_world'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # 3. EE (3) relative to Object (2)
        if r_obj is not None and r_ee is not None:
            pos, quat = self._compute_relative_pose(r_obj, t_obj, r_ee, t_ee)
            obs['aruco_ee_in_object'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # 4. Object (2) relative to EE (3)
        if r_ee is not None and r_obj is not None:
            pos, quat = self._compute_relative_pose(r_ee, t_ee, r_obj, t_obj)
            obs['aruco_object_in_ee'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # Store current observation as previous for next timestep
        self.prev_aruco_obs_dict = {k: v.copy() for k, v in obs.items()}
        
        return obs
    
    def _get_observation(self):
        """
        Get current observation: [aruco_obj_in_world (7D), aruco_obj_in_ee (7D), 
                                 aruco_ee_in_world (7D), gripper (1D)] = 22D
        
        Also stores ArUco observations in self.aruco_obs_dict for access.
        
        Gripper state is read from MuJoCo configuration (same as joint angles).
        """
        # Get ArUco observations as dict
        self.aruco_obs_dict = self._get_aruco_observations_dict()
        
        # Get gripper position from MuJoCo configuration (same as joint angles)
        # configuration.q[5] is the gripper joint in sim units
        # The configuration always has at least 6 elements (5 arm joints + gripper) after initialization
        gripper_pos = self.robot_base.configuration.q[5]
        
        # Extract specific poses for observation array:
        # [aruco_obj_in_world, aruco_obj_in_ee, aruco_ee_in_world, gripper]
        obs = np.concatenate([
            self.aruco_obs_dict['aruco_object_in_world'],  # 7D
            self.aruco_obs_dict['aruco_object_in_ee'],    # 7D
            self.aruco_obs_dict['aruco_ee_in_world'],     # 7D
            [gripper_pos]                                  # 1D (from MuJoCo configuration)
        ])
        
        return obs.astype(np.float32)
    
    def _denormalize_action(self, action):
        """
        Denormalize action from [-1, 1] to actual velocity/gripper ranges.
        
        Args:
            action: [vx, vy, vz, wx, wy, wz, gripper] in [-1, 1]
        
        Returns:
            tuple: (velocity_world, angular_velocity_world, gripper_target)
        """
        # Scale velocities (adjust scales as needed)
        velocity_scale = 0.25  # m/s
        angular_velocity_scale = 1.0  # rad/s
        
        velocity_world = action[:3] * velocity_scale
        angular_velocity_world = action[3:6] * angular_velocity_scale
        
        # Map gripper from [-1, 1] to [closed, open]
        gripper_norm = (action[6] + 1.0) / 2.0  # [0, 1]
        gripper_target = (
            robot_config.gripper_closed_pos * (1.0 - gripper_norm) +
            robot_config.gripper_open_pos * gripper_norm
        )
        
        return velocity_world, angular_velocity_world, gripper_target
    
    def reset(self, seed=None, options=None):
        """
        Reset environment: handle gripper, then move to home.
        
        Args:
            seed: Random seed (unused)
            options: Optional dict (unused)
        
        Returns:
            tuple: (observation, info)
        """
        self.episode_step = 0
        self.episode_return = 0.0
        
        # Clear previous ArUco observations on reset (start fresh)
        self.prev_aruco_obs_dict = {}
        
        # Reset gripper: reboot motor to clear errors, then open and verify
        gripper_motor_id = robot_config.motor_ids[-1]
        
        # Step 1: Reboot gripper motor to clear hardware errors (e.g., over-torque)
        # This also automatically re-enables torque after reboot
        # Try multiple times to ensure successful reboot
        print("Resetting gripper: rebooting motor to clear errors...")
        reboot_successful = False
        max_reboot_attempts = 3
        for attempt in range(max_reboot_attempts):
            try:
                self.robot_base.robot_driver.reboot_motor(gripper_motor_id)
                print(f"✓ Gripper motor rebooted successfully (attempt {attempt+1}/{max_reboot_attempts})")
                reboot_successful = True
                time.sleep(0.3)  # Give motor time to stabilize after reboot
                break
            except Exception as e:
                print(f"⚠️  Warning: Failed to reboot gripper motor (attempt {attempt+1}/{max_reboot_attempts}): {e}")
                if attempt < max_reboot_attempts - 1:
                    time.sleep(0.5)  # Wait before retry
                else:
                    print("⚠️  All reboot attempts failed. Proceeding anyway - motor might still be functional")
                    time.sleep(0.3)
        
        # Command gripper to open
        try:
            self.robot_base.robot_driver.send_motor_positions(
                {gripper_motor_id: robot_config.gripper_encoder_max},
                velocity_limit=robot_config.velocity_limit
            )
            print(f"Gripper commanded to open position: {robot_config.gripper_encoder_max}")
        except Exception as e:
            print(f"Warning: Failed to send gripper open command: {e}")
            return self._get_observation(), {}
        
        # Verify gripper actually moved to open position
        max_verify_attempts = 10
        verify_timeout = 2.0  # seconds
        encoder_tolerance = 50  # Allow some tolerance in encoder reading
        
        gripper_confirmed = False
        for attempt in range(max_verify_attempts):
            time.sleep(verify_timeout / max_verify_attempts)
            try:
                robot_encoders = self.robot_base.robot_driver.read_all_encoders(max_retries=3, retry_delay=0.1)
                if gripper_motor_id in robot_encoders and robot_encoders[gripper_motor_id] is not None:
                    current_encoder = robot_encoders[gripper_motor_id]
                    encoder_error = abs(current_encoder - robot_config.gripper_encoder_max)
                    
                    if encoder_error <= encoder_tolerance:
                        gripper_confirmed = True
                        print(f"✓ Gripper confirmed open: encoder={current_encoder} (target={robot_config.gripper_encoder_max}, error={encoder_error})")
                        break
                    else:
                        print(f"  Gripper verification attempt {attempt+1}/{max_verify_attempts}: encoder={current_encoder} (target={robot_config.gripper_encoder_max}, error={encoder_error})")
            except Exception as e:
                print(f"  Warning: Failed to read gripper encoder (attempt {attempt+1}): {e}")
        
        if not gripper_confirmed:
            print(f"⚠️  Warning: Gripper may not have fully opened. Proceeding anyway...")
            # Try reading one more time to get current state
            try:
                robot_encoders = self.robot_base.robot_driver.read_all_encoders(max_retries=3, retry_delay=0.1)
                if gripper_motor_id in robot_encoders and robot_encoders[gripper_motor_id] is not None:
                    current_encoder = robot_encoders[gripper_motor_id]
                    print(f"  Current gripper encoder: {current_encoder}")
            except:
                pass
        
        # Move to startup home position (not reasonable_home_pose which is for shutdown)
        home_qpos, _, _ = get_sim_home_pose(self.robot_base.model)
        home_motor_positions = get_home_motor_positions(self.robot_base.translator, home_qpos=home_qpos)
        
        # Ensure gripper is set to open position in home_motor_positions
        home_motor_positions[gripper_motor_id] = robot_config.gripper_encoder_max
        
        try:
            # Use move_to_home which handles the movement properly (includes gripper)
            self.robot_base.robot_driver.move_to_home(
                home_motor_positions,
                velocity_limit=robot_config.velocity_limit
            )
            time.sleep(3.0)  # Wait for movement
            
            # Verify gripper is still open after moving to home
            time.sleep(0.2)  # Give encoders time to settle
            try:
                robot_encoders_check = self.robot_base.robot_driver.read_all_encoders(max_retries=3, retry_delay=0.1)
                if gripper_motor_id in robot_encoders_check and robot_encoders_check[gripper_motor_id] is not None:
                    gripper_encoder_after_home = robot_encoders_check[gripper_motor_id]
                    encoder_error_after = abs(gripper_encoder_after_home - robot_config.gripper_encoder_max)
                    if encoder_error_after <= encoder_tolerance:
                        print(f"✓ Gripper confirmed open after home movement: encoder={gripper_encoder_after_home}")
                    else:
                        print(f"⚠️  Warning: Gripper encoder after home movement: {gripper_encoder_after_home} (target={robot_config.gripper_encoder_max}, error={encoder_error_after})")
            except Exception as e:
                print(f"Warning: Failed to verify gripper after home movement: {e}")
        except Exception as e:
            print(f"Warning: Failed to move to home: {e}")
        
        # Sync MuJoCo to actual robot state
        time.sleep(0.2)  # Give encoders time to settle
        try:
            robot_encoders = self.robot_base.robot_driver.read_all_encoders(max_retries=5, retry_delay=0.2)
            robot_joint_angles, actual_position, actual_orientation_quat_wxyz = sync_robot_to_mujoco(
                robot_encoders,
                self.robot_base.translator,
                self.robot_base.model,
                self.robot_base.data,
                self.robot_base.configuration
            )
            
            # Update controller target pose
            self.robot_base.robot_controller.reset_pose(actual_position, actual_orientation_quat_wxyz)
            current_target_pose = self.robot_base.robot_controller.get_target_pose()
            self.robot_base.robot_controller.end_effector_task.set_target(current_target_pose)
            
            # Update gripper state - explicitly set to open position to match what we commanded
            # This ensures SpaceMouse control works correctly after reset
            self.robot_base.gripper_current_position = robot_config.gripper_open_pos
            
            # Verify gripper encoder matches (for debugging)
            if gripper_motor_id in robot_encoders and robot_encoders[gripper_motor_id] is not None:
                actual_gripper_pos = encoder_to_gripper_position(robot_encoders[gripper_motor_id])
                print(f"Reset: Gripper encoder {robot_encoders[gripper_motor_id]} -> position {actual_gripper_pos:.6f}")
                print(f"Reset: Set gripper_current_position to open: {self.robot_base.gripper_current_position:.6f}")
            else:
                print(f"Reset: Could not read gripper encoder, using open position: {self.robot_base.gripper_current_position:.6f}")
        except Exception as e:
            print(f"Warning: Failed to sync robot state: {e}")
            import traceback
            traceback.print_exc()
            # Set gripper to open as fallback
            self.robot_base.gripper_current_position = robot_config.gripper_open_pos
        
        return self._get_observation(), {}
    
    def step(self, action):
        """
        Execute one step: apply action and return observation, reward, done, truncated, info.
        
        Args:
            action: [vx, vy, vz, wx, wy, wz, gripper] in [-1, 1]
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Denormalize action
        velocity_world, angular_velocity_world, gripper_target = self._denormalize_action(action)
        
        # Apply control (same as teleop)
        self.robot_base._execute_control_step(
            velocity_world=velocity_world,
            angular_velocity_world=angular_velocity_world,
            gripper_target=gripper_target,
            dt=self.dt
        )
        
        # Update gripper state
        self.robot_base.gripper_current_position = gripper_target
        
        # Get observation
        obs = self._get_observation()
        
        # Simple reward (can be customized)
        reward = 0.0
        
        # Check termination
        self.episode_step += 1
        terminated = False
        truncated = self.episode_step >= self.max_episode_length
        
        info = {
            "episode_step": self.episode_step,
            "episode_return": self.episode_return
        }
        
        self.episode_return += reward
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        """Render (pass for now)."""
        pass
    
    def close(self):
        """Cleanup and shutdown robot."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        if self.robot_base:
            self.robot_base.shutdown()


class WX200GymEnvWrapper(gym.Env):
    """
    Wrapper for WX200 gym environment with robomimic-compatible interface.
    
    Similar structure to RobomimicLowdimWrapper for compatibility with learning algorithms.
    """
    def __init__(
        self,
        env=None,
        normalization_path=None,
        clamp_obs=False,
        max_episode_length=1000,
        control_frequency=None,
        camera_id=None,
        width=None,
        height=None,
        fps=None,
        enable_aruco=True,
        show_video=True,
        show_axes=True,
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Optional WX200GymEnvBase instance (if None, creates one)
            normalization_path: Path to normalization file (optional)
            clamp_obs: Whether to clamp observations to [-1, 1]
            max_episode_length: Maximum steps per episode
            control_frequency: Control loop frequency (Hz)
            camera_id: Camera device ID
            width: Camera width
            height: Camera height
            fps: Camera FPS
            enable_aruco: Enable ArUco tracking
            show_video: Show camera preview
            show_axes: Draw ArUco axes
        """
        # Create or use provided base environment
        if env is None:
            self.env = WX200GymEnvBase(
                max_episode_length=max_episode_length,
                control_frequency=control_frequency,
                camera_id=camera_id,
                width=width,
                height=height,
                fps=fps,
                enable_aruco=enable_aruco,
                show_video=show_video,
                show_axes=show_axes,
            )
        else:
            self.env = env
        
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.t = 0
        self.episode_return = 0.0
        self.episode_length = 0
        
        # Set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
        
        # Setup spaces - use [-1, 1] for actions
        low = np.full(self.env.action_space.shape[0], fill_value=-1.0)
        high = np.full(self.env.action_space.shape[0], fill_value=1.0)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
        
        # Observation space
        obs_example = self.get_observation()
        if self.normalize:
            low = np.full_like(obs_example, fill_value=-1)
            high = np.full_like(obs_example, fill_value=1)
        else:
            low = np.full_like(obs_example, fill_value=-np.inf)
            high = np.full_like(obs_example, fill_value=np.inf)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )
    
    def normalize_obs(self, obs):
        """Normalize observation to [-1, 1]."""
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1]
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs
    
    def unnormalize_action(self, action):
        """Unnormalize action from [-1, 1] to original range."""
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min
    
    def get_observation(self):
        """Get current observation (normalized if applicable)."""
        raw_obs = self.env._get_observation()
        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs
    
    def seed(self, seed=None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()
    
    def reset(self, options=None, **kwargs):
        """
        Reset environment and return observation and info.
        
        Args:
            options: Optional dict (can contain 'seed', 'video_path', etc.)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            tuple: (observation, info)
        """
        self.t = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes += 1
        
        # Handle None options
        if options is None:
            options = {}
        
        # Call reset on base environment
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        
        obs, info = self.env.reset(seed=new_seed, options=options)
        
        # Normalize if needed
        if self.normalize:
            obs = self.normalize_obs(obs)
        
        return obs, info
    
    def step(self, action):
        """
        Execute one step.
        
        Args:
            action: Action in [-1, 1] range
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Unnormalize action if needed
        if self.normalize:
            action = self.unnormalize_action(action)
        
        # Step base environment
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Normalize observation if needed
        if self.normalize:
            obs = self.normalize_obs(obs)
        
        # Update episode tracking
        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1
        
        # Update info
        info["episode_step"] = self.env_step
        info["episode_return"] = self.episode_return
        info["episode_length"] = self.episode_length
        info["success"] = 0  # Can be customized based on task
        
        # Handle termination
        if terminated:
            return obs, reward, True, False, info
        if self.t >= self.max_episode_length:
            return obs, reward, False, True, info
        return obs, reward, False, False, info
    
    def render(self, mode="human"):
        """Render environment."""
        return self.env.render(mode=mode)
    
    def get_episode_info(self):
        """Get episode information."""
        return {
            "return": self.episode_return,
            "length": self.episode_length
        }
    
    def get_info(self):
        """Get general environment information."""
        return {
            "env_step": self.env_step,
            "n_episodes": self.n_episodes
        }
    
    def close(self):
        """Cleanup and shutdown."""
        if self.env:
            self.env.close()


def make_wx200_env(
    max_episode_length=1000,
    control_frequency=None,
    camera_id=None,
    width=None,
    height=None,
    fps=None,
    enable_aruco=True,
    show_video=True,
    show_axes=True,
    normalization_path=None,
    clamp_obs=False,
    seed=0,
):
    """
    Factory function to create WX200 gym environment.
    
    Similar to make_env in robomimic_env_utils.py.
    
    Args:
        max_episode_length: Maximum steps per episode
        control_frequency: Control loop frequency (Hz)
        camera_id: Camera device ID
        width: Camera width
        height: Camera height
        fps: Camera FPS
        enable_aruco: Enable ArUco tracking
        show_video: Show camera preview
        show_axes: Draw ArUco axes
        normalization_path: Path to normalization file (optional)
        clamp_obs: Whether to clamp observations
        seed: Random seed
    
    Returns:
        WX200GymEnvWrapper: Wrapped environment
    """
    env = WX200GymEnvWrapper(
        env=None,
        normalization_path=normalization_path,
        clamp_obs=clamp_obs,
        max_episode_length=max_episode_length,
        control_frequency=control_frequency,
        camera_id=camera_id,
        width=width,
        height=height,
        fps=fps,
        enable_aruco=enable_aruco,
        show_video=show_video,
        show_axes=show_axes,
    )
    env.seed(seed)
    return env

