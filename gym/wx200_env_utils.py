"""
Gym environment utilities for WX200 robot.

Similar structure to robomimic_env_utils.py for compatibility with learning algorithms.
"""
import os
import time
import cv2
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.spaces import Box
from scipy.spatial.transform import Rotation as R

from envs.hardware.robot_control.robot_config import robot_config
from envs.hardware.robot_control.robot_control_base import RobotControlBase, get_sim_home_pose
from envs.hardware.robot_control.robot_joint_to_motor import sync_robot_to_mujoco, encoder_to_gripper_position
from envs.hardware.robot_control.robot_startup import get_home_motor_positions
from envs.hardware.camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix
from utils.datasets import Dataset

AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale

DATASET_NAME = "merged_data_aruco_pos_ac_targets.npz"

def is_wx200_env(env_name):
    """Determine if an env is WX200 hardware."""
    return env_name.startswith("wx200") or env_name.startswith("WX200")

def _get_max_episode_length(env_name):
    """Get max episode length for WX200 environment."""
    return 500


def make_env(
    env_name,
    control_frequency=None,
    camera_id=None,
    width=None,
    height=None,
    fps=None,
    enable_aruco=True,
    show_video=False,
    show_axes=True,
    normalization_path=None,
    clamp_obs=False,
    seed=0,
):
    """
    Factory function to create WX200 gym environment.
    
    Similar to make_env in robomimic_env_utils.py.
    
    Args:
        env_name: Environment name
        control_frequency: Control loop frequency (Hz)
        camera_id: Camera device ID
        width: Camera width
        height: Camera height
        fps: Camera FPS
        enable_aruco: Enable ArUco tracking
        show_video: Show camera preview (default False)
        show_axes: Draw ArUco axes
        normalization_path: Path to normalization file (optional)
        clamp_obs: Whether to clamp observations
        seed: Random seed
    
    Returns:
        WX200GymEnvWrapper: Wrapped environment
    """
    if not is_wx200_env(env_name):
        raise ValueError(f"Environment {env_name} is not a WX200 environment")

    max_episode_length = _get_max_episode_length(env_name)
    
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



def get_dataset(env, env_name):
    """
    Load dataset from merged_dataset.npz.
    
    Expected format (from merged_data.npz):
    - smoothed_observations: (N, 22)
        - smoothed_aruco_ee_in_world (7)
        - smoothed_aruco_object_in_world (7)
        - smoothed_aruco_object_in_ee (7)
        - states (1: gripper)
    - actions_flat: (N, 7) = [vx, vy, vz, wx, wy, wz, gripper]
        ALWAYS in control space (unnormalized):
        - [vx, vy, vz]: m/s (range: [-0.25, 0.25])
        - [wx, wy, wz]: rad/s (range: [-1.0, 1.0])
        - gripper: meters (range: [-0.026, 0.0])
    
    This function ALWAYS normalizes actions_flat from control space to [-1, 1] for training.
    """
    # Default path to merged dataset
    dataset_path = os.path.join(
        os.path.dirname(__file__),
        "hardware",
        DATASET_NAME
    )
    
    # Allow override via environment variable
    if "WX200_DATASET_PATH" in os.environ:
        dataset_path = os.environ["WX200_DATASET_PATH"]
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Set WX200_DATASET_PATH environment variable to specify path."
        )
    
    print(f"Loading WX200 dataset from: {dataset_path}")
    data = np.load(dataset_path, allow_pickle=True)

    # Extract data
    # observations = data['observations'].astype(np.float32)
    observations = data['smoothed_observations'].astype(np.float32)
    # actions_flat is ALWAYS in control space (unnormalized) - normalize it
    actions = data['actions_flat'].astype(np.float32)
    rewards = data['rewards'].astype(np.float32)
    terminals = data['terminals'].astype(np.float32)
    next_observations = data['next_observations'].astype(np.float32)
    masks = data['masks'].astype(np.float32)

    print(f"Loaded dataset: {len(observations)} transitions")
    print(f"  Observations shape: {observations.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Actions are in control space (unnormalized) - will normalize to [-1, 1]")
    
    # Normalize dataset actions from control ranges to [-1, 1] for training
    # actions_flat is ALWAYS stored in control space (m/s for velocities, meters for gripper)
    # Control ranges (from robot_config):
    # - Linear velocity [vx, vy, vz]: [-0.25, 0.25] m/s per axis
    # - Angular velocity [wx, wy, wz]: [-1.0, 1.0] rad/s per axis
    # - Gripper target: [-0.026, 0.0] meters (open to closed)
    action_low = np.array([
        -robot_config.velocity_scale,  # vx
        -robot_config.velocity_scale,  # vy
        -robot_config.velocity_scale,  # vz
        -robot_config.angular_velocity_scale,  # wx
        -robot_config.angular_velocity_scale,  # wy
        -robot_config.angular_velocity_scale,  # wz
        robot_config.gripper_open_pos,  # gripper (open = -0.026)
    ])
    action_high = np.array([
        robot_config.velocity_scale,  # vx
        robot_config.velocity_scale,  # vy
        robot_config.velocity_scale,  # vz
        robot_config.angular_velocity_scale,  # wx
        robot_config.angular_velocity_scale,  # wy
        robot_config.angular_velocity_scale,  # wz
        robot_config.gripper_closed_pos,  # gripper (closed = 0.0)
    ])
    
    # ALWAYS normalize: (action - low) / (high - low) * 2 - 1 -> [-1, 1]
    # actions_flat is always in control space, never pre-normalized
    normalized_actions = 2.0 * (actions - action_low) / (action_high - action_low) - 1.0
    
    print(f"  Normalized actions from control ranges to [-1, 1]")
    print(f"    Control ranges: low={action_low}, high={action_high}")
    print(f"    Action sample before (control space): {actions[0]}")
    print(f"    Action sample after (normalized): {normalized_actions[0]}")
    
    # Update environment's action/obs space to match dataset if needed
    if hasattr(env, 'set_dims'):
        obs_dim = observations.shape[1]
        action_dim = normalized_actions.shape[1]
        env.set_dims(obs_dim, action_dim)
        print(f"  Updated environment dims: Obs={obs_dim}, Act={action_dim}")
    
    # Return Dataset object with normalized actions (consistent with other environment utils)
    return Dataset.create(
        observations=observations,
        actions=normalized_actions,
        rewards=rewards,
        terminals=terminals,
        masks=masks,
        next_observations=next_observations,
    )

class WX200GymEnvBase(gym.Env):
    """
    Base gym environment for WX200 robot with low-dimensional state observations.
    
    This is the core environment that handles robot control and ArUco tracking.
    """
    # Class-level variable to track which instance has hardware authority
    _hardware_authority = None
    
    def __init__(self, max_episode_length=1000, control_frequency=None, 
                 camera_id=None, width=None, height=None, fps=None, 
                 enable_aruco=True, show_video=False, show_axes=True):
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
            show_video: Show camera preview window (default False)
            show_axes: Draw ArUco axes in preview (default True)
        """
        self.max_episode_length = max_episode_length
        self.control_frequency = control_frequency or robot_config.control_frequency
        self.dt = 1.0 / self.control_frequency
        self.enable_aruco = enable_aruco
        self.show_video = show_video
        self.show_axes = show_axes
        
        # All instances start with no authority - will be claimed when hardware is needed
        self.has_authority = False
        
        # Do NOT initialize hardware in constructor - will be done lazily on first reset/step
        self.robot_base = None
        self._hardware_initialized = False
        
        # Setup camera and ArUco (will be initialized only if has_authority)
        self.camera = None
        self.estimator = None
        self.detector = None
        self.cam_matrix = None
        self.dist_coeffs = None
        self.camera_width = width if width is not None else robot_config.camera_width
        self.camera_height = height if height is not None else robot_config.camera_height
        self.aruco_obs_dict = {}  # Store ArUco observations as dict
        self.prev_aruco_obs_dict = {}  # Store previous ArUco observations for lost tracking
        
        self.early_termination = False
        self.success_trigger = False
        
        # Setup spaces
        self.action_space = Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Observation: aruco_obj_in_world (7D) + robot_state (6D) + ee_pose_debug (7D) = 20D
        obs_example = self._get_observation()
        self.observation_space = Box(
            low=np.full_like(obs_example, -np.inf),
            high=np.full_like(obs_example, np.inf),
            dtype=np.float32
        )
        
        self.episode_step = 0
        self.episode_return = 0.0
        
        # Rate limiting for step frequency
        self.last_step_time = None
        self.min_step_interval = 1.0 / self.control_frequency  # Minimum time between steps
    
    def _initialize_hardware(self, camera_id=None, width=None, height=None, fps=None):
        """Lazily initialize robot and camera hardware. Claims authority if available."""
        if self._hardware_initialized:
            return  # Already initialized
        
        # Claim authority if available (no one else has it)
        authority_claimed_here = False
        if WX200GymEnvBase._hardware_authority is None:
            WX200GymEnvBase._hardware_authority = self
            self.has_authority = True
            authority_claimed_here = True
            print("✓ Hardware authority claimed (hardware was available)")
        elif WX200GymEnvBase._hardware_authority == self:
            # We already have authority (e.g., after close/reopen)
            self.has_authority = True
            print("✓ Hardware authority already held by this instance")
        else:
            # Another instance has authority
            raise RuntimeError(
                f"Hardware authority already claimed by another instance. "
                f"Only one environment instance can control the robot at a time. "
                f"Current authority: {WX200GymEnvBase._hardware_authority}"
            )
        
        try:
            # Initialize robot control base
            print("\nInitializing robot hardware...")
            self.robot_base = RobotControlBase(control_frequency=self.control_frequency)
            self.robot_base.initialize()
            
            # Setup camera and ArUco if enabled
            if self.enable_aruco:
                self._setup_camera(camera_id, width, height, fps)
            
            self._hardware_initialized = True
            print("✓ Hardware initialization complete")
        except Exception as e:
            # If initialization fails, release authority if we claimed it here
            if authority_claimed_here:
                WX200GymEnvBase._hardware_authority = None
                self.has_authority = False
                print(f"⚠️  Hardware initialization failed, authority released: {e}")
            raise  # Re-raise the exception
    
    def _setup_camera(self, camera_id=None, width=None, height=None, fps=None):
        """Initialize camera and ArUco detector (only if has_authority)."""
        if not self.has_authority:
            return  # Skip camera initialization
        
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
            # Print error in red
            RED = '\033[91m'
            RESET = '\033[0m'
            print(f"{RED}⚠️  ERROR: Failed to initialize camera/ArUco: {e}{RESET}")
            import traceback
            print(f"{RED}Traceback:{RESET}")
            traceback.print_exc()
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
        
        if not self.has_authority or not self.camera or not self.enable_aruco:
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
            
            # Debug: print when not all markers are in view
            if not all(obs['aruco_visibility'] == 1.0):
                missing_markers = []
                if obs['aruco_visibility'][0] == 0.0:
                    missing_markers.append(f"world (ID: {robot_config.aruco_world_id})")
                if obs['aruco_visibility'][1] == 0.0:
                    missing_markers.append(f"object (ID: {robot_config.aruco_object_id})")
                if obs['aruco_visibility'][2] == 0.0:
                    missing_markers.append(f"ee (ID: {robot_config.aruco_ee_id})")
                print(f"[DEBUG] Not all ArUco markers in view. Missing: {', '.join(missing_markers)}")
        
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
        
        # Compute relative poses (only update if BOTH markers are CURRENTLY detected)
        # Use visibility flags to ensure we only update with fresh detections, not preserved poses
        # 1. EE (3) relative to World (0) - needs both world and EE markers currently visible
        if (obs['aruco_visibility'][0] == 1.0 and obs['aruco_visibility'][2] == 1.0 and 
            r_world is not None and r_ee is not None):
            pos, quat = self._compute_relative_pose(r_world, t_world, r_ee, t_ee)
            obs['aruco_ee_in_world'] = np.concatenate([pos, quat])
        # Otherwise keep previous value (already set above)
        
        # 2. Object (2) relative to World (0) - needs both world and object markers currently visible
        if (obs['aruco_visibility'][0] == 1.0 and obs['aruco_visibility'][1] == 1.0 and 
            r_world is not None and r_obj is not None):
            pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
            obs['aruco_object_in_world'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # 3. EE (3) relative to Object (2) - needs both object and EE markers currently visible
        if (obs['aruco_visibility'][1] == 1.0 and obs['aruco_visibility'][2] == 1.0 and 
            r_obj is not None and r_ee is not None):
            pos, quat = self._compute_relative_pose(r_obj, t_obj, r_ee, t_ee)
            obs['aruco_ee_in_object'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # 4. Object (2) relative to EE (3) - needs both EE and object markers currently visible
        if (obs['aruco_visibility'][2] == 1.0 and obs['aruco_visibility'][1] == 1.0 and 
            r_ee is not None and r_obj is not None):
            pos, quat = self._compute_relative_pose(r_ee, t_ee, r_obj, t_obj)
            obs['aruco_object_in_ee'] = np.concatenate([pos, quat])
        # Otherwise keep previous value
        
        # Store current observation as previous for next timestep
        self.prev_aruco_obs_dict = {k: v.copy() for k, v in obs.items()}
        
        return obs
    
    def _get_current_ee_pose(self):
        """Get current end-effector pose from MuJoCo (for debugging/inspection only)."""
        if not self.has_authority or not self._hardware_initialized or self.robot_base is None:
            # Return zeros if no hardware access
            return np.zeros(3), np.array([1., 0., 0., 0.])
        
        # Sync MuJoCo data with current configuration
        self.robot_base.data.qpos[:5] = self.robot_base.configuration.q[:5]
        if len(self.robot_base.data.qpos) > 5 and len(self.robot_base.configuration.q) > 5:
            self.robot_base.data.qpos[5] = self.robot_base.configuration.q[5]
        
        mujoco.mj_forward(self.robot_base.model, self.robot_base.data)
        
        # Get site pose
        site_id = mujoco.mj_name2id(self.robot_base.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        site = self.robot_base.data.site(site_id)
        ee_position = site.xpos.copy()
        ee_xmat = site.xmat.reshape(3, 3)
        ee_quat = R.from_matrix(ee_xmat).as_quat()
        
        # Convert from [x, y, z, w] to [w, x, y, z]
        return ee_position, np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
    
    def _get_observation(self):
        """
        Get current observation: [aruco_obj_in_world (7D), robot_state (6D), ee_pose_debug (7D)] = 20D
        
        Also stores ArUco observations in self.aruco_obs_dict for access.
        """
        # Get ArUco observations as dict
        self.aruco_obs_dict = self._get_aruco_observations_dict()

        # Get gripper position and robot state (use dummy values if no hardware authority)
        if not self.has_authority or not self._hardware_initialized or self.robot_base is None:
            gripper_pos = robot_config.gripper_open_pos  # Default to open position
            robot_state = np.zeros(6)  # Dummy robot state (6D: joint angles or position/orientation)
            ee_pose_debug = np.zeros(7)  # Dummy EE pose (3D position + 4D quaternion)
        else:
            gripper_pos = self.robot_base.configuration.q[5]
            robot_state = self.robot_base.configuration.q[:6]  # First 6 elements (assuming 6D state)
            # Get EE pose debug (position + quaternion)
            ee_position, ee_orientation_quat_wxyz = self._get_current_ee_pose()
            ee_pose_debug = np.concatenate([ee_position, ee_orientation_quat_wxyz])
        
        # Extract specific poses for observation array:
        # [aruco_obj_in_world, aruco_obj_in_ee, aruco_ee_in_world, gripper]
        # obs = np.concatenate([
        #     self.aruco_obs_dict['aruco_object_in_world'],  # 7D
        #     self.aruco_obs_dict['aruco_object_in_ee'],    # 7D
        #     self.aruco_obs_dict['aruco_ee_in_world'],     # 7D
        #     [gripper_pos]                                  # 1D (from MuJoCo configuration or dummy)
        # ])

        obs = np.concatenate([
            self.aruco_obs_dict['aruco_object_in_world'],  # 7D
            robot_state,                                    # 6D
            ee_pose_debug,                                  # 7D (position 3D + quaternion 4D)
        ])
        
        return obs.astype(np.float32)
    
    def _denormalize_action(self, action):
        """
        Denormalize action from [-1, 1] to actual control ranges.
        
        Control ranges (from robot_config):
        - Linear velocity [vx, vy, vz]: [-0.25, 0.25] m/s per axis
        - Angular velocity [wx, wy, wz]: [-1.0, 1.0] rad/s per axis
        - Gripper target: [-0.026, 0.0] meters (open to closed)
        
        Denormalization formula: action = (normalized_action + 1) / 2 * (high - low) + low
        
        Args:
            action: [vx, vy, vz, wx, wy, wz, gripper] in [-1, 1] (normalized)
        
        Returns:
            tuple: (velocity_world, angular_velocity_world, gripper_target)
        """
        # Control ranges
        action_low = np.array([
            -robot_config.velocity_scale,  # vx, vy, vz: [-0.25, 0.25] m/s
            -robot_config.velocity_scale,
            -robot_config.velocity_scale,
            -robot_config.angular_velocity_scale,  # wx, wy, wz: [-1.0, 1.0] rad/s
            -robot_config.angular_velocity_scale,
            -robot_config.angular_velocity_scale,
            robot_config.gripper_open_pos,  # gripper: [-0.026, 0.0] meters
        ])
        action_high = np.array([
            robot_config.velocity_scale,
            robot_config.velocity_scale,
            robot_config.velocity_scale,
            robot_config.angular_velocity_scale,
            robot_config.angular_velocity_scale,
            robot_config.angular_velocity_scale,
            robot_config.gripper_closed_pos,
        ])
        
        # Denormalize: (normalized_action + 1) / 2 * (high - low) + low
        denormalized = (action + 1.0) / 2.0 * (action_high - action_low) + action_low
        
        velocity_world = denormalized[:3]
        angular_velocity_world = denormalized[3:6]
        gripper_target = denormalized[6]
        
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
        
        # Reset rate limiter on episode reset
        self.last_step_time = None
        
        # Clear previous ArUco observations on reset (start fresh)
        self.prev_aruco_obs_dict = {}
        
        # Initialize hardware lazily on first reset (claims authority if available)
        if not self._hardware_initialized:
            try:
                self._initialize_hardware()
            except RuntimeError as e:
                # If authority is taken, return dummy observation (for eval env)
                print(f"⚠️  Cannot initialize hardware: {e}")
                return self._get_observation(), {}
        
        # Skip robot operations if no hardware authority or hardware not initialized
        if not self.has_authority or not self._hardware_initialized or self.robot_base is None:
            return self._get_observation(), {}
        
        # CRITICAL: Check if torque was disabled (e.g., from E-stop or interrupt) and re-enable it
        # This ensures the robot can actually move during reset
        print("\n" + "="*60)
        print("RESET: Checking torque status and re-enabling if needed...")
        print("="*60)
        torque_was_disabled = self.robot_base.robot_driver.check_and_reenable_torque()
        if torque_was_disabled:
            print("✓ Torque re-enabled. Proceeding with reset sequence...")
        else:
            print("✓ Torque already enabled. Proceeding with reset sequence...")
        print("="*60 + "\n")
        
        # Reset gripper: reboot, open, and verify with retry logic
        # Note: We do NOT disable torque during normal reset to avoid robot going limp
        # Torque is only disabled on E-stop or during shutdown
        gripper_motor_id = robot_config.motor_ids[-1]
        
        # This retry loop allows the gripper reset to be retried if verification fails
        encoder_tolerance = 50  # Fixed tolerance for open position
        max_sequence_retries = 3
        sequence_successful = False
        
        for sequence_attempt in range(max_sequence_retries):
            if sequence_attempt > 0:
                print(f"\nRetrying gripper reset sequence (attempt {sequence_attempt + 1}/{max_sequence_retries})...")
                time.sleep(2.0)  # Wait longer before retry
            
            # Check for interrupt flag (avoid circular import by checking sys.modules)
            try:
                import sys
                if 'main_ogpo_real' in sys.modules:
                    main_module = sys.modules['main_ogpo_real']
                    if hasattr(main_module, '_interrupt_requested') and main_module._interrupt_requested:
                        print("⚠️  Interrupt requested during gripper reset. Stopping...")
                        self.robot_base.robot_driver.emergency_disable_torque()
                        raise KeyboardInterrupt("Gripper reset interrupted")
            except (AttributeError, KeyError):
                pass  # main_ogpo_real not imported or flag doesn't exist yet
            
            # Gripper reset Step 1: Reboot gripper motor to clear hardware errors
            print("Resetting gripper: rebooting motor to clear errors...")
            reboot_successful = False
            max_reboot_attempts = 3
            for attempt in range(max_reboot_attempts):
                try:
                    self.robot_base.robot_driver.reboot_motor(gripper_motor_id)
                    print(f"✓ Gripper motor rebooted successfully (attempt {attempt+1}/{max_reboot_attempts})")
                    reboot_successful = True
                    time.sleep(0.5)  # Give motor time to stabilize after reboot
                    break
                except Exception as e:
                    print(f"⚠️  Warning: Failed to reboot gripper motor (attempt {attempt+1}/{max_reboot_attempts}): {e}")
                    if attempt < max_reboot_attempts - 1:
                        time.sleep(2.0)  # Wait longer before retry
                    else:
                        print(f"✗ All reboot attempts failed. Retrying sequence...")
                        break
            
            if not reboot_successful:
                continue  # Retry entire sequence
            
            # Gripper reset Step 2: Command gripper to open
            try:
                self.robot_base.robot_driver.send_motor_positions(
                    {gripper_motor_id: robot_config.gripper_encoder_max},
                    velocity_limit=robot_config.velocity_limit
                )
                print(f"Gripper commanded to open position: {robot_config.gripper_encoder_max}")
                time.sleep(1.0)  # Wait for gripper to start moving
            except Exception as e:
                print(f"✗ Failed to send gripper open command: {e}")
                continue  # Retry entire sequence
            
            # Gripper reset Step 3: Verify gripper actually moved to open position
            max_verify_attempts = 10
            verify_timeout = 2.0  # seconds
            
            gripper_confirmed = False
            for attempt in range(max_verify_attempts):
                # Check for interrupt flag (avoid circular import by checking sys.modules)
                try:
                    import sys
                    if 'main_ogpo_real' in sys.modules:
                        main_module = sys.modules['main_ogpo_real']
                        if hasattr(main_module, '_interrupt_requested') and main_module._interrupt_requested:
                            print("⚠️  Interrupt requested during gripper verification. Stopping...")
                            self.robot_base.robot_driver.emergency_disable_torque()
                            raise KeyboardInterrupt("Gripper verification interrupted")
                except (AttributeError, KeyError):
                    pass
                
                time.sleep(verify_timeout / max_verify_attempts)
                try:
                    robot_encoders = self.robot_base.robot_driver.read_all_encoders(max_retries=3, retry_delay=0.1)
                    if gripper_motor_id in robot_encoders and robot_encoders[gripper_motor_id] is not None:
                        current_encoder = robot_encoders[gripper_motor_id]
                        encoder_error = abs(current_encoder - robot_config.gripper_encoder_max)
                        
                        if encoder_error <= encoder_tolerance:
                            gripper_confirmed = True
                            print(f"✓ Gripper confirmed open: encoder={current_encoder} (target={robot_config.gripper_encoder_max}, error={encoder_error})")
                            sequence_successful = True
                            break
                        else:
                            print(f"  Gripper verification attempt {attempt+1}/{max_verify_attempts}: encoder={current_encoder} (target={robot_config.gripper_encoder_max}, error={encoder_error})")
                except Exception as e:
                    print(f"  Warning: Failed to read gripper encoder (attempt {attempt+1}): {e}")
            
            if sequence_successful:
                break
        
        if not sequence_successful:
            # Try reading final state for debugging
            try:
                robot_encoders = self.robot_base.robot_driver.read_all_encoders(max_retries=3, retry_delay=1)
                if gripper_motor_id in robot_encoders and robot_encoders[gripper_motor_id] is not None:
                    current_encoder = robot_encoders[gripper_motor_id]
                    print(f"  Final gripper encoder: {current_encoder}")
            except:
                pass
            
            raise RuntimeError(
                f"Gripper reset failed after {max_sequence_retries} attempts. "
                f"Gripper may not be functional. Please check hardware."
            )
        
        # Move to startup home position (not reasonable_home_pose which is for shutdown)
        # Torque should already be enabled (from initialization or previous operations)
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
        # Rate limiting: enforce control frequency
        current_time = time.perf_counter()
        if self.last_step_time is not None:
            elapsed = current_time - self.last_step_time
            sleep_time = self.min_step_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.last_step_time = time.perf_counter()

        if self.early_termination:
            obs = self._get_observation()
            terminated = True
            truncated = False
            reward = 0.0
            info = {
                "episode_step": self.episode_step,
                "episode_return": self.episode_return
            }
            return obs, reward, terminated, truncated, info
        
        if self.success_trigger:
            obs = self._get_observation()
            terminated = True
            truncated = False
            reward = 0.0
            info = {
                "episode_step": self.episode_step,
                "episode_return": self.episode_return
            }
            return obs, reward, terminated, truncated, info

        # Clip action to valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Denormalize action
        velocity_world, angular_velocity_world, gripper_target = self._denormalize_action(action)
        
        # Initialize hardware lazily on first step (claims authority if available)
        if not self._hardware_initialized:
            try:
                self._initialize_hardware()
            except RuntimeError as e:
                # If authority is taken, skip robot control (for eval env)
                print(f"⚠️  Cannot initialize hardware: {e}")
        
        # Apply control (skip if no hardware authority)
        if self.has_authority and self._hardware_initialized and self.robot_base is not None:
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
        
        # Check termination
        reward = -1.0
        self.episode_step += 1
        terminated = False
        truncated = self.episode_step >= self.max_episode_length
        
        info = {
            "episode_step": self.episode_step,
            "episode_return": self.episode_return,
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
        
        # Only shutdown if this instance has authority and hardware is initialized
        if self.has_authority and self._hardware_initialized and self.robot_base is not None:
            self.robot_base.shutdown()
            self._hardware_initialized = False
        
        # Release hardware authority (will be automatically reclaimed on next reset()/step())
        if WX200GymEnvBase._hardware_authority == self:
            WX200GymEnvBase._hardware_authority = None
            self.has_authority = False
            print("✓ Hardware authority released (will be automatically reclaimed on next reset()/step())")


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
        show_video=False,
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
            show_video: Show camera preview (default False)
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
        # For WX200, actions are always in [-1, 1] from the network
        # The base environment's _denormalize_action() handles conversion to control ranges
        # using config-based bounds. We don't use dataset statistics for actions.
        # So we don't unnormalize here - pass actions through unchanged
        
        # Step base environment
        obs, reward, terminated, _, info = self.env.step(action)
        
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

        # Check for success
        if reward > 0.:
            terminated = True
            info["success"] = 1
        else:
            info["success"] = 0
        
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

