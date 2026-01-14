"""
Robot configuration for WX200.

Centralized configuration for all robot-specific constants.
"""
from dataclasses import dataclass
from typing import List


@dataclass
class RobotConfig:
    """Configuration for WX200 robot."""
    
    # Motor IDs
    motor_ids: List[int] = None
    
    # Dynamixel addresses
    addr_torque_enable: int = 64
    addr_goal_position: int = 116
    addr_profile_velocity: int = 112
    addr_present_position: int = 132
    addr_present_current: int = 126  # 2 bytes, read-only
    
    # Serial communication
    baudrate: int = 4000000  # 4 Mbps (Dynamixel X-series supports up to 4.5 Mbps)
    devicename: str = '/dev/ttyUSB0'
    protocol_version: float = 2.0
    
    # Encoder parameters
    encoder_max: int = 4095  # 12-bit encoder range
    encoder_center: int = 2048  # Middle position (180 degrees)
    
    # Gripper encoder range (measured actual range)
    gripper_encoder_min: int = 1559  # Closed position
    gripper_encoder_max: int = 2776  # Open position
    
    # IK solver parameters (optimized for low latency)
    ik_max_iters: int = 10  # Reduced from 20 for 50Hz control
    ik_pos_threshold: float = 1e-4
    ik_ori_threshold: float = 1e-4
    ik_solver: str = "quadprog"
    ik_lm_damping: float = 0.1
    
    # SpaceMouse deadzones
    velocity_deadzone: float = 0.001
    angular_velocity_deadzone: float = 0.01
    
    # SpaceMouse scaling
    velocity_scale: float = 0.25  # m/s per unit SpaceMouse input
    angular_velocity_scale: float = 1  # rad/s per unit SpaceMouse rotation input
    
    # Gripper configuration
    gripper_open_pos: float = -0.026  # meters
    gripper_closed_pos: float = 0.0  # meters
    gripper_increment_rate: float = 0.0005  # position change per loop when button held
    
    # Robot control parameters
    velocity_limit: int = 40  # Speed limit for movements (0=Max, 30=Slow/Safe)
    control_frequency: float = 10.0  # Outer loop frequency (Hz) - teleop/policy/observations
    inner_control_frequency: float = 120.0  # Inner loop frequency (Hz) - IK solving and motor commands
    inner_loop_skip_threshold: float = 0.002  # Minimum time remaining before outer loop deadline to execute inner loop (seconds)
    use_threaded_control: bool = False  # If True, use separate threads for inner/outer loops (more responsive, but adds threading complexity)
    
    # Control loop timing and profiling parameters
    deadline_threshold_factor: float = 0.8  # Time budget threshold as fraction of dt (80% = 80% of period)
    blocking_outlier_threshold_control_step: float = 0.050  # Control steps exceeding this (seconds) are filtered as blocking ops
    blocking_outlier_threshold_outer_loop: float = 0.100  # Outer loop iterations exceeding this (seconds) are filtered as blocking ops
    blocking_interval_threshold: float = 0.200  # Intervals between iterations exceeding this (seconds) are filtered
    recovery_gap_threshold: float = 0.100  # Gap between control steps exceeding this indicates recovery from blocking (seconds)
    sleep_safety_margin: float = 0.003  # Safety margin for sleep timing to prevent deadline overshoot (seconds)
    min_sleep_time: float = 0.001  # Minimum sleep time - don't sleep if less than this (seconds)
    max_profiling_samples: int = 1000  # Maximum number of profiling samples to keep in memory
    periodic_reporting_interval: int = 500  # Report performance stats every N control steps
    periodic_reporting_sample_window: int = 500  # Number of recent samples to use for periodic reporting
    safe_frequency_margin: float = 0.8  # Safety margin for estimated max frequency (80% of theoretical max)
    lock_wait_threshold: float = 0.001  # Lock wait time threshold for contention tracking (seconds)
    
    # Camera profiling parameters
    profiler_window_size: int = 200  # Sliding window size for camera profiler statistics
    camera_assessment_threshold: float = 0.5  # Camera processing assessment threshold (50% of deadline)
    missed_deadline_warning_threshold: float = 5.0  # Missed deadline percentage threshold for warning (%)
    missed_deadline_error_threshold: float = 10.0  # Missed deadline percentage threshold for error (%)
    frame_downscale_factor: int = 4  # Frame downscale factor for recording (1/4 resolution = 1/16 pixels)
    default_profile_interval: int = 100  # Default interval for printing profiling stats (iterations)
    
    # Data collection verbosity settings
    verbose_profiling: bool = False  # If True, print detailed periodic profiling stats
    warning_only_mode: bool = True  # If True, only print warnings when thresholds exceeded
    encoder_poll_stats_interval: int = 100  # Print encoder poll stats every N polls (0 = disable)
    control_perf_stats_interval: int = 500  # Print control step stats every N steps (0 = disable)
    
    # Vision / camera configuration
    # Note: camera_id maps to /dev/video{camera_id}
    # Use: v4l2-ctl --list-devices to find your camera
    camera_id: int = 1 # UC70 camera capture device (video1 is capture, video2 is metadata)
    camera_width: int = 1920
    camera_height: int = 1080
    camera_fps: int = 30
    
    # ArUco marker configuration
    aruco_marker_size_m: float = 0.030  # meters (default: 30mm)
    aruco_world_id: int = 0
    aruco_object_id: int = 2
    aruco_ee_id: int = 3
    aruco_axis_length_scale: float = 0.5  # fraction of marker size for axis length
    aruco_single_tag_rotation_threshold: float = 0.8
    aruco_single_tag_translation_threshold: float = 0.2
    aruco_max_preserve_frames: int = 8
    aruco_max_rejections_before_force: int = 5
    
    # End-effector workspace bounds (meters) for safety clamping
    # Format: (min, max) for X, Y, Z in world frame
    ee_bound_x: tuple = (0.05, 0.90)   # meters
    ee_bound_y: tuple = (-0.20, 0.20)  # meters
    ee_bound_z: tuple = (0.02, 0.40)   # meters
    
    # Startup home position (encoder values for motors 1-7)
    # If None, will compute from sim keyframe. Otherwise, uses these exact positions.
    # Example: startup_home_positions = [1724, 1388, 2708, 1540, 1416, 2023, 2754]
    startup_home_positions: List[int] = None
    
    # Shutdown sequence poses (-1 means skip that motor)
    reasonable_home_pose: List[int] = None
    base_home_pose: List[int] = None
    folded_home_pose: List[int] = None
    move_delay: float = 1.0  # Seconds to wait between shutdown moves
    
    def __post_init__(self):
        """Set default values for lists."""
        if self.motor_ids is None:
            self.motor_ids = [1, 2, 3, 4, 5, 6, 7]
        if self.startup_home_positions is None:
            # Default: use configured home positions
            # Set to None to compute from sim keyframe instead
            # self.startup_home_positions = [1724, 1388, 2708, 1540, 1416, 2023, 2754]
            # self.startup_home_positions = [2070, 1646, 2453, 1613, 1460, 2011, 2756]
            self.startup_home_positions = [2462, 1496, 2603, 1405, 1400, 2104, 2766]
        if self.reasonable_home_pose is None:
            self.reasonable_home_pose = [-1, 1382, 2712, 1568, 1549, 2058, 1784]
        if self.base_home_pose is None:
            self.base_home_pose = [2040, -1, -1, -1, -1, -1, -1]
        if self.folded_home_pose is None:
            self.folded_home_pose = [2040, 846, 3249, 958, 1944, 2057, 1784]


# Global robot configuration instance
# Can be overridden for different robot models or test configurations
robot_config = RobotConfig()
