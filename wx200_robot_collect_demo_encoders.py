"""
Teleop control with ArUco marker tracking and true encoder state polling.

This version polls actual robot encoder values at high frequency,
uses them to sync MuJoCo state, and records encoder values in the trajectory.
This ensures we're recording the true robot state rather than relying on MuJoCo estimates.

Updated with selective profiling - tracks performance but only prints warnings when issues detected.
Based on wx200_robot_profile_camera.py but optimized for data collection with minimal verbosity.

ArUco polling runs at camera FPS (30 Hz) for maximum update rate, while logging/storage
happens at control frequency (10 Hz).
"""
import cv2
import numpy as np
import argparse
import time
import threading
import mujoco
import mink
from scipy.spatial.transform import Rotation as R
from collections import deque

from robot_control.robot_config import robot_config
from wx200_robot_teleop_control import TeleopControl, save_trajectory
from camera import Camera, is_gstreamer_available, ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix
from robot_control.robot_joint_to_motor import JointToMotorTranslator, encoders_to_joint_angles
from loop_rate_limiters import RateLimiter

# Shorter axes for visualization to reduce chance of going off-frame (and warnings)
AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale


class ArUcoProfiler:
    """
    Profiler for ArUco polling thread that runs at camera FPS.
    
    Tracks ArUco detection performance separately from control loop profiling.
    """
    
    def __init__(self):
        """Initialize ArUco profiler."""
        self.window_size = robot_config.profiler_window_size
        
        # Track ArUco polling timing
        self.poll_times = deque(maxlen=self.window_size)
        self.poll_intervals = deque(maxlen=self.window_size)
        self.poll_count = 0
        self.last_poll_timestamp = None
        
        # Track component timings
        self.detect_times = deque(maxlen=self.window_size)  # ArUco detection time
        self.pose_compute_times = deque(maxlen=self.window_size)  # Pose computation time
    
    def record_poll_iteration(self, total_time, detect_time, pose_time, interval=None):
        """Record a single ArUco poll iteration."""
        self.poll_times.append(total_time)
        if detect_time is not None:
            self.detect_times.append(detect_time)
        if pose_time is not None:
            self.pose_compute_times.append(pose_time)
        if interval is not None:
            self.poll_intervals.append(interval)
        self.poll_count += 1
    
    def get_statistics(self):
        """Get current statistics."""
        stats = {
            'poll_count': self.poll_count,
        }
        
        if self.poll_times:
            times_ms = [t * 1000 for t in self.poll_times]
            stats['poll_time'] = {
                'avg': np.mean(times_ms),
                'min': np.min(times_ms),
                'max': np.max(times_ms),
                'p95': np.percentile(times_ms, 95) if len(times_ms) > 1 else times_ms[0],
                'count': len(times_ms)
            }
        else:
            stats['poll_time'] = None
        
        if self.detect_times:
            times_ms = [t * 1000 for t in self.detect_times]
            stats['detect_time'] = {
                'avg': np.mean(times_ms),
                'max': np.max(times_ms),
                'count': len(times_ms)
            }
        else:
            stats['detect_time'] = None
        
        if self.pose_compute_times:
            times_ms = [t * 1000 for t in self.pose_compute_times]
            stats['pose_time'] = {
                'avg': np.mean(times_ms),
                'max': np.max(times_ms),
                'count': len(times_ms)
            }
        else:
            stats['pose_time'] = None
        
        if self.poll_intervals:
            intervals_ms = [t * 1000 for t in self.poll_intervals]
            avg_interval = np.mean(intervals_ms) / 1000.0
            stats['frequency'] = {
                'actual': 1.0 / avg_interval if avg_interval > 0 else 0,
                'target': robot_config.camera_fps,
                'avg_interval_ms': np.mean(intervals_ms),
                'min_interval_ms': np.min(intervals_ms),
                'max_interval_ms': np.max(intervals_ms),
            }
        else:
            stats['frequency'] = None
        
        return stats
    
    def print_periodic_stats(self):
        """Print periodic statistics (called during polling)."""
        if len(self.poll_times) < 50:
            return
        
        avg_poll_time = np.mean(list(self.poll_times)[-100:])
        max_poll_time = np.max(list(self.poll_times)[-100:])
        
        avg_detect_time = np.mean(list(self.detect_times)[-100:]) if self.detect_times else 0
        avg_pose_time = np.mean(list(self.pose_compute_times)[-100:]) if self.pose_compute_times else 0
        
        if len(self.poll_intervals) >= 50:
            avg_interval = np.mean(list(self.poll_intervals)[-100:])
            actual_freq = 1.0 / avg_interval if avg_interval > 0 else 0
        else:
            actual_freq = 0
        
        print(f"[ARUCO POLL #{self.poll_count}] "
              f"freq={actual_freq:.1f}Hz (target={robot_config.camera_fps:.1f}Hz), "
              f"total={avg_poll_time*1000:.1f}ms (detect={avg_detect_time*1000:.1f}ms, "
              f"pose={avg_pose_time*1000:.1f}ms, max={max_poll_time*1000:.1f}ms)")
    
    def print_final_stats(self):
        """Print final statistics."""
        stats = self.get_statistics()
        
        if stats['poll_count'] == 0:
            return
        
        print(f"\n{'='*60}")
        print(f"ARUCO POLLING THREAD STATISTICS")
        print(f"{'='*60}")
        print(f"  Total polls: {stats['poll_count']}")
        
        if stats['poll_time']:
            pt = stats['poll_time']
            print(f"  Total poll time: avg={pt['avg']:.2f}ms, min={pt['min']:.2f}ms, "
                  f"max={pt['max']:.2f}ms, p95={pt['p95']:.2f}ms")
            
            if stats['detect_time']:
                dt = stats['detect_time']
                print(f"    - Detection: avg={dt['avg']:.2f}ms, max={dt['max']:.2f}ms")
            
            if stats['pose_time']:
                pt = stats['pose_time']
                print(f"    - Pose computation: avg={pt['avg']:.2f}ms, max={pt['max']:.2f}ms")
        
        if stats['frequency']:
            freq = stats['frequency']
            efficiency = (freq['actual'] / freq['target']) * 100 if freq['target'] > 0 else 0
            print(f"  Poll interval: avg={freq['avg_interval_ms']:.2f}ms, "
                  f"min={freq['min_interval_ms']:.2f}ms, max={freq['max_interval_ms']:.2f}ms")
            print(f"  Actual frequency: avg={freq['actual']:.2f}Hz, min={1.0/(freq['max_interval_ms']/1000.0):.2f}Hz, "
                  f"max={1.0/(freq['min_interval_ms']/1000.0):.2f}Hz")
            print(f"  Target frequency: {freq['target']:.1f}Hz")
            print(f"  Efficiency: {efficiency:.1f}% of target")
            
            # Check if keeping up with target
            target_dt = 1.0 / freq['target']
            if stats['poll_time'] and stats['poll_time']['avg'] > target_dt * 800:  # 80% of budget
                print(f"  ⚠️  WARNING: Poll time ({stats['poll_time']['avg']:.1f}ms) close to budget ({target_dt*1000:.1f}ms)")
                print(f"     Consider reducing camera_fps or optimizing detection")
            elif stats['poll_time']:
                budget_pct = (stats['poll_time']['avg'] / (target_dt * 1000)) * 100
                print(f"  ✓ Poll time well within budget (using {budget_pct:.1f}% of period)")
        
        print(f"{'='*60}\n")


class LightweightProfiler:
    """
    Lightweight profiler that tracks performance but only prints warnings when thresholds exceeded.
    
    Tracks timing statistics but doesn't flood console with periodic reports.
    Only alerts when issues detected (excess polling time, control loop failures, etc.)
    """
    
    def __init__(self):
        """Initialize profiler with tracking but minimal output."""
        self.window_size = robot_config.profiler_window_size
        
        # Track control loop iteration timing (for detecting failures)
        self.control_loop_iteration_times = deque(maxlen=self.window_size)
        self.control_loop_intervals = deque(maxlen=self.window_size)
        self.last_control_loop_timestamp = None
        self.total_iterations = 0
        self.missed_deadlines = 0
        self.expected_dt = 1.0 / robot_config.control_frequency
        self.deadline_threshold = robot_config.deadline_threshold_factor
        self._blocking_iteration_count = 0
        
        # Track camera processing time (for detecting issues)
        self.total_camera_time = deque(maxlen=self.window_size)
        self.camera_read_fail_count = 0
        self.camera_read_count = 0
        
        # Track frame storage/polling operations (for performance analysis)
        self.frame_storage_times = deque(maxlen=self.window_size)
        self.frame_poll_times = deque(maxlen=self.window_size)  # Time to read frame from camera
        self.total_frame_ops_time = deque(maxlen=self.window_size)  # Frame poll + storage combined
        
        # Warning tracking (prevent spam)
        self._last_warning_time = {}
        self._warning_cooldown = 5.0  # Only warn once every 5 seconds per issue type
    
    def record_control_loop_iteration(self, elapsed_time):
        """Record control loop iteration timing and check for issues."""
        now = time.perf_counter()
        
        # Filter out blocking operations (save_trajectory, etc.)
        if elapsed_time < robot_config.blocking_outlier_threshold_outer_loop:
            self.control_loop_iteration_times.append(elapsed_time)
            self.total_iterations += 1
            
            # Check for missed deadline
            if elapsed_time > (self.expected_dt * self.deadline_threshold):
                self.missed_deadlines += 1
                self._check_and_warn_control_loop()
        else:
            self._blocking_iteration_count += 1
            self.total_iterations += 1
        
        # Track interval between iterations
        if self.last_control_loop_timestamp is not None:
            interval = now - self.last_control_loop_timestamp
            if interval < robot_config.blocking_interval_threshold:
                self.control_loop_intervals.append(interval)
        self.last_control_loop_timestamp = now
    
    def record_total_camera_time(self, elapsed_time, read_success):
        """Record camera processing time and check for issues."""
        self.total_camera_time.append(elapsed_time)
        self.camera_read_count += 1
        if not read_success:
            self.camera_read_fail_count += 1
            self._check_and_warn_camera()
    
    def record_frame_poll_time(self, elapsed_time):
        """Record camera frame read/poll time."""
        self.frame_poll_times.append(elapsed_time)
    
    def record_frame_storage_time(self, elapsed_time):
        """Record frame storage/downscaling time."""
        self.frame_storage_times.append(elapsed_time)
    
    def record_total_frame_ops_time(self, elapsed_time):
        """Record total frame operations time (poll + storage)."""
        self.total_frame_ops_time.append(elapsed_time)
    
    def _check_and_warn_control_loop(self):
        """Check control loop performance and warn if issues detected."""
        if not robot_config.warning_only_mode:
            return
        
        now = time.perf_counter()
        last_warn = self._last_warning_time.get('control_loop', 0)
        if now - last_warn < self._warning_cooldown:
            return
        
        if len(self.control_loop_iteration_times) < 50:
            return
        
        stats = self._get_control_loop_stats()
        deadline_ms = stats['deadline_threshold_ms']
        
        # Warn if missed deadline rate is high
        if stats['missed_deadline_rate'] >= robot_config.missed_deadline_warning_threshold:
            print(f"\n⚠️  CONTROL LOOP WARNING: {stats['missed_deadline_rate']:.1f}% missed deadlines "
                  f"(threshold: {deadline_ms:.1f}ms, actual: {stats['avg_iteration_time']:.1f}ms)")
            self._last_warning_time['control_loop'] = now
        
        # Warn if frequency is too low
        if stats['actual_frequency'] > 0 and stats['target_frequency'] > 0:
            freq_ratio = stats['actual_frequency'] / stats['target_frequency']
            if freq_ratio < 0.90:  # Running at <90% of target
                print(f"\n⚠️  CONTROL LOOP WARNING: Frequency {stats['actual_frequency']:.2f}Hz "
                      f"(target: {stats['target_frequency']:.2f}Hz, {freq_ratio*100:.1f}% of target)")
                self._last_warning_time['control_loop'] = now
    
    def _check_and_warn_camera(self):
        """Check camera performance and warn if issues detected."""
        if not robot_config.warning_only_mode:
            return
        
        now = time.perf_counter()
        last_warn = self._last_warning_time.get('camera', 0)
        if now - last_warn < self._warning_cooldown:
            return
        
        if len(self.total_camera_time) < 50:
            return
        
        # Warn on camera read failures
        if self.camera_read_count > 0:
            failure_rate = (self.camera_read_fail_count / self.camera_read_count) * 100
            if failure_rate > 5.0:  # >5% failure rate
                print(f"\n⚠️  CAMERA WARNING: {failure_rate:.1f}% read failures "
                      f"({self.camera_read_fail_count}/{self.camera_read_count} failed)")
                self._last_warning_time['camera'] = now
        
        # Warn on excessive camera processing time
        times_ms = [t * 1000 for t in list(self.total_camera_time)[-50:]]
        if times_ms:
            p95_time = np.percentile(times_ms, 95)
            deadline_ms = (self.expected_dt * self.deadline_threshold) * 1000
            
            if p95_time > deadline_ms * robot_config.camera_assessment_threshold:
                print(f"\n⚠️  CAMERA WARNING: Processing time {p95_time:.1f}ms exceeds threshold "
                      f"({deadline_ms * robot_config.camera_assessment_threshold:.1f}ms)")
                self._last_warning_time['camera'] = now
    
    def _get_control_loop_stats(self):
        """Get control loop statistics."""
        if not self.control_loop_iteration_times:
            return {}
        
        times_ms = [t * 1000 for t in self.control_loop_iteration_times]
        intervals_ms = [t * 1000 for t in self.control_loop_intervals] if self.control_loop_intervals else []
        
        stats = {
            'avg_iteration_time': np.mean(times_ms),
            'p95_iteration_time': np.percentile(times_ms, 95) if len(times_ms) > 1 else times_ms[0],
            'missed_deadline_rate': (self.missed_deadlines / max(self.total_iterations, 1)) * 100,
            'deadline_threshold_ms': (self.expected_dt * self.deadline_threshold) * 1000,
            'target_frequency': robot_config.control_frequency,
        }
        
        if intervals_ms:
            avg_interval = np.mean(intervals_ms) / 1000.0
            stats['actual_frequency'] = 1.0 / avg_interval if avg_interval > 0 else 0
        else:
            stats['actual_frequency'] = 0
        
        return stats
    
    def get_final_stats(self):
        """Get final statistics for shutdown reporting."""
        stats = {
            'control_loop': self._get_control_loop_stats(),
            'total_iterations': self.total_iterations,
            'missed_deadlines': self.missed_deadlines,
            'blocking_operations': self._blocking_iteration_count,
            'camera_read_failures': self.camera_read_fail_count,
            'camera_read_count': self.camera_read_count,
        }
        
        # Frame operation statistics
        if self.frame_poll_times:
            times_ms = [t * 1000 for t in list(self.frame_poll_times)]
            stats['frame_poll'] = {
                'avg': np.mean(times_ms),
                'min': np.min(times_ms),
                'max': np.max(times_ms),
                'p95': np.percentile(times_ms, 95) if len(times_ms) > 1 else times_ms[0],
                'p99': np.percentile(times_ms, 99) if len(times_ms) > 1 else times_ms[0],
                'count': len(times_ms)
            }
        else:
            stats['frame_poll'] = None
        
        if self.frame_storage_times:
            times_ms = [t * 1000 for t in list(self.frame_storage_times)]
            stats['frame_storage'] = {
                'avg': np.mean(times_ms),
                'min': np.min(times_ms),
                'max': np.max(times_ms),
                'p95': np.percentile(times_ms, 95) if len(times_ms) > 1 else times_ms[0],
                'p99': np.percentile(times_ms, 99) if len(times_ms) > 1 else times_ms[0],
                'count': len(times_ms)
            }
        else:
            stats['frame_storage'] = None
        
        if self.total_frame_ops_time:
            times_ms = [t * 1000 for t in list(self.total_frame_ops_time)]
            stats['total_frame_ops'] = {
                'avg': np.mean(times_ms),
                'min': np.min(times_ms),
                'max': np.max(times_ms),
                'p95': np.percentile(times_ms, 95) if len(times_ms) > 1 else times_ms[0],
                'p99': np.percentile(times_ms, 99) if len(times_ms) > 1 else times_ms[0],
                'count': len(times_ms)
            }
        else:
            stats['total_frame_ops'] = None
        
        return stats


class TeleopCameraControlEncoders(TeleopControl):
    """
    Teleop control that polls true encoder values and records them.
    
    Key features:
    - Polls actual robot encoders using GroupSyncRead
    - Uses encoder values to compute ground-truth FK (separate MuJoCo instance)
    - Records encoder values in trajectory
    - Records joint angles derived from encoders (not MuJoCo estimates)
    - Lightweight profiling with warning-only output mode
    - High-frequency ArUco polling (camera FPS) with low-frequency logging (control frequency)
    
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
        self.latest_object_pose = None
        self.latest_world_pose = None
        self.latest_gripper_pose = None
        
        # Latest encoder state (updated by encoder polling)
        self.latest_encoder_values = {}
        self.latest_joint_angles_from_encoders = None
        self.latest_ee_pose_from_encoders = None
        
        # Latest ArUco observations (updated every outer loop iteration, for gym env get_obs)
        # Updated at camera FPS (30 Hz) by background thread, read at control frequency (10 Hz) for logging
        self.latest_aruco_obs = {
            'aruco_ee_in_world': np.zeros(7),
            'aruco_object_in_world': np.zeros(7),
            'aruco_ee_in_object': np.zeros(7),
            'aruco_object_in_ee': np.zeros(7),
            'aruco_visibility': np.zeros(3)
        }
        
        # Separate MuJoCo instance for encoder-based forward kinematics only
        self.encoder_fk_model = None
        self.encoder_fk_data = None
        self.encoder_fk_configuration = None
        self.latest_ee_pose_from_encoders_fk = None
        
        # Performance tracking
        self.encoder_poll_times = []
        self.encoder_poll_intervals = []
        self.encoder_poll_count = 0
        self.last_poll_timestamp = None
        self.encoder_lock_wait_times = []
        self.encoder_lock_contention_count = 0
        
        # Lightweight profiler (warning-only mode) - for control loop
        self.profiler = LightweightProfiler()
        
        # ArUco profiler - separate from control loop profiling
        self.aruco_profiler = ArUcoProfiler()
        
        # Frame recording (always enabled when recording)
        self.record_frames = False  # Enabled when actually recording
        
        # High-frequency ArUco polling (runs at camera FPS, separate from control loop)
        self._aruco_polling_active = False
        self._aruco_poll_thread = None
        self._aruco_lock = threading.Lock()  # Protects latest_aruco_obs updates
        self._latest_frame_for_recording = None  # Latest frame for recording (thread-safe)
        
        # Visualization
        self.show_video = True
        self.show_axes = True
        
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
            if self.aruco_profiler.last_poll_timestamp is not None:
                interval = loop_start - self.aruco_profiler.last_poll_timestamp
                if interval >= 1.0:  # Skip large intervals (gaps)
                    interval = None
            self.aruco_profiler.last_poll_timestamp = loop_start
            
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
            
            # Update latest observations and frame (thread-safe)
            with self._aruco_lock:
                self.latest_aruco_obs = {k: v.copy() for k, v in obs.items()}
                # Store latest frame for recording (downscaled)
                if self.is_recording:
                    downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
                    downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
                    self._latest_frame_for_recording = cv2.resize(
                        frame, (downscaled_width, downscaled_height), interpolation=cv2.INTER_AREA
                    )
            
            # Visualization (only in polling thread to avoid conflicts)
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
            total_poll_time = time.perf_counter() - loop_start
            self.aruco_profiler.record_poll_iteration(
                total_poll_time, detect_time, pose_time, interval
            )
            
            # Print periodic stats (every 300 polls = ~10 seconds at 30 Hz)
            if self.aruco_profiler.poll_count % 300 == 0:
                self.aruco_profiler.print_periodic_stats()
            
            rate_limiter.sleep()
    
    def _compute_relative_pose(self, r_ref, t_ref, r_tgt, t_tgt):
        """Compute target pose relative to reference frame. Returns (pos, quat_wxyz) or (zeros, zeros)."""
        if r_ref is None or t_ref is None or r_tgt is None or t_tgt is None:
            return np.zeros(3), np.array([1., 0., 0., 0.])
            
        r_rel, t_rel = self.estimator.get_relative_pose(r_ref, t_ref, r_tgt, t_tgt)
        R_rel, _ = cv2.Rodrigues(r_rel)
        quat_xyzw = R.from_matrix(R_rel).as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return t_rel.flatten(), quat_wxyz
        
    def on_ready(self):
        """Initialize camera and estimator."""
        super().on_ready()
        
        print("\n" + "="*60)
        print("Initializing Camera & ArUco Estimator...")
        print(f"Encoder polling: At recording rate ({robot_config.control_frequency} Hz)")
        print(f"ArUco polling: At camera FPS ({robot_config.camera_fps} Hz) in background thread")
        print(f"Trajectory logging: At control frequency ({robot_config.control_frequency} Hz)")
        
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
            self.camera = Camera(device=self.camera_id, width=self.width, height=self.height, fps=self.fps)
            self.camera.start()
            
            self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
            self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(self.width, self.height)
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
            parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            
            print(f"✓ Camera started: Device {self.camera_id} @ {self.width}x{self.height}")
            print("✓ ArUco Estimator ready (IDs: 0=World, 2=Object)")
            
            # Start high-frequency ArUco polling thread (at camera FPS)
            self._start_aruco_polling()
        except Exception as e:
            print(f"❌ Error starting camera: {e}")
            self.camera = None
            
        print("="*60 + "\n")
    
    def _poll_encoders(self, outer_loop_start_time=None):
        """
        Poll encoder values from robot hardware using fast bulk read.
        
        Uses opportunistic read strategy: skip if insufficient time remaining
        in outer loop period to avoid blocking.
        """
        poll_start = time.perf_counter()
        
        # Opportunistic read strategy
        if outer_loop_start_time is not None:
            outer_loop_period = 1.0 / robot_config.control_frequency
            time_elapsed = poll_start - outer_loop_start_time
            time_remaining = outer_loop_period - time_elapsed
            
            # Safe margin: need at least 30ms to attempt read
            if time_remaining < 0.030:  # 30ms
                if not hasattr(self, '_skipped_reads_count'):
                    self._skipped_reads_count = 0
                self._skipped_reads_count += 1
                # Only warn if skipping becomes frequent
                if self._skipped_reads_count % 100 == 0 and robot_config.warning_only_mode:
                    print(f"⚠️  Skipped {self._skipped_reads_count} encoder reads (insufficient time)")
                return
        
        # Track interval since last poll
        if self.last_poll_timestamp is not None:
            interval = poll_start - self.last_poll_timestamp
            if interval < 1.0:  # Only track reasonable intervals
                self.encoder_poll_intervals.append(interval)
        else:
            interval = 0.0
        
        # Read encoders using bulk read
        if hasattr(self, '_driver_lock') and self._driver_lock is not None:
            lock_acquire_start = time.perf_counter()
            with self._driver_lock:
                lock_acquire_time = time.perf_counter() - lock_acquire_start
                if lock_acquire_time > robot_config.lock_wait_threshold:
                    self.encoder_lock_wait_times.append(lock_acquire_time)
                    self.encoder_lock_contention_count += 1
                    if len(self.encoder_lock_wait_times) > robot_config.max_profiling_samples:
                        self.encoder_lock_wait_times.pop(0)
                encoder_values = self.robot_driver.read_all_encoders(
                    max_retries=1, retry_delay=0.01, use_bulk_read=True
                )
        else:
            encoder_values = self.robot_driver.read_all_encoders(
                max_retries=1, retry_delay=0.01, use_bulk_read=True
            )
        
        # Update encoder state
        if encoder_values:
            self.latest_encoder_values = encoder_values.copy()
            
            try:
                robot_joint_angles = encoders_to_joint_angles(encoder_values, self.translator)
                self.latest_joint_angles_from_encoders = robot_joint_angles.copy()
                
                # Compute EE pose from encoders using temporary MuJoCo computation
                temp_data = mujoco.MjData(self.model)
                temp_data.qpos[:5] = robot_joint_angles[:5]
                if len(temp_data.qpos) > 5:
                    temp_data.qpos[5] = robot_joint_angles[5]
                mujoco.mj_forward(self.model, temp_data)
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                actual_position = temp_data.site(site_id).xpos.copy()
                actual_xmat = temp_data.site(site_id).xmat.reshape(3, 3)
                actual_quat = R.from_matrix(actual_xmat).as_quat()
                actual_orientation_quat_wxyz = np.array([actual_quat[3], actual_quat[0], actual_quat[1], actual_quat[2]])
                self.latest_ee_pose_from_encoders = (actual_position.copy(), actual_orientation_quat_wxyz.copy())
                
                # Update separate FK-only MuJoCo instance
                if self.encoder_fk_model is not None:
                    self.encoder_fk_data.qpos[:5] = robot_joint_angles[:5]
                    if len(self.encoder_fk_data.qpos) > 5:
                        self.encoder_fk_data.qpos[5] = robot_joint_angles[5]
                    
                    mujoco.mj_forward(self.encoder_fk_model, self.encoder_fk_data)
                    self.encoder_fk_configuration.update(self.encoder_fk_data.qpos)
                    
                    site_id = mujoco.mj_name2id(self.encoder_fk_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
                    fk_position = self.encoder_fk_data.site(site_id).xpos.copy()
                    fk_xmat = self.encoder_fk_data.site(site_id).xmat.reshape(3, 3)
                    fk_quat = R.from_matrix(fk_xmat).as_quat()
                    fk_orientation_quat_wxyz = np.array([fk_quat[3], fk_quat[0], fk_quat[1], fk_quat[2]])
                    
                    self.latest_ee_pose_from_encoders_fk = (fk_position.copy(), fk_orientation_quat_wxyz.copy())
                
            except Exception as e:
                if robot_config.warning_only_mode:
                    # Only print warnings occasionally to avoid spam
                    if not hasattr(self, '_last_encoder_error_time'):
                        self._last_encoder_error_time = 0
                    now = time.perf_counter()
                    if now - self._last_encoder_error_time > 5.0:
                        print(f"⚠️  Warning: Failed to sync encoders to MuJoCo: {e}")
                        self._last_encoder_error_time = now
        
        # Track performance
        poll_duration = time.perf_counter() - poll_start
        self.encoder_poll_times.append(poll_duration)
        self.encoder_poll_count += 1
        self.last_poll_timestamp = poll_start
        
        # Print encoder poll stats periodically (only if interval configured and not in warning-only mode)
        if (robot_config.encoder_poll_stats_interval > 0 and 
            not robot_config.warning_only_mode and
            self.encoder_poll_count % robot_config.encoder_poll_stats_interval == 0):
            if len(self.encoder_poll_times) >= 50:
                avg_poll_time = np.mean(self.encoder_poll_times[-50:])
                if len(self.encoder_poll_intervals) >= 50:
                    avg_interval = np.mean(self.encoder_poll_intervals[-50:])
                    avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                    expected_freq = robot_config.control_frequency
                    
                    lock_info = ""
                    if len(self.encoder_lock_wait_times) > 0:
                        recent_waits = self.encoder_lock_wait_times[-50:] if len(self.encoder_lock_wait_times) >= 50 else self.encoder_lock_wait_times
                        if recent_waits:
                            avg_wait = np.mean(recent_waits)
                            max_wait = np.max(recent_waits)
                            contention_pct = (len(recent_waits) / 50) * 100 if len(recent_waits) < 50 else 100
                            lock_info = f" | outer_waits: avg={avg_wait*1000:.1f}ms, max={max_wait*1000:.1f}ms ({contention_pct:.0f}% of reads)"
                    
                    print(f"[ENCODER POLL #{self.encoder_poll_count}] "
                          f"avg_read={avg_poll_time*1000:.1f}ms, "
                          f"avg_freq={avg_freq:.1f}Hz (target={expected_freq:.1f}Hz){lock_info}")
        
        # Warn on excessive encoder read times
        if robot_config.warning_only_mode and len(self.encoder_poll_times) >= 50:
            recent_times = self.encoder_poll_times[-50:]
            avg_time = np.mean(recent_times)
            max_time = np.max(recent_times)
            # Warn if average > 15ms or max > 20ms (indicates potential issues)
            if avg_time > 0.015 or max_time > 0.020:
                if not hasattr(self, '_last_encoder_warning_time'):
                    self._last_encoder_warning_time = 0
                now = time.perf_counter()
                if now - self._last_encoder_warning_time > 5.0:
                    print(f"⚠️  ENCODER WARNING: Slow reads detected (avg={avg_time*1000:.1f}ms, max={max_time*1000:.1f}ms)")
                    self._last_encoder_warning_time = now
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt, outer_loop_start_time=None):
        """
        Poll encoders and update trajectory if recording.
        
        ArUco detection now runs in background thread at camera FPS (30 Hz).
        This method only reads the latest ArUco observations (thread-safe) and stores them at control frequency (10 Hz).
        """
        iteration_start = time.perf_counter()
        
        # Handle GUI commands
        self._handle_control_input()
        
        # Poll encoders
        self._poll_encoders(outer_loop_start_time=outer_loop_start_time)
        
        # Read latest ArUco observations (thread-safe, updated at camera FPS by background thread)
        with self._aruco_lock:
            obs = {k: v.copy() for k, v in self.latest_aruco_obs.items()}
        
        # Compute object_pose_data for backward compatibility
        object_pose_data = np.zeros(7)
        object_visible = 0.0
        if obs['aruco_visibility'][0] and obs['aruco_visibility'][1]:
            object_pose_data = obs['aruco_object_in_world']
            object_visible = 1.0
        
        # Get latest frame for recording (thread-safe, captured by ArUco polling thread at camera FPS)
        frame_to_record = None
        if self.is_recording:
            with self._aruco_lock:
                if self._latest_frame_for_recording is not None:
                    frame_to_record = self._latest_frame_for_recording.copy()
        
        # Recording logic
        if self.is_recording:
            if self.recording_start_time is None:
                self.recording_start_time = time.perf_counter()
                # Always enable frame recording when recording starts
                self.record_frames = True
            
            ik_ee_position, ik_ee_orientation_quat_wxyz = self._get_current_ee_pose()
            ee_pose_target = np.concatenate([ik_ee_position, ik_ee_orientation_quat_wxyz])
            
            if self.latest_joint_angles_from_encoders is not None:
                state = self.latest_joint_angles_from_encoders.copy()
            else:
                state = np.concatenate([self.configuration.q[:5], [gripper_target]])
            
            action = np.concatenate([velocity_world, angular_velocity_world, [gripper_target]])
            
            self.trajectory.append({
                'timestamp': time.perf_counter() - self.recording_start_time,
                'state': state.copy(),
                'action': action.copy(),
                'ee_pose_target': ee_pose_target.copy()
            })
            
            if self.trajectory:
                current_step = self.trajectory[-1]
                
                encoder_array = np.array([
                    self.latest_encoder_values.get(mid, 0) 
                    for mid in robot_config.motor_ids
                ])
                current_step['encoder_values'] = encoder_array
                
                if self.latest_ee_pose_from_encoders_fk is not None:
                    fk_position, fk_orientation = self.latest_ee_pose_from_encoders_fk
                    ee_pose_encoder = np.concatenate([fk_position, fk_orientation])
                    current_step['ee_pose_encoder'] = ee_pose_encoder.copy()
                else:
                    current_step['ee_pose_encoder'] = np.zeros(7)
                
                current_step['object_pose'] = object_pose_data
                current_step['object_visible'] = np.array([object_visible])
                current_step['aruco_ee_in_world'] = obs['aruco_ee_in_world']
                current_step['aruco_object_in_world'] = obs['aruco_object_in_world']
                current_step['aruco_ee_in_object'] = obs['aruco_ee_in_object']
                current_step['aruco_object_in_ee'] = obs['aruco_object_in_ee']
                current_step['aruco_visibility'] = obs['aruco_visibility']
                
                angular_vel = angular_velocity_world
                if dt > 0.0:
                    axis_angle_vec = angular_vel * dt
                else:
                    axis_angle_vec = np.zeros(3)
                
                augmented_actions = np.concatenate([
                    velocity_world,
                    angular_velocity_world,
                    axis_angle_vec,
                    [gripper_target]
                ])
                current_step['augmented_actions'] = augmented_actions
                
                # Record camera frame (captured by ArUco polling thread at camera FPS, stored here at control frequency)
                if self.record_frames:
                    if frame_to_record is not None:
                        # Store frame captured from ArUco polling thread
                        current_step['camera_frame'] = frame_to_record
                    else:
                        # Store empty frame if recording but no frame available yet
                        downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
                        downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
                        current_step['camera_frame'] = np.zeros(
                            (downscaled_height, downscaled_width, 3),
                            dtype=np.uint8
                        )
        
        # Record control loop iteration timing
        iteration_time = time.perf_counter() - iteration_start
        self.profiler.record_control_loop_iteration(iteration_time)
    
    def shutdown(self):
        """Cleanup camera and print final statistics."""
        # Stop ArUco polling thread
        self._aruco_polling_active = False
        if self._aruco_poll_thread is not None:
            self._aruco_poll_thread.join(timeout=1.0)
        
        # Print frame storage statistics if recording
        if self.is_recording and self.record_frames and self.trajectory:
            num_frames = sum(1 for t in self.trajectory if 'camera_frame' in t)
            if num_frames > 0:
                # Frames are saved at 1/4 resolution (1/4 width, 1/4 height = 1/16 pixels)
                downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
                downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
                frame_size_bytes = (downscaled_width * downscaled_height * 3)  # BGR uint8
                frame_size_mb = frame_size_bytes / (1024 * 1024)
                total_size_mb = num_frames * frame_size_mb
                print(f"\n📹 Frame Storage Summary:")
                print(f"  Frames recorded: {num_frames}")
                print(f"  Frame resolution: {downscaled_width}x{downscaled_height} (1/{robot_config.frame_downscale_factor} of {robot_config.camera_width}x{robot_config.camera_height})")
                print(f"  Frame size: {frame_size_mb:.3f} MB per frame (downscaled)")
                print(f"  Total frame data: ~{total_size_mb:.1f} MB (uncompressed)")
                print(f"  Note: NPZ compression will reduce file size significantly")
                print(f"  Note: ArUco detection uses full-resolution frames (not downscaled)")
        
        super().shutdown()
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Print encoder read statistics
        if self.robot_driver:
            encoder_stats = self.robot_driver.get_encoder_read_stats()
            if encoder_stats['total_reads'] > 0:
                print(f"\n{'='*60}")
                print(f"ENCODER READ STATISTICS (Hardware Communication)")
                print(f"{'='*60}")
                print(f"  Total reads: {encoder_stats['total_reads']}")
                print(f"  Successful: {encoder_stats['successful_reads']} ({encoder_stats.get('success_rate', 0):.1f}%)")
                print(f"  Failed: {encoder_stats['failed_reads']} ({encoder_stats.get('failure_rate', 0):.1f}%)")
                
                if 'txrx_times_ms' in encoder_stats:
                    txrx = encoder_stats['txrx_times_ms']
                    print(f"\n  txRxPacket() Timing:")
                    print(f"    avg={txrx['avg']:.2f}ms, min={txrx['min']:.2f}ms, max={txrx['max']:.2f}ms")
                    print(f"    p95={txrx['p95']:.2f}ms, p99={txrx['p99']:.2f}ms")
                    print(f"    Near timeout (18-22ms): {txrx.get('near_timeout_pct', 0):.1f}%")
                
                print(f"{'='*60}\n")
        
        # Print skipped reads count
        if hasattr(self, '_skipped_reads_count') and self._skipped_reads_count > 0:
            print(f"\n{'='*60}")
            print(f"OPPORTUNISTIC READ STATISTICS")
            print(f"{'='*60}")
            print(f"  Skipped reads: {self._skipped_reads_count}")
            skip_rate = (self._skipped_reads_count / (self._skipped_reads_count + self.encoder_poll_count)) * 100
            print(f"  Skip rate: {skip_rate:.1f}%")
            print(f"{'='*60}\n")
        
        # Print final encoder polling stats
        if self.encoder_poll_count > 0:
            avg_poll_time = np.mean(self.encoder_poll_times)
            max_poll_time = np.max(self.encoder_poll_times)
            
            if len(self.encoder_poll_intervals) > 0:
                avg_interval = np.mean(self.encoder_poll_intervals)
                avg_freq = 1.0 / avg_interval if avg_interval > 0 else 0
                efficiency = (avg_freq / robot_config.control_frequency) * 100 if robot_config.control_frequency > 0 else 0
            else:
                avg_freq = 0.0
                efficiency = 0.0
            
            print(f"\n{'='*60}")
            print(f"ENCODER POLLING STATISTICS (Final)")
            print(f"{'='*60}")
            print(f"  Total polls: {self.encoder_poll_count}")
            print(f"  Read time: avg={avg_poll_time*1000:.2f}ms, max={max_poll_time*1000:.2f}ms")
            if len(self.encoder_poll_intervals) > 0:
                print(f"  Actual frequency: avg={avg_freq:.1f}Hz (target={robot_config.control_frequency:.1f}Hz)")
                print(f"  Efficiency: {efficiency:.1f}% of expected")
            print(f"{'='*60}\n")
        
        # Print ArUco polling thread statistics
        self.aruco_profiler.print_final_stats()
        
        # Print final profiler stats (summary)
        final_stats = self.profiler.get_final_stats()
        if final_stats['total_iterations'] > 0:
            print(f"\n{'='*60}")
            print(f"CONTROL LOOP PERFORMANCE SUMMARY")
            print(f"{'='*60}")
            cl_stats = final_stats['control_loop']
            if cl_stats:
                print(f"  Total iterations: {final_stats['total_iterations']}")
                print(f"  Missed deadlines: {final_stats['missed_deadlines']} ({cl_stats.get('missed_deadline_rate', 0):.1f}%)")
                if cl_stats.get('actual_frequency', 0) > 0:
                    print(f"  Frequency: {cl_stats['actual_frequency']:.2f}Hz (target: {cl_stats['target_frequency']:.2f}Hz)")
                if cl_stats.get('avg_iteration_time', 0) > 0:
                    print(f"  Avg iteration time: {cl_stats['avg_iteration_time']:.2f}ms")
                    print(f"    Note: ArUco detection now in separate thread (not included in this time)")
                if final_stats['blocking_operations'] > 0:
                    blocking_pct = (final_stats['blocking_operations'] / final_stats['total_iterations']) * 100
                    print(f"  Blocking operations: {final_stats['blocking_operations']} ({blocking_pct:.1f}%)")
            if final_stats['camera_read_count'] > 0:
                failure_rate = (final_stats['camera_read_failures'] / final_stats['camera_read_count']) * 100
                if failure_rate > 0:
                    print(f"  Camera read failures: {final_stats['camera_read_failures']}/{final_stats['camera_read_count']} ({failure_rate:.1f}%)")
            
            # Frame operation statistics
            if final_stats.get('frame_poll') is not None:
                fp = final_stats['frame_poll']
                print(f"\n📹 Frame Polling (camera.read()):")
                print(f"  avg={fp['avg']:.2f}ms, min={fp['min']:.2f}ms, max={fp['max']:.2f}ms, p95={fp['p95']:.2f}ms, p99={fp['p99']:.2f}ms")
                print(f"  count={fp['count']}")
            
            if final_stats.get('frame_storage') is not None:
                fs = final_stats['frame_storage']
                print(f"\n💾 Frame Storage (resize/copy):")
                print(f"  avg={fs['avg']:.2f}ms, min={fs['min']:.2f}ms, max={fs['max']:.2f}ms, p95={fs['p95']:.2f}ms, p99={fs['p99']:.2f}ms")
                print(f"  count={fs['count']}")
            
            if final_stats.get('total_frame_ops') is not None:
                tfo = final_stats['total_frame_ops']
                print(f"\n⏱️  Total Frame Operations (poll + storage):")
                print(f"  avg={tfo['avg']:.2f}ms, min={tfo['min']:.2f}ms, max={tfo['max']:.2f}ms, p95={tfo['p95']:.2f}ms, p99={tfo['p99']:.2f}ms")
                print(f"  count={tfo['count']}")
                # Show percentage of outer loop budget used
                outer_loop_budget = (1.0 / robot_config.control_frequency) * 1000  # Convert to ms
                budget_pct = (tfo['avg'] / outer_loop_budget) * 100
                print(f"  Avg frame ops use {budget_pct:.1f}% of outer loop budget ({outer_loop_budget:.1f}ms)")
            
            print(f"{'='*60}\n")
        
        super().shutdown()


def main():
    parser = argparse.ArgumentParser(description='WX200 Teleop with Camera and Encoder Polling')
    parser.add_argument('--output', type=str, help='Output filename')
    parser.add_argument('--camera-id', type=int, default=None, help='Camera device ID (defaults to robot_config.camera_id)')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose profiling output (disables warning-only mode)')
    args = parser.parse_args()
    
    # Override warning-only mode if verbose requested
    if args.verbose:
        robot_config.warning_only_mode = False
        robot_config.encoder_poll_stats_interval = 100
        robot_config.control_perf_stats_interval = 500
    
    # Create and run
    controller = TeleopCameraControlEncoders(
        enable_recording=True,  # Always enable recording capability; GUI decides when to actually record
        output_path=args.output,
        camera_id=args.camera_id
    )
    
    if args.no_vis:
        controller.show_video = False
        
    controller.run()


if __name__ == "__main__":
    main()
