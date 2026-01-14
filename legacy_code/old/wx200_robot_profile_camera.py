"""
Profile camera and ArUco processing performance during robot control.

This script profiles the camera and ArUco processing pipeline to assess
feasibility of 20Hz camera polling while running robot control.

Measures timing of:
- Camera frame capture (read())
- Color conversion (BGR to grayscale)
- ArUco marker detection
- Pose estimation (solvePnP) for each marker
- Relative pose computation
- Frame storage (when recording frames to trajectory)
- Total observation computation time
- Control loop jitter and missed deadlines

Usage:
    python wx200_robot_profile_camera.py [--camera-id ID] [--no-vis] [--profile-interval N]
    
    --camera-id: Camera device ID (default: robot_config.camera_id)
    --no-vis: Disable video window (reduces overhead)
    --record: Enable trajectory recording (for comparison with normal operation)
    --record-frames: Record camera frames in trajectory (requires --record, saved at 1/4 resolution, ~0.375MB per frame)
    --output: Output filename if recording
    --profile-interval: Print stats every N iterations (default: 100)

The script will print periodic statistics showing:
- Component-wise timing (camera read, color conversion, ArUco detection, etc.)
- Control loop timing and frequency
- Missed deadline rate
- Performance assessment
"""
import cv2
import numpy as np
import argparse
import time
from collections import deque

from wx200_robot_collect_demo_encoders import TeleopCameraControlEncoders
from robot_control.robot_config import robot_config
from camera import MARKER_SIZE

# Shorter axes for visualization
AXIS_LENGTH = MARKER_SIZE * robot_config.aruco_axis_length_scale


class CameraProfiler:
    """Profiler for camera and ArUco operations."""
    
    def __init__(self, window_size=None):
        """Initialize profiler with sliding window for statistics."""
        self.window_size = window_size if window_size is not None else robot_config.profiler_window_size
        
        # Component timings (in seconds, will convert to ms for reporting)
        self.camera_read_times = deque(maxlen=window_size)
        self.color_convert_times = deque(maxlen=window_size)
        self.aruco_detect_times = deque(maxlen=window_size)
        self.process_tag_times = {'world': deque(maxlen=window_size),
                                  'object': deque(maxlen=window_size),
                                  'ee': deque(maxlen=window_size)}
        self.relative_pose_times = deque(maxlen=window_size)
        self.frame_storage_times = deque(maxlen=window_size)  # Time to copy/store frame
        self.total_camera_time = deque(maxlen=window_size)
        
        # Control loop timing
        self.control_loop_iteration_times = deque(maxlen=window_size)
        self.control_loop_intervals = deque(maxlen=window_size)
        self.last_control_loop_timestamp = None
        
        # Frame tracking
        self.frame_read_count = 0
        self.frame_read_fail_count = 0
        self.total_iterations = 0
        
        # Control loop deadline tracking
        self.expected_dt = 1.0 / robot_config.control_frequency
        self.missed_deadlines = 0
        self.deadline_threshold = robot_config.deadline_threshold_factor
        
    def record_camera_read(self, elapsed_time, success):
        """Record camera read() call timing."""
        self.camera_read_times.append(elapsed_time)
        self.frame_read_count += 1
        if not success:
            self.frame_read_fail_count += 1
    
    def record_color_convert(self, elapsed_time):
        """Record color conversion timing."""
        self.color_convert_times.append(elapsed_time)
    
    def record_aruco_detect(self, elapsed_time):
        """Record ArUco detection timing."""
        self.aruco_detect_times.append(elapsed_time)
    
    def record_process_tag(self, tag_name, elapsed_time):
        """Record tag processing timing."""
        if tag_name in self.process_tag_times:
            self.process_tag_times[tag_name].append(elapsed_time)
    
    def record_relative_pose(self, elapsed_time):
        """Record relative pose computation timing."""
        self.relative_pose_times.append(elapsed_time)
    
    def record_frame_storage(self, elapsed_time):
        """Record frame storage/copy timing."""
        self.frame_storage_times.append(elapsed_time)
    
    def record_total_camera_time(self, elapsed_time):
        """Record total camera processing time."""
        self.total_camera_time.append(elapsed_time)
    
    def record_control_loop_iteration(self, elapsed_time):
        """Record control loop iteration timing."""
        now = time.perf_counter()
        
        # Filter out blocking operations (e.g., save_trajectory) that skew stats
        # Normal outer loop iterations should be < threshold at the configured frequency
        # Blocking operations (save, home, etc.) can take seconds
        if elapsed_time < robot_config.blocking_outlier_threshold_outer_loop:
            # Normal iteration - record it
            self.control_loop_iteration_times.append(elapsed_time)
            self.total_iterations += 1
            
            # Check for missed deadline (only on normal iterations)
            if elapsed_time > (self.expected_dt * self.deadline_threshold):
                self.missed_deadlines += 1
        else:
            # Likely a blocking operation - don't count it but track separately
            if not hasattr(self, '_blocking_iteration_count'):
                self._blocking_iteration_count = 0
            self._blocking_iteration_count += 1
            # Still increment total for accurate percentage calculations
            self.total_iterations += 1
        
        # Track interval between iterations
        if self.last_control_loop_timestamp is not None:
            interval = now - self.last_control_loop_timestamp
            # Also filter interval outliers (gaps above threshold are likely from blocking)
            if interval < robot_config.blocking_interval_threshold:
                self.control_loop_intervals.append(interval)
        self.last_control_loop_timestamp = now
    
    def get_statistics(self):
        """Get current statistics."""
        stats = {}
        
        def _safe_stats(times_list, name):
            if not times_list:
                return {f'{name}_avg': 0, f'{name}_min': 0, f'{name}_max': 0,
                        f'{name}_p50': 0, f'{name}_p95': 0, f'{name}_p99': 0, f'{name}_count': 0}
            times_ms = [t * 1000 for t in times_list]
            return {
                f'{name}_avg': np.mean(times_ms),
                f'{name}_min': np.min(times_ms),
                f'{name}_max': np.max(times_ms),
                f'{name}_p50': np.percentile(times_ms, 50),
                f'{name}_p95': np.percentile(times_ms, 95),
                f'{name}_p99': np.percentile(times_ms, 99),
                f'{name}_count': len(times_ms)
            }
        
        stats.update(_safe_stats(self.camera_read_times, 'camera_read'))
        stats.update(_safe_stats(self.color_convert_times, 'color_convert'))
        stats.update(_safe_stats(self.aruco_detect_times, 'aruco_detect'))
        
        for tag_name in ['world', 'object', 'ee']:
            stats.update(_safe_stats(self.process_tag_times[tag_name], f'process_tag_{tag_name}'))
        
        stats.update(_safe_stats(self.relative_pose_times, 'relative_pose'))
        stats.update(_safe_stats(self.frame_storage_times, 'frame_storage'))
        stats.update(_safe_stats(self.total_camera_time, 'total_camera'))
        stats.update(_safe_stats(self.control_loop_iteration_times, 'control_loop'))
        stats.update(_safe_stats(self.control_loop_intervals, 'control_interval'))
        
        # Frame statistics
        stats['frame_read_success_rate'] = (self.frame_read_count - self.frame_read_fail_count) / max(self.frame_read_count, 1) * 100
        stats['frame_read_count'] = self.frame_read_count
        stats['frame_read_fail_count'] = self.frame_read_fail_count
        
        # Control loop statistics
        stats['total_iterations'] = self.total_iterations
        stats['missed_deadlines'] = self.missed_deadlines
        stats['missed_deadline_rate'] = (self.missed_deadlines / max(self.total_iterations, 1)) * 100
        stats['expected_dt_ms'] = self.expected_dt * 1000
        stats['deadline_threshold_ms'] = self.expected_dt * self.deadline_threshold * 1000
        
        # Actual frequency
        if self.control_loop_intervals:
            avg_interval = np.mean(list(self.control_loop_intervals))
            stats['actual_frequency'] = 1.0 / avg_interval if avg_interval > 0 else 0
            stats['target_frequency'] = robot_config.control_frequency
        else:
            stats['actual_frequency'] = 0
            stats['target_frequency'] = robot_config.control_frequency
        
        return stats
    
    def print_periodic_stats(self):
        """Print periodic statistics."""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print(f"CAMERA PROFILING STATISTICS (Last {self.window_size} samples)")
        print(f"{'='*70}")
        
        print(f"\n📹 Camera Operations:")
        print(f"  Read:      avg={stats['camera_read_avg']:.2f}ms, "
              f"p95={stats['camera_read_p95']:.2f}ms, "
              f"p99={stats['camera_read_p99']:.2f}ms")
        print(f"  Color conv: avg={stats['color_convert_avg']:.2f}ms, "
              f"p95={stats['color_convert_p95']:.2f}ms")
        print(f"  ArUco detect: avg={stats['aruco_detect_avg']:.2f}ms, "
              f"p95={stats['aruco_detect_p95']:.2f}ms, "
              f"p99={stats['aruco_detect_p99']:.2f}ms")
        
        print(f"\n🎯 Pose Estimation (process_tag):")
        for tag_name in ['world', 'object', 'ee']:
            key = f'process_tag_{tag_name}'
            if stats[f'{key}_count'] > 0:
                print(f"  {tag_name:6s}: avg={stats[f'{key}_avg']:.2f}ms, "
                      f"p95={stats[f'{key}_p95']:.2f}ms (count={stats[f'{key}_count']})")
        
        print(f"\n📐 Relative Pose Computation:")
        print(f"  avg={stats['relative_pose_avg']:.2f}ms, "
              f"p95={stats['relative_pose_p95']:.2f}ms")
        
        print(f"\n💾 Frame Storage (if recording):")
        if stats['frame_storage_count'] > 0:
            print(f"  avg={stats['frame_storage_avg']:.2f}ms, "
                  f"p95={stats['frame_storage_p95']:.2f}ms (count={stats['frame_storage_count']})")
        else:
            print(f"  (not recording frames)")
        
        print(f"\n⏱️  Total Camera Processing:")
        print(f"  avg={stats['total_camera_avg']:.2f}ms, "
              f"p95={stats['total_camera_p95']:.2f}ms, "
              f"p99={stats['total_camera_p99']:.2f}ms")
        
        print(f"\n🔄 Control Loop:")
        print(f"  Iteration time: avg={stats['control_loop_avg']:.2f}ms, "
              f"p95={stats['control_loop_p95']:.2f}ms, "
              f"max={stats['control_loop_max']:.2f}ms")
        print(f"  Interval: avg={stats['control_interval_avg']:.2f}ms "
              f"(target={stats['expected_dt_ms']:.2f}ms)")
        print(f"  Frequency: {stats['actual_frequency']:.2f}Hz (target={stats['target_frequency']:.2f}Hz)")
        print(f"  Missed deadlines: {stats['missed_deadlines']}/{stats['total_iterations']} "
              f"({stats['missed_deadline_rate']:.1f}%) "
              f"(threshold={stats['deadline_threshold_ms']:.2f}ms)")
        
        # Report filtered blocking operations if any
        if hasattr(self, '_blocking_iteration_count') and self._blocking_iteration_count > 0:
            blocking_pct = (self._blocking_iteration_count / stats['total_iterations']) * 100
            print(f"  ⚠️  {self._blocking_iteration_count} iterations exceeded {robot_config.blocking_outlier_threshold_outer_loop*1000:.0f}ms (blocking ops like save/home)")
            print(f"     These were excluded from timing stats ({blocking_pct:.1f}% of total)")
        
        print(f"\n📊 Frame Statistics:")
        print(f"  Success rate: {stats['frame_read_success_rate']:.1f}% "
              f"({stats['frame_read_count'] - stats['frame_read_fail_count']}/{stats['frame_read_count']})")
        
        # Performance assessment
        deadline_ms = stats['deadline_threshold_ms']
        total_camera_p95 = stats['total_camera_p95']
        
        print(f"\n✅ Assessment:")
        assessment_threshold = deadline_ms * robot_config.camera_assessment_threshold
        if total_camera_p95 < assessment_threshold:
            print(f"  ✓ Camera processing well within budget ({total_camera_p95:.1f}ms < {assessment_threshold:.1f}ms)")
        elif total_camera_p95 < deadline_ms:
            print(f"  ⚠️  Camera processing close to budget ({total_camera_p95:.1f}ms < {deadline_ms:.1f}ms)")
        else:
            print(f"  ❌ Camera processing exceeds budget ({total_camera_p95:.1f}ms > {deadline_ms:.1f}ms)")
        
        if stats['missed_deadline_rate'] < robot_config.missed_deadline_warning_threshold:
            print(f"  ✓ Control loop timing looks good ({stats['missed_deadline_rate']:.1f}% missed)")
        elif stats['missed_deadline_rate'] < robot_config.missed_deadline_error_threshold:
            print(f"  ⚠️  Some control loop jitter ({stats['missed_deadline_rate']:.1f}% missed)")
        else:
            print(f"  ❌ Significant control loop jitter ({stats['missed_deadline_rate']:.1f}% missed)")
        
        print(f"{'='*70}\n")


class ProfiledCameraControlEncoders(TeleopCameraControlEncoders):
    """Teleop control with camera profiling."""
    
    def __init__(self, enable_recording=False, output_path=None,
                 camera_id=None, width=None, height=None, fps=None,
                 profile_interval=100):
        super().__init__(enable_recording=enable_recording, output_path=output_path,
                        camera_id=camera_id, width=width, height=height, fps=fps)
        
        self.profiler = CameraProfiler(window_size=robot_config.profiler_window_size)
        self.profile_interval = profile_interval  # Print stats every N iterations
        
        # --record flag enables recording capability but doesn't start recording
        # Recording actually starts when GUI 'r' button is pressed
        # So we should NOT set is_recording = True here - let GUI button control it
        # But we do need to track that recording is enabled so the GUI button works
        # The parent class already handles enable_recording properly (enables GUI button)
        
        # Option to record frames (when recording is active via GUI)
        # If --record-frames was passed, enable frame recording when recording starts
        self._record_frames_flag = False  # Will be set by main() if --record-frames passed
        self.record_frames = False  # Only enabled when actually recording
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt, outer_loop_start_time=None):
        """Override to add detailed camera profiling."""
        iteration_start = time.perf_counter()
        camera_start = time.perf_counter()
        
        # Initialize storage (same as parent)
        obs = {
            'aruco_ee_in_world': np.zeros(7),
            'aruco_object_in_world': np.zeros(7),
            'aruco_ee_in_object': np.zeros(7),
            'aruco_object_in_ee': np.zeros(7),
            'aruco_visibility': np.zeros(3)
        }
        object_pose_data = np.zeros(7)
        object_visible = 0.0
        
        # Store frame for recording (if enabled)
        frame_to_record = None
        
        if self.camera:
            # Profile camera.read()
            read_start = time.perf_counter()
            ret, frame = self.camera.read()
            read_time = time.perf_counter() - read_start
            self.profiler.record_camera_read(read_time, ret)
            
            # Store frame for recording if enabled (before any modifications)
            if ret and self.record_frames:
                frame_storage_start = time.perf_counter()
                # Downscale frame for storage (width and height reduced by factor, total pixels reduced by factor^2)
                # ArUco detection still uses full-resolution frame (not downscaled)
                downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
                downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
                frame_to_record = cv2.resize(frame, (downscaled_width, downscaled_height), interpolation=cv2.INTER_AREA)
                frame_storage_time = time.perf_counter() - frame_storage_start
                self.profiler.record_frame_storage(frame_storage_time)
            
            if ret:
                # Profile color conversion
                # Note: ArUco detection uses FULL resolution frame (not downscaled)
                convert_start = time.perf_counter()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                convert_time = time.perf_counter() - convert_start
                self.profiler.record_color_convert(convert_time)
                
                # Profile ArUco detection (on full-resolution gray image)
                detect_start = time.perf_counter()
                corners, ids, _ = self.detector.detectMarkers(gray)
                detect_time = time.perf_counter() - detect_start
                self.profiler.record_aruco_detect(detect_time)
                
                # Profile each tag processing
                # World tag
                tag_start = time.perf_counter()
                r_world, t_world = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id
                )
                tag_time = time.perf_counter() - tag_start
                self.profiler.record_process_tag('world', tag_time)
                
                # Object tag
                tag_start = time.perf_counter()
                r_obj, t_obj = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id
                )
                tag_time = time.perf_counter() - tag_start
                self.profiler.record_process_tag('object', tag_time)
                
                # EE tag
                tag_start = time.perf_counter()
                r_ee, t_ee = self.estimator.process_tag(
                    corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id
                )
                tag_time = time.perf_counter() - tag_start
                self.profiler.record_process_tag('ee', tag_time)
                
                # Update visibility flags
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
                
                # Visualization (same as parent)
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
                    
                    disp = cv2.resize(
                        frame,
                        (robot_config.camera_width // 2, robot_config.camera_height // 2)
                    )
                    cv2.imshow('Robot Camera View', disp)
                    cv2.waitKey(1)
                
                # Profile relative pose computations
                rel_pose_start = time.perf_counter()
                
                # EE in World
                pos, quat = self._compute_relative_pose(r_world, t_world, r_ee, t_ee)
                obs['aruco_ee_in_world'] = np.concatenate([pos, quat])
                
                # Object in World
                pos, quat = self._compute_relative_pose(r_world, t_world, r_obj, t_obj)
                obs['aruco_object_in_world'] = np.concatenate([pos, quat])
                
                if obs['aruco_visibility'][0] and obs['aruco_visibility'][1]:
                    object_pose_data = obs['aruco_object_in_world']
                    object_visible = 1.0
                
                # EE in Object
                pos, quat = self._compute_relative_pose(r_obj, t_obj, r_ee, t_ee)
                obs['aruco_ee_in_object'] = np.concatenate([pos, quat])
                
                # Object in EE
                pos, quat = self._compute_relative_pose(r_ee, t_ee, r_obj, t_obj)
                obs['aruco_object_in_ee'] = np.concatenate([pos, quat])
                
                rel_pose_time = time.perf_counter() - rel_pose_start
                self.profiler.record_relative_pose(rel_pose_time)
        
        # Record total camera processing time
        camera_time = time.perf_counter() - camera_start
        self.profiler.record_total_camera_time(camera_time)
        
        # Call parent's encoder polling and recording (if enabled)
        # Note: parent's on_control_loop_iteration also does camera work, but we've overridden it
        # Poll encoders in outer loop (at 20Hz)
        # With threaded control, this runs in outer loop thread and doesn't block inner loop
        self._poll_encoders()
        
        # Handle recording if enabled (same logic as parent)
        if self.is_recording:
            # Enable frame recording if flag was set and recording just started
            if self.recording_start_time is None:
                self.recording_start_time = time.perf_counter()
                # Enable frame recording if --record-frames was passed
                if hasattr(self, '_record_frames_flag') and self._record_frames_flag:
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
                
                # Record camera frame if enabled
                if self.record_frames:
                    if frame_to_record is not None:
                        # Store already-copied frame (copy was profiled above)
                        current_step['camera_frame'] = frame_to_record
                    else:
                        # Camera read failed - store empty array with downscaled dimensions
                        # This allows np.array conversion to work, but check aruco_visibility to validate
                        # Note: Zero array creation is very fast, not worth profiling
                        downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
                        downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
                        current_step['camera_frame'] = np.zeros(
                            (downscaled_height, downscaled_width, 3),
                            dtype=np.uint8
                        )
        
        # Handle GUI commands
        self._handle_control_input()
        
        # Record control loop iteration timing
        iteration_time = time.perf_counter() - iteration_start
        self.profiler.record_control_loop_iteration(iteration_time)
        
        # Print periodic stats
        if self.profiler.total_iterations % self.profile_interval == 0 and self.profiler.total_iterations > 0:
            self.profiler.print_periodic_stats()
    
    def shutdown(self):
        """Override to print final profiling stats and frame storage info."""
        # Print frame storage statistics if recording
        if self.is_recording and self.record_frames and self.trajectory:
            num_frames = sum(1 for t in self.trajectory if 'camera_frame' in t)
            if num_frames > 0:
                # Frames are saved at 1/4 resolution (1/4 width, 1/4 height = 1/16 pixels)
                downscaled_width = robot_config.camera_width // 4
                downscaled_height = robot_config.camera_height // 4
                frame_size_bytes = (downscaled_width * downscaled_height * 3)  # BGR uint8
                frame_size_mb = frame_size_bytes / (1024 * 1024)
                total_size_mb = num_frames * frame_size_mb
                print(f"\n📹 Frame Storage Summary:")
                print(f"  Frames recorded: {num_frames}")
                print(f"  Frame resolution: {downscaled_width}x{downscaled_height} (1/4 of {robot_config.camera_width}x{robot_config.camera_height})")
                print(f"  Frame size: {frame_size_mb:.3f} MB per frame (downscaled)")
                print(f"  Total frame data: ~{total_size_mb:.1f} MB (uncompressed)")
                print(f"  Note: NPZ compression will reduce file size significantly")
                print(f"  Note: ArUco detection uses full-resolution frames (not downscaled)")
        
        print("\n" + "="*70)
        print("FINAL PROFILING STATISTICS")
        print("="*70)
        self.profiler.print_periodic_stats()
        super().shutdown()


def main():
    parser = argparse.ArgumentParser(description='WX200 Camera Performance Profiling')
    parser.add_argument('--camera-id', type=int, default=None, help='Camera device ID (defaults to robot_config.camera_id)')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    parser.add_argument('--record', action='store_true', help='Enable trajectory recording (for comparison)')
    parser.add_argument('--record-frames', action='store_true', help='Record camera frames in trajectory (only works with --record)')
    parser.add_argument('--output', type=str, help='Output filename if recording')
    parser.add_argument('--profile-interval', type=int, default=robot_config.default_profile_interval, help='Print stats every N iterations')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CAMERA PERFORMANCE PROFILING")
    print("="*70)
    print(f"Control frequency: {robot_config.control_frequency} Hz")
    print(f"Camera: {robot_config.camera_id} @ {robot_config.camera_width}x{robot_config.camera_height}")
    print(f"Profile interval: every {args.profile_interval} iterations")
    if args.record:
        print(f"Recording: ENABLED (output: {args.output or 'auto-generated'})")
        if args.record_frames:
            downscaled_width = robot_config.camera_width // 4
            downscaled_height = robot_config.camera_height // 4
            frame_size_mb = (downscaled_width * downscaled_height * 3) / (1024 * 1024)
            print(f"Frame recording: ENABLED ({downscaled_width}x{downscaled_height}, ~{frame_size_mb:.3f} MB per frame)")
            print(f"  Note: Frames saved at 1/4 resolution, ArUco detection uses full resolution")
        else:
            print(f"Frame recording: disabled (use --record-frames to enable)")
    else:
        print(f"Recording: disabled")
    print("="*70 + "\n")
    
    # Create profiled controller
    controller = ProfiledCameraControlEncoders(
        enable_recording=args.record,
        output_path=args.output,
        camera_id=args.camera_id,
        profile_interval=args.profile_interval
    )
    
    # Enable frame recording if requested (requires --record)
    if args.record_frames:
        if not args.record:
            print("⚠️  Warning: --record-frames requires --record. Enabling recording capability.")
        controller._record_frames_flag = True
        print("✓ Frame recording enabled (will store camera frames when recording starts)")
    
    if args.no_vis:
        controller.show_video = False
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Shutting down...")
    finally:
        controller.shutdown()


if __name__ == "__main__":
    main()
