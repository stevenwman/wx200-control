"""
Profiling utilities for WX200 Gym Environment.
Ported from compact_code/wx200_robot_collect_demo_encoders_compact.py
"""
import time
import numpy as np
from collections import deque
from .robot_config import robot_config

class ArUcoProfiler:
    """
    Profiler for ArUco polling.
    Tracks detection performance and camera frequency.
    """
    
    def __init__(self, window_size=None):
        """Initialize ArUco profiler."""
        self.window_size = window_size or robot_config.profiler_window_size
        
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
    
    def print_periodic_stats(self):
        """Print periodic statistics."""
        if len(self.poll_times) < 10:
            return
        
        avg_poll_time = np.mean(list(self.poll_times)[-50:])
        max_poll_time = np.max(list(self.poll_times)[-50:])
        
        avg_detect_time = np.mean(list(self.detect_times)[-50:]) if self.detect_times else 0
        
        actual_freq = 0
        if len(self.poll_intervals) >= 10:
            avg_interval = np.mean(list(self.poll_intervals)[-50:])
            actual_freq = 1.0 / avg_interval if avg_interval > 0 else 0
        
        print(f"[ARUCO STATS] Freq={actual_freq:.1f}Hz, "
              f"Total={avg_poll_time*1000:.1f}ms (Detect={avg_detect_time*1000:.1f}ms, Max={max_poll_time*1000:.1f}ms)")


class LightweightProfiler:
    """
    Lightweight profiler used for warning-only output on control loop issues.
    """
    def __init__(self, window_size=None):
        self.window_size = window_size or robot_config.profiler_window_size
        self.control_loop_iteration_times = deque(maxlen=self.window_size)
        self.last_control_loop_timestamp = None
        self.missed_deadlines = 0
        self.total_iterations = 0
        
    def record_control_loop_iteration(self, elapsed_time):
        self.control_loop_iteration_times.append(elapsed_time)
        self.total_iterations += 1
        
        # Update timestamp
        self.last_control_loop_timestamp = time.perf_counter()

    def print_stats(self):
        """Print Control Loop Stats."""
        if len(self.control_loop_iteration_times) < 10:
            return
            
        times_ms = [t * 1000 for t in self.control_loop_iteration_times]
        avg_time = np.mean(times_ms)
        max_time = np.max(times_ms)
        
        # Calculate actual frequency from loop timestamps if possible, 
        # but here we just have iteration durations (which might include sleep or not depending on where it's called).
        # Better to track intervals between calls.
        
        print(f"[CONTROL LOOP] Avg={avg_time:.1f}ms, Max={max_time:.1f}ms, Iterations={self.total_iterations}")
