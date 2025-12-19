"""
Threaded control loop implementation for robot control.

This module provides the threaded dual-frequency architecture that runs
outer and inner control loops in separate threads. Only imported when
use_threaded_control = True in robot_config.
"""
import threading
import time
from loop_rate_limiters import RateLimiter

from robot_control.robot_config import robot_config


def run_control_loop_threaded(self):
    """
    Threaded dual-frequency architecture:
    - Outer loop thread: Runs at control_frequency (20Hz) for teleop/policy/observations
    - Inner loop thread: Runs at inner_control_frequency (100Hz) for IK/motor control
    - Commands shared via thread-safe locks
    - More responsive: outer loop timing not affected by inner loop blocking
    
    Subclasses can override on_control_loop_iteration() for custom behavior.
    """
    self._control_active = True
    outer_loop_rate = RateLimiter(frequency=self.control_frequency, warn=False)
    inner_loop_rate = RateLimiter(frequency=self.inner_control_frequency, warn=False)
    
    def outer_loop_thread():
        """Outer loop thread: handles teleop/policy/observations at 20Hz"""
        try:
            while self._control_active:
                loop_start = time.perf_counter()
                dt_outer = outer_loop_rate.dt
                
                if not self._control_active:
                    break
                
                # Get action from input source
                outer_loop_start = time.perf_counter()
                velocity_world, angular_velocity_world, gripper_target = self.get_action(dt_outer)
                
                if velocity_world is None:
                    self._control_active = False
                    break
                
                # Update shared commands (thread-safe)
                with self._command_lock:
                    self._latest_velocity_command[:] = velocity_world
                    self._latest_angular_velocity_command[:] = angular_velocity_world
                    self._latest_gripper_command = gripper_target
                    self.gripper_current_position = gripper_target
                
                # Hook for subclasses (pass outer_loop_start for opportunistic timing)
                # Subclasses can use this to skip expensive operations if time is running out
                self.on_control_loop_iteration(velocity_world, angular_velocity_world, gripper_target, dt_outer, 
                                               outer_loop_start_time=outer_loop_start)
                
                # Sleep to maintain rate
                outer_loop_rate.sleep()
                
        except Exception as e:
            print(f"Error in outer loop thread: {e}")
            self._control_active = False
            import traceback
            traceback.print_exc()
    
    def inner_loop_thread():
        """Inner loop thread: handles IK/motor control at 100Hz"""
        try:
            while self._control_active:
                dt = inner_loop_rate.dt
                
                if not self._control_active:
                    break
                
                # Check if paused for command execution (home/EE reset)
                if self._paused_for_command:
                    # Skip sending commands - let the command execute without interference
                    inner_loop_rate.sleep()
                    continue
                
                # Read shared commands (thread-safe)
                with self._command_lock:
                    vel_cmd = self._latest_velocity_command.copy()
                    ang_vel_cmd = self._latest_angular_velocity_command.copy()
                    gripper_cmd = self._latest_gripper_command
                
                # Execute control step (protect robot_driver with lock - serial port is not thread-safe)
                # Track lock acquisition time to detect contention
                lock_acquire_start = time.perf_counter()
                with self._driver_lock:
                    lock_acquire_time = time.perf_counter() - lock_acquire_start
                    if lock_acquire_time > robot_config.lock_wait_threshold:
                        self._driver_lock_wait_times.append(lock_acquire_time)
                        self._driver_lock_contention_count += 1
                        if len(self._driver_lock_wait_times) > robot_config.max_profiling_samples:
                            self._driver_lock_wait_times.pop(0)
                    self._execute_control_step(vel_cmd, ang_vel_cmd, gripper_cmd, dt)
                
                # Periodic performance reporting
                if self._enable_profiling and self._control_step_count > 0 and self._control_step_count % robot_config.periodic_reporting_interval == 0:
                    self._print_periodic_performance_stats()
                
                # Sleep to maintain rate
                inner_loop_rate.sleep()
                
        except Exception as e:
            print(f"Error in inner loop thread: {e}")
            self._control_active = False
            import traceback
            traceback.print_exc()
    
    try:
        # Start both threads
        outer_thread = threading.Thread(target=outer_loop_thread, daemon=True)
        inner_thread = threading.Thread(target=inner_loop_thread, daemon=True)
        
        outer_thread.start()
        inner_thread.start()
        
        # Main thread waits for threads
        outer_thread.join()
        inner_thread.join()
        
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping control loop...")
        self._control_active = False
    finally:
        self._control_active = False
        # Print final performance statistics
        if self._enable_profiling and self._control_step_count > 0:
            self._print_control_performance_stats()
