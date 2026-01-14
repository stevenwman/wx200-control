"""
Base robot control infrastructure shared between teleop and replay.

Provides common initialization, control loop structure, and shutdown logic.
"""
import time
import numpy as np
import mujoco
import mink
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from robot_control.robot_config import robot_config
from robot_control.robot_driver import RobotDriver
from robot_control.robot_controller import RobotController
from robot_control.robot_joint_to_motor import JointToMotorTranslator
from robot_control.robot_startup import startup_sequence
from robot_control.robot_shutdown import shutdown_sequence, reboot_motors
from loop_rate_limiters import RateLimiter


_XML = Path(__file__).parent.parent / "wx200" / "scene.xml"


def get_sim_home_pose(model):
    """Get the home pose from sim keyframe."""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    home_position = data.site(site_id).xpos.copy()
    home_site_xmat = data.site(site_id).xmat.reshape(3, 3)
    home_site_rot = R.from_matrix(home_site_xmat)
    home_site_quat = home_site_rot.as_quat()
    home_orientation_quat_wxyz = np.array([
        home_site_quat[3], 
        home_site_quat[0], 
        home_site_quat[1], 
        home_site_quat[2]
    ])
    
    qpos = data.qpos.copy()
    return qpos, home_position, home_orientation_quat_wxyz


class RobotControlBase:
    """
    Base class for robot control scripts (teleop, replay, etc.).
    
    Handles common initialization, control loop structure, and shutdown.
    Subclasses implement get_action() to provide velocity commands and gripper target.
    """
    
    def __init__(self, control_frequency=None, inner_control_frequency=None):
        """
        Initialize base robot control infrastructure.
        
        Args:
            control_frequency: Outer loop frequency (Hz) for teleop/policy/observations. If None, uses robot_config.
            inner_control_frequency: Inner loop frequency (Hz) for IK/motor control. If None, uses robot_config.
        """
        self.control_frequency = control_frequency or robot_config.control_frequency
        self.inner_control_frequency = inner_control_frequency or robot_config.inner_control_frequency
        self.model = None
        self.data = None
        self.configuration = None
        self.robot_driver = None
        self.translator = None
        self.robot_controller = None
        self.control_rate = None
        self.inner_control_rate = None
        self.gripper_current_position = None
        
        # For single-loop dual-frequency architecture
        self._outer_loop_counter = 0
        self._outer_loop_period = int(self.inner_control_frequency / self.control_frequency) if self.inner_control_frequency > self.control_frequency else 1
        self._outer_loop_target_time = None  # Track target time for next outer loop call
        
        # Shared control state (used by both single-threaded and threaded modes)
        self._control_active = True  # Flag to stop control loops
        self._paused_for_command = False  # Flag to pause inner loop during home/EE reset commands
        
        # Threading infrastructure (only initialized when threaded mode is used)
        # These are initialized lazily when threaded mode is enabled
        self._command_lock = None  # Protects shared command state (thread-safe access)
        self._driver_lock = None  # Protects robot_driver (serial port) - prevents concurrent read/write
        self._driver_lock_wait_times = []  # Track lock wait times for debugging
        self._driver_lock_contention_count = 0  # Count lock contention events
        
        # Initialize threading locks if threaded mode is enabled
        if robot_config.use_threaded_control:
            import threading
            self._command_lock = threading.Lock()
            self._driver_lock = threading.Lock()
        
        # Performance profiling for control step timing
        self._control_step_times = []  # Execution times for control steps
        self._control_step_count = 0
        self._missed_deadlines = 0  # Count of control steps that exceeded time budget
        self._enable_profiling = True  # Set to False to disable profiling overhead
        self._last_perf_print = 0  # Counter for periodic performance prints
        
    def initialize(self):
        """Initialize MuJoCo model, robot driver, and controller."""
        print("\nInitializing robot control...")
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(_XML.as_posix())
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        
        # Get sim home pose
        home_qpos, home_position, home_orientation_quat_wxyz = get_sim_home_pose(self.model)
        print(f"Sim home pose - EE position: {home_position}")
        
        # Connect to robot
        self.robot_driver = RobotDriver()
        print("\nConnecting to robot...")
        self.robot_driver.connect()
        
        # Create translator
        self.translator = JointToMotorTranslator(
            joint1_motor2_offset=0,
            joint1_motor3_offset=0
        )
        
        # Execute startup sequence
        robot_joint_angles, actual_position, actual_orientation_quat_wxyz, home_motor_positions = startup_sequence(
            self.robot_driver, self.translator, self.model, self.data, self.configuration, home_qpos=home_qpos
        )
        
        print(f"✓ Synced to actual robot position: {actual_position}")
        
        # Initialize robot controller
        self.robot_controller = RobotController(
            model=self.model,
            initial_position=actual_position,
            initial_orientation_quat_wxyz=actual_orientation_quat_wxyz,
            position_cost=1.0,
            orientation_cost=0.1,
            posture_cost=1e-2
        )
        
        # Initialize posture target and pose
        self.robot_controller.initialize_posture_target(self.configuration)
        self.robot_controller.reset_pose(actual_position, actual_orientation_quat_wxyz)
        current_target_pose = self.robot_controller.get_target_pose()
        self.robot_controller.end_effector_task.set_target(current_target_pose)
        
        # Initialize gripper position
        self.gripper_current_position = robot_joint_angles[5] if len(robot_joint_angles) > 5 else robot_config.gripper_open_pos
        
        # Create rate limiter
        # Run at inner loop frequency (fast), do outer loop tasks periodically
        self.control_rate = RateLimiter(frequency=self.inner_control_frequency, warn=False)
        self._outer_loop_period = int(self.inner_control_frequency / self.control_frequency) if self.inner_control_frequency > self.control_frequency else 1
        self._outer_loop_counter = 0
        
        # Store latest commands for inner loop
        self._latest_velocity_command = np.zeros(3)
        self._latest_angular_velocity_command = np.zeros(3)
        self._latest_gripper_command = 0.0
        
        return actual_position, actual_orientation_quat_wxyz
    
    def get_action(self, dt):
        """
        Get action from input source (to be implemented by subclasses).
        
        Args:
            dt: Time step (seconds)
        
        Returns:
            tuple: (velocity_world, angular_velocity_world, gripper_target)
                - velocity_world: [vx, vy, vz] in m/s
                - angular_velocity_world: [wx, wy, wz] in rad/s
                - gripper_target: Gripper target position (meters)
        """
        raise NotImplementedError("Subclasses must implement get_action()")
    
    def _execute_control_step(self, velocity_world, angular_velocity_world, gripper_target, dt):
        """
        Execute a single control step: update controller, solve IK, send to robot.
        
        This is the core control loop logic shared by all control modes.
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in m/s
            angular_velocity_world: [wx, wy, wz] angular velocity in rad/s
            gripper_target: Gripper target position (meters)
            dt: Time step (seconds)
        """
        step_start = time.perf_counter() if self._enable_profiling else None
        
        # Update robot controller with velocity commands
        self.robot_controller.update_from_velocity_command(
            velocity_world=velocity_world,
            angular_velocity_world=angular_velocity_world,
            dt=dt,
            configuration=self.configuration
        )
        
        # Get joint commands from IK solution
        joint_commands_rad = self.configuration.q[:5].copy()
        
        # Convert joint commands to motor positions
        motor_positions = self.translator.joint_commands_to_motor_positions(
            joint_angles_rad=joint_commands_rad,
            gripper_position=gripper_target
        )
        
        # Send commands to robot
        self.robot_driver.send_motor_positions(motor_positions, velocity_limit=robot_config.velocity_limit)
        
        # Performance profiling
        if self._enable_profiling and step_start is not None:
            step_duration = time.perf_counter() - step_start
            
            # Filter out outliers that are likely from blocking operations (e.g., trajectory saving)
            # Normal control steps should be < 10ms (even at slow frequencies)
            # Blocking operations (save, home, etc.) can take seconds and shouldn't be counted
            if step_duration < robot_config.blocking_outlier_threshold_control_step:
                # Normal control step - record it
                self._control_step_times.append(step_duration)
                self._control_step_count += 1
                
                # Check if we exceeded the time budget
                time_budget = dt * robot_config.deadline_threshold_factor
                if step_duration > time_budget:
                    self._missed_deadlines += 1
                
                # Keep only recent samples to avoid memory growth
                if len(self._control_step_times) > robot_config.max_profiling_samples:
                    self._control_step_times.pop(0)
            else:
                # Likely a blocking operation - don't count it but track it separately
                if not hasattr(self, '_blocking_operation_count'):
                    self._blocking_operation_count = 0
                self._blocking_operation_count += 1
                # Still increment count so stats are accurate (but don't include in timing stats)
                self._control_step_count += 1
    
    def on_control_loop_iteration(self, velocity_world, angular_velocity_world, gripper_target, dt, outer_loop_start_time=None):
        """
        Hook called on each control loop iteration before executing control step.
        
        Args:
            outer_loop_start_time: Timestamp when outer loop iteration started (for opportunistic timing)
        
        Subclasses can override to add custom behavior (e.g., recording, progress reporting).
        
        Args:
            velocity_world: [vx, vy, vz] linear velocity in m/s
            angular_velocity_world: [wx, wy, wz] angular velocity in rad/s
            gripper_target: Gripper target position (meters)
            dt: Time step (seconds)
        """
        pass
    
    def run_control_loop(self):
        """
        Main control loop entry point - chooses threaded or single-threaded based on config.
        
        Subclasses can override on_control_loop_iteration() for custom behavior.
        """
        if robot_config.use_threaded_control:
            # Lazy import of threaded implementation
            from robot_control.robot_control_base_threaded import run_control_loop_threaded
            run_control_loop_threaded(self)
        else:
            self._run_control_loop_single_threaded()
    
    def _run_control_loop_single_threaded(self):
        """
        Single-loop dual-frequency architecture (no threading):
        - Runs at inner_control_frequency (100Hz) for smooth motor control
        - Outer loop tasks (teleop/policy/observations) execute every Nth iteration (20Hz)
        - Prioritizes outer loop timing accuracy: skips inner loop iterations if needed
        - Avoids threading overhead and GIL issues while maintaining smooth motion
        
        Subclasses can override on_control_loop_iteration() for custom behavior.
        """
        import time
        control_loop_active = True
        self._control_active = True
        self._paused_for_command = False  # Initialize pause flag for single-threaded mode too
        outer_loop_dt = 1.0 / self.control_frequency  # Outer loop period in seconds (e.g., 0.05s for 20Hz)
        
        try:
            # Initialize outer loop timing
            self._outer_loop_target_time = time.perf_counter() + outer_loop_dt
            
            while control_loop_active:
                current_time = time.perf_counter()
                dt = self.control_rate.dt  # Fast inner loop dt (e.g., 1/100s)
                
                # Check if it's time for outer loop
                # Use ONLY time-based scheduling for accurate 20Hz timing (50ms intervals)
                time_based_trigger = (current_time >= self._outer_loop_target_time)
                
                # Trigger outer loop based on time only (ensures precise 20Hz)
                if time_based_trigger:
                    # Capture the trigger time to maintain exact 50ms spacing
                    outer_loop_start_time = current_time
                    
                    # Sync counter to match (for reporting/compatibility, doesn't affect timing)
                    if self._outer_loop_counter % self._outer_loop_period != 0:
                        self._outer_loop_counter = ((self._outer_loop_counter // self._outer_loop_period) + 1) * self._outer_loop_period
                    
                    # Reset target time for next outer loop call based on START time (not current time after work)
                    # This ensures exactly 50ms between outer loop starts
                    self._outer_loop_target_time = outer_loop_start_time + outer_loop_dt
                    
                    dt_outer = dt * self._outer_loop_period  # Effective outer loop dt (e.g., 1/20s)
                    
                    # Get action from input source (SpaceMouse, trajectory, NN, etc.)
                    velocity_world, angular_velocity_world, gripper_target = self.get_action(dt_outer)
                    
                    # Check for end condition (e.g., replay finished)
                    if velocity_world is None:
                        break
                    
                    # Update latest commands (used by inner loop)
                    self._latest_velocity_command[:] = velocity_world
                    self._latest_angular_velocity_command[:] = angular_velocity_world
                    self._latest_gripper_command = gripper_target
                    
                    # Update gripper current position
                    self.gripper_current_position = gripper_target
                    
                    # Hook for subclasses to add custom behavior (encoder polling, camera, recording)
                    self.on_control_loop_iteration(velocity_world, angular_velocity_world, gripper_target, dt_outer)
                    
                    # If outer loop work took longer than period, skip remaining inner loops in this cycle
                    current_time_after_outer = time.perf_counter()
                    if current_time_after_outer >= self._outer_loop_target_time:
                        # We're already past the next target - fast-forward counter to next cycle
                        self._outer_loop_counter = ((self._outer_loop_counter // self._outer_loop_period) + 1) * self._outer_loop_period
                        # Don't sleep or do inner loop - go straight to next iteration to check outer loop again
                        continue
                
                # Inner loop: Execute control step (IK solving + motor commands) at fast rate
                # Skip if paused for command execution (home/EE reset) or approaching outer loop deadline
                if not self._paused_for_command:
                    time_until_next_outer = self._outer_loop_target_time - time.perf_counter()
                    if time_until_next_outer > robot_config.inner_loop_skip_threshold:
                        # Track time since last control step to detect if we're recovering from blocking operation
                        if not hasattr(self, '_last_control_step_time'):
                            self._last_control_step_time = current_time
                        time_since_last_step = current_time - self._last_control_step_time
                        self._last_control_step_time = current_time
                        
                        # If there was a large gap, we're likely recovering from blocking operation
                        # Don't profile this step to avoid skewing stats (but still execute it)
                        skip_profiling = time_since_last_step > robot_config.recovery_gap_threshold
                        
                        # Temporarily disable profiling if this step is after a blocking operation
                        original_profiling = self._enable_profiling
                        if skip_profiling:
                            self._enable_profiling = False
                        
                        self._execute_control_step(
                            self._latest_velocity_command,
                            self._latest_angular_velocity_command,
                            self._latest_gripper_command,
                            dt
                        )
                        
                        # Restore profiling state
                        self._enable_profiling = original_profiling
                    # else: skip this inner loop iteration to maintain outer loop timing
                
                self._outer_loop_counter += 1
                
                # Periodic performance reporting
                if self._enable_profiling and self._control_step_count > 0 and self._control_step_count % robot_config.periodic_reporting_interval == 0:
                    self._print_periodic_performance_stats()
                
                # Sleep to maintain inner loop frequency, but don't oversleep past outer loop deadline
                # Check time until outer loop deadline BEFORE sleeping to prevent drift
                time_before_sleep = time.perf_counter()
                time_until_outer_deadline = self._outer_loop_target_time - time_before_sleep
                
                # Only sleep if we have sufficient time before the outer loop deadline
                # Use a safety margin to account for sleep() inaccuracy
                if time_until_outer_deadline > robot_config.sleep_safety_margin:
                    # Call RateLimiter normally - it will calculate sleep time
                    # But we need to ensure we don't oversleep past the deadline
                    inner_loop_dt = self.control_rate.dt
                    
                    # If time until deadline is less than inner loop period, don't sleep at all
                    # This prevents oversleeping that causes outer loop drift
                    if time_until_outer_deadline >= inner_loop_dt:
                        # Safe to sleep for full inner loop period
                        self.control_rate.sleep()
                    else:
                        # Too close to deadline - sleep only for remaining time (with margin)
                        max_sleep = time_until_outer_deadline - (robot_config.sleep_safety_margin * 0.5)
                        if max_sleep > robot_config.min_sleep_time:
                            time.sleep(max_sleep)
                        # Update RateLimiter state manually to keep it in sync
                        # (We skip the sleep but still want it to track timing)
                        current_time = time.perf_counter()
                        if not hasattr(self.control_rate, '_last_time'):
                            self.control_rate._last_time = current_time
                        else:
                            # Advance by one inner loop period
                            self.control_rate._last_time = self.control_rate._last_time + inner_loop_dt
                else:
                    # Too close to deadline - skip sleep entirely
                    # Still update RateLimiter state to keep timing correct
                    current_time = time.perf_counter()
                    if not hasattr(self.control_rate, '_last_time'):
                        self.control_rate._last_time = current_time
                    else:
                        # Advance by one inner loop period (simulating a cycle)
                        self.control_rate._last_time = self.control_rate._last_time + inner_loop_dt
                
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected. Stopping control loop...")
            control_loop_active = False
            self._control_active = False
        finally:
            # Print final performance statistics
            if self._enable_profiling and self._control_step_count > 0:
                self._print_control_performance_stats()
    
    def _print_periodic_performance_stats(self):
        """Print periodic performance stats during operation."""
        if not self._control_step_times:
            return
        
        import numpy as np
        
        # Use recent samples for periodic reporting
        recent_times = self._control_step_times[-robot_config.periodic_reporting_sample_window:] if len(self._control_step_times) >= robot_config.periodic_reporting_sample_window else self._control_step_times
        times_ms = np.array(recent_times) * 1000
        avg_time = np.mean(times_ms)
        max_time = np.max(times_ms)
        p95_time = np.percentile(times_ms, 95)
        p99_time = np.percentile(times_ms, 99)
        
        dt_ms = (1.0 / self.inner_control_frequency) * 1000
        time_budget_ms = dt_ms * robot_config.deadline_threshold_factor
        
        recent_missed = sum(1 for t in recent_times if t > (time_budget_ms / 1000.0))
        missed_pct = (recent_missed / len(recent_times)) * 100 if recent_times else 0
        
        status = "✓" if avg_time < time_budget_ms and missed_pct < 5.0 else "⚠️"
        
        # Add lock contention info if available (threaded mode only)
        lock_info = ""
        if (robot_config.use_threaded_control and 
            hasattr(self, '_driver_lock_contention_count') and 
            hasattr(self, '_driver_lock_wait_times') and 
            len(self._driver_lock_wait_times) > 0):
            recent_waits = self._driver_lock_wait_times[-robot_config.periodic_reporting_sample_window:] if len(self._driver_lock_wait_times) >= robot_config.periodic_reporting_sample_window else self._driver_lock_wait_times
            if recent_waits:
                avg_wait = np.mean(recent_waits) * 1000
                max_wait = np.max(recent_waits) * 1000
                wait_pct = (len(recent_waits) / len(recent_times)) * 100 if recent_times else 0
                lock_info = f" | lock_wait: avg={avg_wait:.1f}ms, max={max_wait:.1f}ms ({wait_pct:.0f}% of steps)"
        
        # Show pause status  
        pause_info = ""
        if hasattr(self, '_paused_for_command') and self._paused_for_command:
            pause_info = " | ⚠️ PAUSED for command"
        
        print(f"[CONTROL PERF #{self._control_step_count}] "
              f"avg={avg_time:.1f}ms, max={max_time:.1f}ms, p95={p95_time:.1f}ms, p99={p99_time:.1f}ms, "
              f"budget={time_budget_ms:.1f}ms, missed={missed_pct:.1f}% {status}"
              f"{lock_info}{pause_info}")
    
    
    def _print_control_performance_stats(self):
        """Print final control step performance statistics."""
        if not self._control_step_times:
            return
        
        import numpy as np
        
        times_ms = np.array(self._control_step_times) * 1000  # Convert to ms
        avg_time = np.mean(times_ms)
        min_time = np.min(times_ms)
        max_time = np.max(times_ms)
        p95_time = np.percentile(times_ms, 95)
        p99_time = np.percentile(times_ms, 99)
        
        # Calculate time budget based on inner control frequency
        dt_ms = (1.0 / self.inner_control_frequency) * 1000
        time_budget_ms = dt_ms * robot_config.deadline_threshold_factor  # 80% threshold
        
        missed_pct = (self._missed_deadlines / self._control_step_count) * 100 if self._control_step_count > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"CONTROL STEP PERFORMANCE STATISTICS")
        print(f"{'='*60}")
        print(f"  Total control steps: {self._control_step_count}")
        print(f"  Target frequency: {self.inner_control_frequency:.1f}Hz (dt={dt_ms:.2f}ms)")
        print(f"  Time budget ({robot_config.deadline_threshold_factor*100:.0f}% threshold): {time_budget_ms:.2f}ms")
        print(f"  Execution time: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms")
        print(f"  Percentiles: p95={p95_time:.2f}ms, p99={p99_time:.2f}ms")
        print(f"  Missed deadlines: {self._missed_deadlines}/{self._control_step_count} ({missed_pct:.1f}%)")
        
        # Report if blocking operations were filtered out
        if hasattr(self, '_blocking_operation_count') and self._blocking_operation_count > 0:
            blocking_pct = (self._blocking_operation_count / self._control_step_count) * 100
            print(f"  ⚠️  {self._blocking_operation_count} control steps exceeded {robot_config.blocking_outlier_threshold_control_step*1000:.0f}ms (likely blocking ops like save/home)")
            print(f"     These were excluded from timing stats ({blocking_pct:.1f}% of total steps)")
            print(f"     Stats above reflect only normal control step execution times")
        
        # Performance assessment
        if avg_time > time_budget_ms:
            print(f"  ⚠️  WARNING: Average execution time ({avg_time:.2f}ms) exceeds time budget ({time_budget_ms:.2f}ms)")
            print(f"     Consider reducing inner_control_frequency from {self.inner_control_frequency:.1f}Hz")
        elif p95_time > time_budget_ms:
            print(f"  ⚠️  WARNING: 95th percentile ({p95_time:.2f}ms) exceeds time budget ({time_budget_ms:.2f}ms)")
            print(f"     Some control steps may experience jitter")
        elif missed_pct > 5.0:
            print(f"  ⚠️  WARNING: {missed_pct:.1f}% of control steps exceeded time budget")
        else:
            print(f"  ✓ Performance looks good - execution time well within budget")
        
        # Estimate maximum safe frequency
        safe_freq = 1.0 / (max_time / 1000.0) * robot_config.safe_frequency_margin  # Use max time with safety margin
        print(f"  Estimated max safe frequency: ~{safe_freq:.1f}Hz (based on max execution time)")
        print(f"{'='*60}\n")
    
    def shutdown(self):
        """Execute shutdown sequence and cleanup."""
        try:
            shutdown_sequence(self.robot_driver, velocity_limit=robot_config.velocity_limit)
        except Exception as e:
            print(f"Error during shutdown sequence: {e}")
        
        try:
            reboot_motors(self.robot_driver)
        except Exception as e:
            print(f"Error rebooting motors: {e}")
        
        try:
            self.robot_driver.disconnect()
        except Exception as e:
            print(f"Error disconnecting: {e}")
    
    def run(self):
        """
        Main entry point - initialize, run control loop, and shutdown.
        
        Subclasses can override this for custom behavior.
        """
        try:
            self.initialize()
            self.on_ready()  # Hook for subclasses to do setup after initialization
            self.run_control_loop()
        except Exception as e:
            print(f"\n⚠️  Error: {e}")
            print("Executing emergency shutdown...")
            try:
                shutdown_sequence(self.robot_driver, velocity_limit=robot_config.velocity_limit)
            except:
                pass
            try:
                self.robot_driver.disconnect()
            except:
                pass
            raise
        finally:
            self.shutdown()
    
    def on_ready(self):
        """
        Hook called after initialization, before control loop starts.
        
        Subclasses can override this to print messages, set up recording, etc.
        """
        pass
