"""
Demo Collection Script (Gym Version).

Replicates the functionality of `compact_code/wx200_robot_collect_demo_encoders_compact.py`
but uses the new clean `compact_gym` infrastructure.

Features:
- Teleoperation via SpaceMouse
- Real-time Camera View with ArUco Overlay
- "Start/Stop Recording" via GUI buttons
- Data saving to .npz format compatible with OGPO
"""

import sys
import time
import cv2
import numpy as np
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Local imports
try:
    from loop_rate_limiters import RateLimiter
except ImportError as e:
    print(f"Failed to import loop_rate_limiters: {e}")
    print("Install with: pip install loop-rate-limiters")
    sys.exit(1)

from deployment.gym_env import WX200GymEnv
from deployment.robot_config import robot_config
from deployment.profiling import LightweightProfiler
from collection.spacemouse.spacemouse_driver import SpaceMouseDriver
from collection.utils.robot_control_gui import SimpleControlGUI
from scripts.smooth_aruco_trajectory import smooth_trajectory_file


def auto_fix_usb_latency():
    """Automatically fix USB latency on startup."""
    try:
        from collection.fix_usb_latency import fix_usb_latency
        print("\n" + "="*60)
        print("Checking USB Latency...")
        print("="*60)
        fix_usb_latency(device='/dev/ttyUSB0', target_latency=1, verbose=True)
        print("="*60 + "\n")
    except Exception as e:
        print(f"⚠️  Could not auto-fix USB latency: {e}")
        print(f"   (This is optional - continuing anyway)\n")


class DemoCollector:
    def __init__(self, output_dir="data/gym_demos"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_recording = False
        self.trajectory = []
        self.episode_count = 0
        
        self.profiler = LightweightProfiler()
        self._breakdown_history = []
        self._viz_frame_times = []
        self._viz_last_duration = 0.0
        self._viz_next_time = None
        self._viz_dt = None
        self._viz_axis_length = robot_config.aruco_marker_size_m * robot_config.aruco_axis_length_scale
        
        # Init SpaceMouse
        self.spacemouse = SpaceMouseDriver(
            velocity_scale=robot_config.velocity_scale,
            angular_velocity_scale=robot_config.angular_velocity_scale
        )
        self.spacemouse.start()
        
        # Init GUI
        self.gui = SimpleControlGUI()
        self.gui.start()
        
        # Init Gym Env
        # IMPORTANT: Use inner_control_frequency for dt since env.step() runs at 120Hz
        self.env = WX200GymEnv(
            max_episode_length=99999, # Manual stop
            show_video=False, # We handle viz ourselves to overlay UI info
            enable_aruco=True,
            control_frequency=robot_config.inner_control_frequency  # 120Hz for smooth motor commands
        )
        
        # Gripper state tracking
        self.current_gripper_pos = robot_config.gripper_open_pos
        self.running = True

    def _get_ee_pose_target(self):
        """Get the current IK target pose from the robot controller."""
        if self.env.robot_hardware and self.env.robot_hardware.robot_controller:
            position = self.env.robot_hardware.robot_controller.get_target_position()
            orientation_wxyz = self.env.robot_hardware.robot_controller.get_target_orientation_quat_wxyz()
            # Return in format [px, py, pz, qw, qx, qy, qz]
            return np.concatenate([position, orientation_wxyz])
        else:
            return np.zeros(7)

    def _downscale_frame(self, frame):
        """Downscale frame according to config."""
        downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
        downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
        return cv2.resize(frame, (downscaled_width, downscaled_height), interpolation=cv2.INTER_AREA)

    def _handle_gui_command(self, cmd):
        """Handle GUI command input."""
        if cmd == ' ':  # E-Stop (Spacebar) - Emergency Stop
            print("\n[EMERGENCY STOP TRIGGERED]")
            self.is_recording = False
            self.trajectory = []
            # Disable torque immediately via environment interface
            self.env.emergency_stop()
            sys.exit(1)  # Exit immediately

        elif cmd == 'd':  # Done (Success) - Save and Stop
            if self.is_recording:
                self.stop_recording(success=True)
            else:
                print("Not recording, nothing to save.")

        elif cmd == 'x' or cmd == 'Backspace':  # Discard - Stop and Delete
            if self.is_recording:
                self.stop_recording(success=False)

        elif cmd == 'h':  # Home
            print("Moving Home...")
            try:
                self.env.home()
                self.current_gripper_pos = robot_config.gripper_open_pos
                self.spacemouse.reset_state()
            except Exception as e:
                print(f"Error Homing: {e}")

        elif cmd == 'g':  # Reset EE
            print("Resetting Gripper...")
            try:
                self.env.reset_gripper()
                self.current_gripper_pos = robot_config.gripper_open_pos
                self.spacemouse.reset_state()
            except Exception as e:
                print(f"Error Resetting Gripper: {e}")

        elif cmd == 'r':  # Start Recording
            print("Starting Recording...")
            self.start_recording()

        elif cmd == 's':  # Stop & Save (Legacy key)
            self.stop_recording(success=True)

    def _compute_action(self, dt_outer):
        """Compute normalized action from SpaceMouse input."""
        vel_world = self.spacemouse.get_velocity_command()
        ang_vel_world = self.spacemouse.get_angular_velocity_command()
        self.current_gripper_pos = self.spacemouse.get_gripper_command(self.current_gripper_pos, dt_outer)

        norm_vel = np.clip(vel_world / robot_config.velocity_scale, -1, 1)
        norm_ang_vel = np.clip(ang_vel_world / robot_config.angular_velocity_scale, -1, 1)
        gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
        norm_gripper = np.clip(
            2.0 * (self.current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0,
            -1,
            1,
        )

        latest_action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])
        return vel_world, ang_vel_world, latest_action

    def _record_step(self, info, vel_world, ang_vel_world, latest_action, frame, dt_outer):
        """Record one timestep of data."""
        # 1. State/Qpos
        state = info.get('qpos')
        if state is None:
            state = np.zeros(6)

        # 2. Encoders
        encoders = info.get('encoder_values')
        if encoders is None:
            encoders = {}
        encoder_array = np.array([encoders.get(mid, 0) for mid in robot_config.motor_ids])

        # 3. EE Pose (FK)
        ee_pose_encoder = info.get('ee_pose_fk')
        if ee_pose_encoder is None:
            ee_pose_encoder = np.zeros(7)

        # 4. Raw ArUco
        raw_aruco = info.get('raw_aruco') or {}

        # 5. Augmented Actions
        # vel(3) + ang(3) + axis_angle(3) + grip(1) = 10D
        axis_angle = ang_vel_world * dt_outer
        augmented_action = np.concatenate([
            vel_world, ang_vel_world, axis_angle, [self.current_gripper_pos]
        ])
        legacy_action = np.concatenate([
            vel_world, ang_vel_world, [self.current_gripper_pos]
        ])

        step_data = {
            'timestamp': time.time(),

            # Core State
            'state': state,
            'encoder_values': encoder_array,
            'ee_pose_encoder': ee_pose_encoder,

            # Actions
            'action': legacy_action,
            'action_normalized': latest_action.copy(),
            'augmented_actions': augmented_action,
            'ee_pose_target': self._get_ee_pose_target(),

            # Vision (aruco_visibility order: [world, object, ee])
            'object_pose': raw_aruco.get('aruco_object_in_world', np.zeros(7)),
            'object_visible': np.array([raw_aruco.get('aruco_visibility', np.zeros(3))[1]]),
            'aruco_ee_in_world': raw_aruco.get('aruco_ee_in_world', np.zeros(7)),
            'aruco_object_in_world': raw_aruco.get('aruco_object_in_world', np.zeros(7)),
            'aruco_ee_in_object': raw_aruco.get('aruco_ee_in_object', np.zeros(7)),
            'aruco_object_in_ee': raw_aruco.get('aruco_object_in_ee', np.zeros(7)),
            'aruco_visibility': raw_aruco.get('aruco_visibility', np.zeros(3)),

            # Frame (optional via flag, legacy had record_frames)
            'camera_frame': self._downscale_frame(frame) if frame is not None else np.zeros(
                (
                    robot_config.camera_height // robot_config.frame_downscale_factor,
                    robot_config.camera_width // robot_config.frame_downscale_factor,
                    3,
                ),
                dtype=np.uint8,
            ),
        }
        self.trajectory.append(step_data)

    def _render_visualization(self, frame):
        """Render ArUco overlay and recording status onto the frame."""
        # Draw ArUco overlay
        draw_data = self.env.get_latest_aruco_draw_data()
        if draw_data and draw_data['corners'] is not None and draw_data['ids'] is not None:
            cv2.aruco.drawDetectedMarkers(frame, draw_data['corners'], draw_data['ids'])
            if self.env.cam_matrix is not None and self.env.dist_coeffs is not None:
                if draw_data['world_visible'] and draw_data['r_world'] is not None:
                    cv2.drawFrameAxes(
                        frame, self.env.cam_matrix, self.env.dist_coeffs,
                        draw_data['r_world'], draw_data['t_world'], self._viz_axis_length
                    )
                if draw_data['object_visible'] and draw_data['r_obj'] is not None:
                    cv2.drawFrameAxes(
                        frame, self.env.cam_matrix, self.env.dist_coeffs,
                        draw_data['r_obj'], draw_data['t_obj'], self._viz_axis_length
                    )
                if draw_data['ee_visible'] and draw_data['r_ee'] is not None:
                    cv2.drawFrameAxes(
                        frame, self.env.cam_matrix, self.env.dist_coeffs,
                        draw_data['r_ee'], draw_data['t_ee'], self._viz_axis_length
                    )

        # Draw Recording Status
        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        text = "RECORDING" if self.is_recording else "READY"
        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show
        cv2.imshow("Data Collector", cv2.resize(frame, (960, 540)))
        cv2.waitKey(1)

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.trajectory = []
            print(f"\n[RECORDING STARTED] Episode {self.episode_count}")
            if self.gui.status_label:
                self.gui.status_label.config(text=f"Recording Ep {self.episode_count}...", foreground="red")

    def stop_recording(self, success=True):
        """
        Stop recording and optionally save.

        Args:
            success: If True, save trajectory. If False, discard it.
        """
        if self.is_recording:
            self.is_recording = False
            if success:
                self.save_trajectory()
                self.episode_count += 1
                print("[RECORDING STOPPED] Saved.")
                if self.gui.status_label:
                    self.gui.status_label.config(text="Ready", foreground="green")
            else:
                self.trajectory = []
                print("[RECORDING DISCARDED]")
                if self.gui.status_label:
                    self.gui.status_label.config(text="Discarded", foreground="red")

    def stop_robot(self):
        print("Stopping Robot...")
        self.running = False

    def save_trajectory(self):
        if not self.trajectory:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"demo_{timestamp}.npz"
        
        # Convert list of dicts to dict of arrays
        data = {}
        for key in self.trajectory[0].keys():
            data[key] = np.array([step[key] for step in self.trajectory])

        # Metadata for long-term compatibility
        data['metadata'] = {
            'created_at': timestamp,
            'file_name': filename.name,
            'config_snapshot': asdict(robot_config),
        }
            
        np.savez_compressed(filename, **data)
        print(f"Saved {len(self.trajectory)} steps to {filename}")

        # Smooth ArUco trajectory data (adds smoothed_* keys in-place)
        try:
            smooth_trajectory_file(filename, output_path=filename)
            print(f"✓ Smoothed trajectory updated: {filename}")
        except Exception as e:
            print(f"⚠️  Smoothing failed: {e}")

    def run(self):
        print("Resetting Robot...")
        self.env.reset()
        print("Ready. Use GUI buttons to record.")
        print("  SpaceMouse buttons: left=open gripper, right=close gripper")
        print("  Close the GUI window to quit.")

        # Dual-frequency architecture (like compact_code):
        # - Inner loop at 120Hz: motor commands (env.step)
        # - Outer loop at 10Hz: SpaceMouse input, encoder polling, recording
        inner_rate_limiter = RateLimiter(frequency=robot_config.inner_control_frequency, warn=False)
        self._viz_dt = 1.0 / robot_config.camera_fps if robot_config.camera_fps > 0 else None
        self._viz_next_time = time.perf_counter() + (self._viz_dt or 0.0)

        # Outer loop timing
        outer_loop_dt = 1.0 / robot_config.control_frequency  # 0.1s for 10Hz
        outer_loop_target_time = time.perf_counter() + outer_loop_dt

        # Cache for latest action (reused across inner loop iterations)
        # Initialize gripper to "open" to avoid a brief mid-grip command before first outer loop tick.
        gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
        norm_gripper_init = 2.0 * (self.current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0
        latest_action = np.zeros(7)  # [vel(3), ang_vel(3), gripper(1)]
        latest_action[6] = np.clip(norm_gripper_init, -1.0, 1.0)

        # Timing breakdown storage for outer loop profiling
        t_input = 0.0
        t_gui = 0.0
        t_command = 0.0
        t_record = 0.0
        t_viz = 0.0

        try:
            while self.running and self.gui.is_available():
                current_time = time.perf_counter()

                # Check if it's time for outer loop (time-based trigger)
                time_based_trigger = (current_time >= outer_loop_target_time)

                if time_based_trigger:
                    # === OUTER LOOP (10Hz) ===
                    outer_loop_start_time = current_time
                    outer_loop_target_time = outer_loop_start_time + outer_loop_dt

                    # Poll encoders at 10Hz (outer loop) so env.step() can use fresh state
                    self.env.poll_encoders(outer_loop_start_time=outer_loop_start_time)

                    # Detailed timing breakdown
                    t_input_start = time.perf_counter()

                    # 1. Update SpaceMouse
                    self.spacemouse.update()
                    t_input = time.perf_counter() - t_input_start

                    # 2. Poll GUI Commands
                    t_gui_start = time.perf_counter()
                    if self.gui.is_available():
                        cmd = self.gui.get_command()
                        if cmd:
                            self._handle_gui_command(cmd)

                    t_gui = time.perf_counter() - t_gui_start

                    # 3. Get Control Command from SpaceMouse
                    t_command_start = time.perf_counter()
                    dt_outer = 1.0 / robot_config.control_frequency
                    vel_world, ang_vel_world, latest_action = self._compute_action(dt_outer)
                    t_command = time.perf_counter() - t_command_start

                # === INNER LOOP (120Hz) ===
                # Step env with cached action (IK + motor commands)
                # This runs EVERY iteration for smooth motion
                t_step_start = time.perf_counter()
                _, _, _, _, info = self.env.step(latest_action)
                t_step = time.perf_counter() - t_step_start

                # === VISUALIZATION (camera rate) ===
                if self._viz_dt:
                    now = time.perf_counter()
                    if now >= self._viz_next_time:
                        self._viz_next_time = now + self._viz_dt
                        t_viz_start = time.perf_counter()
                        frame = self.env.get_last_frame_copy()
                        if frame is not None:
                            self._render_visualization(frame)

                            # Track visualization update timestamps
                            self._viz_last_duration = time.perf_counter() - t_viz_start
                            self._viz_frame_times.append(now)
                            window_sec = robot_config.control_perf_window_sec
                            if window_sec and window_sec > 0:
                                cutoff = now - window_sec
                                while self._viz_frame_times and self._viz_frame_times[0] < cutoff:
                                    self._viz_frame_times.pop(0)

                # === OUTER LOOP CONTINUED (10Hz) ===
                # Record data, visualization, and profiling only in outer loop
                if time_based_trigger:
                    frame = self.env.get_last_frame_copy()

                    # 4. Record Data
                    t_record_start = time.perf_counter()
                    if self.is_recording:
                        self._record_step(info, vel_world, ang_vel_world, latest_action, frame, dt_outer)
                    t_record = time.perf_counter() - t_record_start
                    t_viz = self._viz_last_duration

                    # Record breakdown stats for rolling averages
                    now = time.perf_counter()
                    self._breakdown_history.append((now, t_input, t_gui, t_command, t_step, t_record, t_viz))
                    window_sec = robot_config.control_perf_window_sec
                    if window_sec and window_sec > 0:
                        cutoff = now - window_sec
                        while self._breakdown_history and self._breakdown_history[0][0] < cutoff:
                            self._breakdown_history.pop(0)

                    # 6. Profiling (only in outer loop)
                    outer_loop_time = time.perf_counter() - outer_loop_start_time
                    self.profiler.record_control_loop_iteration(outer_loop_time)
                    if self.profiler.total_iterations % 50 == 0:
                        self.profiler.print_stats()
                        # Print averaged breakdown over recent window
                        if self._breakdown_history:
                            avg_input = np.mean([s[1] for s in self._breakdown_history])
                            avg_gui = np.mean([s[2] for s in self._breakdown_history])
                            avg_command = np.mean([s[3] for s in self._breakdown_history])
                            avg_step = np.mean([s[4] for s in self._breakdown_history])
                            avg_record = np.mean([s[5] for s in self._breakdown_history])
                            avg_viz = np.mean([s[6] for s in self._breakdown_history])
                            viz_fps = None
                            if len(self._viz_frame_times) >= 2:
                                span = self._viz_frame_times[-1] - self._viz_frame_times[0]
                                if span > 0:
                                    viz_fps = (len(self._viz_frame_times) - 1) / span
                            viz_fps_text = f", VizFPS={viz_fps:.1f}" if viz_fps is not None else ""
                            print(f"  [BREAKDOWN AVG] Input={avg_input*1000:.1f}ms, GUI={avg_gui*1000:.1f}ms, "
                                  f"Command={avg_command*1000:.1f}ms, Step={avg_step*1000:.1f}ms, "
                                  f"Record={avg_record*1000:.1f}ms, Viz={avg_viz*1000:.1f}ms{viz_fps_text}")
                        print(f"  [INNER LOOP] Running at {robot_config.inner_control_frequency:.0f}Hz (motor commands)")


                # Sleep to maintain inner loop frequency (120Hz)
                inner_rate_limiter.sleep()

        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            print("\nShutting down...")
            if self.is_recording:
                self.stop_recording(success=False)  # Discard on interrupt
            print("Closing environment...")
            self.env.close()  # This already calls robot_hardware.shutdown()
            print("Stopping SpaceMouse...")
            self.spacemouse.stop()
            print("Stopping GUI...")
            self.gui.stop()
            cv2.destroyAllWindows()
            print("✓ Shutdown complete")

if __name__ == "__main__":
    # Auto-fix USB latency on startup
    auto_fix_usb_latency()

    collector = None
    try:
        collector = DemoCollector()
        collector.run()
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if interrupted during initialization
        if collector is not None:
            print("\n[Main] Ensuring cleanup...")
            try:
                if hasattr(collector, 'env') and collector.env is not None:
                    collector.env.close()
                if hasattr(collector, 'spacemouse') and collector.spacemouse is not None:
                    collector.spacemouse.stop()
                if hasattr(collector, 'gui') and collector.gui is not None:
                    collector.gui.stop()
            except Exception as e:
                print(f"Error during final cleanup: {e}")
