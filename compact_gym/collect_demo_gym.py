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
import threading
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
            
        np.savez_compressed(filename, **data)
        print(f"Saved {len(self.trajectory)} steps to {filename}")

    def run(self):
        print("Resetting Robot...")
        obs, _ = self.env.reset()
        print("Ready. Use GUI buttons to record.")
        print("  SpaceMouse buttons: left=open gripper, right=close gripper")
        print("  Close the GUI window to quit.")

        # Dual-frequency architecture (like compact_code):
        # - Inner loop at 120Hz: motor commands (env.step)
        # - Outer loop at 10Hz: SpaceMouse input, encoder polling, recording
        inner_rate_limiter = RateLimiter(frequency=robot_config.inner_control_frequency, warn=False)

        # Outer loop timing
        outer_loop_dt = 1.0 / robot_config.control_frequency  # 0.1s for 10Hz
        outer_loop_period = int(robot_config.inner_control_frequency / robot_config.control_frequency)  # 12 iterations
        outer_loop_target_time = time.perf_counter() + outer_loop_dt
        outer_loop_counter = 0

        # Cache for latest action (reused across inner loop iterations)
        latest_action = np.zeros(7)  # [vel(3), ang_vel(3), gripper(1)]

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
                    if self.env.robot_hardware and self.env.robot_hardware.initialized:
                        self.env.robot_hardware.poll_encoders(outer_loop_start_time=outer_loop_start_time)

                    # Detailed timing breakdown
                    t_input_start = time.perf_counter()

                    # 1. Update SpaceMouse
                    self.spacemouse.update()
                    t_input = time.perf_counter() - t_input_start

                    # Button Logic for Recording
                    left_btn, right_btn = self.spacemouse.get_gripper_button_states()
                    # Simple toggle logic could go here, but let's stick to explicit
                    # We need edge detection for buttons ideally, but driver returns Held state.
                    # Let's rely on GUI for robust toggle or infer from held?
                    # Actually, duplicate logic from legacy:
                    # Left (0) = Open/Start? No, typically buttons are just for gripper in that driver.
                    # Re-reading driver: buttons control gripper increment.
                    # USE GUI ONLY for recording for safety/simplicity,
                    # OR use keyboard if GUI not focused.
                    # SimpleControlGUI usually captures keys too.

                    # 2. Poll GUI Commands
                    t_gui_start = time.perf_counter()
                    if self.gui.is_available():
                        cmd = self.gui.get_command()

                        if cmd == ' ': # E-Stop (Spacebar) - Emergency Stop
                            print("\n[EMERGENCY STOP TRIGGERED]")
                            self.is_recording = False
                            self.trajectory = []
                            # Disable torque immediately via hardware interface if possible
                            if self.env.robot_hardware:
                                 self.env.robot_hardware.robot_driver.disable_torque_all()
                            sys.exit(1) # Exit immediately

                        elif cmd == 'd': # Done (Success) - Save and Stop
                            if self.is_recording:
                                self.stop_recording(success=True)
                            else:
                                print("Not recording, nothing to save.")

                        elif cmd == 'x' or cmd == 'Backspace': # Discard - Stop and Delete
                            if self.is_recording:
                                self.stop_recording(success=False)

                        elif cmd == 'h': # Home
                            print("Moving Home...")
                            # We can call the env's hardware directly if we assume authority
                            if self.env.robot_hardware:
                                # Use loose homing (doesn't reset env state completely, just moves robot)
                                # Or just use env.reset() which is safer/cleaner?
                                # Users usually expect "Home" button to just move robot, not reset episode ID etc.
                                # But for simplicity let's use the driver's move_to_home
                                try:
                                    home_pos = {mid: pos for mid, pos in zip(robot_config.motor_ids, robot_config.startup_home_positions)}
                                    # Open gripper
                                    home_pos[robot_config.motor_ids[-1]] = robot_config.gripper_encoder_max
                                    self.env.robot_hardware.robot_driver.move_to_home(home_pos, velocity_limit=robot_config.velocity_limit)
                                    # Sync after movement
                                    time.sleep(0.5)
                                    # We need to update kinematic state after forced move
                                    # env.step() will do it next iteration via execute_command -> but wait,
                                    # execute_command assumes we are at 'configuration'.
                                    # If we moved robot outside of execute_command, we MUST sync configuration.
                                    # WX200GymEnv doesn't expose a clean "sync" method other than inside reset.
                                    # So actually env.reset() IS the safest way to ensure consistency.
                                    self.env.reset()
                                except Exception as e:
                                    print(f"Error Homing: {e}")

                        elif cmd == 'g': # Reset EE
                            print("Resetting Gripper...")
                            if self.env.robot_hardware:
                                try:
                                    gripper_id = robot_config.motor_ids[-1]
                                    self.env.robot_hardware.robot_driver.reboot_motor(gripper_id)
                                    time.sleep(0.5)
                                    self.env.robot_hardware.robot_driver.send_motor_positions({gripper_id: robot_config.gripper_encoder_max})
                                    self.current_gripper_pos = robot_config.gripper_open_pos
                                except Exception as e:
                                    print(f"Error Resetting Gripper: {e}")

                        elif cmd == 'r': # Start Recording
                            print("Starting Recording...")
                            self.start_recording()
                        elif cmd == 's': # Stop & Save (Legacy key, map to Success too?)
                            # Keep 's' as standard stop/save
                            self.stop_recording(success=True)

                    t_gui = time.perf_counter() - t_gui_start

                    # 3. Get Control Command from SpaceMouse
                    t_command_start = time.perf_counter()
                    vel_world = self.spacemouse.get_velocity_command()
                    ang_vel_world = self.spacemouse.get_angular_velocity_command()
                    dt_outer = 1.0 / robot_config.control_frequency
                    self.current_gripper_pos = self.spacemouse.get_gripper_command(self.current_gripper_pos, dt_outer)

                    # Normalize and cache action for inner loop iterations
                    norm_vel = np.clip(vel_world / robot_config.velocity_scale, -1, 1)
                    norm_ang_vel = np.clip(ang_vel_world / robot_config.angular_velocity_scale, -1, 1)
                    gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
                    norm_gripper = np.clip(2.0 * (self.current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0, -1, 1)

                    latest_action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])
                    t_command = time.perf_counter() - t_command_start

                # === INNER LOOP (120Hz) ===
                # Step env with cached action (IK + motor commands)
                # This runs EVERY iteration for smooth motion
                t_step_start = time.perf_counter()
                next_obs, reward, terminated, truncated, info = self.env.step(latest_action)
                t_step = time.perf_counter() - t_step_start

                # === OUTER LOOP CONTINUED (10Hz) ===
                # Record data, visualization, and profiling only in outer loop
                if time_based_trigger:
                    # 4. Record Data
                    t_record_start = time.perf_counter()
                    if self.is_recording:
                        # Construct data packet matching legacy keys EXACTLY

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
                        # gym action is normalized (8D). We save the UNNORMALIZED world command?
                        # Legacy saved:
                        # 'velocity_world', 'angular_velocity_world', 'axis_angle', 'gripper_target'
                        # Actually legacy saved:
                        # 'action': np.concatenate([velocity_world, angular_velocity_world, [gripper_target]]) (7D)
                        # 'augmented_actions': ... (10D)

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
                            'augmented_actions': augmented_action,
                            'ee_pose_target': self._get_ee_pose_target(),

                            # Vision
                            'object_pose': raw_aruco.get('aruco_object_in_world', np.zeros(7)),
                            'object_visible': np.array([raw_aruco.get('aruco_visibility', np.zeros(3))[1]]),
                            'aruco_ee_in_world': raw_aruco.get('aruco_ee_in_world', np.zeros(7)),
                            'aruco_object_in_world': raw_aruco.get('aruco_object_in_world', np.zeros(7)),
                            'aruco_ee_in_object': raw_aruco.get('aruco_ee_in_object', np.zeros(7)),
                            'aruco_object_in_ee': raw_aruco.get('aruco_object_in_ee', np.zeros(7)),
                            'aruco_visibility': raw_aruco.get('aruco_visibility', np.zeros(3)),

                            # Frame (optional via flag, legacy had record_frames)
                            # We grabbed self.env.last_frame
                            'camera_frame': self._downscale_frame(self.env.last_frame) if self.env.last_frame is not None else np.zeros((robot_config.camera_height // robot_config.frame_downscale_factor, robot_config.camera_width // robot_config.frame_downscale_factor, 3), dtype=np.uint8)
                        }
                        self.trajectory.append(step_data)
                    t_record = time.perf_counter() - t_record_start

                    # 5. Visualization overlay
                    t_viz_start = time.perf_counter()
                    if self.env.last_frame is not None:
                        # Get annotated frame from env
                        frame = self.env.last_frame.copy()

                        # Draw Recording Status
                        color = (0, 0, 255) if self.is_recording else (0, 255, 0)
                        text = "RECORDING" if self.is_recording else "READY"
                        cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                        # Show
                        cv2.imshow("Data Collector", cv2.resize(frame, (960, 540)))
                        cv2.waitKey(1)
                    t_viz = time.perf_counter() - t_viz_start

                    # 6. Profiling (only in outer loop)
                    outer_loop_time = time.perf_counter() - outer_loop_start_time
                    self.profiler.record_control_loop_iteration(outer_loop_time)
                    if self.profiler.total_iterations % 50 == 0:
                        self.profiler.print_stats()
                        # Print detailed breakdown
                        print(f"  [BREAKDOWN] Input={t_input*1000:.1f}ms, GUI={t_gui*1000:.1f}ms, Command={t_command*1000:.1f}ms, "
                              f"Step={t_step*1000:.1f}ms, Record={t_record*1000:.1f}ms, Viz={t_viz*1000:.1f}ms")
                        print(f"  [INNER LOOP] Running at {robot_config.inner_control_frequency:.0f}Hz (motor commands)")

                obs = next_obs
                outer_loop_counter += 1

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
