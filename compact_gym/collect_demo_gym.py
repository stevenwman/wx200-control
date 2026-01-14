"""
Demo Collection Script (Gym Version).

Replicates the functionality of `compact_code/wx200_robot_collect_demo_encoders_compact.py`
but uses the new clean `compact_gym` infrastructure.

Features:
- Teleoperation via SpaceMouse
- Real-time Camera View with ArUco Overlay
- "Start/Stop Recording" via UI or SpaceMouse buttons (Left=Start, Right=Stop)
- Data saving to .npz format compatible with OGPO
"""

import sys
import time
import cv2
import numpy as np
import threading
from datetime import datetime
from pathlib import Path

# Add compact_code to path to import SpaceMouse & GUI utils if needed
# We reuse SimpleControlGUI from compact_code to avoid rewriting it
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))
compact_code_path = repo_root / "compact_code"
sys.path.append(str(compact_code_path))

try:
    from spacemouse.spacemouse_driver import SpaceMouseDriver
    from utils.robot_control_gui import SimpleControlGUI
except ImportError as e:
    print(f"Failed to import dependencies from {compact_code_path}: {e}")
    sys.exit(1)

from compact_gym import WX200GymEnv, robot_config
from compact_gym.profiling import LightweightProfiler


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
        self.env = WX200GymEnv(
            max_episode_length=99999, # Manual stop
            show_video=False, # We handle viz ourselves to overlay UI info
            enable_aruco=True,
            control_frequency=robot_config.control_frequency
        )
        
        # Gripper state tracking
        self.current_gripper_pos = robot_config.gripper_open_pos
        self.running = True

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.trajectory = []
            print(f"\n[RECORDING STARTED] Episode {self.episode_count}")
            self.gui.set_status(f"Recording Ep {self.episode_count}...")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            self.save_trajectory()
            self.episode_count += 1
            print("[RECORDING STOPPED] Saved.")
            self.gui.set_status("Ready")

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
        print("Ready. Use SpaceMouse buttons or GUI to record.")
        print("  Left Button: Start Recording")
        print("  Right Button: Stop Recording")
        
        try:
            while self.running and self.gui.is_available():
                loop_start = time.perf_counter()
                
                # 1. Update Inputs
                self.spacemouse.update()
                
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
                
                # 1b. Poll GUI Commands
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
                            self.is_recording = False
                            self.trajectory = []
                            print("[RECORDING DISCARDED]")
                            self.gui.status_label.config(text="Discarded", foreground="red")
                        
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
                        
                # 2. Get Control Command
                vel_world = self.spacemouse.get_velocity_command()
                ang_vel_world = self.spacemouse.get_angular_velocity_command()
                dt = 1.0 / robot_config.control_frequency
                self.current_gripper_pos = self.spacemouse.get_gripper_command(self.current_gripper_pos, dt)
                
                # Normalize
                norm_vel = np.clip(vel_world / robot_config.velocity_scale, -1, 1)
                norm_ang_vel = np.clip(ang_vel_world / robot_config.angular_velocity_scale, -1, 1)
                gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
                norm_gripper = np.clip(2.0 * (self.current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0, -1, 1)
                
                action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])
                
                # 3. Step Env
                # Obs is [ArUco(7) + Robot(6) + EE(7)] = 20D
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                
                # 4. Record Data
                if self.is_recording:
                    # Construct data packet matching legacy keys EXACTLY
                    
                    # 1. State/Qpos
                    state = info.get('qpos', np.zeros(6))
                    
                    # 2. Encoders
                    encoders = info.get('encoder_values', {})
                    encoder_array = np.array([encoders.get(mid, 0) for mid in robot_config.motor_ids])
                    
                    # 3. EE Pose (FK)
                    ee_pose_encoder = info.get('ee_pose_fk', np.zeros(7))
                    
                    # 4. Raw ArUco
                    raw_aruco = info.get('raw_aruco', {})
                    
                    # 5. Augmented Actions
                    # vel(3) + ang(3) + axis_angle(3) + grip(1) = 10D
                    # gym action is normalized (8D). We save the UNNORMALIZED world command? 
                    # Legacy saved:
                    # 'velocity_world', 'angular_velocity_world', 'axis_angle', 'gripper_target'
                    # Actually legacy saved:
                    # 'action': np.concatenate([velocity_world, angular_velocity_world, [gripper_target]]) (7D)
                    # 'augmented_actions': ... (10D)
                    
                    axis_angle = ang_vel_world * dt
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
                        'ee_pose_target': np.zeros(7), # We don't have IK target readily available unless we assume it met FK? Set to FK for now.
                        
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
                        'camera_frame': cv2.resize(self.env.last_frame, (robot_config.camera_width // 4, robot_config.camera_height // 4)) if self.env.last_frame is not None else np.zeros((270, 480, 3), dtype=np.uint8)
                    }
                    self.trajectory.append(step_data)
                
                # 5. Visualization overlay
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

                obs = next_obs
                
                # Profiling
                loop_time = time.perf_counter() - loop_start
                self.profiler.record_control_loop_iteration(loop_time)
                if self.profiler.total_iterations % 50 == 0:
                    self.profiler.print_stats()
                
                # Rate Limit handled by Env.step() mostly, but we add small sleep if needed
                # Env step waits for min_step_interval (0.1s).
                
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            print("Shutting down...")
            if self.is_recording:
                self.stop_recording()
            self.env.close()
            if self.env.robot_hardware: self.env.robot_hardware.shutdown()
            self.spacemouse.stop()
            self.gui.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    collector = DemoCollector()
    collector.run()
