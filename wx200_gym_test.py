"""
Test script for WX200 gym environment with SpaceMouse and camera.

Uses the gym environment to control the robot while collecting ArUco data.
"""
import time
import numpy as np
import argparse
import sys
import select
import termios
import tty

from wx200_gym_env import WX200GymEnv
from robot_control.robot_config import robot_config
from spacemouse.spacemouse_driver import SpaceMouseDriver

# Simple rate limiter (fallback if loop_rate_limiters not available)
try:
    from loop_rate_limiters import RateLimiter
except ImportError:
    class RateLimiter:
        def __init__(self, frequency, warn=False):
            self.dt = 1.0 / frequency
            self._last_time = time.perf_counter()
        def sleep(self):
            elapsed = time.perf_counter() - self._last_time
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_time = time.perf_counter()


class GymEnvTest:
    """Test gym environment with SpaceMouse and camera."""
    
    def __init__(self, camera_id=None, width=None, height=None, fps=None,
                 max_episode_length=1000, show_video=True, show_axes=True):
        # SpaceMouse
        self.spacemouse = None
        
        # Camera setup (now handled by gym environment)
        self.camera_id = camera_id if camera_id is not None else robot_config.camera_id
        self.width = width if width is not None else robot_config.camera_width
        self.height = height if height is not None else robot_config.camera_height
        self.fps = fps if fps is not None else robot_config.camera_fps
        
        # Simple reset control (keyboard-based)
        self.show_video = show_video
        self.show_axes = show_axes
        self.old_settings = None  # For terminal input
        
        # Initialize gym environment with camera/ArUco support
        self.env = WX200GymEnv(
            max_episode_length=max_episode_length,
            camera_id=camera_id,
            width=width,
            height=height,
            fps=fps,
            enable_aruco=True,
            show_video=self.show_video,
            show_axes=self.show_axes
        )
        
        # Action scaling (to normalize SpaceMouse input to [-1, 1])
        self.velocity_scale = 1.0 / robot_config.velocity_scale  # Inverse of SpaceMouse scale
        self.angular_velocity_scale = 1.0 / robot_config.angular_velocity_scale
    
    def _normalize_action(self, velocity_world, angular_velocity_world, gripper_target):
        """
        Normalize velocity commands to gym action format [-1, 1].
        
        Args:
            velocity_world: [vx, vy, vz] in m/s
            angular_velocity_world: [wx, wy, wz] in rad/s
            gripper_target: Gripper position in meters
        
        Returns:
            np.ndarray: Normalized action [vx, vy, vz, wx, wy, wz, gripper] in [-1, 1]
        """
        # Normalize velocities (inverse of denormalization in env)
        velocity_scale = 0.25  # Must match env._denormalize_action
        angular_velocity_scale = 1.0  # Must match env._denormalize_action
        
        action = np.zeros(7)
        action[:3] = velocity_world / velocity_scale
        action[3:6] = angular_velocity_world / angular_velocity_scale
        
        # Normalize gripper from [closed, open] to [-1, 1]
        gripper_range = robot_config.gripper_open_pos - robot_config.gripper_closed_pos
        if gripper_range != 0:
            gripper_norm = (gripper_target - robot_config.gripper_closed_pos) / gripper_range
            action[6] = gripper_norm * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        else:
            action[6] = -1.0  # Default to closed
        
        return np.clip(action, -1.0, 1.0)
    
    # ArUco tracking is now handled by gym environment
    
    def _handle_reset_command(self):
        """Handle reset command from GUI."""
        print("\n" + "="*60, flush=True)
        print("_handle_reset_command: Resetting environment...", flush=True)
        print("="*60, flush=True)
        
        try:
            obs, info = self.env.reset()
            print("✓ Environment reset complete", flush=True)
            print(f"✓ New observation shape: {obs.shape}", flush=True)
            print("="*60 + "\n", flush=True)
            return obs
        except Exception as e:
            print(f"ERROR during reset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Return current observation if reset fails
            return self.env._get_observation()
    
    def run(self):
        """Run the test control loop."""
        print("\n" + "="*60)
        print("WX200 Gym Environment Test")
        print("="*60)
        print("Using gym environment with SpaceMouse and camera")
        print("="*60 + "\n")
        
        # Setup SpaceMouse
        self.spacemouse = SpaceMouseDriver(
            velocity_scale=robot_config.velocity_scale,
            angular_velocity_scale=robot_config.angular_velocity_scale
        )
        self.spacemouse.start()
        
        # Camera is now handled by gym environment, but we can still visualize if needed
        # Setup keyboard input for reset (non-blocking)
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("✓ Reset control: Press 'R' key in terminal to reset environment")
        except:
            print("⚠️  Terminal input not available - reset functionality disabled")
            self.old_settings = None
        
        # Get initial observation (robot already at home from initialization, don't reset again)
        obs = self.env._get_observation()
        print("✓ Environment ready (robot already at home from initialization)")
        
        # Control loop with rate limiter (same as original teleop)
        control_rate = RateLimiter(frequency=self.env.control_frequency, warn=False)
        try:
            while True:
                dt = control_rate.dt
                
                # Check for keyboard input (non-blocking)
                if self.old_settings is not None:
                    if select.select([sys.stdin], [], [], 0)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 'r':
                            print("\n" + "="*60, flush=True)
                            print("Keyboard: Reset requested (R key pressed)", flush=True)
                            print("="*60, flush=True)
                            obs = self._handle_reset_command()
                            continue
                
                # Get SpaceMouse input
                self.spacemouse.update()
                velocity_world = self.spacemouse.get_velocity_command()
                angular_velocity_world = self.spacemouse.get_angular_velocity_command()
                
                # Get current gripper position from robot_base (same as original teleop)
                # This is the actual tracked state, not from observation
                current_gripper_pos = self.env.robot_base.gripper_current_position
                
                # Get gripper command (increment-based from SpaceMouse)
                gripper_target = self.spacemouse.get_gripper_command(current_gripper_pos, dt)
                
                # Normalize to gym action format
                action = self._normalize_action(velocity_world, angular_velocity_world, gripper_target)
                
                # Step environment (ArUco observations are now included in obs and stored in env.aruco_obs_dict)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                # Get ArUco observations from env (already computed in step)
                aruco_obs = self.env.aruco_obs_dict
                
                # Debug: Print observation and check for lost tracking
                # Check if any marker lost track (visibility is 0 but pose is not all zeros, or pose is all zeros)
                lost_track = False
                visibility = aruco_obs.get('aruco_visibility', np.zeros(3))
                if np.any(visibility < 1.0):
                    # Check if we have non-zero poses but visibility is 0 (lost track)
                    obj_in_world = aruco_obs.get('aruco_object_in_world', np.zeros(7))
                    obj_in_ee = aruco_obs.get('aruco_object_in_ee', np.zeros(7))
                    ee_in_world = aruco_obs.get('aruco_ee_in_world', np.zeros(7))
                    
                    # Lost track if visibility is 0 but we had poses before (or if all poses are zeros when they shouldn't be)
                    if (visibility[1] == 0.0 and np.any(obj_in_world != 0)) or \
                       (visibility[2] == 0.0 and np.any(ee_in_world != 0)) or \
                       (visibility[0] == 0.0 and (np.any(obj_in_world != 0) or np.any(ee_in_world != 0))):
                        lost_track = True
                
                # Print observation
                RED = '\033[91m'
                RESET = '\033[0m'
                color = RED if lost_track else RESET
                # Print observation/visibility with 3 decimal places (standard format)
                obs_fmt = np.array2string(np.array(obs), precision=3, suppress_small=False)
                vis_fmt = np.array2string(np.array(visibility), precision=3, suppress_small=False)
                print(f"{color}Obs: {obs_fmt} | Vis: {vis_fmt} | Lost: {str(lost_track)}{RESET}")
                
                # Handle episode termination (just reset step counter, don't move robot)
                if terminated or truncated:
                    print(f"\nEpisode ended: terminated={terminated}, truncated={truncated}")
                    # Reset episode counter but don't move robot (it's already in a good position)
                    self.env.episode_step = 0
                    self.env.episode_return = 0.0
                    obs = self.env._get_observation()
                    print("✓ Episode counter reset (robot position unchanged)")
                
                # Rate limit (precise timing, same as original teleop)
                control_rate.sleep()
        
        except KeyboardInterrupt:
            print("\n\nKeyboard interrupt detected. Stopping...")
        finally:
            # Cleanup
            if self.old_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
                except:
                    pass
            if self.spacemouse:
                self.spacemouse.stop()
            self.env.close()  # This will cleanup camera too


def main():
    parser = argparse.ArgumentParser(description='WX200 Gym Environment Test')
    parser.add_argument('--camera-id', type=int, help='Camera device ID')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max steps per episode')
    
    args = parser.parse_args()
    
    test = GymEnvTest(
        camera_id=args.camera_id,
        max_episode_length=args.max_episode_length,
        show_video=not args.no_vis,
        show_axes=not args.no_vis
    )
    
    test.run()


if __name__ == "__main__":
    main()

