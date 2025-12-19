"""
Verification Script: Teleop via WX200GymEnv.

Recreates the teleoperation workflow using the new Gym environment and 
legacy SpaceMouse driver.
"""
import sys
import time
import numpy as np
from pathlib import Path

# Add compact_code to path to import SpaceMouse
compact_code_path = Path(__file__).parent.parent / "compact_code"
sys.path.append(str(compact_code_path))

try:
    from spacemouse.spacemouse_driver import SpaceMouseDriver
except ImportError as e:
    print(f"Failed to import SpaceMouseDriver from {compact_code_path}: {e}")
    sys.exit(1)

from compact_gym import WX200GymEnv, robot_config

def main():
    print("Initializing Teleop Verification (Gym + SpaceMouse)...")
    
    # Initialize SpaceMouse with robot config scales
    spacemouse = SpaceMouseDriver(
        velocity_scale=robot_config.velocity_scale,
        angular_velocity_scale=robot_config.angular_velocity_scale
    )
    spacemouse.start()
    print("SpaceMouse initialized.")
    
    # Initialize Gym Env
    env = WX200GymEnv(
        max_episode_length=1000,
        show_video=True,
        enable_aruco=True,
        control_frequency=robot_config.control_frequency
    )
    
    # Track gripper state locally (meters)
    current_gripper_pos = robot_config.gripper_open_pos
    
    try:
        print("\nResetting environment (Moving to Home)...")
        obs, _ = env.reset()
        print("Ready! Control the robot with SpaceMouse.")
        print("Press Ctrl+C to exit.")
        
        # Main Loop
        while True:
            # 1. Update SpaceMouse state
            spacemouse.update()
            
            # 2. Get Velocity Commands (m/s, rad/s)
            # These are already scaled by velocity_scale passed to init
            vel_world = spacemouse.get_velocity_command()
            ang_vel_world = spacemouse.get_angular_velocity_command()
            
            # 3. Get Gripper Command (Meters)
            # Use dt from config
            dt = 1.0 / robot_config.control_frequency
            current_gripper_pos = spacemouse.get_gripper_command(current_gripper_pos, dt)
            
            # 4. Normalize for Gym [-1, 1]
            # Velocity: input is roughly [-scale, scale], so divide by scale
            # (SpaceMouseDriver outputs can technically exceed scale if raw input > 1, but standard is ~1)
            norm_vel = vel_world / robot_config.velocity_scale
            norm_ang_vel = ang_vel_world / robot_config.angular_velocity_scale
            
            # Gripper: Map [open, closed] -> [-1, 1]
            gripper_range = robot_config.gripper_closed_pos - robot_config.gripper_open_pos
            norm_gripper = 2.0 * (current_gripper_pos - robot_config.gripper_open_pos) / gripper_range - 1.0
            
            # Clip to be safe
            norm_vel = np.clip(norm_vel, -1.0, 1.0)
            norm_ang_vel = np.clip(norm_ang_vel, -1.0, 1.0)
            norm_gripper = np.clip(norm_gripper, -1.0, 1.0)
            
            # Construct Action: [vx, vy, vz, wx, wy, wz, gripper]
            action = np.concatenate([norm_vel, norm_ang_vel, [norm_gripper]])
            
            # 5. Step Environment
            obs, _, _, _, _ = env.step(action)
            
            # Rate limiting handled inside env.step()
            
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        if env.robot_hardware: env.robot_hardware.shutdown()
        spacemouse.stop()

if __name__ == "__main__":
    main()
