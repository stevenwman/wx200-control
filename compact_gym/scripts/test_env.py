"""
Test script for WX200GymEnv (Random Policy).

Verifies that the environment can be initialized, reset, stepped, and closed.
Does NOT require SpaceMouse.
Safely tests connection and basic movement limits (random actions).
"""
import time
import numpy as np
from deployment.gym_env import WX200GymEnv

def main():
    print("Testing WX200GymEnv with Random Policy...")
    
    # Create environment
    # Use camera_id=None to use config default
    env = WX200GymEnv(
        max_episode_length=100,
        show_video=True,
        enable_aruco=True
    )
    
    try:
        # Reset
        print("\nResetting environment...")
        obs, _ = env.reset()
        print("Reset complete.")
        print(f"Observation shape: {obs.shape}")
        
        # Run for a few steps
        print("\nRunning random steps...")
        for i in range(50):
            # Sample random action [-1, 1]
            # Scale down to be safe (0.1 scale means slow movement)
            action = np.random.uniform(-0.1, 0.1, size=env.action_space.shape)
            
            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            
            if i % 10 == 0:
                print(f"Step {i}: Obs={obs[:3]}... Reward={reward}")
            
            time.sleep(0.1) # Slow down for visibility
            
            if terminated or truncated:
                print("Episode finished early.")
                break
                
        print("\nTest Complete.")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("Closing environment...")
        env.close()
        # Explicitly call shutdown on hardware if needed (env.close should ideally do this)
        if env.robot_hardware:
             env.robot_hardware.shutdown()

if __name__ == "__main__":
    main()
