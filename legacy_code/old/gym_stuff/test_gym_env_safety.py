"""
Test script to verify IK safety bounds with random actions.

This script tests that the end-effector position stays within the configured
safety bounds when random actions are applied to the gym environment.
"""
import numpy as np
import time
import argparse
from wx200_gym_env_utils import make_wx200_env
from robot_control.robot_config import robot_config


def test_safety_bounds(env, num_steps=1000, verbose=True):
    """
    Test that end-effector stays within safety bounds with random actions.
    
    Args:
        env: Gym environment instance
        num_steps: Number of random action steps to test
        verbose: Whether to print detailed information
    
    Returns:
        dict: Statistics about the test
    """
    # Get safety bounds from config
    bounds = {
        'x': robot_config.ee_bound_x,
        'y': robot_config.ee_bound_y,
        'z': robot_config.ee_bound_z,
    }
    
    if verbose:
        print("\n" + "="*60)
        print("Testing IK Safety Bounds")
        print("="*60)
        print(f"EE Bounds:")
        print(f"  X: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}] m")
        print(f"  Y: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}] m")
        print(f"  Z: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}] m")
        print(f"\nTesting with {num_steps} random actions...")
        print("="*60 + "\n")
    
    # Reset environment
    obs, info = env.reset()
    
    # Statistics
    stats = {
        'total_steps': 0,
        'violations': [],
        'ee_positions': [],
        'min_ee_pos': np.array([np.inf, np.inf, np.inf]),
        'max_ee_pos': np.array([-np.inf, -np.inf, -np.inf]),
        'actions_applied': [],
    }
    
    # Get initial EE position from pose controller (not SE3 object)
    initial_ee_pos = env.env.robot_base.robot_controller.pose_controller.get_target_position()
    stats['ee_positions'].append(initial_ee_pos.copy())
    stats['min_ee_pos'] = np.minimum(stats['min_ee_pos'], initial_ee_pos)
    stats['max_ee_pos'] = np.maximum(stats['max_ee_pos'], initial_ee_pos)
    
    if verbose:
        print(f"Initial EE position: [{initial_ee_pos[0]:.3f}, {initial_ee_pos[1]:.3f}, {initial_ee_pos[2]:.3f}]")
    
    # Test with random actions
    for step in range(num_steps):
        # Generate random action in [-1, 1] range (as neural network would)
        action = np.random.uniform(-1.0, 1.0, size=env.action_space.shape[0])
        stats['actions_applied'].append(action.copy())
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        stats['total_steps'] += 1
        
        # Get current end-effector position from pose controller
        current_ee_pos = env.env.robot_base.robot_controller.pose_controller.get_target_position()
        stats['ee_positions'].append(current_ee_pos.copy())
        
        # Update min/max
        stats['min_ee_pos'] = np.minimum(stats['min_ee_pos'], current_ee_pos)
        stats['max_ee_pos'] = np.maximum(stats['max_ee_pos'], current_ee_pos)
        
        # Check bounds violations
        violation = False
        violation_details = []
        
        if current_ee_pos[0] < bounds['x'][0] or current_ee_pos[0] > bounds['x'][1]:
            violation = True
            violation_details.append(f"X: {current_ee_pos[0]:.3f} (bounds: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}])")
        
        if current_ee_pos[1] < bounds['y'][0] or current_ee_pos[1] > bounds['y'][1]:
            violation = True
            violation_details.append(f"Y: {current_ee_pos[1]:.3f} (bounds: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}])")
        
        if current_ee_pos[2] < bounds['z'][0] or current_ee_pos[2] > bounds['z'][1]:
            violation = True
            violation_details.append(f"Z: {current_ee_pos[2]:.3f} (bounds: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}])")
        
        if violation:
            stats['violations'].append({
                'step': step,
                'ee_pos': current_ee_pos.copy(),
                'action': action.copy(),
                'details': violation_details
            })
            if verbose:
                print(f"⚠️  Step {step}: BOUNDS VIOLATION!")
                for detail in violation_details:
                    print(f"     {detail}")
                print(f"     Action: {action}")
        
        # Print progress every 100 steps
        if verbose and (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{num_steps}: EE pos = [{current_ee_pos[0]:.3f}, {current_ee_pos[1]:.3f}, {current_ee_pos[2]:.3f}]")
        
        # Handle episode termination
        if terminated or truncated:
            if verbose:
                print(f"\nEpisode ended at step {step + 1}: terminated={terminated}, truncated={truncated}")
            obs, info = env.reset()
            if verbose:
                reset_ee_pos = env.env.robot_base.robot_controller.pose_controller.get_target_position()
                print(f"Reset to EE position: [{reset_ee_pos[0]:.3f}, {reset_ee_pos[1]:.3f}, {reset_ee_pos[2]:.3f}]")
    
    # Print summary
    print("\n" + "="*60)
    print("Safety Bounds Test Summary")
    print("="*60)
    print(f"Total steps: {stats['total_steps']}")
    print(f"Bounds violations: {len(stats['violations'])}")
    
    if len(stats['violations']) > 0:
        print(f"\n⚠️  WARNING: {len(stats['violations'])} bounds violations detected!")
        print("\nViolation details:")
        for i, violation in enumerate(stats['violations'][:10]):  # Show first 10
            print(f"  {i+1}. Step {violation['step']}:")
            for detail in violation['details']:
                print(f"     {detail}")
        if len(stats['violations']) > 10:
            print(f"  ... and {len(stats['violations']) - 10} more violations")
    else:
        print("\n✓ All EE positions stayed within safety bounds!")
    
    print(f"\nEE Position Statistics:")
    print(f"  Min: [{stats['min_ee_pos'][0]:.3f}, {stats['min_ee_pos'][1]:.3f}, {stats['min_ee_pos'][2]:.3f}]")
    print(f"  Max: [{stats['max_ee_pos'][0]:.3f}, {stats['max_ee_pos'][1]:.3f}, {stats['max_ee_pos'][2]:.3f}]")
    print(f"  Range X: [{stats['min_ee_pos'][0]:.3f}, {stats['max_ee_pos'][0]:.3f}] (bounds: [{bounds['x'][0]:.3f}, {bounds['x'][1]:.3f}])")
    print(f"  Range Y: [{stats['min_ee_pos'][1]:.3f}, {stats['max_ee_pos'][1]:.3f}] (bounds: [{bounds['y'][0]:.3f}, {bounds['y'][1]:.3f}])")
    print(f"  Range Z: [{stats['min_ee_pos'][2]:.3f}, {stats['max_ee_pos'][2]:.3f}] (bounds: [{bounds['z'][0]:.3f}, {bounds['z'][1]:.3f}])")
    print("="*60 + "\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Test IK safety bounds with random actions')
    parser.add_argument('--num-steps', type=int, default=1000, help='Number of random action steps to test')
    parser.add_argument('--max-episode-length', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--no-vis', action='store_true', help='Disable video window')
    parser.add_argument('--no-aruco', action='store_true', help='Disable ArUco tracking')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    print("Initializing environment...")
    env = make_wx200_env(
        max_episode_length=args.max_episode_length,
        enable_aruco=not args.no_aruco,
        show_video=not args.no_vis,
        show_axes=not args.no_vis,
        seed=args.seed,
    )
    
    try:
        # Run safety bounds test
        stats = test_safety_bounds(env, num_steps=args.num_steps, verbose=True)
        
        # Exit code based on violations
        if len(stats['violations']) > 0:
            print(f"\n❌ Test FAILED: {len(stats['violations'])} bounds violations detected")
            return 1
        else:
            print(f"\n✓ Test PASSED: All EE positions stayed within bounds")
            return 0
    
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return 130
    finally:
        env.close()


if __name__ == "__main__":
    exit(main())

