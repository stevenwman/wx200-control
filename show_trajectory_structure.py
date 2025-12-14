#!/usr/bin/env python3
"""
Quick script to show the structure of saved trajectory files.
"""

import numpy as np
import sys
from pathlib import Path

def show_trajectory_structure(npz_path):
    """Load and display trajectory file structure."""
    data = np.load(npz_path, allow_pickle=True)
    
    print("="*60)
    print(f"Trajectory File: {npz_path}")
    print("="*60)
    
    # Show all keys
    print("\nAvailable keys in .npz file:")
    for key in data.keys():
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"  - {key}: shape {arr.shape}, dtype {arr.dtype}")
        else:
            print(f"  - {key}: {type(arr).__name__}")
    
    # Show main arrays
    print("\n" + "="*60)
    print("Main Data Arrays:")
    print("="*60)
    
    if 'timestamps' in data:
        timestamps = data['timestamps']
        print(f"\ntimestamps: shape {timestamps.shape}")
        print(f"  First 5: {timestamps[:5]}")
        print(f"  Duration: {timestamps[-1]:.2f} seconds")
    
    if 'states' in data:
        states = data['states']
        print(f"\nstates: shape {states.shape}")
        print(f"  First sample: {states[0]}")
        print(f"  Labels: {data.get('metadata', {}).item().get('state_labels', 'N/A') if 'metadata' in data else 'N/A'}")
    
    if 'actions' in data:
        actions = data['actions']
        print(f"\nactions: shape {actions.shape}")
        print(f"  First sample: {actions[0]}")
        print(f"  Labels: {data.get('metadata', {}).item().get('action_labels', 'N/A') if 'metadata' in data else 'N/A'}")
    
    if 'ee_poses_debug' in data:
        ee_poses = data['ee_poses_debug']
        print(f"\nee_poses_debug: shape {ee_poses.shape}")
        print(f"  Format: [x, y, z, qw, qx, qy, qz] (position + quaternion wxyz)")
        print(f"  First sample position: {ee_poses[0, :3]}")
        print(f"  First sample quaternion: {ee_poses[0, 3:]}")
        print(f"  Labels: {data.get('metadata', {}).item().get('ee_pose_debug_labels', 'N/A') if 'metadata' in data else 'N/A'}")
        
        # Show position statistics
        positions = ee_poses[:, :3]
        print(f"\n  Position Statistics:")
        print(f"    X range: [{positions[:, 0].min():.4f}, {positions[:, 0].max():.4f}]")
        print(f"    Y range: [{positions[:, 1].min():.4f}, {positions[:, 1].max():.4f}]")
        print(f"    Z range: [{positions[:, 2].min():.4f}, {positions[:, 2].max():.4f}]")
    
    # Show metadata
    if 'metadata' in data:
        metadata = data['metadata'].item()
        print("\n" + "="*60)
        print("Metadata:")
        print("="*60)
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*60)
    print("Example: How to load for training")
    print("="*60)
    print("""
import numpy as np

# Load trajectory
data = np.load('trajectory_file.npz', allow_pickle=True)

# Access EE positions (for training)
ee_poses = data['ee_poses_debug']  # Shape: (N, 7)
ee_positions = ee_poses[:, :3]     # Shape: (N, 3) - just x, y, z
ee_orientations = ee_poses[:, 3:]   # Shape: (N, 4) - quaternion wxyz

# Access states and actions
states = data['states']      # Shape: (N, 6) - 5 joints + gripper
actions = data['actions']    # Shape: (N, 7) - velocities + gripper target
timestamps = data['timestamps']  # Shape: (N,)

# Example: Create training pairs
# Option 1: Use EE positions as observations
observations = ee_positions  # or ee_poses for full pose

# Option 2: Use joint states as observations
observations = states[:, :5]  # Just joints, no gripper

# Actions are already in the file
targets = actions
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find a trajectory file in data/
        data_dir = Path("data")
        if data_dir.exists():
            npz_files = list(data_dir.glob("*.npz"))
            if npz_files:
                npz_path = npz_files[-1]  # Use most recent
                print(f"No file specified, using most recent: {npz_path}")
            else:
                print("Usage: python show_trajectory_structure.py <trajectory_file.npz>")
                print("Or place a .npz file in the data/ directory")
                sys.exit(1)
        else:
            print("Usage: python show_trajectory_structure.py <trajectory_file.npz>")
            sys.exit(1)
    else:
        npz_path = Path(sys.argv[1])
    
    if not npz_path.exists():
        print(f"Error: File not found: {npz_path}")
        sys.exit(1)
    
    show_trajectory_structure(npz_path)
