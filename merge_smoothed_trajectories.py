#!/usr/bin/env python3
"""
Merge all smoothed trajectory files into a single dataset for training.

Reads all *_smoothed.npz files from a directory and creates a merged dataset
with observations, actions, rewards, terminals, and masks.
"""

import numpy as np
from pathlib import Path
import sys


def merge_smoothed_trajectories(
    input_dir: Path,
    output_path: Path = None,
    use_smoothed: bool = True,
):
    """
    Merge all smoothed trajectory files into a single dataset.
    
    Args:
        input_dir: Directory containing *_smoothed.npz files
        output_path: Path to save merged dataset (default: merged_dataset.npz in input_dir)
        use_smoothed: If True, use smoothed_aruco_* keys; else use original aruco_* keys
    """
    # Find all smoothed trajectory files
    if use_smoothed:
        pattern = "*_smoothed.npz"
        key_prefix = "smoothed_"
    else:
        pattern = "trajectory_*.npz"
        key_prefix = ""
        # Exclude already-smoothed and quarantined files
        def is_valid(f):
            return not f.name.startswith("quarantined_") and "_smoothed" not in f.name
    
    traj_files = sorted(input_dir.glob(pattern))
    
    if use_smoothed:
        traj_files = [f for f in traj_files if not f.name.startswith("quarantined_")]
    else:
        traj_files = [f for f in traj_files if is_valid(f)]
    
    if not traj_files:
        print(f"No trajectory files found in {input_dir}")
        if use_smoothed:
            print("  (Looking for *_smoothed.npz files)")
        return None
    
    print(f"Found {len(traj_files)} trajectory files")
    print("="*60)
    
    datasets = []
    for i, fpath in enumerate(traj_files, 1):
        print(f"[{i}/{len(traj_files)}] Loading: {fpath.name}")
        try:
            data = np.load(fpath, allow_pickle=True)
            datasets.append(data)
        except Exception as e:
            print(f"  ⚠️  Error loading {fpath.name}: {e}")
            continue
    
    if not datasets:
        print("No valid datasets loaded!")
        return None
    
    print(f"\nLoaded {len(datasets)} datasets")
    print(f"Keys in first dataset: {list(datasets[0].keys())}")
    
    # Merge the datasets
    all_observations = []
    all_next_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_masks = []
    
    for i, data in enumerate(datasets):
        try:
            # Extract relevant keys (use smoothed if available, fallback to original)
            if use_smoothed:
                ee_key = "smoothed_aruco_ee_in_world"
                obj_world_key = "smoothed_aruco_object_in_world"
                obj_ee_key = "smoothed_aruco_object_in_ee"
            else:
                ee_key = "aruco_ee_in_world"
                obj_world_key = "aruco_object_in_world"
                obj_ee_key = "aruco_object_in_ee"
            
            # Fallback to original if smoothed not available
            if ee_key not in data:
                ee_key = "aruco_ee_in_world"
            if obj_world_key not in data:
                obj_world_key = "aruco_object_in_world"
            if obj_ee_key not in data:
                obj_ee_key = "aruco_object_in_ee"
            
            aruco_ee_in_world = data[ee_key]           # (T, 7)
            aruco_object_in_world = data[obj_world_key]   # (T, 7)
            aruco_object_in_ee = data[obj_ee_key]         # (T, 7)
            states = data['states']                                 # (T, ?)
            augmented_actions = data['augmented_actions']           # (T, ?)
            actions = data['actions']                               # (T, 7)
            
            T = aruco_ee_in_world.shape[0]
            assert aruco_object_in_world.shape[0] == T, f"Dataset {i}: object_world shape mismatch"
            assert aruco_object_in_ee.shape[0] == T, f"Dataset {i}: object_ee shape mismatch"
            assert states.shape[0] == T, f"Dataset {i}: states shape mismatch"
            assert augmented_actions.shape[0] == T, f"Dataset {i}: augmented_actions shape mismatch"
            assert actions.shape[0] == T, f"Dataset {i}: actions shape mismatch"
            
            # Observations: concat those four features
            obs = np.concatenate(
                [
                    aruco_ee_in_world,
                    aruco_object_in_world,
                    aruco_object_in_ee,
                    states,  # last dimension is gripper position
                ],
                axis=1  # feature concat
            )  # shape: (T, D)
            
            # Next observations (shifted by one)
            next_obs = np.zeros_like(obs)
            next_obs[:-1] = obs[1:]
            next_obs[-1] = obs[-1]  # for last step, next_obs = obs (repeat last)
            
            # Rewards: -1 for all except last action, which is 0
            rewards = -1 * np.ones((T,), dtype=np.float32)
            rewards[-1] = 0
            
            # Terminals: 0 except for last step, which is 1
            dones = np.zeros((T,), dtype=np.float32)
            dones[-1] = 1
            
            # Masks: 1.0 - dones
            masks = 1.0 - dones
            
            all_observations.append(obs)
            all_next_observations.append(next_obs)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_terminals.append(dones)
            all_masks.append(masks)
            
        except Exception as e:
            print(f"  ❌ Error processing dataset {i}: {e}")
            continue
    
    if not all_observations:
        print("No valid observations collected!")
        return None
    
    # Concat along T
    print("\nConcatenating datasets...")
    merged = {
        'observations': np.concatenate(all_observations, axis=0),
        'actions': np.concatenate(all_actions, axis=0),
        'next_observations': np.concatenate(all_next_observations, axis=0),
        'rewards': np.concatenate(all_rewards, axis=0),
        'terminals': np.concatenate(all_terminals, axis=0),
        'masks': np.concatenate(all_masks, axis=0),
    }
    
    # Print summary
    print("\n" + "="*60)
    print("MERGED DATASET SUMMARY")
    print("="*60)
    print(f"Total trajectories: {len(all_observations)}")
    print(f"Total timesteps: {merged['observations'].shape[0]}")
    print(f"Observation dimension: {merged['observations'].shape[1]}")
    print(f"Action dimension: {merged['actions'].shape[1]}")
    print(f"\nShapes:")
    for key, arr in merged.items():
        print(f"  {key}: {arr.shape} ({arr.dtype})")
    
    # Save the merged dataset
    if output_path is None:
        output_path = input_dir / "merged_dataset.npz"
    
    print(f"\nSaving merged dataset: {output_path}")
    np.savez_compressed(output_path, **merged)
    print("Done!")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge smoothed trajectory files into a single dataset"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing smoothed trajectory files (default: data/unsmoothed_data)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for merged dataset (default: merged_dataset.npz in input_dir)",
    )
    parser.add_argument(
        "--no-smoothed",
        action="store_true",
        help="Use original (non-smoothed) ArUco data instead of smoothed",
    )
    
    args = parser.parse_args()
    
    if args.input_dir is None:
        args.input_dir = Path(__file__).parent / "data" / "unsmoothed_data"
    
    if not args.input_dir.exists():
        print(f"Error: Directory not found: {args.input_dir}")
        return
    
    merge_smoothed_trajectories(
        args.input_dir,
        args.output,
        use_smoothed=not args.no_smoothed,
    )


if __name__ == "__main__":
    main()

