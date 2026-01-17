#!/usr/bin/env python3
"""
Merge trajectory files into a single dataset for training.

Supports both legacy compact_code trajectory_*.npz and compact_gym demo_*.npz
files. If smoothed_* keys are present, they are used when --use-smoothed is on.
Outputs a merged dataset with observations, smoothed_observations, actions_flat,
rewards, terminals, masks, and next_observations.
"""

import numpy as np
from pathlib import Path


def merge_smoothed_trajectories(
    input_dir: Path,
    output_path: Path = None,
    use_smoothed: bool = True,
):
    """
    Merge trajectory files into a single dataset.
    
    Args:
        input_dir: Directory containing demo_*.npz or trajectory_*.npz files
        output_path: Path to save merged dataset (default: merged_dataset.npz in input_dir)
        use_smoothed: If True, use smoothed_aruco_* keys when available
    """
    def _is_valid(f):
        return not f.name.startswith("quarantined_")

    # Auto-detect file types: compact_gym demos take priority
    demo_files = [f for f in sorted(input_dir.glob("demo_*.npz")) if _is_valid(f)]
    smoothed_files = [f for f in sorted(input_dir.glob("*_smoothed.npz")) if _is_valid(f)]
    trajectory_files = [f for f in sorted(input_dir.glob("trajectory_*.npz")) if _is_valid(f)]

    if demo_files:
        traj_files = demo_files
    elif smoothed_files:
        traj_files = smoothed_files
    else:
        traj_files = trajectory_files
    
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
    smoothed_observations = []
    
    def _get_first(data, keys, default=None):
        for key in keys:
            if key in data:
                return data[key]
        return default

    for i, data in enumerate(datasets):
        try:
            # Extract keys with compatibility for compact_gym and compact_code
            action = _get_first(data, ["action", "actions"])
            if action is None:
                raise KeyError("Missing action/actions")

            states = _get_first(data, ["state", "states"])
            if states is None:
                raise KeyError("Missing state/states")

            ee_pose_encoder = _get_first(data, ["ee_pose_encoder"])
            if ee_pose_encoder is None:
                raise KeyError("Missing ee_pose_encoder")

            aruco_object_in_world = _get_first(
                data,
                ["smoothed_aruco_object_in_world", "aruco_object_in_world"] if use_smoothed else ["aruco_object_in_world"],
            )
            if aruco_object_in_world is None:
                raise KeyError("Missing aruco_object_in_world")

            aruco_object_in_world_raw = _get_first(data, ["aruco_object_in_world"], aruco_object_in_world)

            T = aruco_object_in_world.shape[0]
            assert states.shape[0] == T, f"Dataset {i}: state shape mismatch"
            assert ee_pose_encoder.shape[0] == T, f"Dataset {i}: ee_pose_encoder shape mismatch"
            assert action.shape[0] == T, f"Dataset {i}: action shape mismatch"

            # Observations (align with wx200_env_utils_position_targets):
            # [aruco_object_in_world (7), state (6), ee_pose_debug (7)] = 20D
            obs = np.concatenate(
                [aruco_object_in_world_raw, states, ee_pose_encoder],
                axis=1,
            )
            smoothed_obs = np.concatenate(
                [aruco_object_in_world, states, ee_pose_encoder],
                axis=1,
            )
            
            # Next observations (shifted by one)
            next_obs = np.zeros_like(smoothed_obs)
            next_obs[:-1] = smoothed_obs[1:]
            next_obs[-1] = smoothed_obs[-1]  # for last step, next_obs = obs (repeat last)
            
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
            all_actions.append(action)
            all_rewards.append(rewards)
            all_terminals.append(dones)
            all_masks.append(masks)
            # Save smoothed observations alongside raw observations
            smoothed_observations.append(smoothed_obs)
            
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
        'smoothed_observations': np.concatenate(smoothed_observations, axis=0),
        'actions_flat': np.concatenate(all_actions, axis=0),
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
    print(f"Action dimension: {merged['actions_flat'].shape[1]}")
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
        help="Directory containing demo/trajectory files (default: data/gym_demos)",
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
        args.input_dir = Path(__file__).parent / "data" / "gym_demos"
    
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

