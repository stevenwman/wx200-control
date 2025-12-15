#!/usr/bin/env python3
"""
Smooth ArUco trajectory data by interpolating missing frames.

For each timestep where ArUco markers are not visible, interpolates the pose
between the previous and next valid frames using:
- Linear interpolation for position (x, y, z)
- SLERP (spherical linear interpolation) for quaternions (qw, qx, qy, qz)

Saves smoothed versions as new keys: smoothed_aruco_ee_in_world, etc.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp


def slerp_quaternion(q1, q2, t):
    """
    Spherical linear interpolation between two quaternions.
    
    Args:
        q1: Quaternion [w, x, y, z] (source)
        q2: Quaternion [w, x, y, z] (target)
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion [w, x, y, z]
    """
    # Convert to scipy format [x, y, z, w]
    q1_xyzw = np.array([q1[1], q1[2], q1[3], q1[0]])
    q2_xyzw = np.array([q2[1], q2[2], q2[3], q2[0]])
    
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)
    
    # SLERP using Slerp class
    key_times = [0, 1]
    key_rots = R.from_quat(np.array([q1_xyzw, q2_xyzw]))
    slerp = Slerp(key_times, key_rots)
    r_interp = slerp(t)
    
    # Convert back to [w, x, y, z]
    q_interp_xyzw = r_interp.as_quat()
    return np.array([q_interp_xyzw[3], q_interp_xyzw[0], q_interp_xyzw[1], q_interp_xyzw[2]])


def interpolate_pose(pose_before, pose_after, t):
    """
    Interpolate a 7D pose [x, y, z, qw, qx, qy, qz] between two poses.
    
    Args:
        pose_before: 7D pose before missing frame
        pose_after: 7D pose after missing frame
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated 7D pose
    """
    pos_before = pose_before[:3]
    pos_after = pose_after[:3]
    quat_before = pose_before[3:7]
    quat_after = pose_after[3:7]
    
    # Linear interpolation for position
    pos_interp = pos_before + t * (pos_after - pos_before)
    
    # SLERP for quaternion
    quat_interp = slerp_quaternion(quat_before, quat_after, t)
    
    return np.concatenate([pos_interp, quat_interp])


def smooth_aruco_poses(poses, visibility_mask):
    """
    Smooth ArUco poses by interpolating missing frames.
    
    Args:
        poses: (N, 7) array of poses [x, y, z, qw, qx, qy, qz]
        visibility_mask: (N,) boolean array, True where visible
    
    Returns:
        smoothed_poses: (N, 7) array with interpolated values for missing frames
    """
    N = len(poses)
    smoothed = poses.copy()
    
    # Find missing frames
    missing = ~visibility_mask
    
    if not np.any(missing):
        return smoothed  # No missing frames
    
    # Find contiguous missing regions
    i = 0
    while i < N:
        if missing[i]:
            # Find start and end of missing region
            start_missing = i
            while i < N and missing[i]:
                i += 1
            end_missing = i - 1
            
            # Find valid frames before and after
            before_idx = start_missing - 1
            after_idx = end_missing + 1
            
            # Handle edge cases
            if before_idx < 0:
                # No valid frame before, use first valid frame after
                if after_idx < N:
                    smoothed[start_missing:end_missing+1] = poses[after_idx]
                continue
            
            if after_idx >= N:
                # No valid frame after, use last valid frame before
                smoothed[start_missing:end_missing+1] = poses[before_idx]
                continue
            
            # Interpolate between before and after
            pose_before = poses[before_idx]
            pose_after = poses[after_idx]
            num_missing = end_missing - start_missing + 1
            
            for j, missing_idx in enumerate(range(start_missing, end_missing + 1)):
                t = (j + 1) / (num_missing + 1)  # t in (0, 1)
                smoothed[missing_idx] = interpolate_pose(pose_before, pose_after, t)
        else:
            i += 1
    
    return smoothed


def smooth_trajectory_file(input_path: Path, output_path: Path = None):
    """
    Load trajectory, smooth ArUco poses, and save with new smoothed_* keys.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output .npz file (default: adds '_smoothed' before .npz)
    """
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_smoothed.npz"
    
    print(f"Loading: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    # Extract ArUco data
    vis = data["aruco_visibility"]  # (N, 3) [world, object, gripper]
    
    aruco_keys = [
        "aruco_ee_in_world",
        "aruco_object_in_world",
        "aruco_ee_in_object",
        "aruco_object_in_ee",
    ]
    
    # Determine visibility for each pose type
    # EE in world: need world AND gripper visible
    ee_world_vis = (vis[:, 0] > 0.5) & (vis[:, 2] > 0.5)
    # Object in world: need world AND object visible
    obj_world_vis = (vis[:, 0] > 0.5) & (vis[:, 1] > 0.5)
    # EE in object: need object AND gripper visible
    ee_obj_vis = (vis[:, 1] > 0.5) & (vis[:, 2] > 0.5)
    # Object in EE: need object AND gripper visible (same as above)
    obj_ee_vis = (vis[:, 1] > 0.5) & (vis[:, 2] > 0.5)
    
    visibility_masks = {
        "aruco_ee_in_world": ee_world_vis,
        "aruco_object_in_world": obj_world_vis,
        "aruco_ee_in_object": ee_obj_vis,
        "aruco_object_in_ee": obj_ee_vis,
    }
    
    # Create new dict with all original data + smoothed ArUco
    new_data = {k: data[k] for k in data.keys()}
    
    print("\nSmoothing ArUco poses...")
    for key in aruco_keys:
        if key in data:
            poses = data[key]
            mask = visibility_masks[key]
            smoothed = smooth_aruco_poses(poses, mask)
            new_key = f"smoothed_{key}"
            new_data[new_key] = smoothed
            
            num_missing = np.sum(~mask)
            num_total = len(mask)
            print(f"  {key}: {num_missing}/{num_total} frames interpolated")
    
    # Update metadata
    if "metadata" in new_data:
        metadata = new_data["metadata"].item().copy()
        metadata["smoothing_note"] = "ArUco poses interpolated for missing frames using linear (position) and SLERP (quaternion)"
        new_data["metadata"] = metadata
    
    print(f"\nSaving smoothed trajectory: {output_path}")
    np.savez_compressed(output_path, **new_data)
    print("Done!")
    
    return output_path


def main():
    import sys
    
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Default: most recent trajectory
        data_dir = Path(__file__).parent / "data"
        traj_files = sorted(data_dir.glob("trajectory_*.npz"))
        if not traj_files:
            print("No trajectory files found in data/")
            return
        input_path = traj_files[-1]
        print(f"No file specified, using most recent: {input_path}")
    
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return
    
    output_path = input_path.parent / f"{input_path.stem}_smoothed.npz"
    smooth_trajectory_file(input_path, output_path)


if __name__ == "__main__":
    main()

