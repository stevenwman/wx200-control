"""
Compare EE poses from IK solver vs encoder-based FK.

This script loads a trajectory file and compares:
- ee_pose_target: Target/commanded EE pose from IK solver (what we're trying to achieve)
- ee_pose_encoder: Actual EE pose from encoder-based FK (ground truth from hardware)

Visualizes the difference in 3D using plotly.
"""
import numpy as np
import argparse
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R


def load_trajectory(trajectory_path):
    """Load trajectory from NPZ file."""
    data = np.load(trajectory_path, allow_pickle=True)
    
    # Get metadata
    metadata = data.get('metadata', None)
    if metadata is not None and isinstance(metadata, np.ndarray):
        metadata = metadata.item()
    
    # Extract trajectory arrays
    timestamps = data['timestamps']
    ee_poses_target = data['ee_poses_target']  # Target/commanded pose from IK solver
    ee_poses_encoder = data.get('ee_pose_encoder', None)  # Ground truth from encoder FK
    
    if ee_poses_encoder is None:
        raise ValueError("Trajectory does not contain 'ee_pose_encoder'. "
                        "Make sure it was recorded with the encoder polling version.")
    
    # Ensure shapes match
    if len(ee_poses_target) != len(ee_poses_encoder):
        min_len = min(len(ee_poses_target), len(ee_poses_encoder))
        print(f"Warning: Mismatched lengths ({len(ee_poses_target)} vs {len(ee_poses_encoder)}), using first {min_len} samples")
        ee_poses_target = ee_poses_target[:min_len]
        ee_poses_encoder = ee_poses_encoder[:min_len]
        timestamps = timestamps[:min_len]
    
    return {
        'timestamps': timestamps,
        'ee_poses_target': ee_poses_target,
        'ee_poses_encoder': ee_poses_encoder,
        'metadata': metadata
    }


def compute_pose_differences(ee_poses_target, ee_poses_encoder):
    """
    Compute differences between two sets of EE poses.
    
    Args:
        ee_poses_target: (N, 7) array [x, y, z, qw, qx, qy, qz] - Target/commanded pose from IK solver
        ee_poses_encoder: (N, 7) array [x, y, z, qw, qx, qy, qz] - Ground truth from encoders
    
    Returns:
        dict with position errors, orientation errors, etc.
    """
    N = len(ee_poses_target)
    
    # Position differences
    positions_target = ee_poses_target[:, :3]
    positions_encoder = ee_poses_encoder[:, :3]
    position_diff = positions_encoder - positions_target
    position_errors = np.linalg.norm(position_diff, axis=1)
    
    # Orientation differences (quaternion)
    quats_target = ee_poses_target[:, 3:]  # [qw, qx, qy, qz]
    quats_encoder = ee_poses_encoder[:, 3:]
    
    # Convert to scipy Rotation format (wxyz -> xyzw)
    quats_target_xyzw = np.column_stack([quats_target[:, 1:4], quats_target[:, 0]])
    quats_encoder_xyzw = np.column_stack([quats_encoder[:, 1:4], quats_encoder[:, 0]])
    
    rots_target = R.from_quat(quats_target_xyzw)
    rots_encoder = R.from_quat(quats_encoder_xyzw)
    
    # Compute relative rotation (encoder relative to target)
    rots_diff = rots_encoder * rots_target.inv()
    
    # Convert to axis-angle for error magnitude
    axis_angles = rots_diff.as_rotvec()
    orientation_errors = np.linalg.norm(axis_angles, axis=1)  # Angle in radians
    
    # Also compute Euler angles for visualization
    euler_diff = rots_diff.as_euler('xyz', degrees=True)
    
    return {
        'position_diff': position_diff,  # (N, 3) xyz differences
        'position_errors': position_errors,  # (N,) Euclidean distance
        'orientation_diff_axis_angle': axis_angles,  # (N, 3) axis-angle representation
        'orientation_errors': orientation_errors,  # (N,) angle in radians
        'orientation_errors_deg': np.degrees(orientation_errors),  # (N,) angle in degrees
        'euler_diff': euler_diff,  # (N, 3) Euler angles in degrees
    }


def create_3d_visualization(trajectory_data, differences):
    """Create interactive 3D visualization using plotly."""
    
    timestamps = trajectory_data['timestamps']
    ee_poses_target = trajectory_data['ee_poses_target']
    ee_poses_encoder = trajectory_data['ee_poses_encoder']
    
    positions_target = ee_poses_target[:, :3]
    positions_encoder = ee_poses_encoder[:, :3]
    
    # Create subplots: 3D trajectory + error plots
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scatter3d", "colspan": 2}, None],
               [{"type": "scatter"}, {"type": "scatter"}]],
        subplot_titles=("3D Trajectory Comparison", 
                        "Position Error Over Time", 
                        "Orientation Error Over Time"),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 3D trajectory plot
    # Target/commanded trajectory (ee_pose_target from IK solver)
    fig.add_trace(
        go.Scatter3d(
            x=positions_target[:, 0],
            y=positions_target[:, 1],
            z=positions_target[:, 2],
            mode='lines+markers',
            name='Target (ee_pose_target)',
            line=dict(color='blue', width=4),
            marker=dict(size=3, color='blue'),
            hovertemplate='<b>Target Pose</b><br>' +
                         'X: %{x:.4f}<br>' +
                         'Y: %{y:.4f}<br>' +
                         'Z: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Encoder-based trajectory (ee_pose_encoder - ground truth from encoders)
    fig.add_trace(
        go.Scatter3d(
            x=positions_encoder[:, 0],
            y=positions_encoder[:, 1],
            z=positions_encoder[:, 2],
            mode='lines+markers',
            name='Encoder (ee_pose_encoder)',
            line=dict(color='red', width=4),
            marker=dict(size=3, color='red'),
            hovertemplate='<b>Encoder Pose</b><br>' +
                         'X: %{x:.4f}<br>' +
                         'Y: %{y:.4f}<br>' +
                         'Z: %{z:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Draw lines connecting corresponding points to show differences
    for i in range(0, len(positions_target), max(1, len(positions_target) // 50)):  # Sample every Nth point
        fig.add_trace(
            go.Scatter3d(
                x=[positions_target[i, 0], positions_encoder[i, 0]],
                y=[positions_target[i, 1], positions_encoder[i, 1]],
                z=[positions_target[i, 2], positions_encoder[i, 2]],
                mode='lines',
                name='Difference' if i == 0 else '',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=(i == 0),
                hovertemplate=f'<b>Difference at t={timestamps[i]:.2f}s</b><br>' +
                             f'Distance: {differences["position_errors"][i]:.4f}m<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Position error over time
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=differences['position_errors'] * 1000,  # Convert to mm
            mode='lines',
            name='Position Error',
            line=dict(color='blue', width=2),
            hovertemplate='Time: %{x:.2f}s<br>' +
                         'Error: %{y:.2f}mm<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Orientation error over time
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=differences['orientation_errors_deg'],
            mode='lines',
            name='Orientation Error',
            line=dict(color='red', width=2),
            hovertemplate='Time: %{x:.2f}s<br>' +
                         'Error: %{y:.2f}°<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title="EE Pose Comparison: IK Solver vs Encoder-Based FK",
        height=900,
        showlegend=True,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode='data'
        )
    )
    
    # Update axes labels for 2D plots
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Position Error (mm)", row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Orientation Error (degrees)", row=2, col=2)
    
    return fig


def print_statistics(differences, timestamps):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("EE POSE COMPARISON STATISTICS")
    print("="*60)
    
    # Position statistics
    pos_errors = differences['position_errors']
    print(f"\nPosition Errors (IK Solver vs Encoder FK):")
    print(f"  Mean: {np.mean(pos_errors)*1000:.3f} mm")
    print(f"  Std:  {np.std(pos_errors)*1000:.3f} mm")
    print(f"  Min:  {np.min(pos_errors)*1000:.3f} mm")
    print(f"  Max:  {np.max(pos_errors)*1000:.3f} mm")
    print(f"  RMS:  {np.sqrt(np.mean(pos_errors**2))*1000:.3f} mm")
    
    # Orientation statistics
    ori_errors_deg = differences['orientation_errors_deg']
    print(f"\nOrientation Errors (IK Solver vs Encoder FK):")
    print(f"  Mean: {np.mean(ori_errors_deg):.3f} degrees")
    print(f"  Std:  {np.std(ori_errors_deg):.3f} degrees")
    print(f"  Min:  {np.min(ori_errors_deg):.3f} degrees")
    print(f"  Max:  {np.max(ori_errors_deg):.3f} degrees")
    print(f"  RMS:  {np.sqrt(np.mean(ori_errors_deg**2)):.3f} degrees")
    
    # Component-wise position differences
    pos_diff = differences['position_diff']
    print(f"\nPosition Differences (Component-wise):")
    print(f"  X: mean={np.mean(pos_diff[:, 0])*1000:.3f} mm, std={np.std(pos_diff[:, 0])*1000:.3f} mm")
    print(f"  Y: mean={np.mean(pos_diff[:, 1])*1000:.3f} mm, std={np.std(pos_diff[:, 1])*1000:.3f} mm")
    print(f"  Z: mean={np.mean(pos_diff[:, 2])*1000:.3f} mm, std={np.std(pos_diff[:, 2])*1000:.3f} mm")
    
    # Time statistics
    duration = timestamps[-1] if len(timestamps) > 0 else 0
    print(f"\nTrajectory Duration: {duration:.2f} seconds")
    print(f"Number of samples: {len(timestamps)}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare EE poses from IK solver vs encoder-based FK',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_ee_poses.py data/trajectory_20251217_223437.npz
  python compare_ee_poses.py data/trajectory.npz --output comparison.html
        """
    )
    parser.add_argument('trajectory', type=str, help='Path to trajectory NPZ file')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output HTML file for plotly visualization (default: trajectory_name_comparison.html)')
    parser.add_argument('--no-show', action='store_true', 
                       help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Load trajectory
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        return
    
    print(f"Loading trajectory: {trajectory_path}")
    try:
        trajectory_data = load_trajectory(trajectory_path)
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        return
    
    # Compute differences
    print("Computing pose differences...")
    differences = compute_pose_differences(
        trajectory_data['ee_poses_target'],
        trajectory_data['ee_poses_encoder']
    )
    
    # Print statistics
    print_statistics(differences, trajectory_data['timestamps'])
    
    # Create visualization
    print("Creating 3D visualization...")
    fig = create_3d_visualization(trajectory_data, differences)
    
    # Save or show
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = trajectory_path.parent / f"{trajectory_path.stem}_comparison.html"
    
    print(f"Saving visualization to: {output_path}")
    fig.write_html(str(output_path))
    
    if not args.no_show:
        print("Opening visualization in browser...")
        fig.show()
    else:
        print(f"Visualization saved. Open {output_path} in a browser to view.")


if __name__ == "__main__":
    main()
