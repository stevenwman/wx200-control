#!/usr/bin/env python3
"""
ArUco marker detection using GStreamer for camera capture.
"""

import cv2
import numpy as np
import time
import argparse
from collections import deque
import matplotlib.pyplot as plt

from camera import GStreamerCamera, is_gstreamer_available

# Configuration
MARKER_SIZE = 0.015  # meters (15mm)
TAG_0_1_OFFSET = 0.061  # meters (61mm) - distance between Tag 0 and Tag 1 centers (along X)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
ENABLE_PROFILING = True
PROFILE_HISTORY_SIZE = 100

ARUCO_DICT_TYPE = None  # None = use default (DICT_4X4_50)


def get_approx_camera_matrix(width, height):
    """Get approximate camera matrix."""
    focal_length = width
    center_x = width / 2.0
    center_y = height / 2.0
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs


def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (ZYX convention)."""
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    
    return np.array([x, y, z])


# Global state for temporal consistency
_prev_solution_choice = {}  # marker_id -> (rvec, tvec) from previous frame
_prev_solution_history = {}  # marker_id -> list of recent (rvec, tvec) for smoother consistency


def solve_pnp_robust(obj_points, img_points, camera_matrix, dist_coeffs, marker_id=None, corners=None):
    """
    Solve PnP robustly by getting both solutions and choosing the correct one.
    Uses temporal consistency if marker_id is provided to prevent flipping.
    """
    # Get both solutions from IPPE_SQUARE
    # solvePnPGeneric returns: success, rvecs, tvecs, reprojectionErrors
    result = cv2.solvePnPGeneric(
        obj_points, img_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    # Handle different OpenCV versions
    if len(result) == 4:
        success, rvecs, tvecs, reprojection_errors = result
    elif len(result) == 3:
        success, rvecs, tvecs = result
        reprojection_errors = None
    else:
        # Fallback to standard solvePnP
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if success:
            return rvec, tvec
        return None, None
    
    if not success or len(rvecs) < 2:
        # Fallback to standard solvePnP if generic doesn't work
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if success:
            return rvec, tvec
        return None, None
    
    # Choose solution where marker's Z-axis points toward camera
    # Strategy: Use multiple criteria in order of reliability
    
    # 1. Reprojection error (most reliable when available)
    if reprojection_errors is not None and len(reprojection_errors) == len(rvecs):
        # Check if reprojection errors are significantly different
        errors = np.array(reprojection_errors)
        if np.max(errors) - np.min(errors) > 0.1:  # Significant difference
            best_idx = np.argmin(errors)
            chosen_rvec, chosen_tvec = rvecs[best_idx], tvecs[best_idx]
        else:
            # Errors too similar, use geometric constraints
            chosen_rvec = None
            chosen_tvec = None
    else:
        chosen_rvec = None
        chosen_tvec = None
    
    # 2. Geometric constraints (if reprojection error not reliable)
    if chosen_rvec is None:
        # Use geometric constraints
        # For ArUco markers viewed from front, the marker normal (Z-axis) should point toward camera
        # In camera coordinates: camera Z points forward, so marker normal pointing toward camera has negative Z
        
        # Primary criterion: marker normal Z-component should be negative
        chosen_rvec = None
        chosen_tvec = None
        
        for rvec, tvec in zip(rvecs, tvecs):
            if tvec[2] <= 0:  # Must be in front of camera
                continue
            
            R, _ = cv2.Rodrigues(rvec)
            marker_normal_z = R[2, 2]  # Z-component of marker's Z-axis (normal)
            
            # For marker viewed from front, normal should point toward camera (negative Z)
            if marker_normal_z < 0:
                chosen_rvec, chosen_tvec = rvec, tvec
                break
        
        # Fallback: if no solution has negative normal Z, use the one with better alignment
        if chosen_rvec is None:
            best_alignment = -np.inf
            for rvec, tvec in zip(rvecs, tvecs):
                if tvec[2] <= 0:
                    continue
                
                R, _ = cv2.Rodrigues(rvec)
                marker_normal = R[:, 2]
                tvec_normalized = tvec.flatten() / np.linalg.norm(tvec)
                alignment = np.dot(marker_normal, tvec_normalized)
                
                if alignment > best_alignment:
                    best_alignment = alignment
                    chosen_rvec, chosen_tvec = rvec, tvec
        
        # Last resort: use first solution
        if chosen_rvec is None:
            chosen_rvec, chosen_tvec = rvecs[0], tvecs[0]
    
    # Temporal consistency: use history to maintain smooth pose
    # This is the most effective way to handle mirror ambiguity for single markers
    if marker_id is not None:
        if marker_id not in _prev_solution_history:
            _prev_solution_history[marker_id] = []
        
        history = _prev_solution_history[marker_id]
        
        if len(history) > 0:
            # Find solution closest to recent history (weighted average of last few frames)
            min_total_diff = np.inf
            best_rvec = chosen_rvec
            best_tvec = chosen_tvec
            
            for rvec, tvec in zip(rvecs, tvecs):
                if tvec[2] <= 0:
                    continue
                
                R, _ = cv2.Rodrigues(rvec)
                total_diff = 0.0
                
                # Compare against recent history (last 3 frames, weighted)
                weights = [0.5, 0.3, 0.2]  # Most recent has highest weight
                for i, (prev_rvec, prev_tvec) in enumerate(history[-3:]):
                    weight = weights[min(i, len(weights)-1)]
                    prev_R, _ = cv2.Rodrigues(prev_rvec)
                    
                    # Rotation difference (angle between rotation axes)
                    R_diff = np.linalg.norm(R - prev_R, 'fro')
                    # Translation difference
                    t_diff = np.linalg.norm(tvec - prev_tvec)
                    total_diff += weight * (R_diff + t_diff * 0.1)
                
                if total_diff < min_total_diff:
                    min_total_diff = total_diff
                    best_rvec = rvec
                    best_tvec = tvec
            
            # Use temporal consistency - prioritize smoothness over geometric constraints
            # This is the most effective way to handle mirror ambiguity for single markers
            if min_total_diff < 10.0:  # Very lenient - prefer temporal consistency
                chosen_rvec, chosen_tvec = best_rvec, best_tvec
            # If temporal difference is too large, still use it (might be fast movement)
            # The history will smooth it out
        
        # Update history (keep last 5 frames)
        history.append((chosen_rvec.copy(), chosen_tvec.copy()))
        if len(history) > 5:
            history.pop(0)
        
        _prev_solution_choice[marker_id] = (chosen_rvec.copy(), chosen_tvec.copy())
    
    return chosen_rvec, chosen_tvec


def main():
    parser = argparse.ArgumentParser(description='ArUco marker detection with GStreamer')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Enable camera preview visualization (default: True)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false',
                        help='Disable camera preview visualization')
    parser.add_argument('--camera-id', type=int, default=1,
                        help='Camera device ID (default: 1)')
    parser.add_argument('--marker-ids', type=int, nargs='+', default=[0, 1, 2],
                        help='Marker IDs to track (default: 0 1 2)')
    args = parser.parse_args()
    
    camera_id = args.camera_id
    marker_size_m = MARKER_SIZE
    dictionary_type = ARUCO_DICT_TYPE if ARUCO_DICT_TYPE is not None else cv2.aruco.DICT_4X4_50
    visualize = args.visualize
    target_marker_ids = args.marker_ids
    
    if not is_gstreamer_available():
        print("ERROR: GStreamer not available!")
        print("Install with: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good")
        return
    
    print("="*60)
    print("ArUco Detection with GStreamer (30 FPS)")
    print("="*60)
    print("Press Ctrl+C to stop or 'q' to quit")
    print()
    
    camera = GStreamerCamera(device=f'/dev/video{camera_id}', width=1920, height=1080, fps=30)
    
    try:
        camera.start()
        frame_w, frame_h = 1920, 1080
        cam_matrix, dist_coeffs = get_approx_camera_matrix(frame_w, frame_h)
        
        # Single marker object points (centered at 0,0,0)
        half = MARKER_SIZE / 2.0
        obj_points_single = np.array([[-half, half, 0], [half, half, 0],
                                      [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        # Combined "Hand" object points (Tag 0 at origin, Tag 1 offset by TAG_0_1_OFFSET)
        # Tag 0 corners
        hand_obj_points_0 = obj_points_single.copy()
        # Tag 1 corners (shifted along X)
        hand_obj_points_1 = obj_points_single.copy()
        hand_obj_points_1[:, 0] += TAG_0_1_OFFSET
        
        print("Using OpenCV ArUco detection with robust pose estimation")
        print("Configuration:")
        print(f"  - Marker Size: {MARKER_SIZE*1000:.1f} mm")
        print(f"  - Tag 0-1 Offset: {TAG_0_1_OFFSET*1000:.1f} mm")
        print(f"Visualization: {'ON' if visualize else 'OFF'}")
        print("Recording orientation trajectory...")
        
        # Trajectory recording
        trajectory = {marker_id: [] for marker_id in target_marker_ids}
        start_time = time.time()
        
        profile_times = {
            'read': deque(maxlen=PROFILE_HISTORY_SIZE),
            'convert': deque(maxlen=PROFILE_HISTORY_SIZE),
            'detect': deque(maxlen=PROFILE_HISTORY_SIZE),
            'pose': deque(maxlen=PROFILE_HISTORY_SIZE),
            'draw': deque(maxlen=PROFILE_HISTORY_SIZE),
            'total': deque(maxlen=PROFILE_HISTORY_SIZE)
        }
        frame_count = 0
        fps_window = deque(maxlen=30)
        last_fps_print = time.time()
        
        print("Starting detection loop...")
        print()
        
        while True:
            frame_start = time.time()
            
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            if ENABLE_PROFILING:
                profile_times['read'].append(time.time() - frame_start)
            
            t0 = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ENABLE_PROFILING:
                profile_times['convert'].append(time.time() - t0)
            
            t0 = time.time()
            aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
            params = cv2.aruco.DetectorParameters()
            stock_detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = stock_detector.detectMarkers(gray)
            if ENABLE_PROFILING:
                profile_times['detect'].append(time.time() - t0)
            
            t0 = time.time()
            if ids is not None:
                # Process "Hand" rigid body (Tags 0 and 1)
                hand_img_points = []
                hand_obj_points = []
                
                # Check for Tag 0
                idx0 = np.where(ids == 0)[0]
                if len(idx0) > 0:
                    hand_img_points.append(corners[idx0[0]][0])
                    hand_obj_points.append(hand_obj_points_0)
                
                # Check for Tag 1
                idx1 = np.where(ids == 1)[0]
                if len(idx1) > 0:
                    hand_img_points.append(corners[idx1[0]][0])
                    hand_obj_points.append(hand_obj_points_1)
                
                # Solve for Hand Pose if any tags found
                if len(hand_img_points) > 0:
                    # Flatten points for solvePnP
                    hand_img_points_flat = np.vstack(hand_img_points)
                    hand_obj_points_flat = np.vstack(hand_obj_points)
                    
                    rvec, tvec = solve_pnp_robust(hand_obj_points_flat, hand_img_points_flat, 
                                                 cam_matrix, dist_coeffs, marker_id=0) # Use ID 0 for temporal consistency key
                    
                    if rvec is not None and tvec is not None:
                        rvec = rvec.reshape(3, 1)
                        tvec = tvec.reshape(3, 1)
                        
                        # Record orientation trajectory (store as ID 0 aka Hand)
                        R, _ = cv2.Rodrigues(rvec)
                        euler = rotation_matrix_to_euler(R)
                        timestamp = time.time() - start_time
                        
                        # Store: timestamp, rvec (3), euler (3), R columns (X, Y, Z axes - 9 values)
                        trajectory[0].append({
                            'time': timestamp,
                            'rvec': rvec.flatten().copy(),
                            'euler': euler.copy(),
                            'R_x': R[:, 0].copy(),
                            'R_y': R[:, 1].copy(),
                            'R_z': R[:, 2].copy(),
                        })
                        
                        print(f"HAND (Tags {0 if len(idx0)>0 else ''}{1 if len(idx1)>0 else ''}) | X: {tvec[0][0]:.4f}, Y: {tvec[1][0]:.4f}, Z: {tvec[2][0]:.4f}")
                        
                        if visualize:
                            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)

                # Process "Base" rigid body (Tag 2)
                idx2 = np.where(ids == 2)[0]
                if len(idx2) > 0:
                    rvec, tvec = solve_pnp_robust(obj_points_single, corners[idx2[0]][0], 
                                                 cam_matrix, dist_coeffs, marker_id=2)
                    
                    if rvec is not None and tvec is not None:
                        rvec = rvec.reshape(3, 1)
                        tvec = tvec.reshape(3, 1)
                        print(f"BASE (Tag 2) | X: {tvec[0][0]:.4f}, Y: {tvec[1][0]:.4f}, Z: {tvec[2][0]:.4f}")
                        if visualize:
                            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE)
                
                if visualize:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            if ENABLE_PROFILING:
                profile_times['pose'].append(time.time() - t0)
            
            t0 = time.time()
            if visualize:
                display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                if ENABLE_PROFILING:
                    profile_times['draw'].append(time.time() - t0)
            else:
                if ENABLE_PROFILING:
                    profile_times['draw'].append(0.0)
            
            if visualize:
                cv2.imshow('ArUco Detection (GStreamer)', display_frame)
            
            frame_time = time.time() - frame_start
            fps_window.append(1.0 / frame_time if frame_time > 0 else 0)
            current_fps = np.mean(fps_window) if fps_window else 0
            
            if time.time() - last_fps_print >= 1.0:
                print(f"FPS: {current_fps:.1f}")
                last_fps_print = time.time()
            
            if ENABLE_PROFILING:
                profile_times['total'].append(time.time() - frame_start)
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print("\n" + "="*60)
                    print("PROFILE (ms)")
                    print("="*60)
                    for key, times in profile_times.items():
                        if times:
                            times_ms = np.array(times) * 1000
                            avg = np.mean(times_ms)
                            min_val = np.min(times_ms)
                            max_val = np.max(times_ms)
                            p95 = np.percentile(times_ms, 95)
                            fps = 1000.0 / avg if avg > 0 else 0
                            print(f"  {key:10s}: avg={avg:6.2f}, max={max_val:6.2f}, min={min_val:6.2f}, p95={p95:6.2f}, fps={fps:5.1f}")
                    print("="*60 + "\n")
            
            if visualize:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                time.sleep(0.001)
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        
        # Save and plot trajectory
        if any(len(trajectory[mid]) > 0 for mid in trajectory):
            print("\nSaving and plotting orientation trajectory...")
            
            # Save to file
            filename = f"orientation_trajectory_{int(time.time())}.npz"
            save_data = {}
            for marker_id in trajectory:
                if len(trajectory[marker_id]) > 0:
                    times = [t['time'] for t in trajectory[marker_id]]
                    rvecs = np.array([t['rvec'] for t in trajectory[marker_id]])
                    eulers = np.array([t['euler'] for t in trajectory[marker_id]])
                    R_xs = np.array([t['R_x'] for t in trajectory[marker_id]])
                    R_ys = np.array([t['R_y'] for t in trajectory[marker_id]])
                    R_zs = np.array([t['R_z'] for t in trajectory[marker_id]])
                    
                    label = "HAND" if marker_id == 0 else "BASE"
                    save_data[f'{label}_time'] = times
                    save_data[f'{label}_rvec'] = rvecs
                    save_data[f'{label}_euler'] = eulers
                    save_data[f'{label}_R_x'] = R_xs
                    save_data[f'{label}_R_y'] = R_ys
                    save_data[f'{label}_R_z'] = R_zs
            
            np.savez(filename, **save_data)
            print(f"Saved trajectory to {filename}")
            
            # Plot
            fig, axes = plt.subplots(3, 2, figsize=(12, 10))
            fig.suptitle('Orientation Trajectory Analysis', fontsize=14)
            
            for marker_id in trajectory:
                if len(trajectory[marker_id]) == 0:
                    continue
                
                label_prefix = "HAND (0+1)" if marker_id == 0 else f"BASE ({marker_id})"
                
                times = [t['time'] for t in trajectory[marker_id]]
                eulers = np.array([t['euler'] for t in trajectory[marker_id]])
                R_zs = np.array([t['R_z'] for t in trajectory[marker_id]])
                
                # Plot Euler angles
                axes[0, 0].plot(times, np.degrees(eulers[:, 0]), label=f'{label_prefix} Roll', alpha=0.7)
                axes[0, 1].plot(times, np.degrees(eulers[:, 1]), label=f'{label_prefix} Pitch', alpha=0.7)
                axes[1, 0].plot(times, np.degrees(eulers[:, 2]), label=f'{label_prefix} Yaw', alpha=0.7)
                
                # Plot Z-axis (normal) components - if these flip, we have a problem
                axes[1, 1].plot(times, R_zs[:, 0], label=f'{label_prefix} Z-x', alpha=0.7)
                axes[2, 0].plot(times, R_zs[:, 1], label=f'{label_prefix} Z-y', alpha=0.7)
                axes[2, 1].plot(times, R_zs[:, 2], label=f'{label_prefix} Z-z', alpha=0.7)
            
            axes[0, 0].set_ylabel('Roll (deg)')
            axes[0, 0].set_title('Roll Angle')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            axes[0, 1].set_ylabel('Pitch (deg)')
            axes[0, 1].set_title('Pitch Angle')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            axes[1, 0].set_ylabel('Yaw (deg)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_title('Yaw Angle')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            axes[1, 1].set_ylabel('Z-axis X component')
            axes[1, 1].set_title('Marker Normal (Z-axis) X')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            axes[2, 0].set_ylabel('Z-axis Y component')
            axes[2, 0].set_xlabel('Time (s)')
            axes[2, 0].set_title('Marker Normal (Z-axis) Y')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            axes[2, 1].set_ylabel('Z-axis Z component')
            axes[2, 1].set_xlabel('Time (s)')
            axes[2, 1].set_title('Marker Normal (Z-axis) Z')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
            
            plt.tight_layout()
            plot_filename = f"orientation_trajectory_{int(time.time())}.png"
            plt.savefig(plot_filename, dpi=150)
            print(f"Saved plot to {plot_filename}")
            plt.show()
        
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
