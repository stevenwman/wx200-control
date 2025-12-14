#!/usr/bin/env python3
"""
ArUco marker detection using GStreamer for camera capture.
Achieves 30 FPS (vs 15 FPS with OpenCV).

Modular design:
- camera/: Camera capture modules (GStreamer)
- aruco/: ArUco detection and pose estimation modules
"""

import cv2
import numpy as np
import time
from collections import deque

from camera import GStreamerCamera, is_gstreamer_available
from aruco import ArucoDetector, PoseFilter

# ============================================================================
# CONFIGURATION - Tune these parameters to improve tracking stability
# ============================================================================

# Marker properties
MARKER_SIZE = 0.015  # meters (15mm) - Must match physical marker size

# Display settings
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Profiling
ENABLE_PROFILING = True
PROFILE_HISTORY_SIZE = 100

# ============================================================================
# POSE FILTERING PARAMETERS (aruco/pose_filter.py)
# ============================================================================
# These control how poses are smoothed and filtered to reduce jitter

POSE_FILTER_ALPHA = 0.7  # Smoothing factor (0-1)
                         # Higher = less smoothing, more responsive (default: 0.7)
                         # Lower = more smoothing, less responsive (try: 0.5-0.6 for stability)

POSE_FILTER_MAX_TRANSLATION_JUMP = 0.1  # meters
                                        # Maximum allowed translation change per frame
                                        # Lower = reject more outliers (default: 0.1m = 10cm)
                                        # Higher = allow faster movements (try: 0.05m for tighter filtering)

POSE_FILTER_MAX_ROTATION_JUMP = 0.5  # radians (~28.6 degrees)
                                    # Maximum allowed rotation change per frame
                                    # Lower = reject more rotation outliers (default: 0.5 rad)
                                    # Higher = allow faster rotations (try: 0.3 rad for tighter filtering)

POSE_FILTER_MAX_CONSECUTIVE_FAILURES = 3  # Frames
                                          # After N consecutive rejected poses, accept the next one
                                          # Lower = stricter (default: 3)
                                          # Higher = more forgiving (try: 5 for noisy environments)

# ============================================================================
# ROI TRACKING PARAMETERS (aruco/detector.py)
# ============================================================================
# These control region-of-interest tracking for faster detection

ROI_EXPAND = 100  # pixels
                 # How much to expand ROI around last known position
                 # Larger = search bigger area, slower but more forgiving (default: 100px)
                 # Smaller = faster detection, but marker must stay closer (try: 150px for stability)

ROI_SEARCH_FRAMES = 5  # frames
                      # Search ROI for N frames before falling back to full-frame search
                      # Higher = more ROI searches, faster but may miss fast movements (default: 5)
                      # Lower = fall back to full-frame sooner, slower but more reliable (try: 3)

# ============================================================================
# ARUCO DETECTION PARAMETERS (aruco/detector.py)
# ============================================================================
# These control the ArUco detector sensitivity and accuracy

ARUCO_DICT_TYPE = None  # None = use default (DICT_4X4_50)
                       # Options: cv2.aruco.DICT_4X4_50, DICT_5X5_50, DICT_6X6_50, etc.
                       # Larger dictionaries = more unique markers but slower detection

ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MIN = 3  # Minimum window size for adaptive thresholding
                                        # Lower = more sensitive to small markers (default: 3)
                                        # Higher = less noise but may miss small markers

ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MAX = 23  # Maximum window size for adaptive thresholding
                                         # Higher = better for large markers (default: 23)
                                         # Lower = faster but may miss large markers

ARUCO_ADAPTIVE_THRESH_WIN_SIZE_STEP = 10  # Step size for window size search
                                          # Lower = more thorough but slower (default: 10)
                                          # Higher = faster but may miss some markers

ARUCO_POLYGONAL_APPROX_ACCURACY_RATE = 0.05  # Accuracy for polygon approximation
                                             # Lower = more accurate but slower (default: 0.05)
                                             # Higher = faster but less accurate

ARUCO_PERSPECTIVE_REMOVE_PIXEL_PER_CELL = 4  # Pixels per cell for perspective removal
                                            # Lower = more accurate but slower (default: 4)
                                            # Higher = faster but less accurate


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


def main():
    camera_id = 1  # /dev/video1
    marker_size_m = MARKER_SIZE
    dictionary_type = ARUCO_DICT_TYPE if ARUCO_DICT_TYPE is not None else cv2.aruco.DICT_4X4_50
    visualize = True
    target_marker_ids = [0, 1]  # Track these marker IDs
    
    if not is_gstreamer_available():
        print("ERROR: GStreamer not available!")
        print("Install with: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good")
        return
    
    print("="*60)
    print("ArUco Detection with GStreamer (30 FPS)")
    print("="*60)
    print("Press Ctrl+C to stop or 'q' to quit")
    print()
    
    # Setup camera
    camera = GStreamerCamera(device=f'/dev/video{camera_id}', width=1920, height=1080, fps=30)
    
    try:
        camera.start()
        frame_w, frame_h = 1920, 1080
        cam_matrix, dist_coeffs = get_approx_camera_matrix(frame_w, frame_h)
        
        # Setup ArUco detector with configurable parameters
        detector = ArucoDetector(
            dictionary_type=dictionary_type, 
            roi_expand=ROI_EXPAND, 
            roi_search_frames=ROI_SEARCH_FRAMES,
            adaptive_thresh_win_size_min=ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MIN,
            adaptive_thresh_win_size_max=ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MAX,
            adaptive_thresh_win_size_step=ARUCO_ADAPTIVE_THRESH_WIN_SIZE_STEP,
            polygonal_approx_accuracy_rate=ARUCO_POLYGONAL_APPROX_ACCURACY_RATE,
            perspective_remove_pixel_per_cell=ARUCO_PERSPECTIVE_REMOVE_PIXEL_PER_CELL
        )
        
        # Object points for pose estimation
        half = marker_size_m / 2.0
        obj_points = np.array([[-half, half, 0], [half, half, 0],
                               [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        # Pose filters for each marker with configurable parameters
        pose_filters = {
            marker_id: PoseFilter(
                alpha=POSE_FILTER_ALPHA,
                max_translation_jump=POSE_FILTER_MAX_TRANSLATION_JUMP,
                max_rotation_jump=POSE_FILTER_MAX_ROTATION_JUMP,
                max_consecutive_failures=POSE_FILTER_MAX_CONSECUTIVE_FAILURES
            )
            for marker_id in target_marker_ids
        }
        
        # Profiling
        profile_times = {
            'read': deque(maxlen=PROFILE_HISTORY_SIZE),
            'convert': deque(maxlen=PROFILE_HISTORY_SIZE),
            'detect': deque(maxlen=PROFILE_HISTORY_SIZE),
            'pose': deque(maxlen=PROFILE_HISTORY_SIZE),
            'draw': deque(maxlen=PROFILE_HISTORY_SIZE),
            'total': deque(maxlen=PROFILE_HISTORY_SIZE)
        }
        frame_count = 0
        
        print("Starting detection loop...")
        
        while True:
            frame_start = time.time()
            
            # Read frame
            t0 = time.time()
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            if ENABLE_PROFILING:
                profile_times['read'].append(time.time() - t0)
            
            # Convert to grayscale
            t0 = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ENABLE_PROFILING:
                profile_times['convert'].append(time.time() - t0)
            
            # Detect ArUco markers (with ROI tracking)
            t0 = time.time()
            corners, ids = detector.detect(gray, frame_w, frame_h, target_marker_ids)
            if ENABLE_PROFILING:
                profile_times['detect'].append(time.time() - t0)
            
            # Process poses
            t0 = time.time()
            if ids is not None:
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    if marker_id in target_marker_ids:
                        _, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], 
                                                     cam_matrix, dist_coeffs, 
                                                     flags=cv2.SOLVEPNP_IPPE_SQUARE)
                        
                        # Filter pose (using marker-specific filter)
                        rvec, tvec = pose_filters[marker_id].filter_pose(rvec, tvec)
                        
                        # Update ROI position based on detected marker
                        detector.update_position(marker_id, corners[i])
                        
                        # Print pose
                        print(f"Processing ID: {marker_id} | X: {tvec[0][0]:.4f}, Y: {tvec[1][0]:.4f}, Z: {tvec[2][0]:.4f}")
                        
                        # Draw on frame
                        if visualize:
                            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_size_m)
                            
                            # Draw ROI if exists
                            roi = detector.get_roi(marker_id)
                            if roi is not None:
                                x1, y1, x2, y2 = roi
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
            if ENABLE_PROFILING:
                profile_times['pose'].append(time.time() - t0)
            
            # Resize for display
            t0 = time.time()
            display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            if ENABLE_PROFILING:
                profile_times['draw'].append(time.time() - t0)
            
            # Show frame
            cv2.imshow('ArUco Detection (GStreamer)', display_frame)
            
            if ENABLE_PROFILING:
                profile_times['total'].append(time.time() - frame_start)
                frame_count += 1
                
                # Print profiling every 100 frames
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
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
