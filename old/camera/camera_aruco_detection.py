#!/usr/bin/env python3
"""
Simple OpenCV script for camera feed with ArUco marker detection.
Press 'q' to quit.
"""

import cv2
import numpy as np
import time

# Marker size in meters (15mm = 0.015m)
MARKER_SIZE = 0.015  # meters

# Pose filtering parameters
USE_POSE_FILTERING = True
POSE_FILTER_ALPHA = 0.7  # Exponential smoothing factor (0-1), higher = less smoothing

# Display window size (processing still uses full resolution)
DISPLAY_WIDTH = 1280  # Display window width
DISPLAY_HEIGHT = 720  # Display window height

# Detection resolution (lower = faster detection, but markers must be larger in frame)
# Pose estimation still uses full resolution for accuracy
DETECTION_WIDTH = 480   # Detection resolution width (lower for speed)
DETECTION_HEIGHT = 270  # Detection resolution height

# Frame skipping for detection (detect every N frames, use previous pose in between)
DETECT_EVERY_N_FRAMES = 2  # Detect every 2nd frame (30fps detection, 60fps display)

# Profiling
ENABLE_PROFILING = True  # Set to True to see timing breakdown


def detect_camera(max_cameras=10):
    """
    Detect available camera by trying different indices and backends.
    Returns the first working camera index, or None if none found.
    """
    print("Detecting available cameras...")
    
    # Try auto-detect first (CAP_ANY), then try V4L2 specifically
    # CAP_ANY often works better when cameras are available
    backends = [
        (cv2.CAP_ANY, "Auto-detect"),      # Let OpenCV choose
        (cv2.CAP_V4L2, "V4L2"),            # Linux Video4Linux2
    ]
    
    for backend, backend_name in backends:
        print(f"Trying backend: {backend_name}")
        for i in range(max_cameras):
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    # Try to read a frame to confirm it's working
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        print(f"Found camera at index {i} (backend: {backend_name}) - Resolution: {width}x{height}")
                        cap.release()
                        return i, backend
                    cap.release()
            except Exception as e:
                # Continue trying other indices if one fails
                pass
    
    print("No working cameras detected")
    return None, None


def get_camera_matrix(frame_width, frame_height):
    """
    Get camera matrix and distortion coefficients.
    For accurate results, you should calibrate your camera.
    This provides a basic estimate that may work reasonably well.
    
    Returns:
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: distortion coefficients (can be zeros if unknown)
    """
    # Estimate focal length (rough approximation)
    # A common estimate is: focal_length = image_width (or height) * sensor_size / sensor_width
    # For many webcams, this is close to the image width
    focal_length = frame_width
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    
    camera_matrix = np.array([
        [focal_length, 0, center_x],
        [0, focal_length, center_y],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Distortion coefficients (radial and tangential)
    # These should be calibrated, but zeros can work if distortion is minimal
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    return camera_matrix, dist_coeffs


def draw_axis(frame, camera_matrix, dist_coeffs, rvec, tvec, length=0.01):
    """
    Draw coordinate axes on the marker to visualize pose.
    X: red, Y: green, Z: blue
    """
    # Create points for axis drawing (in marker coordinate system)
    axis_points = np.float32([
        [0, 0, 0],           # Origin
        [length, 0, 0],      # X axis
        [0, length, 0],      # Y axis
        [0, 0, -length]      # Z axis (negative because camera looks down -Z)
    ]).reshape(-1, 3)
    
    # Project 3D points to 2D
    img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = np.int32(img_points).reshape(-1, 2)
    
    # Draw axes
    origin = tuple(img_points[0])
    cv2.line(frame, origin, tuple(img_points[1]), (0, 0, 255), 3)  # X - Red
    cv2.line(frame, origin, tuple(img_points[2]), (0, 255, 0), 3)  # Y - Green
    cv2.line(frame, origin, tuple(img_points[3]), (255, 0, 0), 3)  # Z - Blue


def estimate_marker_pose(corners, marker_size, camera_matrix, dist_coeffs, use_new_api=False):
    """
    Estimate pose of ArUco markers.
    Handles both old API (estimatePoseSingleMarkers) and new API (solvePnP).
    """
    # Check if old API function exists
    has_estimate_pose = hasattr(cv2.aruco, 'estimatePoseSingleMarkers')
    
    if not has_estimate_pose or use_new_api:
        # New API (OpenCV 4.7+): use solvePnP directly
        # Define marker corners in marker coordinate system (centered at origin)
        # Marker coordinate system: origin at center, X-right, Y-up, Z-toward camera
        # Corner order matches ArUco: top-left, top-right, bottom-right, bottom-left
        half_size = marker_size / 2.0
        marker_points = np.array([
            [-half_size, half_size, 0],   # top-left
            [half_size, half_size, 0],    # top-right
            [half_size, -half_size, 0],   # bottom-right
            [-half_size, -half_size, 0]   # bottom-left
        ], dtype=np.float32)
        
        rvecs = []
        tvecs = []
        
        for corner in corners:
            # corner is (1, 4, 2) - reshape to (4, 2) for solvePnP
            image_points = corner[0].astype(np.float32)
            
            # Use solvePnP to estimate pose
            # SOLVEPNP_IPPE_SQUARE is optimized for square markers but can have ambiguity
            # SOLVEPNP_IPPE returns both solutions which can help with ambiguity
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE  # Optimized for square markers
            )
            
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
            else:
                # Fallback: try generic solvePnP
                success2, rvec2, tvec2 = cv2.solvePnP(
                    marker_points, image_points, camera_matrix, dist_coeffs
                )
                if success2:
                    rvecs.append(rvec2)
                    tvecs.append(tvec2)
                else:
                    rvecs.append(np.zeros((3, 1), dtype=np.float32))
                    tvecs.append(np.zeros((3, 1), dtype=np.float32))
        
        return np.array(rvecs), np.array(tvecs)
    else:
        # Old API: use estimatePoseSingleMarkers
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )
        return rvecs, tvecs


class PoseFilter:
    """
    Simple exponential smoothing filter for ArUco marker poses.
    Helps reduce jitter and handle pose flips.
    """
    def __init__(self, alpha=0.7, max_z_flip_threshold=0.8):
        """
        Args:
            alpha: Smoothing factor (0-1). Higher = less smoothing, more responsive.
            max_z_flip_threshold: Maximum cosine distance to detect a Z-axis flip.
        """
        self.alpha = alpha
        self.max_z_flip_threshold = max_z_flip_threshold
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_rmat = None
        
    def filter_pose(self, rvec, tvec):
        """
        Filter pose to reduce jitter and handle flips.
        Returns filtered rvec and tvec.
        """
        if self.prev_rvec is None:
            # First pose, just store it
            self.prev_rvec = rvec.copy()
            self.prev_tvec = tvec.copy()
            rmat, _ = cv2.Rodrigues(rvec)
            self.prev_rmat = rmat
            return rvec, tvec
        
        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Check for Z-axis flip by comparing the Z-axis direction
        # The Z-axis is the third column of the rotation matrix
        prev_z_axis = self.prev_rmat[:, 2]
        curr_z_axis = rmat[:, 2]
        
        # Dot product: if negative, axes point in opposite directions (flip detected)
        z_dot = np.dot(prev_z_axis, curr_z_axis)
        
        if z_dot < -self.max_z_flip_threshold:
            # Z-axis flipped! This is the ambiguous solution problem.
            # The correct fix is to flip both the Z-axis and the X-axis to maintain
            # a right-handed coordinate system, then re-orthonormalize
            rmat[:, 0] = -rmat[:, 0]  # Flip X
            rmat[:, 2] = -rmat[:, 2]  # Flip Z
            # Re-orthonormalize the rotation matrix using SVD
            u, _, vt = np.linalg.svd(rmat)
            rmat = u @ vt
            # Ensure determinant is +1 (proper rotation)
            if np.linalg.det(rmat) < 0:
                u[:, -1] = -u[:, -1]
                rmat = u @ vt
            
            # Convert back to rotation vector
            rvec_corrected, _ = cv2.Rodrigues(rmat)
            rvec = rvec_corrected
        
        # Apply exponential smoothing
        rvec_filtered = self.alpha * rvec + (1 - self.alpha) * self.prev_rvec
        tvec_filtered = self.alpha * tvec + (1 - self.alpha) * self.prev_tvec
        
        # Update previous values
        self.prev_rvec = rvec_filtered.copy()
        self.prev_tvec = tvec_filtered.copy()
        rmat_filtered, _ = cv2.Rodrigues(rvec_filtered)
        self.prev_rmat = rmat_filtered
        
        return rvec_filtered, tvec_filtered
    
    def reset(self):
        """Reset the filter state."""
        self.prev_rvec = None
        self.prev_tvec = None
        self.prev_rmat = None


def rotation_vector_to_euler(rvec):
    """
    Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees.
    """
    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)
    
    # Extract Euler angles (ZYX convention)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(rmat[2, 1], rmat[2, 2])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        roll = np.arctan2(-rmat[1, 2], rmat[1, 1])
        pitch = np.arctan2(-rmat[2, 0], sy)
        yaw = 0
    
    return np.degrees([roll, pitch, yaw])


def main():
    # Detect available camera
    camera_index, backend = detect_camera()
    
    if camera_index is None:
        print("Error: No available cameras found")
        print("Please check:")
        print("  1. Camera is connected via USB")
        print("  2. Camera drivers are installed")
        print("  3. Camera is not being used by another application")
        return
    
    # Initialize the camera with the detected backend
    if backend is not None:
        cap = cv2.VideoCapture(camera_index, backend)
    else:
        cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        # Try without backend specification as fallback
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Camera {camera_index} failed with both methods")
            return
    
    # Set camera properties
    # Request 1080p resolution at 60fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set to 60fps
    
    # Try to maximize FOV (if camera supports these controls)
    # Setting zoom to 0 typically gives widest FOV (if supported)
    cap.set(cv2.CAP_PROP_ZOOM, 0)
    # Some cameras support focus, setting to minimum might give widest FOV
    # (Not all cameras support this - will be ignored if not supported)
    try:
        cap.set(cv2.CAP_PROP_FOCUS, 0)
    except:
        pass
    
    # Get actual frame size (may be different if camera doesn't support requested resolution)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {frame_width}x{frame_height}")
    print(f"Detection resolution: {DETECTION_WIDTH}x{DETECTION_HEIGHT} (for speed)")
    print(f"Detection every {DETECT_EVERY_N_FRAMES} frames (for speed)")
    print(f"Display window size: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
    print("(Detection at lower res for speed, pose estimation at full res for accuracy)")
    
    # Get actual FPS
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera FPS: {actual_fps}")
    
    # Check if we got the requested resolution
    if frame_width != 1920 or frame_height != 1080:
        print(f"Warning: Requested 1920x1080 but got {frame_width}x{frame_height}")
        print("Camera may not support 1080p resolution")
    
    # Get camera calibration parameters
    # NOTE: For accurate measurements, you should calibrate your camera
    # This uses an estimated camera matrix - results may not be perfectly accurate
    camera_matrix, dist_coeffs = get_camera_matrix(frame_width, frame_height)
    print(f"Using estimated camera matrix (resolution: {frame_width}x{frame_height})")
    print("NOTE: For accurate pose estimation, calibrate your camera using chessboard calibration")
    print(f"Marker size: {MARKER_SIZE*1000}mm")
    
    # Initialize ArUco detector - handle both old and new OpenCV APIs
    # Using DICT_4X4_50 as default - change if needed
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Optimize parameters for speed
    aruco_params.adaptiveThreshWinSizeMin = 3
    aruco_params.adaptiveThreshWinSizeMax = 23
    aruco_params.adaptiveThreshWinSizeStep = 10
    aruco_params.minMarkerPerimeterRate = 0.03  # Smaller = faster (detects larger markers only)
    aruco_params.maxMarkerPerimeterRate = 4.0
    aruco_params.polygonalApproxAccuracyRate = 0.03  # Larger = faster (less accurate but faster)
    aruco_params.minCornerDistanceRate = 0.05
    aruco_params.minDistanceToBorder = 3
    aruco_params.minOtsuStdDev = 5.0
    aruco_params.perspectiveRemovePixelPerCell = 4  # Smaller = faster
    aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.13
    aruco_params.maxErroneousBitsInBorderRate = 0.35
    aruco_params.errorCorrectionRate = 0.6
    
    # Check if using new API (OpenCV 4.7+)
    use_new_api = hasattr(cv2.aruco, 'ArucoDetector')
    if use_new_api:
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        print("Using new ArUco API (OpenCV 4.7+)")
    else:
        print("Using legacy ArUco API")
    
    # Initialize pose filters for each marker (if filtering is enabled)
    pose_filters = {}  # Dictionary to store filters for each marker ID
    
    if USE_POSE_FILTERING:
        print("Pose filtering enabled (alpha={:.2f})".format(POSE_FILTER_ALPHA))
    print("\nCamera feed started. Press 'q' to quit.")
    print("Pose data will be displayed on detected markers.")
    if ENABLE_PROFILING:
        print("Profiling enabled - timing info will be shown every 60 frames.\n")
    else:
        print("(Pose data is printed every 30 frames to reduce console spam)\n")
    
    # Pre-calculate text scaling factors (once, not every frame)
    text_scale_factor = frame_height / 480.0
    text_scale = 0.8 * text_scale_factor
    text_thickness = max(1, int(1.5 * text_scale_factor))
    text_offset_base = int(50 * text_scale_factor)
    
    # FPS tracking
    frame_count = 0
    fps_start_time = time.time()
    fps_frame_count = 0
    fps_display = 0.0
    
    # Profiling
    profile_times = {
        'read': [],
        'convert': [],
        'resize': [],
        'detect': [],
        'scale_corners': [],
        'pose': [],
        'draw': [],
        'display': []
    }
    
    # Cache for detection results (to skip detection on some frames)
    cached_corners = None
    cached_ids = None
    cached_rvecs = None
    cached_tvecs = None
    
    while True:
        frame_start = time.time()
        # Read frame from camera
        t0 = time.time()
        ret, frame = cap.read()
        if ENABLE_PROFILING:
            profile_times['read'].append(time.time() - t0)
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Only detect every N frames to improve FPS
        should_detect = (frame_count % DETECT_EVERY_N_FRAMES == 0)
        
        if should_detect:
            # Create downscaled version for faster detection
            t0 = time.time()
            detection_scale_x = frame_width / DETECTION_WIDTH
            detection_scale_y = frame_height / DETECTION_HEIGHT
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if ENABLE_PROFILING:
                profile_times['convert'].append(time.time() - t0)
            
            t0 = time.time()
            gray_detection = cv2.resize(gray_full, (DETECTION_WIDTH, DETECTION_HEIGHT))
            if ENABLE_PROFILING:
                profile_times['resize'].append(time.time() - t0)
            
            # Detect ArUco markers on downscaled image (much faster)
            t0 = time.time()
            if use_new_api:
                corners_detection, ids, rejected = detector.detectMarkers(gray_detection)
            else:
                corners_detection, ids, rejected = cv2.aruco.detectMarkers(gray_detection, aruco_dict, parameters=aruco_params)
            if ENABLE_PROFILING:
                profile_times['detect'].append(time.time() - t0)
            
            # Scale corners back to full resolution for accurate pose estimation
            t0 = time.time()
            corners = None
            if ids is not None and len(corners_detection) > 0:
                corners = []
                for corner_set in corners_detection:
                    scaled_corners = corner_set.copy()
                    scaled_corners[0][:, 0] *= detection_scale_x
                    scaled_corners[0][:, 1] *= detection_scale_y
                    corners.append(scaled_corners)
                corners = np.array(corners)
            else:
                corners = None
                ids = None
            if ENABLE_PROFILING:
                profile_times['scale_corners'].append(time.time() - t0)
            
            # Cache results
            cached_corners = corners
            cached_ids = ids
        else:
            # Use cached detection results
            corners = cached_corners
            ids = cached_ids
        
        # Estimate pose and draw results (use cached if no new detection)
        detected_ids = set()
        if ids is not None:
            detected_ids = set(ids.flatten())
            
            # Reset filters for markers that are no longer detected
            if USE_POSE_FILTERING:
                missing_ids = set(pose_filters.keys()) - detected_ids
                for missing_id in missing_ids:
                    pose_filters[missing_id].reset()
            
            # Draw marker outlines and IDs
            t0 = time.time()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Estimate pose for each detected marker (only if we have new detection)
            if should_detect and corners is not None:
                t0 = time.time()
                has_estimate_pose = hasattr(cv2.aruco, 'estimatePoseSingleMarkers')
                rvecs, tvecs = estimate_marker_pose(
                    corners, MARKER_SIZE, camera_matrix, dist_coeffs, use_new_api=not has_estimate_pose
                )
                if ENABLE_PROFILING:
                    profile_times['pose'].append(time.time() - t0)
                # Cache pose results
                cached_rvecs = rvecs
                cached_tvecs = tvecs
            else:
                # Use cached pose results
                if cached_rvecs is not None and cached_tvecs is not None:
                    rvecs = cached_rvecs
                    tvecs = cached_tvecs
                else:
                    rvecs = None
                    tvecs = None
            
            # Process each detected marker (only if we have valid pose data)
            if rvecs is not None and tvecs is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    # Handle different array shapes between old and new API
                    if not has_estimate_pose:
                        rvec = rvecs[i].flatten()
                        tvec = tvecs[i].flatten()
                    else:
                        rvec = rvecs[i][0]
                        tvec = tvecs[i][0]
                    
                    # Ensure rvec and tvec are the right shape
                    rvec_3x1_raw = rvec.reshape(3, 1)
                    tvec_3x1_raw = tvec.reshape(3, 1)
                    
                    # Apply pose filtering if enabled
                    if USE_POSE_FILTERING:
                        # Initialize filter for this marker ID if not exists
                        if marker_id not in pose_filters:
                            pose_filters[marker_id] = PoseFilter(alpha=POSE_FILTER_ALPHA)
                        
                        # Filter the pose
                        rvec_3x1, tvec_3x1 = pose_filters[marker_id].filter_pose(rvec_3x1_raw, tvec_3x1_raw)
                    else:
                        rvec_3x1 = rvec_3x1_raw
                        tvec_3x1 = tvec_3x1_raw
                    
                    # Draw coordinate axes on marker
                    draw_axis(frame, camera_matrix, dist_coeffs, rvec_3x1, tvec_3x1, length=MARKER_SIZE * 0.8)
                    
                    # Extract position (translation vector in meters)
                    # tvec[0]: X (right), tvec[1]: Y (down), tvec[2]: Z (forward, away from camera)
                    x_pos = tvec[0] * 1000  # Convert to mm
                    y_pos = tvec[1] * 1000
                    z_pos = tvec[2] * 1000  # Distance from camera
                    
                    # Extract rotation (convert to Euler angles)
                    roll, pitch, yaw = rotation_vector_to_euler(rvec_3x1)
                    
                    # Display position and pose info on frame (using pre-calculated scale factors)
                    corner = tuple(corners[i][0][0].astype(int))
                    
                    cv2.putText(frame, f"ID:{marker_id}", 
                               (corner[0], corner[1] - int(text_offset_base * 1.8)), 
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness)
                    cv2.putText(frame, f"X:{x_pos:.1f} Y:{y_pos:.1f} Z:{z_pos:.1f}mm",
                               (corner[0], corner[1] - int(text_offset_base * 1.3)),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (255, 255, 255), text_thickness)
                    cv2.putText(frame, f"R:{roll:.0f} P:{pitch:.0f} Y:{yaw:.0f}",
                               (corner[0], corner[1] - text_offset_base),
                               cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (255, 255, 255), text_thickness)
                    
                    # Print to console (every 30 frames to reduce spam)
                    if frame_count % 30 == 0 and not ENABLE_PROFILING:
                        print(f"Marker {marker_id}: Pos=({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})mm, "
                              f"Rot=(Roll:{roll:.1f}, Pitch:{pitch:.1f}, Yaw:{yaw:.1f})deg")
            
            # End of drawing operations
            if ENABLE_PROFILING:
                profile_times['draw'].append(time.time() - t0)
        
        frame_count += 1
        fps_frame_count += 1
        
        # Resize frame for display (processing was done on full resolution)
        t0 = time.time()
        display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        
        # Add FPS to display
        cv2.putText(display_frame, f"FPS: {fps_display:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('ArUco Marker Detection', display_frame)
        if ENABLE_PROFILING:
            profile_times['display'].append(time.time() - t0)
        
        # Calculate FPS every second
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            fps_display = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()
            
            # Print profiling info every 60 frames
            if ENABLE_PROFILING and frame_count % 60 == 0:
                print("\n=== Performance Profile (ms) ===")
                for key, times in profile_times.items():
                    if times:
                        avg_ms = np.mean(times) * 1000
                        max_ms = np.max(times) * 1000
                        print(f"  {key:15s}: avg={avg_ms:6.2f}ms, max={max_ms:6.2f}ms")
                print(f"Total FPS: {fps_display:.1f}")
                print("=" * 35 + "\n")
                # Clear profiling data
                for key in profile_times:
                    profile_times[key] = []
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


if __name__ == "__main__":
    main()
