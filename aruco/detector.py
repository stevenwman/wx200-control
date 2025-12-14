"""
ArUco marker detection with ROI-based tracking for efficiency.
"""

import cv2
import numpy as np


class ArucoDetector:
    """ArUco marker detector with ROI-based tracking."""
    
    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_50, 
                 roi_expand=100, roi_search_frames=5,
                 adaptive_thresh_win_size_min=3,
                 adaptive_thresh_win_size_max=23,
                 adaptive_thresh_win_size_step=10,
                 polygonal_approx_accuracy_rate=0.05,
                 perspective_remove_pixel_per_cell=4):
        """
        Initialize ArUco detector.
        
        Args:
            dictionary_type: ArUco dictionary type
            roi_expand: Pixels to expand ROI around last position
            roi_search_frames: Search ROI for N frames, then full frame
            adaptive_thresh_win_size_min: Minimum window size for adaptive thresholding
            adaptive_thresh_win_size_max: Maximum window size for adaptive thresholding
            adaptive_thresh_win_size_step: Step size for window size search
            polygonal_approx_accuracy_rate: Accuracy for polygon approximation (lower = more accurate)
            perspective_remove_pixel_per_cell: Pixels per cell for perspective removal
        """
        self.dictionary_type = dictionary_type
        self.roi_expand = roi_expand
        self.roi_search_frames = roi_search_frames
        
        # Setup ArUco detector with configurable parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = adaptive_thresh_win_size_min
        params.adaptiveThreshWinSizeMax = adaptive_thresh_win_size_max
        params.adaptiveThreshWinSizeStep = adaptive_thresh_win_size_step
        params.polygonalApproxAccuracyRate = polygonal_approx_accuracy_rate
        params.perspectiveRemovePixelPerCell = perspective_remove_pixel_per_cell
        
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        # ROI tracking state
        self.last_positions = {}  # marker_id -> (x, y, w, h)
        self.roi_frame_count = {}  # marker_id -> frame count
    
    def detect(self, gray, frame_width, frame_height, target_marker_ids=None):
        """
        Detect ArUco markers with ROI-based tracking.
        
        Args:
            gray: Grayscale image
            frame_width: Frame width
            frame_height: Frame height
            target_marker_ids: List of marker IDs to track (None = all)
        
        Returns:
            (corners, ids): Tuple of detected corners and IDs, or (None, None)
        """
        if target_marker_ids is None:
            target_marker_ids = []
        
        all_corners = []
        all_ids = []
        detected_markers = set()
        
        # First, try ROI-based detection for each known marker
        for marker_id in target_marker_ids:
            if marker_id in self.last_positions and self.last_positions[marker_id] is not None:
                if self.roi_frame_count.get(marker_id, 0) < self.roi_search_frames:
                    # Search in ROI around last known position
                    x, y, roi_w, roi_h = self.last_positions[marker_id]
                    x1 = max(0, x - self.roi_expand)
                    y1 = max(0, y - self.roi_expand)
                    x2 = min(frame_width, x + roi_w + self.roi_expand)
                    y2 = min(frame_height, y + roi_h + self.roi_expand)
                    
                    roi_gray = gray[y1:y2, x1:x2]
                    if roi_gray.size > 0:
                        corners_roi, ids_roi, _ = self.detector.detectMarkers(roi_gray)
                        if ids_roi is not None:
                            # Adjust corner coordinates back to full frame
                            for j, corner in enumerate(corners_roi):
                                if ids_roi[j][0] == marker_id:
                                    corner[0][:, 0] += x1
                                    corner[0][:, 1] += y1
                                    all_corners.append(corner)
                                    all_ids.append(ids_roi[j])
                                    detected_markers.add(marker_id)
                                    self.roi_frame_count[marker_id] = 0  # Reset counter
                                    break
        
        # Full frame detection for markers not found in ROI or if ROI search expired
        full_frame_needed = False
        for marker_id in target_marker_ids:
            if marker_id not in detected_markers:
                if marker_id not in self.last_positions or self.last_positions[marker_id] is None:
                    full_frame_needed = True
                elif self.roi_frame_count.get(marker_id, 0) >= self.roi_search_frames:
                    full_frame_needed = True
                    self.roi_frame_count[marker_id] = 0  # Reset for next ROI search
        
        if full_frame_needed:
            corners_full, ids_full, _ = self.detector.detectMarkers(gray)
            if ids_full is not None:
                for j, marker_id_arr in enumerate(ids_full):
                    marker_id = marker_id_arr[0]
                    if marker_id in target_marker_ids and marker_id not in detected_markers:
                        all_corners.append(corners_full[j])
                        all_ids.append(marker_id_arr)
                        detected_markers.add(marker_id)
        
        # Increment ROI frame counters for missed detections
        for marker_id in target_marker_ids:
            if marker_id not in detected_markers:
                if marker_id in self.last_positions and self.last_positions[marker_id] is not None:
                    self.roi_frame_count[marker_id] = self.roi_frame_count.get(marker_id, 0) + 1
        
        # Convert to format expected by rest of code
        if all_corners:
            return all_corners, np.array(all_ids)
        else:
            return None, None
    
    def update_position(self, marker_id, corners):
        """Update the last known position of a marker."""
        corner_points = corners[0]
        x_min = int(np.min(corner_points[:, 0]))
        y_min = int(np.min(corner_points[:, 1]))
        x_max = int(np.max(corner_points[:, 0]))
        y_max = int(np.max(corner_points[:, 1]))
        self.last_positions[marker_id] = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def get_roi(self, marker_id):
        """Get ROI rectangle for a marker, or None if not tracked."""
        if marker_id in self.last_positions and self.last_positions[marker_id] is not None:
            x, y, w, h = self.last_positions[marker_id]
            return (x - self.roi_expand, y - self.roi_expand, 
                   x + w + self.roi_expand, y + h + self.roi_expand)
        return None
