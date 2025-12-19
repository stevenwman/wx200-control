import cv2
import numpy as np
from robot_control.robot_config import robot_config

# Suppress noisy OpenCV WARN logs from drawFrameAxes / solvePnP when axes go off-frame
try:
    # Newer OpenCV
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try:
        # Older OpenCV fallback (3 = ERROR)
        cv2.setLogLevel(3)
    except Exception:
        pass

# --- Configuration (sourced from robot_config) ---
MARKER_SIZE = robot_config.aruco_marker_size_m  # meters - 5x5 Tags (World, Object, Gripper)
SINGLE_TAG_ROTATION_THRESHOLD = robot_config.aruco_single_tag_rotation_threshold
SINGLE_TAG_TRANSLATION_THRESHOLD = robot_config.aruco_single_tag_translation_threshold
MAX_PRESERVE_FRAMES = robot_config.aruco_max_preserve_frames
MAX_REJECTIONS_BEFORE_FORCE = robot_config.aruco_max_rejections_before_force

def get_approx_camera_matrix(width, height):
    """Get approximate camera matrix based on frame dimensions."""
    focal_length = width
    center_x = width / 2.0
    center_y = height / 2.0
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs

class ArUcoPoseEstimator:
    def __init__(self, marker_size=MARKER_SIZE):
        self.marker_size = marker_size
        
        # Object points for a single marker
        half = marker_size / 2.0
        self.obj_points = np.array([[-half, half, 0], [half, half, 0],
                                    [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
        
        # State tracking
        self.history = {}  # marker_id -> list of recent poses
        self.tag_states = {}  # id -> state dict

    def _get_tag_state(self, tag_id):
        if tag_id not in self.tag_states:
            self.tag_states[tag_id] = {
                'last_valid_rvec': None,
                'last_valid_tvec': None,
                'consecutive_misses': 0,
                'consecutive_rejections': 0
            }
        return self.tag_states[tag_id]

    def process_tag(self, corners, ids, cam_matrix, dist_coeffs, target_id):
        """Process a single tag with robust validation and smoothing."""
        # Guard against None or scalar ids from OpenCV
        if ids is None:
            return None, None
        ids_arr = np.atleast_1d(ids).ravel()
        idx = np.where(ids_arr == target_id)[0]
        state = self._get_tag_state(target_id)
        
        # 1. Handle Missing
        if len(idx) == 0:
            if (state['last_valid_rvec'] is not None and 
                state['consecutive_misses'] < MAX_PRESERVE_FRAMES):
                state['consecutive_misses'] += 1
                return state['last_valid_rvec'], state['last_valid_tvec']
            return None, None

        # 2. Get Candidates (IPPE gives 2 solutions)
        result = cv2.solvePnPGeneric(
            self.obj_points, corners[idx[0]][0], cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        # Unpack result
        candidates = []
        if len(result) >= 3 and result[0]: # Success check
             # result is (success, rvecs, tvecs, errs) or (success, rvecs, tvecs)
             rvecs, tvecs = result[1], result[2]
             for r, t in zip(rvecs, tvecs):
                 if t[2] > 0: candidates.append((r, t))
        
        # 3. Select Best Candidate (Z-Normal < 0)
        chosen_rvec, chosen_tvec = self._select_best_candidate(candidates)
        
        # 4. Validate & Update State
        if chosen_rvec is not None:
            if self._validate_update(chosen_rvec, chosen_tvec, state):
                # Valid new pose -> Smooth and return
                return self._apply_smoothing(target_id, state['last_valid_rvec'], state['last_valid_tvec'])
            elif state['last_valid_rvec'] is not None:
                # Invalid (flip) -> Return Preserved
                return self._apply_smoothing(target_id, state['last_valid_rvec'], state['last_valid_tvec'])
                
        return None, None

    def _select_best_candidate(self, candidates):
        """Pick the solution where Z-normal points to camera."""
        if not candidates: return None, None
        
        # Prefer solution with negative Z normal (facing camera)
        for rvec, tvec in candidates:
            R, _ = cv2.Rodrigues(rvec)
            if R[2, 2] < 0: return rvec, tvec
            
        # Fallback: align with view vector
        best, best_score = candidates[0], -np.inf
        for rvec, tvec in candidates:
            R, _ = cv2.Rodrigues(rvec)
            score = np.dot(R[:, 2], tvec.flatten() / np.linalg.norm(tvec))
            if score > best_score:
                best, best_score = (rvec, tvec), score
        return best

    def _validate_update(self, rvec, tvec, state):
        """Prevent flips by checking consistency with history."""
        if state['last_valid_rvec'] is None:
            state['last_valid_rvec'] = rvec
            state['last_valid_tvec'] = tvec
            return True

        R_new, _ = cv2.Rodrigues(rvec)
        R_last, _ = cv2.Rodrigues(state['last_valid_rvec'])
        
        r_diff = np.linalg.norm(R_new - R_last, 'fro')
        t_diff = np.linalg.norm(tvec - state['last_valid_tvec'])
        
        # Check thresholds
        if r_diff < SINGLE_TAG_ROTATION_THRESHOLD and t_diff < SINGLE_TAG_TRANSLATION_THRESHOLD:
            state['last_valid_rvec'] = rvec
            state['last_valid_tvec'] = tvec
            state['consecutive_rejections'] = 0
            state['consecutive_misses'] = 0
            return True
        else:
            # Rejection logic (persistence check)
            state['consecutive_rejections'] += 1
            if state['consecutive_rejections'] > MAX_REJECTIONS_BEFORE_FORCE:
                # Force accept after too many rejected frames
                state['last_valid_rvec'] = rvec
                state['last_valid_tvec'] = tvec
                state['consecutive_rejections'] = 0
                state['consecutive_misses'] = 0
                return True
            return False

    def _apply_smoothing(self, mid, rvec, tvec):
        """Simple history smoothing."""
        if mid not in self.history: self.history[mid] = []
        
        self.history[mid].append((rvec.copy(), tvec.copy()))
        if len(self.history[mid]) > 5: self.history[mid].pop(0)
        
        # Return current (smoothing can be added here if needed, but current is most responsive)
        return rvec, tvec

    def get_relative_pose(self, rvec_ref, tvec_ref, rvec_tgt, tvec_tgt):
        """Compute T_target relative to T_reference."""
        if rvec_ref is None or rvec_tgt is None: return None, None
        
        R_ref, _ = cv2.Rodrigues(rvec_ref)
        T_ref = np.eye(4); T_ref[:3,:3] = R_ref; T_ref[:3,3] = tvec_ref.flatten()
        
        R_tgt, _ = cv2.Rodrigues(rvec_tgt)
        T_tgt = np.eye(4); T_tgt[:3,:3] = R_tgt; T_tgt[:3,3] = tvec_tgt.flatten()
        
        T_rel = np.linalg.inv(T_ref) @ T_tgt
        r_rel, _ = cv2.Rodrigues(T_rel[:3,:3])
        return r_rel, T_rel[:3,3].reshape(3,1)
