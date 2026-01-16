"""
Camera and ArUco pose estimation module.

Combines GStreamer-based camera access (MANDATORY) and ArUco marker detection.
OpenCV fallback has been removed to ensure high performance (30 FPS vs 15 FPS).
"""
import sys
import os
import time
from pathlib import Path

# Fix GStreamer environment before any imports that use it
from . import fix_gstreamer_env  # Must be imported BEFORE cv2 and GStreamer

import cv2
import numpy as np
import threading
from scipy.spatial.transform import Rotation as R
from .robot_config import robot_config

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GSTREAMER_AVAILABLE = True
except (ImportError, ValueError) as e:
    GSTREAMER_AVAILABLE = False
    GSTREAMER_ERROR = str(e)

# Suppress noisy OpenCV WARN logs
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    try:
        cv2.setLogLevel(3)
    except Exception:
        pass


class GStreamerCamera:
    """GStreamer-based camera capture for high-performance video capture (30fps)."""
    
    def __init__(self, device='/dev/video1', width=1920, height=1080, fps=30):
        if not GSTREAMER_AVAILABLE:
            raise RuntimeError(
                f"GStreamer not available: {GSTREAMER_ERROR}\n"
                "Mandatory for high performance. Install with: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good"
            )
        
        # Ensure device string
        if isinstance(device, int):
            self.device = f"/dev/video{device}"
        else:
            self.device = device
            
        Gst.init(None)
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsink = None
        self.frame = None
        self.frame_available = False
        
    def _on_new_sample(self, appsink):
        """Callback when a new frame is available."""
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Create numpy array from buffer
                # Note: GStreamer buffer is likely BGR due to pipeline conversion
                arr = np.ndarray(
                    (self.height, self.width, 3),
                    buffer=map_info.data,
                    dtype=np.uint8
                )
                self.frame = arr.copy()
                self.frame_available = True
                buffer.unmap(map_info)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR
    
    def start(self):
        """Start the GStreamer pipeline."""
        pipeline_str = (
            f"v4l2src device={self.device} ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=True max-buffers=1 drop=True"
        )
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self._on_new_sample)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"Failed to start GStreamer pipeline for {self.device}. Check if device is busy/exists.")
        
        # Async wait
        if ret == Gst.StateChangeReturn.ASYNC:
            ret, state, pending = self.pipeline.get_state(timeout=5 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(f"Failed to start GStreamer pipeline for {self.device} (State failure).")
            elif ret == Gst.StateChangeReturn.ASYNC:
                 raise RuntimeError(f"GStreamer startup timed out for {self.device}.")

    def read(self):
        """
        Read a frame (non-blocking).
        Returns (success, frame).
        """
        if self.frame_available:
            frame = self.frame.copy()
            self.frame_available = False
            return True, frame
        return False, None
    
    def release(self):
        """Stop and cleanup."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            try:
                self.pipeline.get_state(timeout=Gst.SECOND)
            except Exception:
                pass
            self.pipeline = None
            self.frame = None
            self.frame_available = False

def Camera(device=0, width=1920, height=1080, fps=30):
    """
    Factory function. Now strictly enforcing GStreamer usage.
    """
    return GStreamerCamera(device=device, width=width, height=height, fps=fps)


# --- ArUco Logic ---

MARKER_SIZE = robot_config.aruco_marker_size_m
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
        
        self.history = {}
        self.tag_states = {}

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

        # 2. Get Candidates
        result = cv2.solvePnPGeneric(
            self.obj_points, corners[idx[0]][0], cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        candidates = []
        if len(result) >= 3 and result[0]:
             rvecs, tvecs = result[1], result[2]
             for r, t in zip(rvecs, tvecs):
                 if t[2] > 0: candidates.append((r, t))
        
        # 3. Select Best Candidate
        chosen_rvec, chosen_tvec = self._select_best_candidate(candidates)
        
        # 4. Validate & Update State
        if chosen_rvec is not None:
            if self._validate_update(chosen_rvec, chosen_tvec, state):
                return self._apply_smoothing(target_id, state['last_valid_rvec'], state['last_valid_tvec'])
            elif state['last_valid_rvec'] is not None:
                return self._apply_smoothing(target_id, state['last_valid_rvec'], state['last_valid_tvec'])
                
        return None, None

    def _select_best_candidate(self, candidates):
        if not candidates: return None, None
        
        for rvec, tvec in candidates:
            R, _ = cv2.Rodrigues(rvec)
            if R[2, 2] < 0: return rvec, tvec
            
        best, best_score = candidates[0], -np.inf
        for rvec, tvec in candidates:
            R, _ = cv2.Rodrigues(rvec)
            score = np.dot(R[:, 2], tvec.flatten() / np.linalg.norm(tvec))
            if score > best_score:
                best, best_score = (rvec, tvec), score
        return best

    def _validate_update(self, rvec, tvec, state):
        if state['last_valid_rvec'] is None:
            state['last_valid_rvec'] = rvec
            state['last_valid_tvec'] = tvec
            return True

        R_new, _ = cv2.Rodrigues(rvec)
        R_last, _ = cv2.Rodrigues(state['last_valid_rvec'])
        
        r_diff = np.linalg.norm(R_new - R_last, 'fro')
        t_diff = np.linalg.norm(tvec - state['last_valid_tvec'])
        
        if r_diff < SINGLE_TAG_ROTATION_THRESHOLD and t_diff < SINGLE_TAG_TRANSLATION_THRESHOLD:
            state['last_valid_rvec'] = rvec
            state['last_valid_tvec'] = tvec
            state['consecutive_rejections'] = 0
            state['consecutive_misses'] = 0
            return True
        else:
            state['consecutive_rejections'] += 1
            if state['consecutive_rejections'] > MAX_REJECTIONS_BEFORE_FORCE:
                state['last_valid_rvec'] = rvec
                state['last_valid_tvec'] = tvec
                state['consecutive_rejections'] = 0
                state['consecutive_misses'] = 0
                return True
            return False

    def _apply_smoothing(self, mid, rvec, tvec):
        if mid not in self.history: self.history[mid] = []
        self.history[mid].append((rvec.copy(), tvec.copy()))
        if len(self.history[mid]) > 5: self.history[mid].pop(0)
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


class ThreadedArUcoCamera:
    """
    Threaded wrapper for Camera + ArUco Estimator.
    Polls at camera FPS (30Hz) in background thread, decoupling it from control loop (10Hz).
    """
    def __init__(self, device=0, width=1920, height=1080, fps=30):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.detector = None
        self.estimator = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Latest data (Thread Safe)
        self.latest_frame = None
        self.latest_obs = {
            'aruco_ee_in_world': np.zeros(7),
            'aruco_object_in_world': np.zeros(7),
            'aruco_ee_in_object': np.zeros(7),
            'aruco_object_in_ee': np.zeros(7),
            'aruco_visibility': np.zeros(3)
        }
        
        # Init components
        self.camera = Camera(device=device, width=width, height=height, fps=fps)
        self.camera.start()
        
        self.estimator = ArUcoPoseEstimator(MARKER_SIZE)
        self.cam_matrix, self.dist_coeffs = get_approx_camera_matrix(width, height)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._poll_loop, daemon=True)
        self.thread.start()
        print(f"✓ Started ThreadedArUcoCamera polling at {self.fps}Hz")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.camera:
            self.camera.release()

    def release(self):
        self.stop()

    def _poll_loop(self):
        # We manually limit rate to fps
        dt = 1.0 / self.fps
        
        while self.running:
            start = time.perf_counter()
            
            # 1. Read Frame
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.001)
                continue
                
            # 2. Detect
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            # 3. Process Tags
            r_world, t_world = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_world_id)
            r_obj, t_obj = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_object_id)
            r_ee, t_ee = self.estimator.process_tag(corners, ids, self.cam_matrix, self.dist_coeffs, robot_config.aruco_ee_id)
            
            # 4. Compute Obs
            obs = {}
            # Visibility
            world_vis = 1.0 if (ids is not None and robot_config.aruco_world_id in ids) else 0.0
            obj_vis = 1.0 if (ids is not None and robot_config.aruco_object_id in ids) else 0.0
            ee_vis = 1.0 if (ids is not None and robot_config.aruco_ee_id in ids) else 0.0
            obs['aruco_visibility'] = np.array([world_vis, obj_vis, ee_vis])
            
            # Relative Poses
            def pack(r, t): 
                if r is None: return np.zeros(3), np.array([1.,0.,0.,0.])
                R_mat, _ = cv2.Rodrigues(r)
                quat = np.array([0,0,0,1]) # dummy
                # We need proper conversion if using this helper, but ArUcoPoseEstimator has specific helper?
                # ThreadedArUcoCamera should reuse logic. 
                # Let's use self.estimator.get_relative_pose helper if possible? 
                # Wait, I can't call a method I haven't defined on this class easily if I didn't copy it.
                # ArUcoPoseEstimator.get_relative_pose IS defined in this file.
                return np.zeros(3), np.array([1.,0.,0.,0.])

            # Actually, let's copy the compute_relative_pose logic or move it to helper.
            # Ideally ArUcoPoseEstimator shouldn't be responsible for "relative between two tags".
            # It is in `camera.py` lines 280-292. `get_relative_pose` returns r_rel, t_rel.
            # We need to convert to pos, quat_wxyz.
            
            def compute_rel(r1, t1, r2, t2):
                if r1 is None or r2 is None: return np.zeros(7)
                r_rel, t_rel = self.estimator.get_relative_pose(r1, t1, r2, t2)
                R_rel, _ = cv2.Rodrigues(r_rel)
                from scipy.spatial.transform import Rotation as R
                q = R.from_matrix(R_rel).as_quat() # xyzw
                return np.concatenate([t_rel.flatten(), [q[3], q[0], q[1], q[2]]])

            obs['aruco_ee_in_world'] = compute_rel(r_world, t_world, r_ee, t_ee)
            obs['aruco_object_in_world'] = compute_rel(r_world, t_world, r_obj, t_obj)
            obs['aruco_ee_in_object'] = compute_rel(r_obj, t_obj, r_ee, t_ee)
            obs['aruco_object_in_ee'] = compute_rel(r_ee, t_ee, r_obj, t_obj)
            
            # 5. Update Shared State
            with self.lock:
                self.latest_frame = frame.copy()
                self.latest_obs = obs
                # Draw markers on frame for Viz?
                # Ideally we store "annotated frame" or "clean frame"?
                # Legacy stored "frame_to_record" (clean, downscaled) and Viz showed annotated.
                # Let's just store clean frame. Viz can annotate if needed, or we annotate a copy.
                if ids is not None:
                     cv2.aruco.drawDetectedMarkers(self.latest_frame, corners, ids)
            
            # Sleep remainder
            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def get_data(self):
        """Return latest frame and observation safely."""
        with self.lock:
            return (
                self.latest_frame.copy() if self.latest_frame is not None else None, 
                self.latest_obs.copy()
            )

