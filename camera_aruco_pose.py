#!/usr/bin/env python3
"""
ArUco marker detection main loop (Simplified).
Tracks IDs 0 (World), 2 (Object), 3 (Gripper) using 5x5 markers.
"""
import cv2
import numpy as np
import time
import argparse
from collections import deque

from camera import GStreamerCamera, is_gstreamer_available
from camera import ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-visualize', dest='visualize', action='store_false', help='Disable vis')
    parser.add_argument('--camera-id', type=int, default=1)
    parser.add_argument('--width', type=int, default=1920, help='Frame width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Frame height (default: 1080)')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS (default: 30)')
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    
    if not is_gstreamer_available():
        print("GStreamer not found.")
        return

    # Setup Camera
    camera = GStreamerCamera(device=f'/dev/video{args.camera_id}', width=args.width, height=args.height, fps=args.fps)
    estimator = ArUcoPoseEstimator(MARKER_SIZE)
    
    print(f"Starting ArUco (5x5). Tracking IDs: 0(World), 2(Object), 3(Gripper).")
    print(f"Camera: {args.width}x{args.height} @ {args.fps} FPS")
    
    try:
        camera.start()
        cam_matrix, dist_coeffs = get_approx_camera_matrix(args.width, args.height)
        
        # Setup Single 5x5 Detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        fps_window = deque(maxlen=30)
        last_time = time.time()
        last_print = time.time()
        
        while True:
            ret, frame = camera.read()
            if not ret: time.sleep(0.001); continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            # --- Draw Raw Axes for ALL Detected Tags ---
            if ids is not None and args.visualize:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # Quick viz of raw inputs
                for i in range(len(ids)):
                    # Stateless solve for viz
                    rvec, tvec = estimator.process_tag(corners, ids, cam_matrix, dist_coeffs, ids[i][0])
                    # Note: process_tag uses state, so this updates state. 
                    # But we want to process specific IDs next. 
                    # Actually, let's just process the target IDs specifically below to avoid double-processing state.
                    pass

            # --- Process Target Tags ---
            # World (ID 0)
            r_world, t_world = estimator.process_tag(corners, ids, cam_matrix, dist_coeffs, 0)
            if r_world is not None and args.visualize:
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, r_world, t_world, MARKER_SIZE)

            # Object (ID 2)
            r_obj, t_obj = estimator.process_tag(corners, ids, cam_matrix, dist_coeffs, 2)
            if r_obj is not None and args.visualize:
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, r_obj, t_obj, MARKER_SIZE)

            # Gripper (ID 3)
            r_grip, t_grip = estimator.process_tag(corners, ids, cam_matrix, dist_coeffs, 3)
            if r_grip is not None and args.visualize:
                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, r_grip, t_grip, MARKER_SIZE)

            # --- Compute Relative Poses ---
            if r_world is not None:
                if r_obj is not None:
                    _, t_rel = estimator.get_relative_pose(r_world, t_world, r_obj, t_obj)
                    print(f"OBJECT in WORLD  | X: {t_rel[0][0]:.3f} Y: {t_rel[1][0]:.3f} Z: {t_rel[2][0]:.3f}")
                
                if r_grip is not None:
                    _, t_rel = estimator.get_relative_pose(r_world, t_world, r_grip, t_grip)
                    print(f"GRIPPER in WORLD | X: {t_rel[0][0]:.3f} Y: {t_rel[1][0]:.3f} Z: {t_rel[2][0]:.3f}")

            if args.visualize:
                disp = cv2.resize(frame, (1280, 720))
                cv2.imshow('ArUco', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # FPS
            now = time.time()
            fps_window.append(1.0/(now - last_time) if now - last_time > 0 else 0)
            last_time = now
            if now - last_print > 1.0:
                 print(f"FPS: {np.mean(fps_window):.1f}")
                 last_print = now

    except KeyboardInterrupt: pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
