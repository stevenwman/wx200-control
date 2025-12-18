#!/usr/bin/env python3
"""
Simple script to identify ArUco marker IDs using the camera.
Scans for both DICT_4X4_50 and DICT_5X5_100 by default.
"""

import cv2
import time
import numpy as np
import argparse

from camera import GStreamerCamera, is_gstreamer_available

def main():
    parser = argparse.ArgumentParser(description='Identify ArUco marker IDs')
    parser.add_argument('--camera-id', type=int, default=1,
                        help='Camera device ID (default: 1)')
    args = parser.parse_args()

    if not is_gstreamer_available():
        print("ERROR: GStreamer not available!")
        return

    # Initialize detectors for both 4x4 and 5x5
    detectors = []
    
    # DICT_4X4_50
    dict_4x4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params_4x4 = cv2.aruco.DetectorParameters()
    detectors.append(('4x4', cv2.aruco.ArucoDetector(dict_4x4, params_4x4)))
    
    # DICT_5X5_100
    dict_5x5 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    params_5x5 = cv2.aruco.DetectorParameters()
    detectors.append(('5x5', cv2.aruco.ArucoDetector(dict_5x5, params_5x5)))

    print(f"Starting ArUco ID detection...")
    print(f"Scanning for: {[name for name, _ in detectors]}")
    print(f"Using Camera {args.camera_id}")
    print("Press 'q' to quit")

    # Initialize camera
    camera = GStreamerCamera(device=f'/dev/video{args.camera_id}', width=1920, height=1080, fps=30)
    
    try:
        camera.start()
        
        while True:
            ret, frame = camera.read()
            if not ret:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            all_detected_text = []
            y_offset = 30
            
            # Run each detector
            for name, detector in detectors:
                corners, ids, rejected = detector.detectMarkers(gray)
                
                if ids is not None:
                    # Draw markers
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    
                    # Process IDs
                    unique_ids = np.unique(ids)
                    id_list = ", ".join([str(id) for id in unique_ids])
                    text = f"{name}: {id_list}"
                    all_detected_text.append(text)
                    
                    # Draw ID list on screen
                    color = (0, 255, 0) if name == '4x4' else (255, 255, 0)
                    cv2.putText(frame, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    y_offset += 30

            if all_detected_text:
                print(f"Found: {' | '.join(all_detected_text)}")

            cv2.imshow('ArUco Identifier', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
