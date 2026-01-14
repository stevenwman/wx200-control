import cv2
import numpy as np
import multiprocessing
import time
from collections import deque

# Display window size (smaller than actual camera resolution)
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

# Profiling
ENABLE_PROFILING = True
PROFILE_HISTORY_SIZE = 100  # Keep last N measurements

# --- Helper Functions (Same as before) ---
def get_approx_camera_matrix(width, height):
    focal_length = width
    center_x = width / 2.0
    center_y = height / 2.0
    camera_matrix = np.array([[focal_length, 0, center_x],
                              [0, focal_length, center_y],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    return camera_matrix, dist_coeffs

def vision_worker(camera_id, marker_size_m, dictionary_type, visualize, pose_queue, frame_queue, stop_event, profile_queue=None):
    """
    Worker process that captures camera frames and detects ArUco markers.
    Must be at module level for multiprocessing to work.
    """
    # 1. Setup Camera (Inside the process)
    # First, try to reset camera to factory defaults using v4l2-ctl
    import subprocess
    try:
        print("Resetting camera to factory defaults...")
        # Reset exposure to auto (3 = Auto Mode, 1 = Manual Mode)
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=auto_exposure=3'], 
                      capture_output=True, timeout=2, check=False)
        # Reset focus to auto
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=focus_automatic_continuous=1'], 
                      capture_output=True, timeout=2, check=False)
        # Reset white balance to auto
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=white_balance_automatic=1'], 
                      capture_output=True, timeout=2, check=False)
        # Reset brightness, contrast, saturation to defaults
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=brightness=0'], 
                      capture_output=True, timeout=2, check=False)
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=contrast=0'], 
                      capture_output=True, timeout=2, check=False)
        subprocess.run(['v4l2-ctl', '-d', f'/dev/video{camera_id}', '--set-ctrl=saturation=50'], 
                      capture_output=True, timeout=2, check=False)
        print("Camera reset complete")
    except Exception as e:
        print(f"Could not reset camera via v4l2-ctl: {e}")
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Set MJPEG format for much better performance (15 FPS vs 1 FPS)
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        stop_event.set()
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"Camera opened: {w}x{h}, Format: {actual_str}")
    cam_matrix, dist_coeffs = get_approx_camera_matrix(w, h)

    # 2. Setup Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.polygonalApproxAccuracyRate = 0.05
    params.perspectiveRemovePixelPerCell = 4
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # 3. Object Points
    half = marker_size_m / 2.0
    obj_points = np.array([[-half, half, 0], [half, half, 0],
                           [half, -half, 0], [-half, -half, 0]], dtype=np.float32)

    print("Vision process started.")
    
    # Profiling
    frame_count = 0
    profile_times = {
        'read': deque(maxlen=PROFILE_HISTORY_SIZE),
        'convert': deque(maxlen=PROFILE_HISTORY_SIZE),
        'detect': deque(maxlen=PROFILE_HISTORY_SIZE),
        'pose': deque(maxlen=PROFILE_HISTORY_SIZE),
        'draw': deque(maxlen=PROFILE_HISTORY_SIZE),
        'queue_pose': deque(maxlen=PROFILE_HISTORY_SIZE),
        'queue_frame': deque(maxlen=PROFILE_HISTORY_SIZE),
        'total': deque(maxlen=PROFILE_HISTORY_SIZE)
    }
    
    try:
        while not stop_event.is_set():
            frame_start = time.time()
            try:
                # Read frame - use grab() + retrieve() for potentially better performance
                t0 = time.time()
                # Try grab() first (non-blocking metadata capture)
                if cap.grab():
                    ret, frame = cap.retrieve()
                else:
                    ret, frame = cap.read()  # Fallback
                if ENABLE_PROFILING:
                    profile_times['read'].append(time.time() - t0)
                if not ret: 
                    break

                # Convert to grayscale
                t0 = time.time()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if ENABLE_PROFILING:
                    profile_times['convert'].append(time.time() - t0)

                # Detect ArUco markers
                t0 = time.time()
                corners, ids, _ = detector.detectMarkers(gray)
                if ENABLE_PROFILING:
                    profile_times['detect'].append(time.time() - t0)

                poses_this_frame = []
                draw_start = time.time()
                
                if ids is not None:
                    # Process poses
                    pose_start = time.time()
                    for i in range(len(ids)):
                        marker_id = ids[i][0]
                        if marker_id in [0, 1]:
                            _, rvec, tvec = cv2.solvePnP(obj_points, corners[i][0], cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                            
                            # Store pose data
                            poses_this_frame.append({
                                'id': marker_id,
                                'rvec': rvec,
                                'tvec': tvec
                            })
                            
                            # Draw on frame if visualizing
                            if visualize:
                                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                                cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, marker_size_m)
                    
                    if ENABLE_PROFILING:
                        profile_times['pose'].append(time.time() - pose_start)
                        profile_times['draw'].append(time.time() - draw_start)

                # Send poses to main process (non-blocking, drop if queue full)
                if poses_this_frame:
                    t0 = time.time()
                    try:
                        pose_queue.put_nowait(poses_this_frame)
                    except:
                        pass  # Queue full, drop old data
                    if ENABLE_PROFILING:
                        profile_times['queue_pose'].append(time.time() - t0)
                
                # Send frame to main process for visualization (if needed)
                # Skip frames to reduce queue overhead (send every 2nd frame)
                if visualize and frame_queue is not None and frame_count % 2 == 0:
                    t0 = time.time()
                    try:
                        # Copy is needed for multiprocessing, but only if queue has space
                        frame_queue.put_nowait(frame.copy())
                    except:
                        pass  # Queue full, drop old frame
                    if ENABLE_PROFILING:
                        profile_times['queue_frame'].append(time.time() - t0)
                
                if ENABLE_PROFILING:
                    profile_times['total'].append(time.time() - frame_start)
                    frame_count += 1
                    
                    # Print profiling every 5 frames
                    if frame_count % 5 == 0 and profile_queue is not None:
                        stats = {}
                        for key, times in profile_times.items():
                            if times:
                                stats[key] = {
                                    'avg_ms': np.mean(times) * 1000,
                                    'max_ms': np.max(times) * 1000,
                                    'min_ms': np.min(times) * 1000
                                }
                        try:
                            profile_queue.put_nowait(stats)
                        except:
                            pass
                
                # Check stop event more frequently
                if stop_event.is_set():
                    break
            except KeyboardInterrupt:
                # Handle interrupt in worker process
                stop_event.set()
                break
                    
    except Exception as e:
        print(f"Vision process error: {e}")
    finally:
        cap.release()
        print("Vision process stopped.")

def stream_aruco_poses(camera_id=0, marker_size_m=0.015, dictionary_type=cv2.aruco.DICT_4X4_50, visualize=True):
    """
    Multiprocessing generator. 
    - Process 1: Captures camera, detects, and sends poses.
    - Main Process: Yields the poses to your code and handles visualization.
    """
    
    # Queue to pass data from Vision Process -> Main Process
    # maxsize=1 ensures we always get the *freshest* data (dropping old frames if logic is slow)
    pose_queue = multiprocessing.Queue(maxsize=1)
    frame_queue = multiprocessing.Queue(maxsize=1) if visualize else None
    profile_queue = multiprocessing.Queue(maxsize=1) if ENABLE_PROFILING else None
    stop_event = multiprocessing.Event()

    # Start the process (pass all necessary parameters)
    p = multiprocessing.Process(
        target=vision_worker,
        args=(camera_id, marker_size_m, dictionary_type, visualize, pose_queue, frame_queue, stop_event, profile_queue),
        daemon=True
    )
    p.start()

    # Main Generator Loop
    main_profile_times = {
        'get_frame': deque(maxlen=PROFILE_HISTORY_SIZE),
        'resize': deque(maxlen=PROFILE_HISTORY_SIZE),
        'imshow': deque(maxlen=PROFILE_HISTORY_SIZE),
        'get_pose': deque(maxlen=PROFILE_HISTORY_SIZE),
        'total': deque(maxlen=PROFILE_HISTORY_SIZE)
    }
    main_frame_count = 0
    
    try:
        while not stop_event.is_set():
            loop_start = time.time()
            
            # Handle visualization in main process (OpenCV windows need to be in main process)
            if visualize and frame_queue is not None:
                try:
                    t0 = time.time()
                    frame = frame_queue.get_nowait()
                    if ENABLE_PROFILING:
                        main_profile_times['get_frame'].append(time.time() - t0)
                    
                    # Resize frame for display (processing still uses full resolution)
                    t0 = time.time()
                    display_frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
                    if ENABLE_PROFILING:
                        main_profile_times['resize'].append(time.time() - t0)
                    
                    t0 = time.time()
                    cv2.imshow("ArUco Stream (Multiprocessing)", display_frame)
                    if ENABLE_PROFILING:
                        main_profile_times['imshow'].append(time.time() - t0)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        stop_event.set()
                        break
                except:
                    pass  # No frame available
            
            # Get pose data from the process
            try:
                t0 = time.time()
                poses = pose_queue.get(timeout=0.1)
                if ENABLE_PROFILING:
                    main_profile_times['get_pose'].append(time.time() - t0)
                # Yield each pose
                for pose in poses:
                    yield pose['id'], pose['rvec'], pose['tvec']
            except:
                continue  # No pose data available
            
            # Check for profiling data from worker
            if ENABLE_PROFILING and profile_queue is not None:
                try:
                    worker_stats = profile_queue.get_nowait()
                    print("\n=== WORKER PROCESS PROFILE (ms) ===")
                    for key, stats in worker_stats.items():
                        print(f"  {key:15s}: avg={stats['avg_ms']:6.2f}, max={stats['max_ms']:6.2f}, min={stats['min_ms']:6.2f}")
                    print(f"  Estimated FPS: {1000.0 / worker_stats['total']['avg_ms']:.1f}")
                    print("=" * 40)
                except:
                    pass
            
            if ENABLE_PROFILING:
                main_profile_times['total'].append(time.time() - loop_start)
                main_frame_count += 1
                
                if main_frame_count % 5 == 0:
                    print("\n=== MAIN PROCESS PROFILE (ms) ===")
                    for key, times in main_profile_times.items():
                        if times:
                            avg_ms = np.mean(times) * 1000
                            max_ms = np.max(times) * 1000
                            min_ms = np.min(times) * 1000
                            print(f"  {key:15s}: avg={avg_ms:6.2f}, max={max_ms:6.2f}, min={min_ms:6.2f}")
                    total_avg_ms = np.mean(main_profile_times['total']) * 1000
                    print(f"  Estimated FPS: {1000.0 / total_avg_ms:.1f}")
                    print("=" * 40)
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
    finally:
        # Cleanup
        stop_event.set()  # Ensure stop event is set
        
        # Clear queues to unblock any waiting operations
        try:
            while not pose_queue.empty():
                pose_queue.get_nowait()
        except:
            pass
        if frame_queue is not None:
            try:
                while not frame_queue.empty():
                    frame_queue.get_nowait()
            except:
                pass
        
        if visualize:
            cv2.destroyAllWindows()
        
        # Wait for process to finish, then terminate if needed
        p.join(timeout=1.0)
        if p.is_alive():
            print("Process not responding, terminating...")
            p.terminate()
            p.join(timeout=0.5)
        if p.is_alive():
            print("Force killing process...")
            p.kill()
            p.join()
        
        print("Cleanup complete.")

if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    multiprocessing.set_start_method('spawn', force=True)
    
    print("Starting multiprocessing detection...")
    print("Press Ctrl+C to stop")
    
    try:
        # NOTE: camera_id=0 is usually the webcam. Change to 1 if using external USB.
        for m_id, rvec, tvec in stream_aruco_poses(camera_id=1, visualize=True):
            
            # --- SIMULATING HEAVY LOGIC ---
            # Even if we sleep here, the CV window will stay smooth!
            print(f"Processing ID: {m_id} | Z: {tvec[2][0]:.4f}")
            # time.sleep(0.5) # Uncomment this to test that the UI doesn't freeze
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Exiting...")