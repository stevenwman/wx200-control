#!/usr/bin/env python3
"""
Simple script to profile camera read performance.
Tests different methods and settings to find the fastest approach.
"""

import cv2
import time
import numpy as np
from collections import deque

def test_camera_read(camera_id=1, num_frames=100, method='read'):
    """
    Test camera read performance with different methods.
    
    Methods:
    - 'read': Standard cv2.read()
    - 'grab_retrieve': Use grab() + retrieve() separately
    - 'grab_retrieve_flush': Use grab() multiple times to flush buffer
    """
    print(f"\n=== Testing method: {method} ===")
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return None
    
    # Set basic properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Resolution: {w}x{h}, Reported FPS: {fps}")
    
    # Warm up - read a few frames
    print(f"  Warming up (5 frames)...")
    for _ in range(5):
        cap.read()
    
    # Profile
    print(f"  Profiling {num_frames} frames...")
    times = []
    start_time = time.time()
    
    for i in range(num_frames):
        # Show progress every frame after first 3, with time estimate
        if i >= 3:
            elapsed = time.time() - start_time
            avg_time_per_frame = elapsed / (i + 1)
            remaining = avg_time_per_frame * (num_frames - i - 1)
            progress = ((i + 1) / num_frames) * 100
            print(f"    Frame {i+1}/{num_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
        elif i == 0:
            print(f"    Starting... (frame 1/{num_frames})", end='\r', flush=True)
        
        frame_start = time.time()
        
        if method == 'read':
            ret, frame = cap.read()
        elif method == 'grab_retrieve':
            if cap.grab():
                ret, frame = cap.retrieve()
            else:
                ret = False
        elif method == 'grab_retrieve_flush':
            # Flush buffer by grabbing multiple times
            for _ in range(2):
                if not cap.grab():
                    ret = False
                    break
            else:
                ret, frame = cap.retrieve()
        else:
            ret, frame = cap.read()
        
        if not ret:
            print(f"\n  Failed to read frame {i}")
            break
        
        elapsed = time.time() - frame_start
        times.append(elapsed)
    
    print()  # New line after progress
    
    total_time = time.time() - start_time
    cap.release()
    
    if times:
        times_ms = np.array(times) * 1000
        stats = {
            'avg_ms': np.mean(times_ms),
            'min_ms': np.min(times_ms),
            'max_ms': np.max(times_ms),
            'std_ms': np.std(times_ms),
            'median_ms': np.median(times_ms),
            'p95_ms': np.percentile(times_ms, 95),
            'p99_ms': np.percentile(times_ms, 99),
            'total_fps': num_frames / total_time,
            'avg_fps': 1000.0 / np.mean(times_ms)
        }
        return stats
    return None


def test_different_backends(camera_id=1):
    """Test different OpenCV backends."""
    backends = [
        (cv2.CAP_ANY, "ANY"),
        (cv2.CAP_V4L2, "V4L2"),
    ]
    
    # Check if GStreamer is available
    try:
        # Try to create a test capture with GStreamer
        test_cap = cv2.VideoCapture(f"v4l2src device=/dev/video{camera_id} ! video/x-raw,format=MJPG,width=1920,height=1080,framerate=30/1 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)
        if test_cap.isOpened():
            test_cap.release()
            backends.append((cv2.CAP_GSTREAMER, "GSTREAMER"))
            print("  GStreamer backend available!")
        else:
            print("  GStreamer backend not available (camera didn't open)")
    except Exception as e:
        print(f"  GStreamer backend not available: {e}")
    
    results = {}
    for idx, (backend, name) in enumerate(backends, 1):
        print(f"\n[{idx}/{len(backends)}] Testing backend: {name}")
        print(f"\n{'='*50}")
        print(f"Testing backend: {name}")
        print(f"{'='*50}")
        
        cap = cv2.VideoCapture(camera_id, backend)
        if not cap.isOpened():
            print(f"  Backend {name} failed to open camera")
            continue
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Resolution: {w}x{h}, FPS: {fps}")
        
        # Test read speed
        num_test_frames = 30  # Reduced from 50
        print(f"  Testing read speed ({num_test_frames} frames)...")
        times = []
        start_time = time.time()
        for i in range(num_test_frames):
            # Show progress every frame after first 3, with time estimate
            if i >= 3:
                elapsed = time.time() - start_time
                avg_time_per_frame = elapsed / (i + 1)
                remaining = avg_time_per_frame * (num_test_frames - i - 1)
                progress = ((i + 1) / num_test_frames) * 100
                print(f"    Frame {i+1}/{num_test_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
            elif i == 0:
                print(f"    Starting... (frame 1/{num_test_frames})", end='\r', flush=True)
            
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            times.append(time.time() - t0)
        print()  # New line after progress
        
        cap.release()
        
        if times:
            times_ms = np.array(times) * 1000
            results[name] = {
                'avg_ms': np.mean(times_ms),
                'min_ms': np.min(times_ms),
                'max_ms': np.max(times_ms),
                'fps': 1000.0 / np.mean(times_ms)
            }
            print(f"  Avg read time: {results[name]['avg_ms']:.2f}ms ({results[name]['fps']:.1f} FPS)")
    
    return results


def test_different_formats(camera_id=1):
    """Test different pixel formats."""
    formats = [
        ('MJPEG', cv2.VideoWriter_fourcc('M','J','P','G')),
        ('YUYV', cv2.VideoWriter_fourcc('Y','U','Y','V')),
    ]
    
    results = {}
    for idx, (format_name, fourcc) in enumerate(formats, 1):
        print(f"\n[{idx}/{len(formats)}] Testing format: {format_name}")
        print(f"\n{'='*50}")
        print(f"Testing format: {format_name}")
        print(f"{'='*50}")
        
        cap = cv2.VideoCapture(camera_id, cv2.CAP_ANY)
        if not cap.isOpened():
            continue
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 60)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Try to set format
        if cap.set(cv2.CAP_PROP_FOURCC, fourcc):
            actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
            print(f"  Set format to {format_name} (got: {actual_str})")
        else:
            print(f"  Failed to set format to {format_name}")
            cap.release()
            continue
        
        # Warm up
        print(f"  Warming up (5 frames)...")
        for _ in range(5):
            cap.read()
        
        # Test read speed
        num_test_frames = 20  # Reduced from 30
        print(f"  Testing read speed ({num_test_frames} frames)...")
        times = []
        start_time = time.time()
        for i in range(num_test_frames):
            # Show progress every frame after first 3, with time estimate
            if i >= 3:
                elapsed = time.time() - start_time
                avg_time_per_frame = elapsed / (i + 1)
                remaining = avg_time_per_frame * (num_test_frames - i - 1)
                progress = ((i + 1) / num_test_frames) * 100
                print(f"    Frame {i+1}/{num_test_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
            elif i == 0:
                print(f"    Starting... (frame 1/{num_test_frames})", end='\r', flush=True)
            
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            times.append(time.time() - t0)
        print()  # New line after progress
        
        cap.release()
        
        if times:
            times_ms = np.array(times) * 1000
            results[format_name] = {
                'avg_ms': np.mean(times_ms),
                'min_ms': np.min(times_ms),
                'max_ms': np.max(times_ms),
                'fps': 1000.0 / np.mean(times_ms)
            }
            print(f"  Avg read time: {results[format_name]['avg_ms']:.2f}ms ({results[format_name]['fps']:.1f} FPS)")
    
    return results


def main():
    camera_id = 1
    
    print("="*60)
    print("Camera Read Performance Profiling")
    print("="*60)
    print("Estimated time: ~2-3 minutes")
    print("="*60)
    
    overall_start = time.time()
    
    # Test 1: Different backends
    print("\n### TEST 1: Backend Comparison (2 backends) ###")
    print("This will test 2 backends × 30 frames each")
    test1_start = time.time()
    backend_results = test_different_backends(camera_id)
    print(f"  ✓ Test 1 completed in {time.time() - test1_start:.1f}s")
    
    # Test 2: Different formats
    print("\n### TEST 2: Format Comparison (2 formats) ###")
    print("This will test 2 formats × 20 frames each")
    test2_start = time.time()
    format_results = test_different_formats(camera_id)
    print(f"  ✓ Test 2 completed in {time.time() - test2_start:.1f}s")
    
    # Test 3: Different read methods
    print("\n### TEST 3: Read Method Comparison (3 methods) ###")
    print("This will test 3 methods × 30 frames each")
    methods = ['read', 'grab_retrieve', 'grab_retrieve_flush']
    method_results = {}
    test3_start = time.time()
    
    for idx, method in enumerate(methods, 1):
        print(f"\n[{idx}/{len(methods)}] Testing method: {method}")
        result = test_camera_read(camera_id, num_frames=30, method=method)  # Reduced to 30 frames
        if result:
            method_results[method] = result
            print(f"  ✓ Completed: {result['avg_ms']:.2f}ms avg, {result['avg_fps']:.1f} FPS")
    
    print(f"  ✓ Test 3 completed in {time.time() - test3_start:.1f}s")
    print(f"\n  Total time: {time.time() - overall_start:.1f}s")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if backend_results:
        print("\nBackend Results:")
        for name, stats in backend_results.items():
            print(f"  {name:10s}: {stats['avg_ms']:6.2f}ms avg, {stats['fps']:5.1f} FPS")
    
    if format_results:
        print("\nFormat Results:")
        for name, stats in format_results.items():
            print(f"  {name:10s}: {stats['avg_ms']:6.2f}ms avg, {stats['fps']:5.1f} FPS")
    
    if method_results:
        print("\nMethod Results:")
        for name, stats in method_results.items():
            print(f"  {name:20s}: {stats['avg_ms']:6.2f}ms avg, {stats['avg_fps']:5.1f} FPS")
            print(f"  {'':20s}  min={stats['min_ms']:6.2f}ms, max={stats['max_ms']:6.2f}ms, p95={stats['p95_ms']:6.2f}ms")


if __name__ == "__main__":
    main()
