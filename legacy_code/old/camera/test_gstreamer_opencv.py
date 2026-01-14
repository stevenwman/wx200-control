#!/usr/bin/env python3
"""
Test GStreamer backend in OpenCV vs default backend.
Cheese uses GStreamer, so this might be why it's faster.
"""

import cv2
import time
import numpy as np

def test_gstreamer_pipeline(camera_id=1, num_frames=30):
    """Test GStreamer pipeline directly."""
    print("\n=== Testing GStreamer Pipeline ===")
    
    # Try different GStreamer pipeline formats
    pipelines = [
        # Pipeline 1: MJPEG format
        (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=1920,height=1080,framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink"
        ),
        # Pipeline 2: Direct MJPEG with different caps
        (
            f"v4l2src device=/dev/video{camera_id} ! "
            "video/x-raw,format=MJPG,width=1920,height=1080,framerate=30/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "appsink"
        ),
        # Pipeline 3: Simpler pipeline
        (
            f"v4l2src device=/dev/video{camera_id} ! "
            "image/jpeg,width=1920,height=1080 ! "
            "jpegdec ! "
            "videoconvert ! "
            "appsink"
        ),
    ]
    
    for idx, pipeline in enumerate(pipelines, 1):
        print(f"\n  Trying pipeline {idx}:")
        print(f"  {pipeline}")
        
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"  ✓ Pipeline {idx} opened successfully!")
                break
            else:
                cap.release()
                print(f"  ✗ Pipeline {idx} failed to open")
                continue
        except Exception as e:
            print(f"  ✗ Pipeline {idx} error: {e}")
            continue
    else:
        print("\n  All GStreamer pipelines failed")
        return None
    
    try:
        # Warm up
        print("  Warming up (5 frames)...")
        for _ in range(5):
            ret, frame = cap.read()
            if not ret:
                print("  Failed to read during warmup")
                cap.release()
                return None
        
        # Test read speed
        print(f"  Testing read speed ({num_frames} frames)...")
        times = []
        start_time = time.time()
        
        for i in range(num_frames):
            if i >= 3:
                elapsed = time.time() - start_time
                avg_time_per_frame = elapsed / (i + 1)
                remaining = avg_time_per_frame * (num_frames - i - 1)
                progress = ((i + 1) / num_frames) * 100
                print(f"    Frame {i+1}/{num_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
            elif i == 0:
                print(f"    Starting... (frame 1/{num_frames})", end='\r', flush=True)
            
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print(f"\n  Failed to read frame {i}")
                break
            times.append(time.time() - frame_start)
        
        print()  # New line
        cap.release()
        
        if times:
            times_ms = np.array(times) * 1000
            avg_ms = np.mean(times_ms)
            fps = 1000.0 / avg_ms
            print(f"  Result: {avg_ms:.2f}ms avg ({fps:.1f} FPS)")
            return {'avg_ms': avg_ms, 'fps': fps}
        
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    return None


def test_opencv_default(camera_id=1, num_frames=30):
    """Test OpenCV default backend with MJPEG."""
    print("\n=== Testing OpenCV Default Backend (MJPEG) ===")
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("  Failed to open camera")
        return None
    
    # Set to MJPEG format
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Try to set MJPEG format
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    actual_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    print(f"  Resolution: {w}x{h}, FPS: {fps}, Format: {actual_str}")
    
    # Warm up
    print("  Warming up (5 frames)...")
    for _ in range(5):
        cap.read()
    
    # Test read speed
    print(f"  Testing read speed ({num_frames} frames)...")
    times = []
    start_time = time.time()
    
    for i in range(num_frames):
        if i >= 3:
            elapsed = time.time() - start_time
            avg_time_per_frame = elapsed / (i + 1)
            remaining = avg_time_per_frame * (num_frames - i - 1)
            progress = ((i + 1) / num_frames) * 100
            print(f"    Frame {i+1}/{num_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
        elif i == 0:
            print(f"    Starting... (frame 1/{num_frames})", end='\r', flush=True)
        
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print(f"\n  Failed to read frame {i}")
            break
        times.append(time.time() - frame_start)
    
    print()  # New line
    cap.release()
    
    if times:
        times_ms = np.array(times) * 1000
        avg_ms = np.mean(times_ms)
        fps = 1000.0 / avg_ms
        print(f"  Result: {avg_ms:.2f}ms avg ({fps:.1f} FPS)")
        return {'avg_ms': avg_ms, 'fps': fps}
    
    return None


def main():
    camera_id = 1
    
    print("="*60)
    print("GStreamer vs OpenCV Backend Comparison")
    print("="*60)
    print("Cheese uses GStreamer - let's see if it's faster!")
    print("="*60)
    
    # Check if GStreamer is available
    has_gstreamer = hasattr(cv2, 'CAP_GSTREAMER')
    print(f"\nOpenCV GStreamer support: {has_gstreamer}")
    
    # Test OpenCV default
    opencv_result = test_opencv_default(camera_id, num_frames=30)
    
    # Test GStreamer if available
    if has_gstreamer:
        gstreamer_result = test_gstreamer_pipeline(camera_id, num_frames=30)
    else:
        print("\n⚠ GStreamer backend not available in OpenCV")
        print("  You may need to rebuild OpenCV with GStreamer support")
        gstreamer_result = None
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if opencv_result:
        print(f"OpenCV Default (MJPEG): {opencv_result['avg_ms']:.2f}ms ({opencv_result['fps']:.1f} FPS)")
    
    if gstreamer_result:
        print(f"GStreamer Pipeline:      {gstreamer_result['avg_ms']:.2f}ms ({gstreamer_result['fps']:.1f} FPS)")
        if opencv_result:
            speedup = opencv_result['avg_ms'] / gstreamer_result['avg_ms']
            print(f"\nGStreamer is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than OpenCV default")
    else:
        print("\n💡 Tip: If GStreamer is not available, consider:")
        print("  1. Installing OpenCV with GStreamer support")
        print("  2. Using PyGObject (gi.repository.Gst) directly")
        print("  3. Using other libraries like picamera2 or v4l2-python")


if __name__ == "__main__":
    main()
