#!/usr/bin/env python3
"""
Utility script to check GStreamer pipeline status and restart if necessary.

Usage:
    python check_gstreamer_pipeline.py                    # Check status
    python check_gstreamer_pipeline.py --restart           # Check and restart if needed
    python check_gstreamer_pipeline.py --device /dev/video1  # Specify device
    python check_gstreamer_pipeline.py --test              # Test pipeline with a few frames
"""

import fix_gstreamer_env  # Must be imported BEFORE camera module
import argparse
import sys
import time
from camera import GStreamerCamera, is_gstreamer_available
from robot_control.robot_config import robot_config


def print_status(status):
    """Print pipeline status in a readable format."""
    print("\n" + "="*60)
    print("GStreamer Pipeline Status")
    print("="*60)
    print(f"Device:        {status['device']}")
    print(f"Pipeline Exists: {status['has_pipeline']}")
    
    if status['has_pipeline']:
        print(f"State:         {status['state']}")
        print(f"Running:       {'✓ YES' if status['is_running'] else '✗ NO'}")
        if status['error']:
            print(f"Error:         {status['error']}")
    else:
        print("State:         No pipeline initialized")
    
    print("="*60 + "\n")


def check_pipeline_status(device=None, width=None, height=None, fps=None):
    """
    Check the status of a GStreamer pipeline.
    
    Args:
        device: Camera device path (defaults to config)
        width: Frame width (defaults to config)
        height: Frame height (defaults to config)
        fps: Frame rate (defaults to config)
    
    Returns:
        tuple: (camera_instance, status_dict) or (None, None) if failed
    """
    if not is_gstreamer_available():
        print("❌ GStreamer is not available!")
        print("   Install with: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good")
        return None, None
    
    # Use config defaults if not specified
    device = device or f"/dev/video{robot_config.camera_id}"
    width = width or robot_config.camera_width
    height = height or robot_config.camera_height
    fps = fps or robot_config.camera_fps
    
    try:
        camera = GStreamerCamera(device=device, width=width, height=height, fps=fps)
        status = camera.get_status()
        return camera, status
    except Exception as e:
        print(f"❌ Error creating camera instance: {e}")
        return None, None


def test_pipeline(camera, num_frames=10):
    """
    Test the pipeline by reading a few frames.
    
    Args:
        camera: GStreamerCamera instance
        num_frames: Number of frames to read
    
    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\nTesting pipeline: Reading {num_frames} frames...")
    
    if not camera.is_running():
        print("❌ Pipeline is not running. Cannot test.")
        return False
    
    success_count = 0
    for i in range(num_frames):
        ret, frame = camera.read()
        if ret:
            success_count += 1
            print(f"  Frame {i+1}/{num_frames}: ✓ ({frame.shape if frame is not None else 'None'})")
        else:
            print(f"  Frame {i+1}/{num_frames}: ✗ No frame available")
        time.sleep(0.1)  # Small delay between reads
    
    success_rate = success_count / num_frames
    print(f"\nTest Results: {success_count}/{num_frames} frames successful ({success_rate*100:.1f}%)")
    
    return success_rate > 0.5  # Consider successful if >50% frames work


def main():
    parser = argparse.ArgumentParser(
        description='Check and manage GStreamer pipeline status',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check status
  python check_gstreamer_pipeline.py
  
  # Check and restart if not running
  python check_gstreamer_pipeline.py --restart
  
  # Test pipeline with frame reads
  python check_gstreamer_pipeline.py --test
  
  # Use specific device
  python check_gstreamer_pipeline.py --device /dev/video0
        """
    )
    
    parser.add_argument('--device', type=str, help='Camera device path (e.g., /dev/video1)')
    parser.add_argument('--width', type=int, help='Frame width')
    parser.add_argument('--height', type=int, help='Frame height')
    parser.add_argument('--fps', type=int, help='Frame rate')
    parser.add_argument('--restart', action='store_true', 
                       help='Restart pipeline if it exists but is not running')
    parser.add_argument('--start', action='store_true',
                       help='Start pipeline if it does not exist')
    parser.add_argument('--test', action='store_true',
                       help='Test pipeline by reading frames')
    parser.add_argument('--test-frames', type=int, default=10,
                       help='Number of frames to read during test (default: 10)')
    
    args = parser.parse_args()
    
    # Check status
    camera, status = check_pipeline_status(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps
    )
    
    if camera is None:
        sys.exit(1)
    
    print_status(status)
    
    # Handle restart
    if args.restart:
        if status['has_pipeline'] and not status['is_running']:
            print("🔄 Restarting pipeline...")
            if camera.restart():
                print("✓ Pipeline restarted successfully")
                # Get updated status
                status = camera.get_status()
                print_status(status)
            else:
                print("❌ Failed to restart pipeline")
                sys.exit(1)
        elif not status['has_pipeline']:
            print("ℹ️  No pipeline exists. Use --start to create one.")
        elif status['is_running']:
            print("ℹ️  Pipeline is already running. No restart needed.")
    
    # Handle start
    if args.start:
        if not status['has_pipeline'] or not status['is_running']:
            print("🔄 Starting pipeline...")
            try:
                camera.start()
                print("✓ Pipeline started successfully")
                status = camera.get_status()
                print_status(status)
            except Exception as e:
                print(f"❌ Failed to start pipeline: {e}")
                sys.exit(1)
        else:
            print("ℹ️  Pipeline is already running.")
    
    # Handle test
    if args.test:
        if not status['has_pipeline'] or not status['is_running']:
            print("⚠️  Pipeline is not running. Starting it first...")
            try:
                camera.start()
                time.sleep(0.5)  # Give it a moment to start
                status = camera.get_status()
            except Exception as e:
                print(f"❌ Failed to start pipeline: {e}")
                sys.exit(1)
        
        test_passed = test_pipeline(camera, num_frames=args.test_frames)
        if not test_passed:
            print("\n❌ Pipeline test failed. Consider restarting with --restart")
            sys.exit(1)
        else:
            print("\n✓ Pipeline test passed!")
    
    # Cleanup
    if camera:
        camera.release()


if __name__ == "__main__":
    main()

