#!/usr/bin/env python3
"""
Camera capture using GStreamer directly (via PyGObject).
This is what Cheese uses, so it should be faster than OpenCV.

Installation:
  sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0
  # OR for conda:
  conda install -c conda-forge pygobject
"""

try:
    import os
    # Try to use system GStreamer introspection files if conda doesn't have them
    system_typelib_paths = [
        '/usr/lib/x86_64-linux-gnu/girepository-1.0',
        '/usr/lib/girepository-1.0',
        '/usr/share/gir-1.0'
    ]
    for path in system_typelib_paths:
        if os.path.exists(path) and 'GI_TYPELIB_PATH' not in os.environ:
            current_path = os.environ.get('GI_TYPELIB_PATH', '')
            os.environ['GI_TYPELIB_PATH'] = f"{path}:{current_path}" if current_path else path
    
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    GSTREAMER_AVAILABLE = True
except (ImportError, ValueError) as e:
    print("ERROR: PyGObject/GStreamer not available!")
    print(f"Error: {e}")
    print("\nInstallation options:")
    print("  1. System packages: sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0")
    print("  2. Conda: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good")
    print("  3. Set GI_TYPELIB_PATH to point to system GStreamer typelib files")
    GSTREAMER_AVAILABLE = False

import numpy as np
import cv2
import time
from collections import deque

class GStreamerCamera:
    """GStreamer-based camera capture."""
    
    def __init__(self, device='/dev/video1', width=1920, height=1080, fps=30):
        Gst.init(None)
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.appsink = None
        self.loop = None
        self.frame = None
        self.frame_available = False
        
    def _on_new_sample(self, appsink):
        """Callback when a new frame is available."""
        sample = appsink.emit('pull-sample')
        if sample:
            buffer = sample.get_buffer()
            
            # Get frame data
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if success:
                # Create numpy array from buffer
                arr = np.ndarray(
                    (self.height, self.width, 3),
                    buffer=map_info.data,
                    dtype=np.uint8
                )
                # Copy to avoid issues with buffer being unmapped
                self.frame = arr.copy()
                self.frame_available = True
                buffer.unmap(map_info)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.ERROR
    
    def start(self):
        """Start the GStreamer pipeline."""
        # Build pipeline string
        pipeline_str = (
            f"v4l2src device={self.device} ! "
            f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink name=sink emit-signals=True max-buffers=1 drop=True"
        )
        
        print(f"GStreamer pipeline: {pipeline_str}")
        
        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get appsink
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self._on_new_sample)
        
        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")
        
        print("GStreamer pipeline started")
        
    def read(self):
        """Read a frame (non-blocking)."""
        if self.frame_available:
            frame = self.frame.copy()
            self.frame_available = False
            return True, frame
        return False, None
    
    def release(self):
        """Stop and cleanup."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None


def test_gstreamer_camera():
    """Test GStreamer camera capture performance."""
    print("="*60)
    print("Testing GStreamer Camera Capture")
    print("="*60)
    
    camera = GStreamerCamera(device='/dev/video1', width=1920, height=1080, fps=30)
    
    try:
        camera.start()
        
        # Warm up
        print("Warming up (5 frames)...")
        warmup_count = 0
        start_time = time.time()
        while warmup_count < 5:
            ret, frame = camera.read()
            if ret:
                warmup_count += 1
            time.sleep(0.01)  # Small delay
        
        # Test read speed - measure time between NEW frames
        num_frames = 30
        print(f"Testing read speed ({num_frames} frames)...")
        times = []
        start_time = time.time()
        last_frame_time = start_time
        frame_count = 0
        last_frame_id = None
        
        while frame_count < num_frames:
            ret, frame = camera.read()
            if ret:
                current_time = time.time()
                # Only count if this is a new frame (check by frame data hash or timing)
                # For simplicity, measure time between successful reads
                if frame_count > 0:  # Skip first frame timing
                    elapsed = current_time - last_frame_time
                    times.append(elapsed)
                
                last_frame_time = current_time
                frame_count += 1
                
                if frame_count >= 3:
                    elapsed_total = time.time() - start_time
                    avg_time_per_frame = elapsed_total / frame_count
                    remaining = avg_time_per_frame * (num_frames - frame_count)
                    progress = (frame_count / num_frames) * 100
                    print(f"  Frame {frame_count}/{num_frames} ({progress:.0f}%) - ~{remaining:.1f}s remaining", end='\r', flush=True)
            else:
                time.sleep(0.001)  # Small delay if no frame available
        
        print()  # New line
        
        if times:
            times_ms = np.array(times) * 1000
            avg_ms = np.mean(times_ms)
            fps = 1000.0 / avg_ms
            print(f"Result: {avg_ms:.2f}ms avg ({fps:.1f} FPS)")
            print(f"  Min: {np.min(times_ms):.2f}ms, Max: {np.max(times_ms):.2f}ms")
            return {'avg_ms': avg_ms, 'fps': fps}
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        camera.release()
    
    return None


if __name__ == "__main__":
    if not GSTREAMER_AVAILABLE:
        print("\nCannot run without PyGObject/GStreamer installed.")
        print("See INSTALL_GSTREAMER.md for installation instructions.")
        exit(1)
    
    result = test_gstreamer_camera()
    if result:
        print(f"\n✓ GStreamer achieved {result['fps']:.1f} FPS")
        print("  This should be faster than OpenCV!")
        print(f"  OpenCV with MJPEG gets ~15 FPS, camera supports 30 FPS")
