"""
GStreamer-based camera capture.
Achieves 30 FPS (vs 15 FPS with OpenCV).
"""

import sys
import os

# Setup GStreamer paths for conda environments
system_typelib_paths = [
    '/usr/lib/x86_64-linux-gnu/girepository-1.0',
    '/usr/lib/girepository-1.0',
    '/usr/share/gir-1.0'
]
for path in system_typelib_paths:
    if os.path.exists(path) and 'GI_TYPELIB_PATH' not in os.environ:
        current_path = os.environ.get('GI_TYPELIB_PATH', '')
        os.environ['GI_TYPELIB_PATH'] = f"{path}:{current_path}" if current_path else path

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
    GSTREAMER_AVAILABLE = True
except (ImportError, ValueError) as e:
    GSTREAMER_AVAILABLE = False
    GSTREAMER_ERROR = str(e)

import numpy as np


def is_gstreamer_available():
    """Check if GStreamer is available."""
    return GSTREAMER_AVAILABLE


class GStreamerCamera:
    """GStreamer-based camera capture for high-performance video capture."""
    
    def __init__(self, device='/dev/video1', width=1920, height=1080, fps=30):
        """
        Initialize GStreamer camera.
        
        Args:
            device: Camera device path (e.g., '/dev/video1')
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frame rate
        """
        if not GSTREAMER_AVAILABLE:
            raise RuntimeError(
                f"GStreamer not available: {GSTREAMER_ERROR}\n"
                "Install with: conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good"
            )
        
        Gst.init(None)
        self.device = device
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
        
        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get appsink
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.connect('new-sample', self._on_new_sample)
        
        # Start pipeline and wait for state change to complete
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(
                f"Failed to start GStreamer pipeline for device {self.device}. "
                f"Device may be busy or unavailable. Check with: lsof {self.device}"
            )
        
        # Wait for async state changes to complete (timeout after 5 seconds)
        if ret == Gst.StateChangeReturn.ASYNC:
            ret, state, pending = self.pipeline.get_state(timeout=5 * Gst.SECOND)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError(
                    f"Failed to start GStreamer pipeline for device {self.device}. "
                    f"Device may be busy or unavailable. Check with: lsof {self.device}"
                )
            elif ret == Gst.StateChangeReturn.ASYNC:
                # Still pending after timeout - this indicates the device may be busy
                raise RuntimeError(
                    f"Pipeline state change timed out for device {self.device}. "
                    f"Device may be busy. Current state: {state}, pending: {pending}. "
                    f"Check with: lsof {self.device}"
                )
            elif state != Gst.State.PLAYING:
                raise RuntimeError(
                    f"Pipeline did not reach PLAYING state for device {self.device}. "
                    f"Current state: {state}"
                )
        
    def read(self):
        """
        Read a frame (non-blocking).
        
        Returns:
            (success, frame): Tuple of (bool, numpy array or None)
        """
        if self.frame_available:
            frame = self.frame.copy()
            self.frame_available = False
            return True, frame
        return False, None
    
    def get_status(self):
        """
        Get the current status of the GStreamer pipeline.
        
        Returns:
            dict: Status information including:
                - state: Current pipeline state (string)
                - state_enum: GStreamer state enum
                - is_running: Boolean indicating if pipeline is playing
                - has_pipeline: Boolean indicating if pipeline exists
                - device: Camera device path
                - error: Error message if any
        """
        status = {
            'has_pipeline': self.pipeline is not None,
            'is_running': False,
            'state': 'NULL',
            'state_enum': None,
            'device': self.device,
            'error': None
        }
        
        if self.pipeline:
            try:
                ret, state, pending = self.pipeline.get_state(timeout=Gst.SECOND)
                status['state_enum'] = state
                
                if ret == Gst.StateChangeReturn.SUCCESS:
                    if state == Gst.State.PLAYING:
                        status['state'] = 'PLAYING'
                        status['is_running'] = True
                    elif state == Gst.State.PAUSED:
                        status['state'] = 'PAUSED'
                    elif state == Gst.State.READY:
                        status['state'] = 'READY'
                    elif state == Gst.State.NULL:
                        status['state'] = 'NULL'
                elif ret == Gst.StateChangeReturn.FAILURE:
                    status['state'] = 'FAILURE'
                    status['error'] = 'Pipeline state change failed'
                elif ret == Gst.StateChangeReturn.ASYNC:
                    status['state'] = f'ASYNC (pending: {pending})'
            except Exception as e:
                status['error'] = str(e)
        
        return status
    
    def is_running(self):
        """
        Quick check if pipeline is running.
        
        Returns:
            bool: True if pipeline is in PLAYING state, False otherwise
        """
        if not self.pipeline:
            return False
        try:
            ret, state, _ = self.pipeline.get_state(timeout=Gst.SECOND)
            return ret == Gst.StateChangeReturn.SUCCESS and state == Gst.State.PLAYING
        except Exception:
            return False
    
    def restart(self):
        """
        Restart the GStreamer pipeline.
        
        This stops the current pipeline (if running) and starts it again.
        Useful for recovering from errors or reinitializing after device issues.
        
        Returns:
            bool: True if restart was successful, False otherwise
        """
        try:
            # Stop current pipeline if it exists
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                # Wait for state change to complete
                self.pipeline.get_state(timeout=Gst.SECOND)
                self.pipeline = None
                self.appsink = None
                self.frame = None
                self.frame_available = False
            
            # Start fresh pipeline
            self.start()
            return True
        except Exception as e:
            print(f"Error restarting pipeline: {e}")
            return False
    
    def release(self):
        """Stop and cleanup."""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            # Wait for state change to complete
            try:
                self.pipeline.get_state(timeout=Gst.SECOND)
            except Exception:
                pass  # Ignore errors during cleanup
            self.pipeline = None
            self.appsink = None
            self.frame = None
            self.frame_available = False
