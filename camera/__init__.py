"""
Camera capture modules.
Supports GStreamer for high-performance camera access, with OpenCV fallback.
"""

from .gstreamer_camera import GStreamerCamera, is_gstreamer_available
from .opencv_camera import OpenCVCamera

def Camera(device=0, width=1920, height=1080, fps=30):
    """Factory function to return the best available camera implementation."""
    if is_gstreamer_available():
        # GStreamer expects device string like '/dev/video1'
        dev_str = f"/dev/video{device}" if isinstance(device, int) else device
        try:
            return GStreamerCamera(device=dev_str, width=width, height=height, fps=fps)
        except Exception as e:
            print(f"Warning: GStreamer initialization failed: {e}. Falling back to OpenCV.")
            pass # Fallback
            
    # Fallback to OpenCV
    return OpenCVCamera(device=device, width=width, height=height, fps=fps)

__all__ = ['GStreamerCamera', 'is_gstreamer_available', 'OpenCVCamera', 'Camera']
