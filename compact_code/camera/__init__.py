"""
Camera capture modules.
Supports GStreamer for high-performance camera access, with OpenCV fallback.
"""

from .gstreamer_camera import GStreamerCamera, is_gstreamer_available
from .opencv_camera import OpenCVCamera
from .aruco_pose_estimator import ArUcoPoseEstimator, MARKER_SIZE, get_approx_camera_matrix

def Camera(device=0, width=1920, height=1080, fps=30):
    """
    Factory function to return GStreamer camera implementation.

    Raises:
        RuntimeError: If GStreamer is not available or initialization fails.
    """
    if not is_gstreamer_available():
        raise RuntimeError(
            "GStreamer is not available. This application requires GStreamer for camera access.\n"
            "Please install GStreamer and ensure fix_gstreamer_env.py is imported before camera initialization."
        )

    # GStreamer expects device string like '/dev/video1'
    dev_str = f"/dev/video{device}" if isinstance(device, int) else device
    try:
        camera = GStreamerCamera(device=dev_str, width=width, height=height, fps=fps)
        return camera
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize GStreamer camera on device {dev_str}: {e}\n"
            "GStreamer is available but camera initialization failed. Check camera device and permissions."
        ) from e

__all__ = ['GStreamerCamera', 'is_gstreamer_available', 'OpenCVCamera', 'Camera', 
           'ArUcoPoseEstimator', 'MARKER_SIZE', 'get_approx_camera_matrix']
