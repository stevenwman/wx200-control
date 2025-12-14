"""
Camera capture modules.
Supports GStreamer for high-performance camera access.
"""

from .gstreamer_camera import GStreamerCamera, is_gstreamer_available

__all__ = ['GStreamerCamera', 'is_gstreamer_available']
