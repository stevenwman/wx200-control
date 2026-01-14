"""
Auto-configure environment to use system GStreamer with conda's PyGObject.

Import this module at the very top of your script (before any other imports)
to automatically fix GStreamer paths.

Usage:
    import fix_gstreamer_env  # This line must be FIRST
    import cv2
    import numpy as np
    # ... rest of your imports
"""
import os
import sys

# System GStreamer plugin paths
SYSTEM_GST_PLUGIN_PATH = '/usr/lib/x86_64-linux-gnu/gstreamer-1.0'
SYSTEM_LIB_PATH = '/usr/lib/x86_64-linux-gnu'
SYSTEM_GI_TYPELIB_PATH = '/usr/lib/x86_64-linux-gnu/girepository-1.0'

def setup_gstreamer_environment():
    """Configure environment to use system GStreamer instead of conda's incomplete installation."""

    # Set GStreamer plugin paths to system locations
    os.environ['GST_PLUGIN_SYSTEM_PATH'] = SYSTEM_GST_PLUGIN_PATH
    os.environ['GST_PLUGIN_PATH'] = SYSTEM_GST_PLUGIN_PATH

    # Prepend system library path (for GStreamer libraries)
    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    if SYSTEM_LIB_PATH not in ld_library_path:
        os.environ['LD_LIBRARY_PATH'] = f"{SYSTEM_LIB_PATH}:{ld_library_path}"

    # Use system GObject introspection
    gi_typelib_path = os.environ.get('GI_TYPELIB_PATH', '')
    if SYSTEM_GI_TYPELIB_PATH not in gi_typelib_path:
        os.environ['GI_TYPELIB_PATH'] = f"{SYSTEM_GI_TYPELIB_PATH}:{gi_typelib_path}"

    # Suppress GStreamer warnings about failed plugin loads from conda
    os.environ.setdefault('GST_DEBUG', '1')  # Only show critical errors

# Automatically run setup when module is imported
setup_gstreamer_environment()

# Verify setup
if __name__ == '__main__':
    print("GStreamer environment configured:")
    print(f"  GST_PLUGIN_SYSTEM_PATH: {os.environ.get('GST_PLUGIN_SYSTEM_PATH')}")
    print(f"  GST_PLUGIN_PATH: {os.environ.get('GST_PLUGIN_PATH')}")
    print(f"  LD_LIBRARY_PATH (first entry): {os.environ.get('LD_LIBRARY_PATH', '').split(':')[0]}")
    print(f"  GI_TYPELIB_PATH (first entry): {os.environ.get('GI_TYPELIB_PATH', '').split(':')[0]}")

    # Test GStreamer import
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        print(f"\n✓ GStreamer imported successfully: {Gst.version_string()}")

        # Test v4l2src availability
        v4l2src = Gst.ElementFactory.make('v4l2src', None)
        if v4l2src:
            print("✓ v4l2src plugin available")
        else:
            print("✗ v4l2src plugin NOT found")
    except Exception as e:
        print(f"\n✗ Failed to import GStreamer: {e}")
