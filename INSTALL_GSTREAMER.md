# Installing GStreamer Python Bindings

To use GStreamer with Python (like Cheese does), you need to install PyGObject.

## Installation

On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0
```

For conda environments, you may need to install system packages or use conda-forge:
```bash
conda install -c conda-forge pygobject
```

## Verify Installation

```bash
python -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; print('GStreamer version:', Gst.version_string())"
```

## Why GStreamer?

- **Cheese uses it**: Cheese (the camera app) uses GStreamer, which is why it's so smooth
- **Better performance**: GStreamer can achieve closer to the camera's native 30 FPS
- **More control**: Direct access to camera capabilities
- **OpenCV limitation**: OpenCV's GStreamer backend wrapper may not be optimal

## Testing

After installation, test with:
```bash
python camera_gstreamer.py
```

This should show better FPS than OpenCV (closer to 30 FPS instead of 15 FPS).
