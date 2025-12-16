# GStreamer Installation and Debugging Guide

To use GStreamer with Python (like Cheese does), you need to install PyGObject.

## Installation

### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0
```

### For conda environments:
You may need to install system packages or use conda-forge:
```bash
conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good
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

---

## Debugging and Troubleshooting

### Check Available Camera Devices

First, verify which video devices are available on your system:

```bash
# List all video devices
ls -la /dev/video*

# Get detailed information about devices
v4l2-ctl --list-devices
```

**Example output:**
```
Dummy video device (0x0000) (platform:v4l2loopback-000):
	/dev/video0

UC70: UC70 (usb-0000:0a:00.0-3):
	/dev/video2
	/dev/video3
```

### Check if Device is in Use

If you get "device busy" errors, check what's using the camera:

```bash
# Check what process is using the device
lsof /dev/video2

# Or use fuser
fuser /dev/video2
```

### Using the Pipeline Status Checker

We provide a utility script to check and manage GStreamer pipelines:

#### Check Pipeline Status
```bash
python check_gstreamer_pipeline.py
```

This will show:
- Current pipeline state (PLAYING, PAUSED, NULL, etc.)
- Whether the pipeline is running
- Any error messages
- The device being used

#### Restart a Failed Pipeline
```bash
python check_gstreamer_pipeline.py --restart
```

This will restart the pipeline if it exists but is not running.

#### Start a New Pipeline
```bash
python check_gstreamer_pipeline.py --start
```

#### Test Pipeline with Frame Reads
```bash
python check_gstreamer_pipeline.py --test
```

This will read a few frames to verify the pipeline is working correctly.

#### Use a Specific Device
```bash
python check_gstreamer_pipeline.py --device /dev/video2 --start --test
```

### Common Issues and Solutions

#### Issue: "Failed to start GStreamer pipeline"

**Possible causes:**
1. **Wrong device path**: The configured `camera_id` maps to a device that doesn't exist
   - **Solution**: Check available devices with `ls -la /dev/video*` and update `robot_config.camera_id` in `robot_control/robot_config.py`
   - Or use `--device` flag: `python check_gstreamer_pipeline.py --device /dev/video2`

2. **Device is busy**: Another process is using the camera
   - **Solution**: Check with `lsof /dev/video2` and close the process, or restart your system

3. **GStreamer not installed**: PyGObject bindings are missing
   - **Solution**: Install with `conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good`

4. **Camera permissions**: User doesn't have access to the device
   - **Solution**: Add user to `video` group: `sudo usermod -a -G video $USER` (requires logout/login)

#### Issue: "GStreamer not available"

**Solution**: Install GStreamer Python bindings:
```bash
conda install -c conda-forge pygobject gstreamer gst-plugins-base gst-plugins-good
```

Or on Ubuntu:
```bash
sudo apt-get install python3-gi python3-gi-cairo gir1.2-gstreamer-1.0
```

#### Issue: Pipeline starts but no frames are received

**Possible causes:**
1. Camera format mismatch: The camera may not support the requested format/resolution
   - **Solution**: Try different resolutions or check camera capabilities with `v4l2-ctl --device=/dev/video2 --list-formats-ext`

2. Pipeline state issue: Pipeline may be in wrong state
   - **Solution**: Restart with `python check_gstreamer_pipeline.py --restart`

### Programmatic Status Checking

You can also check pipeline status programmatically:

```python
from camera import GStreamerCamera

camera = GStreamerCamera(device='/dev/video2', width=1920, height=1080, fps=30)

# Get detailed status
status = camera.get_status()
print(f"State: {status['state']}")
print(f"Running: {status['is_running']}")

# Quick check
if camera.is_running():
    print("Pipeline is running!")

# Restart if needed
if not camera.is_running():
    camera.restart()
```

### Configuration

The default camera device is configured in `robot_control/robot_config.py`:

```python
camera_id: int = 2  # Maps to /dev/video2
camera_width: int = 1920
camera_height: int = 1080
camera_fps: int = 30
```

You can override these when running scripts:
```bash
python wx200_robot_collect_demo.py --camera-id 2
```

### Quick Diagnostic Checklist

When troubleshooting GStreamer issues, check:

1. ✅ GStreamer is installed: `python -c "import gi; gi.require_version('Gst', '1.0')"`
2. ✅ Camera device exists: `ls -la /dev/video*`
3. ✅ Device is not in use: `lsof /dev/video2`
4. ✅ Correct device in config: Check `robot_config.camera_id`
5. ✅ Pipeline status: `python check_gstreamer_pipeline.py`
6. ✅ Test pipeline: `python check_gstreamer_pipeline.py --test`

