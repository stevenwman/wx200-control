# GStreamer Fix Summary

## Problem
Your conda environment had incomplete GStreamer plugins (missing `v4l2src` needed for camera access), causing camera initialization to fail with:
```
gst_parse_error: no element "v4l2src"
```

## Solution Applied

### 1. Removed Conda's Incomplete GStreamer
```bash
conda remove gstreamer gst-plugins-base gst-plugins-good gstreamer-orc
```

### 2. Created Environment Fix Module
**File:** `fix_gstreamer_env.py`

This module automatically configures Python to use system GStreamer (which has all plugins) instead of conda's incomplete installation.

**What it does:**
- Points GStreamer plugin path to system location (`/usr/lib/x86_64-linux-gnu/gstreamer-1.0`)
- Adds system library paths so Python can load GStreamer libraries
- Sets up GObject introspection to work with system GStreamer

### 3. Updated Scripts to Use Fix Module

**Modified files:**
- `compact_code/wx200_robot_collect_demo_encoders_compact.py` - Added import at top
- `compact_gym/camera.py` - Added import at top
- `check_gstreamer_pipeline.py` - Added import at top

**Pattern used:**
```python
import fix_gstreamer_env  # Must be FIRST, before cv2/GStreamer imports
import cv2
import numpy as np
# ... rest of imports
```

### 4. Fixed Camera Device ID
Changed `robot_config.camera_id` from 2 to 1:
- `/dev/video1` = UC70 camera (capture device) ✓
- `/dev/video2` = UC70 metadata device ✗

## How to Use

### Running Your Scripts
You can now run scripts **directly** without the wrapper:

```bash
# Data collection with camera and encoders
python compact_code/wx200_robot_collect_demo_encoders_compact.py

# Check GStreamer pipeline
python check_gstreamer_pipeline.py --test

# Other scripts that use camera
python your_script.py
```

### For New Scripts
If you create new scripts that use the camera, add this at the very top:

```python
import fix_gstreamer_env  # Must be FIRST import
import cv2
# ... rest of your code
```

## Testing

Verify GStreamer is working:
```bash
python check_gstreamer_pipeline.py --test
```

Should show:
```
✓ Pipeline test passed!
```

## Files Created/Modified

**New files:**
- `fix_gstreamer_env.py` - Environment configuration module
- `run_with_system_gstreamer.sh` - Wrapper script (no longer needed, but kept as backup)

**Modified files:**
- `compact_code/wx200_robot_collect_demo_encoders_compact.py`
- `compact_gym/camera.py`
- `check_gstreamer_pipeline.py`
- `robot_control/robot_config.py` (camera_id: 2 → 1)

## Technical Details

**Why this works:**
- System GStreamer (`/usr/lib/x86_64-linux-gnu/gstreamer-1.0`) has ALL plugins including `v4l2src`
- Conda's PyGObject works fine with system GStreamer when paths are configured correctly
- Environment variables must be set BEFORE GStreamer is initialized (hence import first)

**Why conda's GStreamer was incomplete:**
- Conda's GStreamer packages don't include all plugins (especially platform-specific ones like `v4l2src` for Linux)
- Building GStreamer plugins requires system dependencies that conda packages don't always include
- System package managers (apt) provide complete GStreamer installations

## Troubleshooting

If you still get GStreamer errors:

1. **Check system GStreamer is installed:**
   ```bash
   gst-inspect-1.0 v4l2src
   ```

2. **Verify conda GStreamer is removed:**
   ```bash
   conda list | grep gstreamer
   # Should show nothing
   ```

3. **Test the fix module:**
   ```bash
   python fix_gstreamer_env.py
   ```

4. **Check camera device:**
   ```bash
   v4l2-ctl --list-devices
   v4l2-ctl --device=/dev/video1 --list-formats-ext
   ```
