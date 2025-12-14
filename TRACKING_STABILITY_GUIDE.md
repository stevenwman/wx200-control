# ArUco Tracking Stability Tuning Guide

This guide explains all the parameters you can tune to improve tracking stability.

## Quick Recommendations

### For Maximum Stability (Less Jitter)
```python
POSE_FILTER_ALPHA = 0.5  # More smoothing
POSE_FILTER_MAX_TRANSLATION_JUMP = 0.05  # Tighter filtering (5cm)
POSE_FILTER_MAX_ROTATION_JUMP = 0.3  # Tighter filtering (~17 degrees)
ROI_EXPAND = 150  # Larger search area
```

### For Responsiveness (Faster Updates)
```python
POSE_FILTER_ALPHA = 0.8  # Less smoothing
POSE_FILTER_MAX_TRANSLATION_JUMP = 0.15  # Allow faster movements
POSE_FILTER_MAX_ROTATION_JUMP = 0.7  # Allow faster rotations
ROI_EXPAND = 75  # Smaller search area (faster)
```

### For Noisy Environments
```python
POSE_FILTER_ALPHA = 0.6  # Moderate smoothing
POSE_FILTER_MAX_TRANSLATION_JUMP = 0.08  # Moderate filtering
POSE_FILTER_MAX_CONSECUTIVE_FAILURES = 5  # More forgiving
ROI_SEARCH_FRAMES = 3  # Fall back to full-frame sooner
```

## Parameter Details

### 1. Pose Filtering Parameters

#### `POSE_FILTER_ALPHA` (0.0 - 1.0)
**What it does:** Exponential smoothing factor for pose updates.

- **Higher (0.8-0.9):** Less smoothing, more responsive to changes, but more jittery
- **Lower (0.4-0.6):** More smoothing, less jittery, but slower to respond to real movements
- **Default:** 0.7 (balanced)
- **For stability:** Try 0.5-0.6

#### `POSE_FILTER_MAX_TRANSLATION_JUMP` (meters)
**What it does:** Maximum allowed translation change per frame. Larger jumps are rejected as outliers.

- **Lower (0.03-0.05):** Very strict, rejects most outliers, but may reject fast but valid movements
- **Higher (0.15-0.2):** More forgiving, allows faster movements, but may accept some outliers
- **Default:** 0.1m (10cm)
- **For stability:** Try 0.05-0.08m

#### `POSE_FILTER_MAX_ROTATION_JUMP` (radians)
**What it does:** Maximum allowed rotation change per frame. Larger rotations are rejected as outliers.

- **Lower (0.2-0.3):** Very strict, rejects rotation outliers (~11-17 degrees)
- **Higher (0.7-1.0):** More forgiving, allows fast rotations (~40-57 degrees)
- **Default:** 0.5 rad (~28.6 degrees)
- **For stability:** Try 0.3-0.4 rad

#### `POSE_FILTER_MAX_CONSECUTIVE_FAILURES` (frames)
**What it does:** After N consecutive rejected poses, accept the next one (even if it's an outlier).

- **Lower (1-2):** Very strict, may cause tracking to "freeze" if marker moves quickly
- **Higher (5-7):** More forgiving, allows recovery from temporary tracking loss
- **Default:** 3
- **For stability:** Try 4-5

### 2. ROI Tracking Parameters

#### `ROI_EXPAND` (pixels)
**What it does:** How much to expand the search region around the last known marker position.

- **Larger (150-200):** Bigger search area, more forgiving if marker moves, but slower detection
- **Smaller (50-75):** Faster detection, but marker must stay very close to last position
- **Default:** 100px
- **For stability:** Try 120-150px

#### `ROI_SEARCH_FRAMES` (frames)
**What it does:** How many frames to search in ROI before falling back to full-frame search.

- **Higher (7-10):** More ROI searches (faster), but may miss fast movements
- **Lower (2-3):** Fall back to full-frame sooner (more reliable), but slower
- **Default:** 5
- **For stability:** Try 3-4

### 3. ArUco Detection Parameters

#### `ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MIN` (pixels)
**What it does:** Minimum window size for adaptive thresholding (marker detection).

- **Lower (3):** More sensitive to small markers
- **Higher (5-7):** Less noise, but may miss small markers
- **Default:** 3
- **For stability:** Usually keep at 3

#### `ARUCO_ADAPTIVE_THRESH_WIN_SIZE_MAX` (pixels)
**What it does:** Maximum window size for adaptive thresholding.

- **Lower (15-20):** Faster, but may miss large markers
- **Higher (25-30):** Better for large markers, but slower
- **Default:** 23
- **For stability:** Try 25-27 if markers are large

#### `ARUCO_ADAPTIVE_THRESH_WIN_SIZE_STEP` (pixels)
**What it does:** Step size when searching for optimal window size.

- **Lower (5-7):** More thorough search, slower but more accurate
- **Higher (15-20):** Faster search, but may miss optimal settings
- **Default:** 10
- **For stability:** Try 7-8

#### `ARUCO_POLYGONAL_APPROX_ACCURACY_RATE` (0.0 - 1.0)
**What it does:** Accuracy for polygon approximation of marker corners.

- **Lower (0.02-0.03):** More accurate corner detection, slower
- **Higher (0.08-0.1):** Faster, but less accurate corners
- **Default:** 0.05
- **For stability:** Try 0.03-0.04

#### `ARUCO_PERSPECTIVE_REMOVE_PIXEL_PER_CELL` (pixels)
**What it does:** Pixels per cell when removing perspective distortion.

- **Lower (2-3):** More accurate perspective correction, slower
- **Higher (6-8):** Faster, but less accurate
- **Default:** 4
- **For stability:** Try 3

## Tuning Workflow

1. **Start with pose filtering:**
   - If jittery: Lower `POSE_FILTER_ALPHA` to 0.5-0.6
   - If missing fast movements: Increase `POSE_FILTER_MAX_TRANSLATION_JUMP` to 0.15

2. **Adjust ROI tracking:**
   - If losing track: Increase `ROI_EXPAND` to 150
   - If too slow: Decrease `ROI_SEARCH_FRAMES` to 3

3. **Fine-tune detection:**
   - If missing markers: Lower `ARUCO_POLYGONAL_APPROX_ACCURACY_RATE` to 0.03
   - If too slow: Increase `ARUCO_PERSPECTIVE_REMOVE_PIXEL_PER_CELL` to 6

4. **Test and iterate:**
   - Make one change at a time
   - Test with your typical marker movements
   - Monitor the profiling output for performance impact

## Monitoring

Watch the profiling output every 100 frames:
- **detect** time: Should be < 2ms for good performance
- **pose** time: Should be < 0.5ms
- **total** FPS: Should be close to 30 FPS

If detection time increases significantly, you may have made detection too sensitive.
