# ArUco Background Thread Implementation - Phase 2 Complete

## Overview

Successfully ported ArUco marker polling to a background thread that runs at camera FPS (30 Hz), independent of the control loop frequency (10 Hz). This decouples visual tracking from robot control, improving both performance and reliability.

## Key Changes

### Thread Architecture

**Before (Phase 1)**:
- ArUco polling in main control loop
- Blocking camera reads at 10 Hz
- Control loop had to wait for camera/detection

**After (Phase 2)**:
- Background thread polls at camera FPS (30 Hz)
- Non-blocking reads via thread-safe shared state
- Control loop independent of camera timing

## Implementation Details

### 1. Thread Initialization (`gym_env.py` lines 52-65)

```python
# ArUco background thread (runs at camera FPS)
self._aruco_polling_active = False
self._aruco_poll_thread = None
self._aruco_lock = threading.Lock()  # Protects latest_aruco_obs updates
self.latest_aruco_obs = {
    'aruco_ee_in_world': np.zeros(7),
    'aruco_object_in_world': np.zeros(7),
    'aruco_ee_in_object': np.zeros(7),
    'aruco_object_in_ee': np.zeros(7),
    'aruco_visibility': np.zeros(3)
}
```

### 2. Thread Startup (`_start_aruco_polling()` lines 138-145)

```python
def _start_aruco_polling(self):
    """Start background thread for high-frequency ArUco polling at camera FPS."""
    if self.camera is None:
        return

    self._aruco_polling_active = True
    self._aruco_poll_thread = threading.Thread(target=self._aruco_poll_loop, daemon=True)
    self._aruco_poll_thread.start()
    print(f"✓ Started ArUco polling thread at {robot_config.camera_fps} Hz")
```

### 3. Background Polling Loop (`_aruco_poll_loop()` lines 147-256)

**Key features:**
- Runs at camera FPS (30 Hz) with rate limiter
- Detects ArUco markers on every frame
- Computes all relative poses (ee_in_world, object_in_world, etc.)
- Thread-safe updates with lock
- Handles visualization (video display) in background thread
- Profiling and performance tracking

**Critical Pattern - Thread-Safe Update:**
```python
# Update latest observations (thread-safe)
with self._aruco_lock:
    self.latest_aruco_obs = {k: v.copy() for k, v in obs.items()}
    self.last_frame = frame.copy()
```

### 4. Thread-Safe Read (`_get_aruco_observations()` lines 271-289)

Simplified from ~70 lines to ~20 lines:

```python
def _get_aruco_observations(self):
    """
    Get latest ArUco observations from background thread (thread-safe).

    The background thread polls at camera FPS (30 Hz), while this method
    reads the latest observations at control frequency (10 Hz).
    """
    if not self.has_authority or not self.camera or not self.enable_aruco:
        return {
            'aruco_ee_in_world': np.zeros(7),
            'aruco_object_in_world': np.zeros(7),
            # ... all fields ...
        }

    # Read latest observations from background thread (thread-safe)
    with self._aruco_lock:
        obs = {k: v.copy() for k, v in self.latest_aruco_obs.items()}

    return obs
```

### 5. Thread Cleanup (`close()` lines 453-473)

```python
def close(self):
    """Clean up resources including background thread."""
    # Stop ArUco polling thread
    self._aruco_polling_active = False
    if self._aruco_poll_thread is not None:
        self._aruco_poll_thread.join(timeout=1.0)

    # ... cleanup camera, hardware, etc ...
```

## Benefits

### Performance

✅ **Higher update rate**: ArUco tracking at 30 Hz vs 10 Hz
✅ **Decoupled timing**: Control loop no longer blocked by camera
✅ **Reduced latency**: Latest observations always available
✅ **Smoother tracking**: More frequent pose updates

### Architecture

✅ **Thread-safe**: Lock-protected shared state
✅ **Clean separation**: Vision in background, control in main thread
✅ **Daemon thread**: Auto-terminates with main program
✅ **Graceful shutdown**: Proper join() on close

### Code Quality

✅ **Simplified main loop**: No camera polling in `step()`
✅ **Better profiling**: Separate tracking for ArUco performance
✅ **Cleaner visualization**: All rendering in background thread

## Thread Safety Guarantees

### Protected Resources

1. **`latest_aruco_obs`**: Locked during read/write
2. **`last_frame`**: Locked during updates
3. **Background thread**: Separate rate limiter, profiler tracking

### Safe Access Patterns

**Writer (Background Thread):**
```python
with self._aruco_lock:
    self.latest_aruco_obs = {k: v.copy() for k, v in obs.items()}
    self.last_frame = frame.copy()
```

**Reader (Main Thread):**
```python
with self._aruco_lock:
    obs = {k: v.copy() for k, v in self.latest_aruco_obs.items()}
```

**Key Point**: Always `.copy()` data inside the lock to prevent race conditions.

## Performance Characteristics

### Expected Metrics

- **ArUco polling frequency**: 30 Hz (camera FPS)
- **Control loop frequency**: 10 Hz (unchanged)
- **Lock contention**: Minimal (writes at 30 Hz, reads at 10 Hz, non-overlapping)
- **CPU overhead**: ~5-10% for ArUco thread (depends on resolution, marker count)

### Profiling

The `ArUcoProfiler` tracks:
- Poll times (total time per frame)
- Detection times (marker detection)
- Pose computation times (relative pose calculations)
- Poll intervals (actual vs target frequency)

Stats printed every 300 polls (~10 seconds at 30 Hz).

## Rate Limiter Implementation

Added simple `RateLimiter` class (lines 23-38):

```python
class RateLimiter:
    """Simple rate limiter for background threads."""

    def __init__(self, frequency, warn=False):
        self.period = 1.0 / frequency
        self.last_time = None
        self.warn = warn

    def sleep(self):
        """Sleep to maintain target frequency."""
        now = time.perf_counter()
        if self.last_time is not None:
            elapsed = now - self.last_time
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif self.warn and sleep_time < -0.01:
                print(f"⚠️  Rate limiter: {-sleep_time*1000:.1f}ms late")
        self.last_time = time.perf_counter()
```

Used in background thread:
```python
rate_limiter = RateLimiter(frequency=robot_config.camera_fps, warn=False)
while self._aruco_polling_active and self.camera:
    # ... do work ...
    rate_limiter.sleep()
```

## Integration with Data Collection

The background thread updates `latest_aruco_obs` at 30 Hz, but data collection reads it at 10 Hz (control frequency). This means:

1. **Higher tracking fidelity**: Smoother pose trajectories in recorded data
2. **Latest observations**: Always get most recent ArUco data
3. **No blocking**: Control loop never waits for camera
4. **Flexible recording rate**: Can read observations at any frequency

## Visualization

Video display moved to background thread (lines 219-233):
- Only renders when `show_video=True`
- Draws markers and axes directly in background
- No impact on control loop timing
- Downscaled display for performance (50% resolution)

## Testing

### Syntax Verification (No Hardware Required)

```bash
python compact_gym/verify_aruco_thread_syntax.py
```

**Checks:**
- ✓ Thread attributes in `__init__`
- ✓ `_start_aruco_polling()` method
- ✓ `_aruco_poll_loop()` background thread
- ✓ Thread-safe access in `_get_aruco_observations()`
- ✓ Thread cleanup in `close()`

### Runtime Testing (Hardware Required)

```bash
python compact_gym/test_encoder_polling.py
```

Now also tests ArUco thread:
- Background thread starts automatically
- Observations updated at camera FPS
- No blocking in main loop

## Files Modified

### `compact_gym/gym_env.py`

**Lines changed:** ~100 lines modified/added
- Added threading import (line 7)
- Added RateLimiter class (lines 23-38)
- Added thread attributes in `__init__` (lines 52-65)
- Added `_start_aruco_polling()` (lines 138-145)
- Added `_aruco_poll_loop()` (lines 147-256)
- Simplified `_get_aruco_observations()` (lines 271-289, ~70 lines → ~20 lines)
- Added `close()` method (lines 453-473)

## Configuration

All parameters in `compact_gym/robot_config.py`:

```python
camera_fps: int = 30  # ArUco polling frequency
profiler_window_size: int = 200  # Stats window size
```

## Next Steps

### Phase 3: Full Data Collection Testing
- Test `collect_demo_gym.py` end-to-end with hardware
- Verify NPZ data format matches `compact_code/` output
- Compare ArUco tracking quality (30 Hz vs old 10 Hz)
- Validate encoder + ArUco data in trajectories

### Phase 4: Hardware Authority Manager
- Copy `gym/hardware/authority_manager.py` to `compact_gym/`
- Integrate into `gym_env.py` for multi-env support
- Enable training + evaluation simultaneous operation

## Implementation Status

✅ **Phase 1 Complete**: Encoder Polling (opportunistic reads)
✅ **Phase 2 Complete**: ArUco Background Thread (30 Hz independent polling)
⏳ **Phase 3 Pending**: Full Data Collection Testing
⏳ **Phase 4 Pending**: Hardware Authority Manager

---

**Implementation Date:** 2026-01-14
**Lines Added:** ~120 (background thread + rate limiter + cleanup)
**Lines Removed/Simplified:** ~50 (old synchronous ArUco polling)
**Net Change:** +70 lines, better architecture
