# Camera Polling at 20Hz: Feasibility Analysis & Plan

## Current Architecture

### Control Loop Timing
- **Outer loop (policy/observations)**: 20 Hz (50ms per step)
- **Inner loop (IK/motor control)**: 100 Hz (10ms per step)
- **Single-threaded architecture**: Non-blocking, event-driven

### Camera System
- **Frame rate**: 30 FPS (33ms per frame)
- **Implementation**: GStreamer (preferred) with OpenCV fallback
- **Capture method**: Non-blocking callback with `appsink`
- **Resolution**: 1920x1080 (2MP frames)
- **Frame size**: ~6MB per frame (BGR, uint8)

### Current Camera Usage
- Called in `_get_aruco_observations_dict()` → `camera.read()` (non-blocking)
- ArUco detection + pose estimation (~3 tags: world, object, ee)
- Called during `step()` → `get_obs()` → every 50ms at 20Hz

## Feasibility Assessment

### ✅ **Why 20Hz Camera Polling Should Work**

1. **Frame Rate Mismatch is Favorable**
   - Camera: 30 FPS (new frame every 33ms)
   - Control: 20 Hz (poll every 50ms)
   - **Result**: At 20Hz, each poll should always get a "fresh" frame (within 33ms)
   - Camera produces frames faster than we consume them → no blocking

2. **Non-Blocking Architecture**
   - GStreamer `appsink` uses callback pattern
   - `camera.read()` just copies latest frame (if available)
   - No blocking I/O wait for camera hardware
   - Frame available check is O(1)

3. **Existing Performance Profile**
   - Control loop already runs at 50ms period
   - Current teleop code (`wx200_robot_collect_demo_encoders.py`) already polls camera at 20Hz
   - No reported issues with camera blocking control

### ⚠️ **Potential Bottlenecks**

1. **ArUco Processing Cost** (Primary concern)
   - `cv2.cvtColor(frame, COLOR_BGR2GRAY)`: ~5-10ms for 1920x1080
   - `detector.detectMarkers()`: ~10-30ms (depends on marker count/visibility)
   - `process_tag()` × 3: ~5-15ms each (solvePnP + validation)
   - **Total estimated**: 30-75ms per observation call
   - **Risk**: Could exceed 50ms budget at 20Hz

2. **Frame Copy Overhead**
   - `camera.read()` copies ~6MB frame buffer
   - Memory copy: ~5-10ms for 1920x1080 BGR image
   - Currently copies frame even if we don't need it

3. **Image Processing Pipeline**
   - Color conversion (BGR→GRAY): ~5-10ms
   - Resize for display (if enabled): ~5ms
   - Visualization drawing: ~2-5ms

4. **Control Loop Budget**
   - 50ms budget per outer loop iteration (20Hz)
   - IK solver: ~5-10ms
   - Motor commands: ~2-5ms
   - Encoder polling: ~5-10ms
   - **Remaining for camera/ArUco**: ~25-35ms

## Optimization Strategies

### Strategy 1: **Asynchronous Camera Processing** (Recommended)
- **Approach**: Process camera frames in background thread
- **Implementation**:
  - Camera thread continuously captures frames and runs ArUco detection
  - Main thread reads latest ArUco results (non-blocking)
  - Use thread-safe queue with latest-frame policy
- **Benefit**: Decouples camera processing from control loop
- **Risk**: Adds threading complexity, potential sync issues

### Strategy 2: **Optimize ArUco Detection**
- **Approach**: Reduce computational cost per frame
- **Techniques**:
  - Downscale image before detection (e.g., 960x540) → 4x faster detection
  - Use image pyramid: detect at low-res, refine at high-res
  - Skip detection if markers haven't moved much (motion threshold)
  - Limit search area (ROI) if marker positions are predictable
- **Benefit**: 2-4x speedup with minimal accuracy loss
- **Trade-off**: Slight reduction in detection robustness

### Strategy 3: **Frame Skipping / Stale Frame Tolerance**
- **Approach**: Don't process every frame, allow some staleness
- **Implementation**:
  - Only process camera every N control steps (e.g., every 2nd step = 10Hz)
  - Reuse ArUco results from last processed frame
  - Mark observations with "freshness" timestamp
- **Benefit**: Guarantees control loop timing
- **Trade-off**: Slightly stale observations (50-100ms old)

### Strategy 4: **Reduce Resolution**
- **Approach**: Lower camera resolution
- **Options**:
  - 1280x720 (HD): ~3MB/frame, 2.25x faster processing
  - 960x540 (qHD): ~1.5MB/frame, 4x faster processing
- **Benefit**: Linear speedup in image processing
- **Trade-off**: Reduced marker detection range/accuracy

### Strategy 5: **Selective Processing**
- **Approach**: Only process markers that are actually needed
- **Implementation**:
  - Check marker visibility first (quick check)
  - Skip `solvePnP` if marker not visible
  - Use cached pose from previous frame
- **Benefit**: Avoids expensive processing when markers are occluded

## Recommended Implementation Plan

### Phase 1: **Baseline Profiling** (Do First)
1. **Add timing instrumentation** to gym environment:
   ```python
   - Time `camera.read()` call
   - Time ArUco detection pipeline
   - Time total `_get_observation()` call
   - Track over 1000+ steps
   ```
2. **Profile under realistic conditions**:
   - Robot moving
   - Multiple markers visible
   - Control loop running at full speed
3. **Measure**:
   - Mean, median, p95, p99 latencies
   - Frame drops (when camera.read() returns False)
   - Control loop jitter (missed 50ms deadlines)

### Phase 2: **Quick Wins** (Low Risk)
1. **Conditional visualization**:
   - Only draw/display when `show_video=True`
   - Skip `cv2.waitKey(1)` when not needed
2. **Image downscaling for detection**:
   - Detect markers at 960x540, keep full-res for display
   - 4x speedup with minimal accuracy loss
3. **Cached color conversion**:
   - Convert to grayscale once, reuse if same frame
4. **Frame freshness check**:
   - Track frame timestamps
   - Skip processing if frame hasn't changed

### Phase 3: **If Needed: Async Processing**
If profiling shows ArUco processing exceeds 30ms consistently:
1. **Implement camera thread**:
   - Separate thread for camera capture + ArUco
   - Thread-safe queue for latest results
   - Main thread reads results (non-blocking)
2. **Careful synchronization**:
   - Use locks/queues properly
   - Handle thread startup/shutdown
   - Monitor thread health

### Phase 4: **Fallback: Frame Skipping**
If async is too complex or still too slow:
1. **Process every Nth frame**:
   - `_get_observation()` processes camera every 2nd call
   - Reuse previous ArUco results otherwise
2. **Add staleness metadata**:
   - Include timestamp in observation
   - Let policy handle stale data

## Expected Performance

### Best Case (Current Setup + Optimizations)
- **Camera read**: 1-2ms (non-blocking copy)
- **ArUco detection**: 15-25ms (with downscaling)
- **Pose estimation**: 5-10ms (3 tags)
- **Total**: 20-35ms per `get_obs()` call
- **Verdict**: ✅ Feasible at 20Hz (fits in 50ms budget)

### Worst Case (No Optimizations)
- **Camera read**: 5-10ms (full frame copy)
- **ArUco detection**: 30-50ms (full resolution)
- **Pose estimation**: 15-30ms (3 tags, worst case)
- **Total**: 50-90ms per `get_obs()` call
- **Verdict**: ⚠️ Risky (may exceed 50ms budget, causing jitter)

### Realistic Case (With Quick Wins)
- **Camera read**: 1-2ms
- **ArUco detection**: 20-30ms (downscaled)
- **Pose estimation**: 8-12ms
- **Total**: 30-45ms per `get_obs()` call
- **Verdict**: ✅ Should work, but monitor p95/p99 latencies

## Testing Plan

### 1. **Latency Benchmark**
```python
# Add to wx200_gym_env.py
import time
self._obs_times = []

def _get_observation(self):
    start = time.perf_counter()
    obs = self._get_aruco_observations_dict()
    elapsed = (time.perf_counter() - start) * 1000  # ms
    self._obs_times.append(elapsed)
    
    if len(self._obs_times) % 100 == 0:
        print(f"Obs latency: avg={np.mean(self._obs_times[-100:]):.1f}ms, "
              f"p95={np.percentile(self._obs_times[-100:], 95):.1f}ms")
    # ... rest of method
```

### 2. **Control Loop Jitter Test**
- Run gym environment for 10,000 steps
- Measure time between `step()` calls
- Check for missed deadlines (>55ms indicates problem)

### 3. **Frame Freshness Test**
- Track camera frame timestamps
- Verify frames are <50ms old when used
- Monitor frame drops (camera.read() returns False)

## Conclusion

**Recommendation**: **20Hz camera polling is likely feasible** with current architecture, but requires optimization to guarantee real-time performance.

**Action Items**:
1. ✅ **Profile first** - Add timing instrumentation to measure actual latency
2. ✅ **Implement quick wins** - Image downscaling, conditional visualization
3. ⚠️ **Monitor p95/p99 latencies** - Ensure <45ms for 90% of calls
4. 🔄 **Consider async if needed** - Only if profiling shows consistent >40ms latency

**Key Insight**: The non-blocking GStreamer architecture is well-suited for this use case. The main risk is ArUco processing cost, which can be mitigated through standard computer vision optimizations (downscaling, selective processing).
