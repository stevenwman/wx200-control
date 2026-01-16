# Phase 3 Preparation Fixes

**Date**: 2026-01-14
**Status**: ✅ Complete - Ready for Phase 3 Testing

---

## Overview

Fixed 4 critical issues identified in code review to prepare for Phase 3 (Data Collection Testing). These fixes ensure `collect_demo_gym.py` will work correctly and produce high-quality data.

---

## Fixes Applied

### ✅ Fix #1: Self-Contained scene.xml Path

**Issue**: Scene XML path referenced `compact_code/` directory, making `compact_gym` not self-contained.

**Solution**: Copied entire `wx200/` directory to `compact_gym/` and updated path.

**Files Modified**:
- Created `compact_gym/wx200/` directory with all MuJoCo files
- Updated [robot_hardware.py:31](robot_hardware.py#L31):
  ```python
  # Before:
  _XML = Path(__file__).parent.parent / "compact_code" / "wx200" / "scene.xml"

  # After:
  _XML = Path(__file__).parent / "wx200" / "scene.xml"
  ```

**Benefit**: `compact_gym` is now self-contained and portable.

---

### ✅ Fix #3: stop_recording() Signature Mismatch

**Issue**: `stop_recording()` was called with `success=True` parameter but didn't accept it, causing crashes when saving demos.

**Solution**: Added `success` parameter and unified discard logic.

**Files Modified**: [collect_demo_gym.py:81-98](collect_demo_gym.py#L81-L98)

```python
def stop_recording(self, success=True):
    """
    Stop recording and optionally save.

    Args:
        success: If True, save trajectory. If False, discard it.
    """
    if self.is_recording:
        self.is_recording = False
        if success:
            self.save_trajectory()
            self.episode_count += 1
            print("[RECORDING STOPPED] Saved.")
            self.gui.set_status("Ready")
        else:
            self.trajectory = []
            print("[RECORDING DISCARDED]")
            self.gui.set_status("Discarded")
```

**Also Updated**: 'x' key handler now uses `stop_recording(success=False)` instead of duplicating logic.

**Benefit**: No crash when pressing 'd' or 's' keys to save demos.

---

### ✅ Fix #2: Record Actual ee_pose_target

**Issue**: `ee_pose_target` was recording zeros instead of actual IK solver target, losing critical data for analyzing IK behavior.

**Solution**: Added helper method to extract target from robot controller.

**Files Modified**: [collect_demo_gym.py:74-84, 287](collect_demo_gym.py#L74-L84)

```python
def _get_ee_pose_target(self):
    """Get the current IK target pose from the robot controller."""
    if self.env.robot_hardware and self.env.robot_hardware.robot_controller:
        position = self.env.robot_hardware.robot_controller.get_target_position()
        orientation_wxyz = self.env.robot_hardware.robot_controller.get_target_orientation_quat_wxyz()
        # Return in format [px, py, pz, qw, qx, qy, qz]
        return np.concatenate([position, orientation_wxyz])
    else:
        return np.zeros(7)
```

**Recording Updated**:
```python
'ee_pose_target': self._get_ee_pose_target(),  # Now gets actual IK target
```

**Benefit**: Recorded data now includes real IK solver targets for analysis.

---

### ✅ Fix #4: Use frame_downscale_factor Config

**Issue**: Hardcoded `// 4` for frame downscaling instead of using config parameter, making it unmaintainable.

**Solution**: Added helper method using `robot_config.frame_downscale_factor`.

**Files Modified**: [collect_demo_gym.py:86-91, 310](collect_demo_gym.py#L86-L91)

```python
def _downscale_frame(self, frame):
    """Downscale frame according to config."""
    downscaled_width = robot_config.camera_width // robot_config.frame_downscale_factor
    downscaled_height = robot_config.camera_height // robot_config.frame_downscale_factor
    return cv2.resize(frame, (downscaled_width, downscaled_height), interpolation=cv2.INTER_AREA)
```

**Recording Updated**:
```python
'camera_frame': self._downscale_frame(self.env.last_frame) if self.env.last_frame is not None else np.zeros((robot_config.camera_height // robot_config.frame_downscale_factor, robot_config.camera_width // robot_config.frame_downscale_factor, 3), dtype=np.uint8)
```

**Benefit**: Frame downscaling now controlled by `robot_config.frame_downscale_factor` (default: 4).

---

## Deferred Issues

These were identified but deferred to future work:

### 🟡 Issue #5: ThreadedArUcoCamera Dead Code (Future)

**Status**: Deferred - doesn't affect functionality

**Description**: `camera.py:292-429` contains 137 lines of unused `ThreadedArUcoCamera` class that duplicates `gym_env.py:_aruco_poll_loop()`.

**Future Action**: Delete when doing code cleanup pass.

---

### 🟡 Issue #6: Observation Space Design Question (Future)

**Status**: Deferred - needs user input

**Current**: Observation space is 20D with only `aruco_object_in_world` included.

**Question**: Should we expand to 48D to include all ArUco data?
- `aruco_ee_in_world` (7D)
- `aruco_object_in_world` (7D) ← currently included
- `aruco_ee_in_object` (7D)
- `aruco_object_in_ee` (7D)
- `aruco_visibility` (3D)
- `robot_state` (6D)
- `ee_pose_debug` (7D)

All data is already in `info` dict, just not in observation space.

**Future Action**: Decide based on policy needs during RL training.

---

### 🟢 Issue #7: Test File Organization (Future)

**Status**: Deferred - cosmetic only

**Description**: Test files mixed with production code:
- `test_encoder_polling.py`
- `verify_aruco_thread_syntax.py`
- `verify_encoder_implementation.py`
- `verify_encoder_syntax.py`
- `verify_teleop_gym.py`

**Future Action**:
```bash
mkdir -p compact_gym/tests
mv compact_gym/test_*.py compact_gym/tests/
mv compact_gym/verify_*.py compact_gym/tests/
```

---

## Testing Checklist for Phase 3

After these fixes, Phase 3 data collection should work:

```bash
cd compact_gym
python collect_demo_gym.py
```

**Expected Behavior**:
1. ✅ Env resets without crashing (scene.xml loads from local copy)
2. ✅ ArUco polling starts at 30 Hz (background thread)
3. ✅ Robot moves in response to SpaceMouse
4. ✅ Press 'r' to start recording (no crash)
5. ✅ Press 'd' to save (no crash, ee_pose_target != zeros)
6. ✅ Press 'x' to discard (no crash)
7. ✅ Saved NPZ contains all expected fields with correct shapes

**Data Validation**:
```python
import numpy as np

# After collecting a demo
data = np.load("data/gym_demos/demo_*.npz")

# Critical checks
assert 'ee_pose_target' in data
assert not np.allclose(data['ee_pose_target'], 0)  # Should NOT be all zeros
assert data['encoder_values'].shape[1] == 7  # All 7 motors
assert data['camera_frame'].shape[1:] == (270, 480, 3)  # 1/4 resolution (1920x1080 / 4)
```

---

## Files Modified Summary

| File | Changes | Lines |
|------|---------|-------|
| `robot_hardware.py` | Scene XML path → local | 1 line |
| `collect_demo_gym.py` | stop_recording() signature | ~15 lines |
| `collect_demo_gym.py` | _get_ee_pose_target() helper | ~10 lines |
| `collect_demo_gym.py` | _downscale_frame() helper | ~5 lines |
| `collect_demo_gym.py` | Recording updates | ~3 lines |
| `wx200/` (new dir) | Copied MuJoCo files | ~4 files |

**Total**: ~34 lines changed/added, 1 directory created

---

## Next Phase: Phase 3 - Data Collection Testing

With these fixes complete, we're ready for Phase 3:

**Phase 3 Goals**:
1. Test `collect_demo_gym.py` end-to-end with hardware
2. Verify NPZ data format matches `compact_code/` output
3. Compare ArUco tracking quality (30 Hz vs old 10 Hz)
4. Validate encoder + ArUco data synchronization
5. Test data collection under various conditions

**Phase 4 (Future)**: Hardware Authority Manager for multi-env support

---

**Status**: ✅ Ready for Phase 3 hardware testing
**Date**: 2026-01-14
**All Critical Fixes**: Complete
