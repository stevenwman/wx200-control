# compact_gym Implementation Review & Fix Guide

**Date**: 2026-01-14
**Reviewer**: Claude Code (Analysis Agent)
**Status**: Architecture is good, but **7 critical issues** block production use

---

## Executive Summary

**compact_gym** is a well-architected gym wrapper for the WX200 robot that successfully refactors compact_code into a cleaner structure. However, it has **incomplete implementation** in several areas that prevent it from working correctly.

**Overall Assessment**:
- ✅ Architecture: Excellent (better than compact_code)
- ⚠️ Implementation: 70% complete
- ❌ Production Ready: **NO** - blocking issues must be fixed first

---

## Critical Issues Requiring Fixes

### 🔴 **ISSUE #1: Scene XML Path is Fragile**

**File**: `robot_hardware.py:31`

**Current Code**:
```python
_XML = Path(__file__).parent.parent / "compact_code" / "wx200" / "scene.xml"
```

**Problem**:
- Assumes `compact_code/` and `compact_gym/` are siblings
- Will fail if directory structure changes
- No fallback if file doesn't exist

**Fix Required**:
```python
# Try multiple locations with fallback
_XML_PATHS = [
    Path(__file__).parent.parent / "compact_code" / "wx200" / "scene.xml",  # Primary
    Path(__file__).parent / "wx200" / "scene.xml",  # If copied locally
    Path(__file__).parent.parent / "wx200" / "scene.xml",  # If at repo root
]
_XML = next((p for p in _XML_PATHS if p.exists()), None)
if _XML is None:
    raise FileNotFoundError(
        f"scene.xml not found. Searched:\n" +
        "\n".join(f"  - {p}" for p in _XML_PATHS)
    )
```

**Priority**: 🔴 **CRITICAL** - env.reset() will crash without this

---

### 🔴 **ISSUE #2: Missing ee_pose_target (Data Loss)**

**File**: `collect_demo_gym.py:268`

**Current Code**:
```python
'ee_pose_target': np.zeros(7), # We don't have IK target readily available
```

**Problem**:
- Records zeros instead of actual IK solver target pose
- compact_code records this correctly, gym version loses this data
- This is critical for analyzing IK solver behavior

**Fix Required**:
```python
# In collect_demo_gym.py, get actual target from controller
if self.env.robot_hardware and self.env.robot_hardware.robot_controller:
    target_pose_se3 = self.env.robot_hardware.robot_controller.get_target_pose()
    # target_pose_se3 is mink.SE3 with format [w,x,y,z, x,y,z]
    ee_pose_target = target_pose_se3.np  # This gives [qw,qx,qy,qz, px,py,pz]
    # Reorder to [px,py,pz, qw,qx,qy,qz]
    ee_pose_target = np.concatenate([
        ee_pose_target[4:7],  # position
        ee_pose_target[0:4]   # quaternion
    ])
else:
    ee_pose_target = np.zeros(7)

step_data['ee_pose_target'] = ee_pose_target
```

**Priority**: 🟠 **HIGH** - Data quality issue, but not a crash

---

### 🔴 **ISSUE #3: stop_recording() Signature Mismatch**

**File**: `collect_demo_gym.py`

**Current Code**:
```python
# Line 81: Definition
def stop_recording(self):
    if self.is_recording:
        self.is_recording = False
        self.save_trajectory()
        # ...

# Line 149, 202: Calls
self.stop_recording(success=True)  # ❌ CRASHES - missing parameter
```

**Problem**: Function doesn't accept `success` parameter but is called with it

**Fix Required**:
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
        else:
            self.trajectory = []
            print("[RECORDING DISCARDED]")
        self.gui.set_status("Ready")
```

**Priority**: 🔴 **CRITICAL** - Will crash when user presses 'd' or 's' key

---

### 🟡 **ISSUE #4: Frame Downscaling Inconsistency**

**File**: `collect_demo_gym.py:280`

**Current Code**:
```python
cv2.resize(self.env.last_frame, (robot_config.camera_width // 4, robot_config.camera_height // 4))
```

**Problem**: Hardcoded `// 4` instead of using config parameter

**Fix Required**:
```python
downscale_factor = robot_config.frame_downscale_factor
downscaled_width = robot_config.camera_width // downscale_factor
downscaled_height = robot_config.camera_height // downscale_factor
frame_to_store = cv2.resize(
    self.env.last_frame,
    (downscaled_width, downscaled_height),
    interpolation=cv2.INTER_AREA
)
step_data['camera_frame'] = frame_to_store
```

**Priority**: 🟡 **MEDIUM** - Works but not maintainable

---

### 🟡 **ISSUE #5: ThreadedArUcoCamera is Dead Code**

**File**: `camera.py:292-429`

**Problem**:
- `ThreadedArUcoCamera` class is defined but **never used**
- Duplicates logic already in `gym_env.py:167-280`
- 137 lines of dead code creating maintenance burden

**Fix Required**:
```python
# DELETE the entire ThreadedArUcoCamera class (lines 292-429)
# It's unused and duplicates gym_env._aruco_poll_loop()
```

**Justification**:
- `gym_env.py` already has ArUco polling thread
- No other file imports `ThreadedArUcoCamera`
- Violates DRY principle

**Priority**: 🟡 **MEDIUM** - Code smell but doesn't affect functionality

---

### 🟡 **ISSUE #6: Observation Space Design Question**

**File**: `gym_env.py:99-103, 346-351`

**Current State**:
```python
# Observation space: 20D
self.observation_space = Box(low=np.full(20, -np.inf, ...), ...)

# But _get_observation() only includes:
obs = np.concatenate([
    self.aruco_obs_dict['aruco_object_in_world'], # 7D
    robot_state, # 6D
    ee_pose_debug # 7D
])  # Total: 20D
```

**Full ArUco data available but not in obs**:
- ✅ `aruco_object_in_world` - IN observation
- ❌ `aruco_ee_in_world` - in info dict only
- ❌ `aruco_ee_in_object` - in info dict only
- ❌ `aruco_object_in_ee` - in info dict only
- ❌ `aruco_visibility` - in info dict only

**Question for User**:
Is this intentional? If policies need full ArUco state, observation should be expanded to 34D:
- `aruco_ee_in_world` (7D)
- `aruco_object_in_world` (7D)
- `aruco_ee_in_object` (7D)
- `aruco_object_in_ee` (7D)
- `aruco_visibility` (3D)
- `robot_state` (6D)
- `ee_pose_debug` (7D)
= **48D total**

Or keep current 20D if policy doesn't need all ArUco data.

**Priority**: 🟡 **MEDIUM** - Design decision needed

---

### 🟢 **ISSUE #7: Test Files in Production Directory**

**Files**:
- `test_encoder_polling.py`
- `verify_aruco_thread_syntax.py`
- `verify_encoder_implementation.py`
- `verify_encoder_syntax.py`
- `verify_teleop_gym.py`

**Problem**: Test/verification files mixed with production code

**Fix Required**:
```bash
mkdir -p compact_gym/tests
mv compact_gym/test_*.py compact_gym/tests/
mv compact_gym/verify_*.py compact_gym/tests/
```

**Priority**: 🟢 **LOW** - Organizational issue only

---

## What's Already Good ✅

### 1. **Driver Improvements** (Better than compact_code!)

**File**: `robot_driver.py:136-156`

```python
# Selective motor reboot - only reboots motors that fail torque enable
for dxl_id in robot_config.motor_ids:
    dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(...)
    if needs_reboot:
        self.reboot_motor(dxl_id)
```

This is **safer** than compact_code which reboots all motors on startup.

### 2. **Shutdown Sequence** (Fixed in recent edits)

**File**: `robot_hardware.py:354-356`

```python
# 4. Extra delay before disabling torque (critical for preventing slam!)
print("Waiting for movement to complete before disabling torque...")
time.sleep(robot_config.move_delay)
```

Good addition - prevents robot from dropping due to premature torque disable.

### 3. **Architecture**

- Clean separation: `gym_env.py` (interface) vs `robot_hardware.py` (lifecycle)
- Thread-safe ArUco polling at 30 Hz decoupled from 10 Hz control
- Encoder polling with opportunistic reads preserved from compact_code

---

## Implementation Checklist

Use this to track progress:

- [ ] **Fix #1**: Robust scene.xml path with fallback
- [ ] **Fix #2**: Record actual `ee_pose_target` instead of zeros
- [ ] **Fix #3**: Add `success` parameter to `stop_recording()`
- [ ] **Fix #4**: Use `frame_downscale_factor` config instead of hardcoded `// 4`
- [ ] **Fix #5**: Delete `ThreadedArUcoCamera` dead code
- [ ] **Fix #6**: Decide on observation space design (keep 20D or expand to 48D)
- [ ] **Fix #7**: Move test files to `tests/` subdirectory

---

## Testing After Fixes

### Minimum Viable Test:
```bash
cd compact_gym
python collect_demo_gym.py
```

**Expected**:
1. ✅ Env resets without crashing (scene.xml loads)
2. ✅ ArUco polling starts at 30 Hz
3. ✅ Robot moves in response to SpaceMouse
4. ✅ Press 'r' to start recording (no crash)
5. ✅ Press 'd' to save (no crash, ee_pose_target != zeros)

### Data Validation Test:
```python
import numpy as np

# After collecting a demo
data = np.load("data/gym_demos/demo_*.npz")

# Check critical fields
assert 'ee_pose_target' in data
assert not np.allclose(data['ee_pose_target'], 0)  # Should NOT be all zeros
assert data['encoder_values'].shape[1] == 7  # All 7 motors
assert data['camera_frame'].shape[1:] == (270, 480, 3)  # Downscaled correctly
```

---

## Cross-Reference: compact_code vs compact_gym

| Feature | compact_code | compact_gym | Status |
|---------|--------------|-------------|--------|
| Scene XML loading | Hardcoded path | Hardcoded path | ⚠️ Both fragile, fix gym |
| Motor reboot | All on startup | Selective | ✅ Gym better |
| ee_pose_target recording | ✅ Correct | ❌ Zeros | 🔴 Fix gym |
| Shutdown safety delay | ✅ Yes | ✅ Fixed | ✅ Both good now |
| ArUco polling thread | ✅ 30 Hz | ✅ 30 Hz | ✅ Both good |
| Code duplication | Minimal | ThreadedArUcoCamera | 🟡 Clean up gym |
| Self-contained | ✅ Yes | ❌ Imports SpaceMouse | ℹ️ Known dependency |

---

## Recommendations

### For Immediate Fix (Priority Order):

1. **Fix #1** - Scene XML path (will crash otherwise)
2. **Fix #3** - `stop_recording()` signature (will crash on save)
3. **Fix #2** - Record actual ee_pose_target (data quality)
4. **Fix #4** - Use frame_downscale_factor config
5. **Fix #5** - Delete ThreadedArUcoCamera
6. **Fix #6** - Decide on observation space (user input needed)
7. **Fix #7** - Move test files

### For Long-Term:

- Consider copying SpaceMouse/GUI into compact_gym to make it truly self-contained
- Add proper unit tests in `tests/` directory
- Consolidate `.md` docs into single README

---

## Final Verdict

**compact_gym has excellent architecture but incomplete execution.**

Once Issues #1-#5 are fixed, this will be **production-ready** and **better than compact_code** for gym-based RL training.

The refactoring work is solid - just needs the implementation to catch up to the design.

---

**Good luck with the fixes! The architecture is already better than compact_code - you're 70% there.**
