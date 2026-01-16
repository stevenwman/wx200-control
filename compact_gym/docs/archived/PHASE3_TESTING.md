# Phase 3: Data Collection Testing

**Status**: 🚀 Ready to Start
**Date**: 2026-01-14

---

## Prerequisites Complete

✅ **Phase 1**: Encoder polling with opportunistic reads
✅ **Phase 2**: ArUco background thread at 30 Hz
✅ **Phase 2.5**: Shutdown timing fix (double-delay)
✅ **Phase 3 Prep**: All 4 critical fixes applied

---

## Phase 3 Goals

1. **End-to-End Testing**: Verify full data collection pipeline with hardware
2. **Data Format Validation**: Ensure NPZ matches `compact_code` format
3. **ArUco Quality Check**: Compare 30 Hz vs 10 Hz tracking
4. **Encoder Sync Verification**: Validate encoder + ArUco alignment
5. **Robustness Testing**: Various scenarios (occlusion, fast movement, etc.)

---

## Testing Script

### Basic Test (with hardware)

```bash
cd /home/steven/Desktop/work/research/openarm_control/compact_gym
python collect_demo_gym.py
```

### What to Test

#### 1. **Startup Sequence**
- [ ] Scene XML loads from local `wx200/scene.xml`
- [ ] Robot initializes and homes
- [ ] Camera starts (1920x1080 at 30 Hz)
- [ ] ArUco polling thread starts
- [ ] GUI displays with status "Ready"

#### 2. **Teleoperation**
- [ ] SpaceMouse controls robot smoothly
- [ ] Gripper opens/closes with left/right buttons
- [ ] No lag or stuttering
- [ ] ArUco markers tracked in real-time

#### 3. **Recording Workflow**
- [ ] Press 'r' - Recording starts, status shows "Recording Ep 0..."
- [ ] Move robot through trajectory
- [ ] Press 'd' - Recording stops and saves (no crash!)
- [ ] Check output: `data/gym_demos/demo_TIMESTAMP.npz` created
- [ ] Press 'r' again - Episode count increments
- [ ] Press 'x' - Recording discarded (no crash!)

#### 4. **Shutdown**
- [ ] Press 'q' - Program exits
- [ ] Robot moves through: reasonable_home → base_home → folded_home
- [ ] Extra delay before torque disable
- [ ] Robot does NOT slam
- [ ] Motors de-energized (can move by hand)
- [ ] Terminal shows "✓ Robot shutdown complete"

---

## Data Validation

After collecting a demo, validate the NPZ file:

```python
import numpy as np
import os

# Load most recent demo
demo_files = sorted(glob.glob("data/gym_demos/demo_*.npz"))
data = np.load(demo_files[-1])

print("=== Data Validation ===")
print(f"File: {demo_files[-1]}")
print(f"Size: {os.path.getsize(demo_files[-1]) / 1024 / 1024:.2f} MB")
print()

# Check all expected fields
expected_fields = [
    'timestamp', 'state', 'encoder_values', 'ee_pose_encoder',
    'action', 'augmented_actions', 'ee_pose_target',
    'object_pose', 'object_visible',
    'aruco_ee_in_world', 'aruco_object_in_world',
    'aruco_ee_in_object', 'aruco_object_in_ee', 'aruco_visibility',
    'camera_frame'
]

print("Field Check:")
for field in expected_fields:
    if field in data:
        shape = data[field].shape
        print(f"  ✓ {field:25s} {shape}")
    else:
        print(f"  ✗ {field:25s} MISSING!")

print()

# Critical validations
print("Critical Checks:")

# 1. ee_pose_target should NOT be all zeros
ee_target_nonzero = not np.allclose(data['ee_pose_target'], 0)
print(f"  {'✓' if ee_target_nonzero else '✗'} ee_pose_target not all zeros: {ee_target_nonzero}")

# 2. Encoder values shape
encoder_ok = data['encoder_values'].shape[1] == 7
print(f"  {'✓' if encoder_ok else '✗'} encoder_values has 7 motors: {data['encoder_values'].shape}")

# 3. Camera frame shape (1/4 resolution)
expected_frame_shape = (270, 480, 3)  # 1080/4 x 1920/4
frame_ok = data['camera_frame'].shape[1:] == expected_frame_shape
print(f"  {'✓' if frame_ok else '✗'} camera_frame downscaled correctly: {data['camera_frame'].shape}")

# 4. Trajectory length
traj_len = len(data['timestamp'])
duration = data['timestamp'][-1] - data['timestamp'][0] if traj_len > 1 else 0
freq = traj_len / duration if duration > 0 else 0
print(f"  ✓ Trajectory: {traj_len} steps, {duration:.2f}s, {freq:.1f} Hz")

print()
print("=== Validation Complete ===")
```

Save as `validate_demo.py` and run:
```bash
python validate_demo.py
```

---

## Expected Output

### Good Data Example

```
=== Data Validation ===
File: data/gym_demos/demo_20260114_174523.npz
Size: 45.23 MB

Field Check:
  ✓ timestamp                  (523,)
  ✓ state                      (523, 6)
  ✓ encoder_values             (523, 7)
  ✓ ee_pose_encoder            (523, 7)
  ✓ action                     (523, 7)
  ✓ augmented_actions          (523, 6)
  ✓ ee_pose_target             (523, 7)
  ✓ object_pose                (523, 7)
  ✓ object_visible             (523, 1)
  ✓ aruco_ee_in_world          (523, 7)
  ✓ aruco_object_in_world      (523, 7)
  ✓ aruco_ee_in_object         (523, 7)
  ✓ aruco_object_in_ee         (523, 7)
  ✓ aruco_visibility           (523, 3)
  ✓ camera_frame               (523, 270, 480, 3)

Critical Checks:
  ✓ ee_pose_target not all zeros: True
  ✓ encoder_values has 7 motors: (523, 7)
  ✓ camera_frame downscaled correctly: (523, 270, 480, 3)
  ✓ Trajectory: 523 steps, 52.3s, 10.0 Hz

=== Validation Complete ===
```

---

## ArUco Quality Comparison

To compare ArUco tracking at 30 Hz (gym) vs 10 Hz (old code):

1. **Collect demo with compact_gym**:
   ```bash
   cd compact_gym
   python collect_demo_gym.py
   # Save as demo_gym.npz
   ```

2. **Collect similar demo with compact_code** (for reference):
   ```bash
   cd ../compact_code
   python wx200_robot_collect_demo_encoders_compact.py
   # Save as demo_old.npz
   ```

3. **Compare ArUco smoothness**:
   ```python
   import numpy as np
   import matplotlib.pyplot as plt

   gym_data = np.load("demo_gym.npz")
   old_data = np.load("demo_old.npz")

   # Plot object position over time
   fig, axes = plt.subplots(3, 1, figsize=(12, 8))

   for i, label in enumerate(['X', 'Y', 'Z']):
       axes[i].plot(gym_data['aruco_object_in_world'][:, i],
                    label='compact_gym (30 Hz)', linewidth=2)
       axes[i].plot(old_data['object_pose'][:, i],
                    label='compact_code (10 Hz)', linewidth=2, alpha=0.7)
       axes[i].set_ylabel(f'{label} position (m)')
       axes[i].legend()
       axes[i].grid(True)

   axes[2].set_xlabel('Step')
   plt.suptitle('ArUco Object Tracking Comparison')
   plt.tight_layout()
   plt.show()
   ```

**Expected**: 30 Hz should show smoother trajectories with less jitter.

---

## Stress Tests

Once basic testing passes, try these edge cases:

### 1. **Fast Movement**
- Move robot at maximum SpaceMouse speed
- Verify no frame drops or ArUco loss
- Check encoder polling doesn't skip

### 2. **Marker Occlusion**
- Cover EE marker, verify `aruco_visibility[0] == 0`
- Cover object marker, verify `aruco_visibility[1] == 0`
- Uncover, verify tracking resumes

### 3. **Long Recording**
- Record for 2+ minutes (1200+ steps)
- Verify no memory leaks
- Check NPZ file size is reasonable

### 4. **Multiple Episodes**
- Record 5+ demos in one session
- Verify episode counter increments
- Check all files saved correctly

### 5. **Emergency Stop**
- Start recording
- Press Ctrl+C mid-recording
- Verify robot shuts down cleanly
- Check try/finally cleanup works

---

## Troubleshooting

### Robot Slams During Shutdown
- Check `robot_config.move_delay` value (should be 1.0+)
- Verify `robot_hardware.py:354-356` has extra sleep
- Test with longer delay (2.0 or 3.0)

### Camera Not Found
- Check `camera_index` in `robot_config.py`
- Verify GStreamer installed: `gst-inspect-1.0 --version`
- Try different index (0, 2, 4, etc.)

### ArUco Not Detecting
- Check marker IDs match config (EE=1, Object=2, World=3)
- Verify marker size in config (default: 0.04m)
- Check camera calibration matrix

### Encoder Read Timeouts
- Check serial port permissions
- Verify `motor_ids` in config match hardware
- Increase encoder read timeout if needed

### NPZ File Too Large
- Check `frame_downscale_factor` (should be 4)
- Verify frames are compressed (using `cv2.INTER_AREA`)
- Consider disabling frame recording for testing

---

## Success Criteria

Phase 3 is complete when:

- [ ] Can collect 5+ demos without crashes
- [ ] All NPZ fields present with correct shapes
- [ ] `ee_pose_target` contains real IK targets (not zeros)
- [ ] ArUco tracking at 30 Hz shows improved quality
- [ ] Encoder + ArUco data properly synchronized
- [ ] Robot shuts down cleanly every time
- [ ] Data format matches `compact_code` for compatibility

---

## Next Phase: Phase 4

Once Phase 3 is validated, we can tackle:

**Phase 4: Hardware Authority Manager**
- Port `gym/hardware/authority_manager.py` to `compact_gym/`
- Replace simple class-level authority check
- Enable multi-env support (train + eval simultaneously)
- Test with 2 gym instances

---

**Ready to start Phase 3 testing with hardware!**
**All prerequisites complete, fixes applied, syntax validated.**
