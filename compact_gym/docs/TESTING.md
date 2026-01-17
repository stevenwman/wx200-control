# Testing Guide

Guide for testing compact_gym functionality.

## Quick Verification

### Data Collection Test

```bash
cd compact_gym
python collect_demo_gym.py
```

**What to verify:**
- ✓ Robot initializes and homes
- ✓ SpaceMouse controls work smoothly
- ✓ Recording starts/stops correctly via GUI buttons
- ✓ Shutdown is clean (Ctrl+C)

### Demo Validation

```bash
# Validate most recent demo
python collection/validate_demo.py

# Validate specific demo
python collection/validate_demo.py --file data/gym_demos/demo_0.npz

# Validate all demos
python collection/validate_demo.py --all
```

**What it checks:**
- ✓ All expected fields present (state, actions, observations)
- ✓ Correct data shapes and types
- ✓ No NaN values in critical fields
- ✓ Recording frequency (~10 Hz)
- ✓ ArUco visibility statistics
- ✓ `ee_pose_target` is not all zeros

## Teleoperation Verification

```bash
python scripts/verify_teleop_gym.py
```

**What it tests:**
- Environment initialization
- SpaceMouse control (30 seconds)
- Action normalization
- Motor commands at 120Hz
- Clean shutdown

## Testing Checklist

### Startup
- [ ] Scene XML loads correctly
- [ ] Robot homes without errors
- [ ] USB latency auto-fixed (or warning shown)
- [ ] Camera initializes (or graceful failure)
- [ ] GUI displays "Ready" status

### Control
- [ ] SpaceMouse translation moves robot smoothly
- [ ] SpaceMouse rotation works correctly
- [ ] Gripper opens/closes with buttons
- [ ] No jitter or stuttering
- [ ] Control frequency ~120Hz (check profiling output)

### Recording
- [ ] Start Recording button starts recording (red status)
- [ ] Stop & Save button saves demo (green status)
- [ ] Stop & Discard button discards demo
- [ ] NPZ file created in `data/gym_demos/`
- [ ] Validation passes for recorded demo
- [ ] `smoothed_aruco_*` keys present in saved demo (in-place)

### Shutdown
- [ ] Close GUI window exits cleanly
- [ ] Ctrl+C triggers proper cleanup
- [ ] Robot returns to home
- [ ] Motors disable (torque off)
- [ ] No hanging processes

## Common Issues

### Robot moves 12x too fast
**Cause:** `env.step()` using wrong dt value
**Fix:** Verify `gym_env.py` uses `inner_control_frequency` (120Hz) not `control_frequency` (10Hz)

### Jittery motion
**Cause:** Motor commands running at 10Hz instead of 120Hz
**Fix:** See [RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #6 - Dual-frequency architecture

### USB latency warnings
**Cause:** USB latency timer at 16ms (default)
**Fix:** Run `python collection/fix_usb_latency.py` or it should auto-fix on startup

### Camera initialization fails
**Expected behavior:** System continues without camera (ArUco disabled)
**Fix:** Check `/dev/video*` devices, try different camera_id in `robot_config.py`

### Keyboard interrupt doesn't clean up robot
**Cause:** Missing finally block in outer cleanup
**Fix:** See [RUNTIME_FIXES.md](RUNTIME_FIXES.md) Issue #2 - Keyboard interrupt cleanup

## Performance Benchmarks

**Expected timings (from profiling):**
- Control loop: ~8ms avg @ 120Hz
- env.step(): 1-3ms (motor commands + IK)
- Encoder polling: 8-12ms @ 10Hz (outer loop only)
- ArUco detection: 30Hz (background thread)
- Recording: < 1ms (outer loop only)

**Frequencies:**
- Inner loop (motor commands): 120Hz
- Outer loop (input/recording): 10Hz
- ArUco thread: 30Hz
- Recorded data: 10Hz

## Validation Criteria

A valid demo should have:
- ✅ All expected fields present
- ✅ Trajectory length > 1 step
- ✅ Recording frequency ~10 Hz
- ✅ `ee_pose_target` non-zero (IK target being tracked)
- ✅ No NaN values in critical fields
- ✅ Camera frames recorded (if camera enabled)

Run `python collection/validate_demo.py --all` to check all collected demos.

## Development Testing

For development and debugging:

```bash
# Syntax verification (no hardware required)
python scripts/verify_encoder_syntax.py
python scripts/verify_aruco_thread_syntax.py

# Hardware tests
python scripts/test_encoder_polling.py
python scripts/test_env.py
```

See archived testing docs for Phase 1/2 history:
- [archived/TESTING.md](archived/TESTING.md) - Phase 1/2 verification scripts

## Troubleshooting

See [RUNTIME_FIXES.md](RUNTIME_FIXES.md) for known issues and solutions.

---

**Status**: ✅ All tests passing as of 2026-01-16
