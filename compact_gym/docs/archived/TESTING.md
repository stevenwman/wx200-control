# Testing Guide for compact_gym Implementation

## Overview

This directory contains verification and testing scripts for Phase 1 (Encoder Polling) and Phase 2 (ArUco Background Thread).

## Phase 1: Encoder Polling

### 1.1 Syntax Verification (No Hardware Required)

**Script:** `verify_encoder_syntax.py`

Checks Python syntax and implementation structure without requiring hardware or dependencies.

```bash
python verify_encoder_syntax.py
```

**What it checks:**
- ✓ Python syntax is valid
- ✓ `poll_encoders()` method exists with `outer_loop_start_time` parameter
- ✓ `get_encoder_state()` method exists
- ✓ Encoder state attributes in `RobotHardware.__init__`
- ✓ `poll_encoders()` called in `gym_env.step()`
- ✓ Encoder data added to info dict

**Expected output:**
```
======================================================================
✓ ALL SYNTAX CHECKS PASSED

Implementation structure verified:
  - robot_hardware.py: poll_encoders() and get_encoder_state() methods
  - gym_env.py: Integration with step() and info dict

Note: Runtime testing on hardware required for full verification.
======================================================================
```

### 1.2 Runtime Testing (Hardware Required)

**Script:** `test_encoder_polling.py`

Full runtime test that requires:
- Robot hardware connected
- Full Python environment (gymnasium, mujoco, opencv, scipy)

```bash
# Ensure you're in the correct Python environment
# conda activate ogpo  # or your environment name

python test_encoder_polling.py
```

**What it tests:**
- Hardware initialization
- 20 control steps with encoder polling
- Encoder poll performance metrics
- Encoder state updates
- Skipped read tracking

**Expected output:**
```
======================================================================
ENCODER POLLING TEST
======================================================================

[1/4] Creating environment...
[2/4] Resetting environment (initializing hardware)...
[3/4] Running 20 steps with encoder polling...
    Step  0: encoder_poll_count=1, step_time=105.2ms
    Step  5: encoder_poll_count=6, step_time=102.1ms
    ...

[4/4] Encoder Polling Statistics:
======================================================================
✓ Total encoder polls: 20
✓ Encoder states collected: 20
✓ Average encoder read time: 8.3ms
✓ Max encoder read time: 12.1ms
  → Performance: GOOD (avg < 15ms)
✓ Average polling frequency: 9.8Hz
  (Target: 10.0Hz)
✓ No skipped reads

TEST COMPLETE
```

## Phase 2: ArUco Background Thread

### 2.1 Syntax Verification (No Hardware Required)

**Script:** `verify_aruco_thread_syntax.py`

Checks that ArUco background thread is properly implemented.

```bash
python verify_aruco_thread_syntax.py
```

**What it checks:**
- ✓ Thread attributes in `__init__` (_aruco_polling_active, _aruco_lock, etc.)
- ✓ `_start_aruco_polling()` creates and starts thread
- ✓ `_aruco_poll_loop()` background thread method
- ✓ Thread-safe access in `_get_aruco_observations()`
- ✓ `close()` properly stops thread
- ✓ threading module imported
- ✓ RateLimiter available

**Expected output:**
```
======================================================================
✓ ALL CHECKS PASSED

ArUco background thread implementation verified:
  - Thread attributes initialized in __init__
  - _start_aruco_polling() creates and starts thread
  - _aruco_poll_loop() runs at camera FPS
  - _get_aruco_observations() uses thread-safe access
  - close() properly stops thread

Phase 2 implementation is structurally correct!
======================================================================
```

### 2.2 Runtime Testing (Hardware Required)

The same `test_encoder_polling.py` script now also tests ArUco background thread:
- Background thread starts automatically with camera
- ArUco observations updated at 30 Hz (camera FPS)
- Control loop reads observations at 10 Hz (non-blocking)
- Thread-safe access verified

## Running Without Full Environment

If you don't have the full Python environment (gymnasium, etc.), you can still verify the implementation:

1. **Syntax verification** (always works):
   ```bash
   python verify_encoder_syntax.py
   ```

2. **Manual code review**:
   - Check [robot_hardware.py](robot_hardware.py:159-261) for `poll_encoders()` implementation
   - Check [gym_env.py](gym_env.py:317-358) for integration

## Common Issues

### ImportError: No module named 'gymnasium'

This means you're not in the correct Python environment. The runtime test requires:

```bash
pip install gymnasium mujoco opencv-python scipy numpy
```

Or activate your existing environment:
```bash
conda activate ogpo  # or your environment name
```

### ModuleNotFoundError: No module named 'fix_gstreamer_env'

The `fix_gstreamer_env.py` file should exist in `compact_gym/`. If missing, copy it:

```bash
cp ../compact_code/fix_gstreamer_env.py .
```

### Hardware not connected

The runtime test (`test_encoder_polling.py`) requires the robot hardware to be connected. If you just want to verify the code structure, use `verify_encoder_syntax.py` instead.

## What's Been Verified

✅ **Syntax and structure** (via `verify_encoder_syntax.py`):
- All methods exist with correct signatures
- Integration points are in place
- No Python syntax errors

⏳ **Runtime behavior** (requires hardware):
- Encoder polling performance
- State updates
- Thread safety
- Real-world timing

## Next Steps

After verifying the implementation:

1. **Hardware testing**: Run `test_encoder_polling.py` with robot connected
2. **Data collection testing**: Test `collect_demo_gym.py` for full workflow
3. **Phase 2**: Implement ArUco background thread
4. **Phase 3**: Test complete data collection pipeline

## Files

- `verify_encoder_syntax.py` - Syntax/structure verification (no deps)
- `test_encoder_polling.py` - Full runtime test (hardware required)
- `ENCODER_POLLING_IMPLEMENTATION.md` - Implementation details
- `TESTING.md` - This file
