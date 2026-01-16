# Shutdown Fix for Test Script

## Problem

The Phase 1 test script (`test_encoder_polling.py`) was not properly cleaning up the robot hardware, requiring manual power-down after tests.

## Root Cause

The cleanup code `env.close()` was only in the normal execution path. If an exception occurred during testing, the cleanup would not run, leaving the robot motors energized.

## Solution

Wrapped the test code in a try/finally block to ensure cleanup ALWAYS runs:

```python
def test_encoder_polling():
    env = WX200GymEnv(...)

    try:
        # Test code here
        obs, info = env.reset()
        # ... run tests ...

    finally:
        # Cleanup - ALWAYS runs even if there's an error
        print()
        print("Cleaning up and shutting down robot...")
        env.close()
        print("✓ Robot shutdown complete")
```

## Shutdown Sequence

When `env.close()` is called, it triggers:

### 1. ArUco Thread Shutdown (`gym_env.py:473-476`)
```python
self._aruco_polling_active = False
if self._aruco_poll_thread is not None:
    self._aruco_poll_thread.join(timeout=1.0)
```

### 2. Camera Cleanup (`gym_env.py:478-480`)
```python
if self.camera:
    self.camera.stop()
```

### 3. Robot Hardware Shutdown (`gym_env.py:482-484`)
```python
if self.robot_hardware:
    self.robot_hardware.shutdown()
```

### 4. Hardware Shutdown Sequence (`robot_hardware.py:325-356`)

The robot follows a safe shutdown sequence:

1. **Reasonable Home** - Move to safe intermediate position
2. **Base Home** - Rotate base to center
3. **Folded Home** - Fold arm into compact position
4. **Disable Torque** - Turn off all motor torque
5. **Disconnect** - Close serial connection

Each move waits for a configured delay (`move_delay` in robot_config.py, default 2.5 seconds) to allow the movement to complete before proceeding.

**IMPORTANT**: The shutdown uses fixed time delays, NOT encoder-based completion detection. If the robot "slams" during shutdown, increase `move_delay` in robot_config.py.

```python
def shutdown(self):
    """Safe shutdown sequence."""
    print("\n[RobotHardware] Shutting down...")
    if self.robot_driver and self.robot_driver.connected:
        try:
            # 1. Reasonable Home
            reasonable_pose = list(robot_config.reasonable_home_pose)
            if len(reasonable_pose) > 6:
                reasonable_pose[6] = robot_config.gripper_encoder_max  # Open gripper
            self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, reasonable_pose) if pos != -1})
            time.sleep(robot_config.move_delay)

            # 2. Base Home
            base_pose = list(robot_config.base_home_pose)
            if len(base_pose) > 6:
                base_pose[6] = robot_config.gripper_encoder_max
            self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, base_pose) if pos != -1})
            time.sleep(robot_config.move_delay)

            # 3. Folded Home
            folded_pose = list(robot_config.folded_home_pose)
            if len(folded_pose) > 6:
                folded_pose[6] = robot_config.gripper_encoder_max
            self._failsafe_move({mid: pos for mid, pos in zip(robot_config.motor_ids, folded_pose) if pos != -1})
            time.sleep(robot_config.move_delay)

            # 4. Extra delay before disabling torque (critical!)
            print("Waiting for movement to complete before disabling torque...")
            time.sleep(robot_config.move_delay)

            # 5. Disable Torque
            self.robot_driver.disable_torque_all()
            self.robot_driver.disconnect()

        except Exception as e:
            print(f"Error during shutdown: {e}")
```

### 5. Release Hardware Authority (`gym_env.py:486-489`)
```python
if self.has_authority:
    WX200GymEnv._hardware_authority = None
    self.has_authority = False
```

### 6. Close OpenCV Windows (`gym_env.py:491`)
```python
cv2.destroyAllWindows()
```

## Failsafe Moves

All shutdown moves use `_failsafe_move()` which catches exceptions:

```python
def _failsafe_move(self, motor_positions):
    """Helper to move motors with broad try/catch."""
    try:
        self.robot_driver.send_motor_positions(motor_positions, velocity_limit=30)
    except Exception as e:
        print(f"Move failed: {e}")
```

This ensures that even if one move fails, subsequent moves are attempted.

## Error Handling

The test script also handles keyboard interrupts and exceptions:

```python
if __name__ == "__main__":
    try:
        test_encoder_polling()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
```

**Note**: The `finally` block inside `test_encoder_polling()` ensures cleanup happens regardless of how the function exits.

## Testing

To verify the shutdown works correctly:

```bash
# Normal completion
python compact_gym/test_encoder_polling.py

# Interrupt during test (Ctrl+C)
python compact_gym/test_encoder_polling.py
# Press Ctrl+C during execution
# Should still see "Cleaning up and shutting down robot..."
```

## Verification

After running the test, verify:
- ✓ Robot motors move through shutdown sequence
- ✓ Motors are de-energized (you can move them by hand)
- ✓ No need to manually power down robot
- ✓ Terminal shows "✓ Robot shutdown complete"

## Shutdown Timing Issue (Follow-up Fix)

### Problem
After initial fix, user reported: "the robot kinda slams a bit" during shutdown, suggesting torque was disabled before the folded_home movement completed. Testing with `move_delay=5.0` still showed the issue.

### Root Cause
The shutdown sequence was **MISSING A CRITICAL DELAY** that exists in the original `compact_code/robot_control/robot_shutdown.py`.

The original code has:
1. Reasonable Home + sleep
2. Base Home + sleep
3. Folded Home + sleep
4. **Extra sleep** ← MISSING!
5. Disable torque

The `compact_gym` implementation was immediately disabling torque after the folded_home sleep, without the extra delay.

### Solution
Added the missing extra sleep before disabling torque in `robot_hardware.py:354-356`:
```python
# 3. Folded Home
self._failsafe_move({...folded_pose...})
time.sleep(robot_config.move_delay)

# 4. Extra delay before disabling torque (critical for preventing slam!)
print("Waiting for movement to complete before disabling torque...")
time.sleep(robot_config.move_delay)  # ← THIS WAS MISSING!

# 5. Disable Torque
self.robot_driver.disable_torque_all()
```

This gives **TWO full delays** after the folded_home command before torque is disabled.

### Why This Works
The extra delay ensures that:
1. The robot has time to complete the folded_home movement
2. Any residual motion has stopped
3. The arm is stable before torque is removed
4. Gravity doesn't cause the arm to slam down when torque is disabled

With `move_delay=1.0`, this gives **2 full seconds** between the move command and torque disable.

### Testing
Run the test and verify the robot no longer "slams" during shutdown:
```bash
python compact_gym/test_encoder_polling.py
```

Watch the shutdown sequence and ensure all movements complete smoothly before torque is disabled.

## Files Modified

- `compact_gym/test_encoder_polling.py` - Added try/finally for guaranteed cleanup
- `compact_gym/robot_hardware.py` - Added missing extra delay before disabling torque (lines 354-356)

## Related Files

- `compact_gym/gym_env.py` - `close()` method (lines 471-491)
- `compact_gym/robot_hardware.py` - `shutdown()` method (lines 325-361)

---

**Fix Date:** 2026-01-14
**Issues:**
1. Robot not shutting down after tests
2. Robot "slamming" during shutdown (torque disabled too early)

**Solutions:**
1. Added try/finally block to guarantee cleanup
2. Added missing extra delay before disabling torque (matching original compact_code behavior)

**Status:** ✅ Fixed - shutdown now has proper double-delay before torque disable
