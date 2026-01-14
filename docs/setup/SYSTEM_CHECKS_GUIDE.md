# System Checks Integration Guide

## Overview

Your data collection script now automatically checks critical system settings before running:

✅ **GStreamer** - Camera pipeline availability
✅ **USB Latency Timer** - Dynamixel communication speed
✅ **Camera Device** - Camera hardware availability

## Usage

### Basic Usage (Recommended)
```bash
python compact_code/wx200_robot_collect_demo_encoders_compact.py
```

This will automatically:
1. Run system checks before starting
2. Display any warnings
3. Continue running (even if some checks fail)

### With Auto-Fix
```bash
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix
```

This will:
1. Run system checks
2. **Automatically fix USB latency** if it's not optimal (requires sudo password)
3. Continue running

### Skip Checks (Not Recommended)
```bash
python compact_code/wx200_robot_collect_demo_encoders_compact.py --skip-checks
```

Use this only if you're confident everything is configured correctly and want to save a few seconds at startup.

## Example Output

### All Checks Pass ✓
```
======================================================================
SYSTEM PRE-FLIGHT CHECKS
======================================================================

✓ GStreamer
  Current:  Available with v4l2src
  Expected: Available
  Details:  GStreamer GStreamer 1.24.2

✓ USB Latency Timer
  Current:  1ms
  Expected: 1ms
  Details:  Optimal for high-frequency control

✓ Camera Device
  Current:  /dev/video1
  Expected: Capture device
  Details:  Camera is a valid capture device

======================================================================
✓ ALL CHECKS PASSED (3/3)
======================================================================
```

### USB Latency Warning ⚠️
```
======================================================================
SYSTEM PRE-FLIGHT CHECKS
======================================================================

✓ GStreamer
  Current:  Available with v4l2src
  Expected: Available
  Details:  GStreamer GStreamer 1.24.2

✗ USB Latency Timer
  Current:  16ms
  Expected: 1ms
  Details:  High latency will cause slow encoder reads (16-20ms instead of 10-12ms)
  Fix:      echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

✓ Camera Device
  Current:  /dev/video1
  Expected: Capture device
  Details:  Camera is a valid capture device

======================================================================
⚠ SOME CHECKS FAILED (2/3 passed)

⚠️  CRITICAL WARNINGS:
  • USB Latency Timer: High latency will cause slow encoder reads (16-20ms instead of 10-12ms)
    Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer

⚠️  Robot will run but performance may be degraded.
   Encoder reads will be slow (16-20ms instead of 10-12ms).
======================================================================
```

**Note:** The script will continue running even with warnings. The warnings tell you what performance to expect.

## Standalone System Checks

You can also run system checks independently:

```bash
# Check all systems
python system_checks.py

# Check and auto-fix USB latency
python system_checks.py --auto-fix

# Check with custom camera ID
python system_checks.py --camera-id 2

# Exit with error if any check fails (useful for scripts)
python system_checks.py --require-all
```

## What Each Check Does

### 1. GStreamer Check
- Verifies GStreamer is available in Python
- Checks that `v4l2src` plugin exists (needed for camera access)
- Uses `fix_gstreamer_env.py` to configure paths automatically

**Fix:** Already integrated - GStreamer fix is permanent

### 2. USB Latency Timer Check
- Reads `/sys/bus/usb-serial/devices/ttyUSB0/latency_timer`
- Optimal value: **1ms** (fast encoder reads ~10-12ms)
- Default value: **16ms** (slow encoder reads ~16-20ms)

**Fix:**
```bash
echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

Or use `--auto-fix` flag.

### 3. Camera Device Check
- Verifies `/dev/video{id}` exists
- Checks it's a capture device (not metadata)
- Uses `v4l2-ctl` if available for detailed check

**Fix:** If this fails, check:
```bash
v4l2-ctl --list-devices
ls /dev/video*
```

## Integration Details

### In Your Data Collection Script

The checks run in `main()` before creating the controller:

```python
def main():
    # ... parse arguments ...

    # Run system pre-flight checks (unless skipped)
    if not args.skip_checks:
        from system_checks import run_system_checks, check_and_fix_usb_latency

        # Auto-fix USB latency if requested
        if args.auto_fix:
            check_and_fix_usb_latency(device='/dev/ttyUSB0', auto_fix=True)

        # Run all checks
        run_system_checks(
            camera_device_id=camera_id,
            usb_device='/dev/ttyUSB0',
            verbose=True,
            require_all=False  # Don't fail, just warn
        )

    # ... rest of main() ...
```

### Files Created

**New files:**
- `system_checks.py` - System check module
- `SYSTEM_CHECKS_GUIDE.md` - This guide

**Modified files:**
- `compact_code/wx200_robot_collect_demo_encoders_compact.py` - Added checks to `main()`

## Impact on Warnings

### Before Integration
You would see encoder warnings during runtime:
```
⚠️  ENCODER WARNING: Slow reads detected (avg=16.5ms, max=22.6ms)
```

But you wouldn't know why until you checked manually.

### After Integration
You'll see the warning **before** the robot starts:
```
⚠️  CRITICAL WARNINGS:
  • USB Latency Timer: High latency will cause slow encoder reads (16-20ms instead of 10-12ms)
    Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
```

So you can fix it immediately, or at least know what to expect.

## Making USB Latency Fix Permanent

The system checks will always warn you if USB latency isn't optimal. To make the fix permanent (survive reboots), see [ENCODER_WARNING_ANALYSIS.md](ENCODER_WARNING_ANALYSIS.md) for udev rule setup.

**Quick permanent fix:**
```bash
# Create udev rule
sudo nano /etc/udev/rules.d/99-usb-serial-latency.rules

# Add this line (replace vendor/product IDs with yours):
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", RUN+="/bin/sh -c 'echo 1 > /sys/bus/usb-serial/devices/%k/latency_timer'"

# Reload
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Find your IDs:
```bash
lsusb
udevadm info --name=/dev/ttyUSB0 | grep -E "ID_VENDOR_ID|ID_MODEL_ID"
```

## Command Reference

```bash
# Normal run with checks
python compact_code/wx200_robot_collect_demo_encoders_compact.py

# Auto-fix USB latency
python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix

# Skip checks (faster startup)
python compact_code/wx200_robot_collect_demo_encoders_compact.py --skip-checks

# Verbose profiling with checks
python compact_code/wx200_robot_collect_demo_encoders_compact.py --verbose

# Standalone system check
python system_checks.py

# Standalone check with auto-fix
python system_checks.py --auto-fix
```

## Summary

🎯 **What Changed:**
- Your data collection script now runs automatic system checks
- You'll be warned **before** running if USB latency is suboptimal
- You can use `--auto-fix` to fix USB latency automatically
- GStreamer is always configured correctly (permanent fix)

🚀 **Recommended Workflow:**
1. Run with `--auto-fix` the first time:
   ```bash
   python compact_code/wx200_robot_collect_demo_encoders_compact.py --auto-fix
   ```
2. If you see USB latency warnings frequently, set up the permanent udev rule
3. After that, just run normally (checks are fast, ~0.5 seconds)

✅ **Benefits:**
- No more mystery encoder warnings
- Catch configuration issues before they affect data collection
- Clear instructions on how to fix any issues
- Optional auto-fix for USB latency
